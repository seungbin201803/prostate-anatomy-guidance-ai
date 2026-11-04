"""
Train a channel-stacked mpMRI classifier (T2/DWI/ADC [+ segmentation mask] -> 3D ViT / ResNet ->
mean-pooled tokens -> CORAL or softmax head), from scratch or fine-tuned from a previous run.

From scratch (defaults are the recipe used for the internal dataset). Without --seg_dir the input is
T2/DWI/ADC only; with it, the case's mask is a 4th channel, its --seg_labels spread evenly over (0, 1]
(background 0 stays 0; e.g. 0,1,2,3,4,5 -> 0, 0.2, 0.4, 0.6, 0.8, 1.0):

    python train.py --data_csv data/processed/internal/dataset.csv \
        --label_csv data/csv/internal.csv --label_col pirads --classes 1,2,3,4,5 --loss coral \
        --model resnet18 --seg_dir data/processed/internal/seg --seg_labels 0,1,2,3,4,5 \
        --out_dir runs/scratch --run_name pirads_resnet18_mask

Fine-tune: the backbone is initialized from another run's backbone_<--finetune_ckpt>.pt (its
architecture, --img_size and segmentation channel are taken from it: --seg_dir is required if it has
one, and --seg_labels defaults to its labels); the head is always new:

    python train.py --finetune_from runs/scratch/pirads_resnet18_mask \
        --seg_dir data/processed/external/seg \
        --data_csv data/processed/external/dataset.csv --split_dir data/splits/external \
        --label_csv data/csv/external.csv --label_col pirads --classes 3,4,5 --loss coral \
        --epochs 50 --lr_backbone 2e-5 --weight_decay 1e-2 --warmup_epochs 2 \
        --early_stopping_min_delta 0.01 --early_stopping_min_epochs 10 \
        --out_dir runs/finetune/external --run_name pirads_resnet18_mask

Resume an interrupted run (every other argument comes from the run; --epochs may be raised):

    python train.py --resume runs/scratch/pirads_resnet18_mask

Outputs in <out_dir>/<run_name>/: log.txt, args.json, metrics.jsonl (one row per epoch),
tensorboard/, {backbone,heads}_{max_val_score,min_val,last}.pt, train_state_last.pt,
final_metrics.json and, once training ends, eval_<checkpoint>/predictions_<split>.csv + scores.json for
the best checkpoint on --eval_splits.
"""

import os
import json
import math
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pagai.utils import parse_tuple, parse_str_list, str2bool, log_print, format_duration
from pagai.data import MPMRDataset, seg_channel_encoding, compute_label_space
from pagai.models import MODEL_CHOICES, VIT_DEFAULTS, build_backbone
from pagai.tasks import LOSS_CHOICES, build_heads, selection_score, format_metrics
from pagai.checkpoint import CHECKPOINT_NAMES, model_config, describe_mask, save_pair, load_model
from pagai.engine import train_one_epoch, evaluate

# Kept from the command line on --resume; everything else comes from the run being resumed.
RESUME_KEEP_ARGS = {"resume", "epochs", "num_workers"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--data_csv", type=str, help="dataset.csv written by preprocess_data.py (see data/README.md)")
    g.add_argument("--image_dir", type=str, default=None, help="Root of the image paths in --data_csv. Default: its directory.")
    g.add_argument("--split_dir", type=str, default=None,
                   help="Directory with train.csv/val.csv/test.csv (case_id column). Default: --data_csv's split column.")
    g.add_argument("--label_csv", type=str, help="CSV with case_id and the --label_col column.")
    g.add_argument("--label_col", type=str, help="Label column = task name, e.g. pirads or cspca.")
    g.add_argument("--classes", type=parse_str_list, default=None,
                   help="Comma list of label values to use, in class order (the ordinal order for --loss coral), "
                        "e.g. 1,2,3,4,5. Rows with other values are dropped. Default: every value, sorted.")
    g.add_argument("--require_t2", type=str2bool, default=True, help="Drop cases without T2 (default true).")
    g.add_argument("--img_size", type=parse_tuple, default=(112, 112, 24), help="Volume size X,Y,Z on disk.")
    g.add_argument("--seg_dir", type=str, default=None,
                   help="Directory of {case_id}.nii.gz segmentation masks on the T2 grid (preprocess_data.py's seg/). "
                        "Given: the mask is a 4th input channel and cases without one are dropped. Default: no mask.")
    g.add_argument("--seg_labels", type=parse_tuple, default=None,
                   help="Comma list of the mask's integer labels, e.g. 0,1,2,3,4,5. Background 0 -> 0; the others, "
                        "in this order, -> k/K (here 0.2, 0.4, ..., 1.0). Required with --seg_dir, unless "
                        "fine-tuning (default: the --finetune_from backbone's labels).")

    g = p.add_argument_group("model")
    g.add_argument("--model", type=str, default=None, choices=MODEL_CHOICES)
    g.add_argument("--vit_patch_size", type=parse_tuple, default=VIT_DEFAULTS["vit_patch_size"])
    g.add_argument("--vit_hidden_size", type=int, default=VIT_DEFAULTS["vit_hidden_size"])
    g.add_argument("--vit_mlp_dim", type=int, default=VIT_DEFAULTS["vit_mlp_dim"])
    g.add_argument("--vit_num_layers", type=int, default=VIT_DEFAULTS["vit_num_layers"])
    g.add_argument("--vit_num_heads", type=int, default=VIT_DEFAULTS["vit_num_heads"])
    g.add_argument("--vit_dropout_rate", type=float, default=VIT_DEFAULTS["vit_dropout_rate"])
    g.add_argument("--hidden_dim", type=int, default=None, help="Head hidden size. Default: embed_dim // 2.")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--loss", type=str, default="coral", choices=LOSS_CHOICES,
                   help="coral: ordinal CORAL head (PI-RADS); ce: softmax cross-entropy (binary csPCa).")
    g.add_argument("--finetune_from", type=str, default=None,
                   help="Run directory to initialize the backbone from (its backbone_<--finetune_ckpt>.pt).")
    g.add_argument("--finetune_ckpt", type=str, default="max_val_score", choices=CHECKPOINT_NAMES)

    g = p.add_argument_group("optimization")
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--grad_accum_steps", type=int, default=16, help="Effective batch = batch_size * this.")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--lr_head", type=float, default=1e-3)
    g.add_argument("--lr_backbone", type=float, default=2e-4)
    g.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay.")
    g.add_argument("--warmup_epochs", type=float, default=0,
                   help="Linear warmup epochs, then cosine decay to 0 at --epochs. Negative: constant LR.")
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"], help="bf16 autocast (CUDA only).")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("early stopping")
    g.add_argument("--early_stopping_patience", type=int, default=15, help="0 disables early stopping.")
    g.add_argument("--early_stopping_metric", type=str, default="val_score", choices=["val_score", "val_loss"],
                   help="val_score: validation auc (binary) / qwk (multi-class), higher is better.")
    g.add_argument("--early_stopping_min_delta", type=float, default=0.005)
    g.add_argument("--early_stopping_min_epochs", type=int, default=0)

    g = p.add_argument_group("output")
    g.add_argument("--out_dir", type=str, default="runs")
    g.add_argument("--run_name", type=str, default=None)
    g.add_argument("--overwrite", action="store_true", help="Allow writing into an existing run directory.")
    g.add_argument("--eval_splits", type=parse_str_list, default=("val", "test"),
                   help="Splits to predict with the best checkpoint once training ends.")
    g.add_argument("--resume", type=str, default=None, help="Run directory to resume (needs train_state_last.pt).")
    return p, p.parse_args()


def setup_new_run(parser, args):
    for key in ["data_csv", "label_csv", "label_col", "run_name"]:
        if not getattr(args, key):
            parser.error(f"--{key} is required")
    args.image_dir = args.image_dir or os.path.dirname(os.path.abspath(args.data_csv))
    if args.seg_labels and not args.seg_dir:
        parser.error("--seg_labels needs --seg_dir")
    if args.seg_dir and not os.path.isdir(args.seg_dir):
        parser.error(f"--seg_dir {args.seg_dir} not found")
    try:
        encoding = seg_channel_encoding(args.seg_labels) if args.seg_labels else None
    except ValueError as e:
        parser.error(str(e))

    if args.finetune_from:
        path = os.path.join(args.finetune_from, f"backbone_{args.finetune_ckpt}.pt")
        if not os.path.isfile(path):
            parser.error(f"--finetune_from: {path} not found")
        src = model_config(torch.load(path, map_location="cpu", weights_only=False)["args"])
        if args.model and args.model != src["model"]:
            parser.error(f"--model {args.model} differs from the --finetune_from backbone ({src['model']})")
        args.model = src["model"]
        args.img_size = src["img_size"]
        for key, value in src["vit_kwargs"].items():
            setattr(args, key, value)
        src_encoding = (src["mask_labels"], src["mask_values"]) if src["mask_labels"] else None
        if src_encoding and not args.seg_dir:
            parser.error(f"the --finetune_from backbone was trained with a {describe_mask(src)}; pass --seg_dir")
        if args.seg_dir and not src_encoding:
            parser.error("the --finetune_from backbone was trained without a segmentation channel; drop --seg_dir")
        if src_encoding and encoding is None:
            encoding = src_encoding
        elif src_encoding and not (encoding[0] == src_encoding[0]
                                   and all(abs(a - b) < 1e-6 for a, b in zip(encoding[1], src_encoding[1]))):
            parser.error(f"--seg_labels {args.seg_labels} must encode the mask like the --finetune_from backbone "
                         f"({describe_mask(src)}); omit --seg_labels to use its labels")
    elif not args.model:
        parser.error("--model is required (unless --finetune_from or --resume)")
    elif args.seg_dir and encoding is None:
        parser.error("--seg_dir needs --seg_labels, e.g. --seg_labels 0,1,2,3,4,5")

    # Saved with the run's args; checkpoint.model_config rebuilds the input channels from them.
    args.mask_labels, args.mask_values = encoding if encoding else (None, None)
    if encoding and not args.seg_labels:  # inherited from --finetune_from: record it in args.json too
        args.seg_labels = (0, *encoding[0])

    run_dir = os.path.join(args.out_dir, args.run_name)
    if os.path.isfile(os.path.join(run_dir, "train_state_last.pt")) and not args.overwrite:
        parser.error(f"{run_dir} already holds a run; use --resume {run_dir}, another --run_name, or --overwrite")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_dataset(args, split, classes, require_label=True):
    return MPMRDataset(
        data_csv=args.data_csv,
        image_dir=args.image_dir,
        img_size=args.img_size,
        split=split,
        split_csv=os.path.join(args.split_dir, f"{split}.csv") if args.split_dir else None,
        label_csv=args.label_csv,
        label_col=args.label_col,
        classes=classes,
        require_label=require_label,
        require_t2=args.require_t2,
        mask_dir=args.seg_dir,
        mask_labels=args.mask_labels,
        mask_values=args.mask_values,
    )


def build_scheduler(optimizer, args, steps_per_epoch):
    """Linear warmup over --warmup_epochs, then cosine decay to 0 at --epochs (per optimizer step)."""
    if args.warmup_epochs < 0:
        return None
    total = max(1, args.epochs * steps_per_epoch)
    warmup = round(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_final_eval(args, run_dir, classes, device, log_path):
    """Predicts --eval_splits with the best checkpoint (max_val_score, else min_val)."""
    for name in ["max_val_score", "min_val"]:
        heads_path = os.path.join(run_dir, f"heads_{name}.pt")
        if os.path.isfile(heads_path):
            break
    else:
        log_print("no best checkpoint saved, skipping the final evaluation", log_path)
        return

    backbone, heads, info = load_model(heads_path)
    backbone.to(device)
    heads.to(device)
    out_dir = os.path.join(run_dir, f"eval_{name}")
    os.makedirs(out_dir, exist_ok=True)
    log_print(f"\nFinal evaluation with {name} (epoch {info['epoch']}) -> {out_dir}", log_path)

    scores = {"checkpoint": name, "epoch": info["epoch"]}
    for split in args.eval_splits:
        ds = make_dataset(args, split, classes, require_label=False)
        if len(ds) == 0:
            log_print(f"  [{split}] no cases", log_path)
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        metrics, preds = evaluate(backbone, heads, info["task"], classes, loader, device, args.precision)
        preds.to_csv(os.path.join(out_dir, f"predictions_{split}.csv"), index=False)
        scores[split] = metrics
        log_print(f"  [{split}] loss={metrics['loss']:.4f} {format_metrics(metrics)}", log_path)
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)


def main():
    parser, args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = None
    if args.resume:
        state = torch.load(os.path.join(args.resume, "train_state_last.pt"), map_location="cpu", weights_only=False)
        for key, value in state["args"].items():
            if key not in RESUME_KEEP_ARGS:
                setattr(args, key, value)
        if not args.seg_dir and state["args"].get("mask_dir"):  # run from before --seg_dir replaced --mask_dir
            args.seg_dir = state["args"]["mask_dir"]
        run_dir = args.resume
    else:
        run_dir = setup_new_run(parser, args)

    log_path = os.path.join(run_dir, "log.txt")
    if state is None:
        log_print("\n".join(f"--{k} {v}" for k, v in vars(args).items()), log_path, mode="w")
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    else:
        log_print(f"\nResuming {run_dir} after epoch {state['epoch']} (--epochs {args.epochs})", log_path)
    log_print(f"device: {device}", log_path)

    torch.manual_seed(args.seed + (state["epoch"] if state else 0))

    # ------------------------------------------------------------ data
    classes = compute_label_space(args.label_csv, args.label_col, args.classes)
    num_classes = len(classes)
    task = args.label_col
    log_print(f"task {task!r}: {num_classes} classes {classes} (loss {args.loss})", log_path)

    train_set = make_dataset(args, "train", classes)
    val_set = make_dataset(args, "val", classes)
    for ds in (train_set, val_set):
        ds.log_summary(log_path)
    if len(train_set) == 0 or len(val_set) == 0:
        raise SystemExit("empty train or val split; check --data_csv/--split_dir/--label_csv")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if len(train_loader) == 0:
        raise SystemExit(f"train split has {len(train_set)} cases, fewer than --batch_size {args.batch_size}")

    # ------------------------------------------------------------ model
    in_channels = 3 + (1 if args.mask_labels else 0)
    vit_kwargs = {k: getattr(args, k) for k in VIT_DEFAULTS}
    backbone = build_backbone(args.model, args.img_size, in_channels, vit_kwargs).to(device)
    heads = build_heads(backbone.embed_dim, task, num_classes, args.hidden_dim, args.dropout, args.loss).to(device)

    if state is not None:
        backbone.load_state_dict(torch.load(os.path.join(run_dir, "backbone_last.pt"), map_location="cpu",
                                            weights_only=False)["model_state_dict"])
        heads.load_state_dict(torch.load(os.path.join(run_dir, "heads_last.pt"), map_location="cpu",
                                         weights_only=False)["heads_state_dict"])
    elif args.finetune_from:
        path = os.path.join(args.finetune_from, f"backbone_{args.finetune_ckpt}.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt["model_state_dict"])
        log_print(f"backbone initialized from {path} (epoch {ckpt['epoch']}); new {task!r} head", log_path)

    n_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in heads.parameters())
    mask_info = describe_mask({"mask_labels": args.mask_labels, "mask_values": args.mask_values})
    log_print(f"model {args.model}, {in_channels} input channels ({mask_info}), {n_params / 1e6:.2f}M parameters",
              log_path)

    optimizer = torch.optim.AdamW([
        {"params": heads.parameters(), "lr": args.lr_head, "name": "head"},
        {"params": backbone.parameters(), "lr": args.lr_backbone, "name": "backbone"},
    ], weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))

    start_epoch, global_step = 1, 0
    best = {"val_loss": float("inf"), "val_score": float("-inf")}
    stop = {"best": None, "epoch": None, "since": 0}
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
            # The optimizer's saved LR follows the old schedule; recompute it (--epochs may have changed).
            for group, base, lam in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
                group["lr"] = base * lam(scheduler.last_epoch)
        start_epoch, global_step, best, stop = state["epoch"] + 1, state["global_step"], state["best"], state["stop"]

    # ------------------------------------------------------------ train
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    maximize = args.early_stopping_metric == "val_score"
    status = "completed"
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        train_metrics, global_step = train_one_epoch(
            backbone, heads, task, train_loader, optimizer, scheduler, device, args,
            writer=writer, global_step=global_step,
        )
        val_metrics, val_preds = evaluate(backbone, heads, task, classes, val_loader, device, args.precision)
        val_score = selection_score(val_metrics, num_classes)

        lrs = " ".join(f"lr_{g['name']}={g['lr']:.2e}" for g in optimizer.param_groups)
        log_print(
            f"Epoch {epoch}/{args.epochs} | train loss={train_metrics['loss']:.4f} {format_metrics(train_metrics)} | "
            f"val loss={val_metrics['loss']:.4f} {format_metrics(val_metrics)} score={val_score:.4f} | {lrs} | "
            f"{format_duration(time.time() - t_epoch)} (total {format_duration(time.time() - t_start)})",
            log_path,
        )
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics, "val_score": val_score}) + "\n")
        for split, m in [("train", train_metrics), ("val", val_metrics)]:
            for key, value in m.items():
                if key != "n":
                    writer.add_scalar(f"{key}/{split}", value, epoch)

        if val_metrics["loss"] < best["val_loss"]:
            best["val_loss"] = val_metrics["loss"]
            save_pair(run_dir, "min_val", backbone, heads, epoch, val_metrics, task, classes, args)
            log_print(f"  saved min_val (epoch {epoch})", log_path)
        if val_score > best["val_score"]:  # nan never compares greater
            best["val_score"] = val_score
            save_pair(run_dir, "max_val_score", backbone, heads, epoch, val_metrics, task, classes, args)
            val_preds.to_csv(os.path.join(run_dir, "val_predictions_max_val_score.csv"), index=False)
            log_print(f"  saved max_val_score (epoch {epoch}, score {val_score:.4f})", log_path)

        value = val_score if maximize else val_metrics["loss"]
        reference = stop["best"] if stop["best"] is not None else (float("-inf") if maximize else float("inf"))
        improvement = value - reference if maximize else reference - value
        if improvement > args.early_stopping_min_delta:
            stop.update(best=value, epoch=epoch, since=0)
        else:
            stop["since"] += 1

        save_pair(run_dir, "last", backbone, heads, epoch, val_metrics, task, classes, args)
        state_path = os.path.join(run_dir, "train_state_last.pt")
        torch.save({
            "epoch": epoch, "global_step": global_step, "best": best, "stop": stop, "args": vars(args),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
        }, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

        if 0 < args.early_stopping_patience <= stop["since"] and epoch >= args.early_stopping_min_epochs:
            log_print(f"Early stopping at epoch {epoch}: {args.early_stopping_metric} has not improved by more than "
                      f"{args.early_stopping_min_delta:g} for {stop['since']} epochs (best {stop['best']:.4f} at "
                      f"epoch {stop['epoch']})", log_path)
            status = "early_stopped"
            break

    writer.close()
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump({"status": status, "best": best, "early_stopping": stop}, f, indent=2)

    run_final_eval(args, run_dir, classes, device, log_path)


if __name__ == "__main__":
    main()
