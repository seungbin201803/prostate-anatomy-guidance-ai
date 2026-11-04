"""
Run a trained checkpoint on any preprocessed dataset, with or without labels.

    python inference.py --checkpoint runs/scratch/pirads_resnet18_5zone \
        --data_csv data/processed/external/dataset.csv \
        --label_csv data/csv/external.csv --label_col pirads \
        --out_dir results/pirads_resnet18_5zone__external

--checkpoint is a heads_*.pt file or a run directory (then heads_<--ckpt>.pt). The architecture, input
size, mask channel and classes come from the checkpoint. A checkpoint trained with a zone mask needs the
masks of the new data (--mask_dir, default <image_dir>/zones2 or zones5); one trained without must not
get one.

Labels are optional: every case gets a prediction; cases whose label is one of the checkpoint's classes
are also scored (other label values count as unlabeled). --label_col defaults to the checkpoint's task
name.

Outputs in --out_dir, per split (--splits all = every row of --data_csv, no split needed):
    predictions_<split>.csv   case_id, <task>_gt, <task>_pred and class probabilities
    confusion_<split>.csv/png with labels: rows ground truth, columns prediction
    scores.json               with labels: n, accuracy, mae, mse, qwk (+ auc if binary)
    inference_log.txt
"""

import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from pagai.utils import parse_str_list, str2bool, log_print
from pagai.data import MPMRDataset
from pagai.checkpoint import CHECKPOINT_NAMES, resolve_heads_path, load_model
from pagai.engine import evaluate
from pagai.tasks import format_metrics


def plot_confusion(cm, classes, title, path):
    """Row-normalized confusion matrix with counts."""
    row_sums = cm.sum(axis=1, keepdims=True)
    ratio = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    size = max(5, len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{ratio[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if ratio[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="heads_*.pt, or a run directory.")
    p.add_argument("--ckpt", default="max_val_score", choices=CHECKPOINT_NAMES,
                   help="Which checkpoint of a run directory. Default: max_val_score.")
    p.add_argument("--data_csv", required=True)
    p.add_argument("--image_dir", default=None, help="Default: --data_csv's directory.")
    p.add_argument("--splits", type=parse_str_list, default=("all",),
                   help="'all' or a comma list of train,val,test (needs --split_dir or a split column).")
    p.add_argument("--split_dir", default=None)
    p.add_argument("--label_csv", default=None)
    p.add_argument("--label_col", default=None, help="Default: the checkpoint's task name.")
    p.add_argument("--mask_dir", default=None)
    p.add_argument("--require_t2", type=str2bool, default=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    if args.splits != ("all",) and set(args.splits) - {"train", "val", "test"}:
        p.error(f"--splits must be 'all' or from train,val,test, got {args.splits}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.image_dir = args.image_dir or os.path.dirname(os.path.abspath(args.data_csv))

    backbone, heads, info = load_model(resolve_heads_path(args.checkpoint, args.ckpt))
    backbone.to(device)
    heads.to(device)
    task, classes = info["task"], info["classes"]

    if info["mask_labels"] and not args.mask_dir:
        default = os.path.join(args.image_dir, f"zones{len(info['mask_labels'])}")
        if not os.path.isdir(default):
            p.error(f"the checkpoint uses a {info['mask']} zone-mask channel; pass --mask_dir ({default} not found)")
        args.mask_dir = default
    if args.mask_dir and not info["mask_labels"]:
        p.error("--mask_dir given, but the checkpoint was trained without a zone mask")
    label_col = args.label_col or task if args.label_csv else None
    if label_col and label_col not in pd.read_csv(args.label_csv, nrows=0).columns:
        if args.label_col:
            p.error(f"--label_csv {args.label_csv} has no column {label_col!r}")
        print(f"{args.label_csv} has no {label_col!r} column: predicting without labels", flush=True)
        args.label_csv = label_col = None

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "inference_log.txt")
    log_print("\n".join(f"--{k} {v}" for k, v in vars(args).items()), log_path, mode="w")
    log_print(f"model {info['model']} (mask {info['mask']}), task {task!r} classes {classes}, loss {info['loss']}, "
              f"epoch {info['epoch']}\n  {info['heads_path']}\n  {info['backbone_path']}", log_path)

    scores = {}
    for split in args.splits:
        ds = MPMRDataset(
            data_csv=args.data_csv,
            image_dir=args.image_dir,
            img_size=info["img_size"],
            split=split,
            split_csv=os.path.join(args.split_dir, f"{split}.csv") if args.split_dir and split != "all" else None,
            label_csv=args.label_csv,
            label_col=label_col,
            classes=classes,
            require_label=False,
            require_t2=args.require_t2,
            mask_dir=args.mask_dir,
            mask_labels=info["mask_labels"],
            mask_values=info["mask_values"],
        )
        ds.log_summary(log_path)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        metrics, preds = evaluate(backbone, heads, task, classes, loader, device, args.precision)
        preds.to_csv(os.path.join(args.out_dir, f"predictions_{split}.csv"), index=False)
        log_print(f"[{split}] {len(preds)} cases -> predictions_{split}.csv", log_path)

        if metrics["n"] > 0:
            scores[split] = metrics
            log_print(f"[{split}] {format_metrics(metrics)}", log_path)
            labeled = preds[f"{task}_gt"] != ""
            to_idx = {c: i for i, c in enumerate(classes)}
            cm = confusion_matrix(preds.loc[labeled, f"{task}_gt"].map(to_idx), preds.loc[labeled, f"{task}_pred"].map(to_idx),
                                  labels=list(range(len(classes))))
            np.savetxt(os.path.join(args.out_dir, f"confusion_{split}.csv"), cm, fmt="%d", delimiter=",",
                       header=",".join(f"pred_{c}" for c in classes) + "  (rows: gt " + ",".join(classes) + ")")
            title = f"{task} ({split}, n={metrics['n']})\n" + format_metrics(metrics).split(" ", 1)[1]
            plot_confusion(cm, classes, title, os.path.join(args.out_dir, f"confusion_{split}.png"))

    if scores:
        with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
    log_print(f"done -> {args.out_dir}", log_path)


if __name__ == "__main__":
    main()
