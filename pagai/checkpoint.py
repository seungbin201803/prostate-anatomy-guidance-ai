"""
Checkpoints come in pairs written to a run directory:

    backbone_<name>.pt  {"epoch", "model_state_dict", "args"}
    heads_<name>.pt     {"epoch", "heads_state_dict", "metrics", "num_classes_dict", "label_mapping_dict", "args"}

<name> is max_val_score (best validation auc/qwk), min_val (lowest validation loss) or last; "args" is
the training script's arguments, from which the architecture is rebuilt. The same layout (and its
argument names --ve_name/--seg_dir/--seg_labels/--seg_label_values) was used by the original research
code, so its checkpoints load here too.
"""

import os

import torch

from .models import MODEL_CHOICES, LEGACY_MODEL_NAMES, VIT_DEFAULTS, build_backbone
from .tasks import build_heads
from .utils import whitelist_str

CHECKPOINT_NAMES = ("max_val_score", "min_val", "last")


def save_backbone(path, backbone, epoch, args):
    torch.save({"epoch": epoch, "model_state_dict": backbone.state_dict(), "args": vars(args)}, path)


def save_heads(path, heads, epoch, metrics, task, classes, args):
    torch.save(
        {
            "epoch": epoch,
            "heads_state_dict": heads.state_dict(),
            "metrics": metrics,
            "num_classes_dict": {task: len(classes)},
            "label_mapping_dict": {task: {"column": task, "csv": args.label_csv, "classes": list(classes)}},
            "args": vars(args),
        },
        path,
    )


def save_pair(run_dir, name, backbone, heads, epoch, metrics, task, classes, args):
    """Writes both files as .tmp first, then renames, so a kill never leaves a half-written or mixed pair."""
    paths = [os.path.join(run_dir, f"{kind}_{name}.pt") for kind in ("backbone", "heads")]
    save_backbone(paths[0] + ".tmp", backbone, epoch, args)
    save_heads(paths[1] + ".tmp", heads, epoch, metrics, task, classes, args)
    for path in paths:
        os.replace(path + ".tmp", path)


def model_config(saved):
    """Normalized architecture/input config from a checkpoint's saved args (this repo's or legacy names)."""
    model = saved.get("model") or LEGACY_MODEL_NAMES.get(saved.get("ve_name"), saved.get("ve_name"))
    if model not in MODEL_CHOICES:
        raise ValueError(f"checkpoint model {model!r} is not supported here ({MODEL_CHOICES})")

    if "mask_labels" in saved:  # this repo
        mask_labels, mask_values = saved.get("mask_labels"), saved.get("mask_values")
    elif saved.get("seg_dir"):  # legacy
        mask_labels = saved.get("seg_labels")
        mask_values = saved.get("seg_label_values") or (1.0,) * len(mask_labels)
    else:
        mask_labels = mask_values = None
    mask_labels = tuple(int(v) for v in mask_labels) if mask_labels else None
    mask_values = tuple(float(v) for v in mask_values) if mask_values else None

    return {
        "model": model,
        "img_size": tuple(saved["img_size"]),
        "vit_kwargs": {k: saved.get(k, v) for k, v in VIT_DEFAULTS.items()},
        "mask": saved.get("mask") or ("custom" if mask_labels else "none"),
        "mask_labels": mask_labels,
        "mask_values": mask_values,
        "hidden_dim": saved.get("hidden_dim"),
        "dropout": saved.get("dropout", 0.1),
        "loss": saved.get("loss", "coral"),
    }


def build_backbone_from_config(cfg):
    in_channels = 3 + (1 if cfg["mask_labels"] else 0)
    return build_backbone(cfg["model"], cfg["img_size"], in_channels, cfg["vit_kwargs"])


def resolve_heads_path(checkpoint, name="max_val_score"):
    """
    checkpoint: a heads_*.pt file, or a run directory (then heads_<name>.pt in it). max_val_score falls back
    to min_val, which is all a run has when its validation score was never defined (e.g. one-class val set).
    """
    if not os.path.isdir(checkpoint):
        return checkpoint
    path = os.path.join(checkpoint, f"heads_{name}.pt")
    fallback = os.path.join(checkpoint, "heads_min_val.pt")
    if name == "max_val_score" and not os.path.isfile(path) and os.path.isfile(fallback):
        print(f"{path} not found (validation score never defined), using {fallback}", flush=True)
        return fallback
    return path


def load_model(heads_path, backbone_path=None, map_location="cpu"):
    """
    Rebuilds backbone + heads from a heads_*.pt and its backbone_*.pt (default: the one next to it with
    the same suffix). Returns (backbone, heads, info); info has task, classes, epoch and model_config's keys.
    """
    heads_ckpt = torch.load(heads_path, map_location=map_location, weights_only=False)
    if backbone_path is None:
        name = os.path.basename(heads_path).replace("heads_", "backbone_", 1)
        backbone_path = os.path.join(os.path.dirname(heads_path), name)
    backbone_ckpt = torch.load(backbone_path, map_location=map_location, weights_only=False)

    tasks = list(heads_ckpt["num_classes_dict"])
    if len(tasks) != 1:
        raise ValueError(f"{heads_path} has {len(tasks)} heads {tasks}; only single-task checkpoints are supported")
    task = tasks[0]
    # whitelist_str: legacy checkpoints may store float-formatted classes ("3.0")
    classes = [whitelist_str(c) for c in heads_ckpt["label_mapping_dict"][task]["classes"]]

    cfg = model_config(backbone_ckpt["args"])
    head_cfg = model_config(heads_ckpt["args"])
    backbone = build_backbone_from_config(cfg)
    backbone.load_state_dict(backbone_ckpt["model_state_dict"])
    heads = build_heads(backbone.embed_dim, task, len(classes), head_cfg["hidden_dim"], head_cfg["dropout"], head_cfg["loss"])
    heads.load_state_dict(heads_ckpt["heads_state_dict"])

    info = {**cfg, "loss": head_cfg["loss"], "task": task, "classes": classes, "epoch": heads_ckpt["epoch"],
            "heads_path": heads_path, "backbone_path": backbone_path}
    return backbone, heads, info
