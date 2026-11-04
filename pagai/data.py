"""
Dataset for the channel-stacked classifier: per case, T2/DWI/ADC (and optionally a segmentation mask)
on a common grid (img_size, 1x1x3 mm in our experiments, written by preprocess_data.py) are
z-scored and stacked as channels -> [3 or 4, X, Y, Z]. No augmentation.

Data CSV (--data_csv, see README.md, section 1. Data):
    case_id, t2_path, dwi_path, adc_path   paths relative to --image_dir; a blank DWI/ADC cell (or a
                                           missing file) becomes an all-zero channel
    split                                  only when no --split_dir is given
Segmentation masks (--seg_dir): {case_id}.nii.gz on the case's T2 grid, integer labels (--seg_labels).
Labels (--label_csv): case_id plus one column per task.
"""

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .utils import ID_COL, read_table, log_print, whitelist_str

MODALITIES = ("t2", "dwi", "adc")

def seg_channel_encoding(seg_labels):
    """
    --seg_labels (e.g. 0,1,2,3,4,5) -> (labels, values) of the segmentation channel. Background 0 is
    always 0 (listing it is optional). The other labels, in the order given, are spread evenly over
    (0, 1]: the k-th of K becomes k/K, so every label is nonzero and differs from background.
        0,1,2,3,4,5 -> labels (1, 2, 3, 4, 5), values (0.2, 0.4, 0.6, 0.8, 1.0)
        0,1,2       -> labels (1, 2), values (0.5, 1.0)
    """
    labels = [int(v) for v in seg_labels]
    if len(set(labels)) != len(labels):
        raise ValueError(f"--seg_labels {tuple(labels)} has duplicates")
    if any(v < 0 for v in labels):
        raise ValueError(f"--seg_labels {tuple(labels)} must be non-negative integers")
    foreground = [v for v in labels if v != 0]
    if not foreground:
        raise ValueError("--seg_labels needs at least one label other than background 0")
    k = len(foreground)
    return tuple(foreground), tuple(i / k for i in range(1, k + 1))


# ============================================================
# Volume IO
# ============================================================

def robust_zscore(arr):
    """
    Clips to the 0.5-99.5 percentile of the nonzero finite voxels, then z-scores with their mean/std
    (std floored at 1e-6 -> 1). Falls back to all finite voxels when fewer than 100 are nonzero.
    """
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    mask = finite & (np.abs(arr) > 1e-6)
    if mask.sum() < 100:
        mask = finite

    vals = arr[mask]
    if vals.size < 100:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(vals, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    mean, std = arr[mask].mean(), arr[mask].std()
    if std < 1e-6:
        std = 1.0
    arr = np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def load_volume(path, expected_shape):
    """NIfTI -> [1, X, Y, Z] z-scored float tensor. No reorientation, no resampling."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape} for {path}")
    arr = np.transpose(arr, (2, 1, 0))  # -> [X, Y, Z]
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"{path} has shape {tuple(arr.shape)}, expected {tuple(expected_shape)} (--img_size); "
            f"run preprocess_data.py with the same --size"
        )
    return torch.from_numpy(robust_zscore(arr)[None])


def load_mask_on_grid(mask_path, ref_path, labels, values):
    """
    Segmentation mask already on ref_path's (the T2's) voxel grid -> [1, X, Y, Z] float tensor with label
    labels[i] set to values[i] and everything else 0. Raises ValueError if the mask is not on that grid
    or contains a nonzero label outside `labels`.
    """
    ref = sitk.ImageFileReader()
    ref.SetFileName(ref_path)
    ref.ReadImageInformation()

    mask = sitk.ReadImage(mask_path)
    if not (
        mask.GetSize() == ref.GetSize()
        and np.allclose(mask.GetSpacing(), ref.GetSpacing(), atol=1e-2)
        and np.allclose(mask.GetOrigin(), ref.GetOrigin(), atol=1.0)
        and np.allclose(mask.GetDirection(), ref.GetDirection(), atol=1e-2)
    ):
        raise ValueError(
            f"mask {mask_path} is not on the grid of {ref_path}: size {mask.GetSize()} vs {ref.GetSize()}, "
            f"spacing {mask.GetSpacing()} vs {ref.GetSpacing()}, origin {mask.GetOrigin()} vs {ref.GetOrigin()}"
        )

    arr = sitk.GetArrayFromImage(mask)  # [Z, Y, X]
    unexpected = sorted(set(np.unique(arr).tolist()) - {0} - set(labels))
    if unexpected:
        raise ValueError(f"mask {mask_path} has labels {unexpected} outside --seg_labels {(0, *labels)}")

    out = np.zeros(arr.shape, dtype=np.float32)
    for label, value in zip(labels, values):
        out[arr == label] = value
    return torch.from_numpy(np.ascontiguousarray(np.transpose(out, (2, 1, 0)))[None])


def is_missing_path(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip() == ""


# ============================================================
# Labels
# ============================================================

def _class_sort_key(v):
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v)


def compute_label_space(label_csv, label_col, classes=None):
    """
    Class list (canonical strings, class index = position) of label_col in label_csv: `classes` in the
    given order if set (other values are then treated as unlabeled), else every distinct value sorted
    (numerically when numeric). For --loss coral the order is the ordinal order.
    """
    df = read_table(label_csv, required=[label_col])
    values = df[label_col].dropna().map(whitelist_str)
    if classes:
        classes = [whitelist_str(c) for c in classes]
        absent = [c for c in classes if c not in set(values)]
        if absent:
            print(f"warning: classes {absent} of {label_col!r} have no rows in {label_csv}", flush=True)
        return classes
    return sorted(values.unique().tolist(), key=_class_sort_key)


# ============================================================
# Dataset
# ============================================================

class MPMRDataset(Dataset):
    """
    One split of --data_csv, joined with an optional label column. Items:
        image       [3 or 4, X, Y, Z] float, channels T2, DWI, ADC (, segmentation mask)
        label       class index, -1 if unlabeled
        label_raw   raw label as a string ('' if none)
        case_id
    Rows are excluded, in this order (each step is counted and logged):
        1. require_t2: no T2 (blank path or no file)
        2. mask_dir: no {case_id}.nii.gz segmentation mask
        3. require_label: no label, or a label outside `classes`
    split: 'train'/'val'/'test' (rows of split_csv's case_ids if given, else of the `split` column),
    or 'all' (every row).
    """

    def __init__(
        self,
        data_csv,
        image_dir,
        img_size,
        split="all",
        split_csv=None,
        label_csv=None,
        label_col=None,
        classes=None,
        require_label=True,
        require_t2=True,
        mask_dir=None,
        mask_labels=None,
        mask_values=None,
    ):
        self.image_dir = image_dir
        self.img_size = tuple(img_size)
        self.split = split
        self.mask_dir = mask_dir
        self.mask_labels = tuple(mask_labels) if mask_labels else None
        self.mask_values = tuple(mask_values) if mask_values else None
        if mask_dir and not self.mask_labels:
            raise ValueError("mask_labels are required with mask_dir")
        self.label_col = label_col
        self.class_to_idx = {c: i for i, c in enumerate(classes or [])}
        self.steps = []

        df = read_table(data_csv, required=[f"{m}_path" for m in MODALITIES])
        if split_csv is not None:
            ids = set(read_table(split_csv)[ID_COL])
            df = df[df[ID_COL].isin(ids)]
        elif split != "all":
            if "split" not in df.columns:
                raise ValueError(f"{data_csv} has no 'split' column; pass --split_dir")
            df = df[df["split"] == split]
        df = df.reset_index(drop=True)
        self.n_start = len(df)

        for m in MODALITIES:  # full path if the file exists, else None
            full = [None if is_missing_path(p) else os.path.join(image_dir, str(p)) for p in df[f"{m}_path"]]
            df[f"_{m}_file"] = pd.Series([p if p and os.path.isfile(p) else None for p in full], index=df.index, dtype=object)

        if label_csv is not None:
            labels = read_table(label_csv, required=[label_col])[[ID_COL, label_col]].dropna()
            labels = labels.drop_duplicates(ID_COL).rename(columns={label_col: "_label_raw"})
            df = df.merge(labels, on=ID_COL, how="left")
        else:
            df["_label_raw"] = np.nan
        df["_label"] = df["_label_raw"].map(
            lambda v: self.class_to_idx.get(whitelist_str(v), -1) if pd.notna(v) else -1
        )
        self.df_start = df

        if require_t2:
            df = self._exclude(df, df["_t2_file"].isna(), "no T2 image")
        if mask_dir:
            has_mask = df[ID_COL].map(lambda c: os.path.isfile(os.path.join(mask_dir, f"{c}.nii.gz")))
            df = self._exclude(df, ~has_mask, f"no segmentation mask in {mask_dir}")
        if require_label:
            df = self._exclude(df, df["_label"] < 0, f"no {label_col!r} label in {classes}")
        self.df = df.reset_index(drop=True)

    def _exclude(self, df, drop, reason):
        self.steps.append((reason, int(drop.sum()), int((~drop).sum())))
        return df[~drop]

    def log_summary(self, log_path=None):
        tag = f"[data:{self.split}]"
        log_print(f"{tag} start {self.n_start} cases", log_path)
        for i, (reason, n_drop, n_left) in enumerate(self.steps, 1):
            log_print(f"{tag}   step {i}: excluded {n_drop} ({reason}) -> {n_left}", log_path)
        present = ", ".join(f"{m}={int(self.df[f'_{m}_file'].notna().sum())}" for m in MODALITIES)
        log_print(f"{tag} final {len(self.df)} cases; images present: {present}", log_path)
        if self.label_col:
            labeled = self.df["_label"] >= 0
            counts = self.df.loc[labeled, "_label_raw"].map(whitelist_str).value_counts().sort_index()
            log_print(
                f"{tag} {self.label_col!r}: labeled {int(labeled.sum())}, unlabeled {int((~labeled).sum())}; "
                + ", ".join(f"{k}: {v}" for k, v in counts.items()),
                log_path,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        channels = []
        for m in MODALITIES:
            path = row[f"_{m}_file"]
            if isinstance(path, str):
                channels.append(load_volume(path, self.img_size))
            else:
                channels.append(torch.zeros((1, *self.img_size), dtype=torch.float32))

        if self.mask_dir:
            if isinstance(row["_t2_file"], str):
                channels.append(load_mask_on_grid(
                    os.path.join(self.mask_dir, f"{row[ID_COL]}.nii.gz"), row["_t2_file"],
                    self.mask_labels, self.mask_values,
                ))
            else:
                channels.append(torch.zeros((1, *self.img_size), dtype=torch.float32))

        raw = row["_label_raw"]
        return {
            "image": torch.cat(channels, dim=0),
            "label": int(row["_label"]),
            "label_raw": whitelist_str(raw) if pd.notna(raw) else "",
            ID_COL: row[ID_COL],
        }
