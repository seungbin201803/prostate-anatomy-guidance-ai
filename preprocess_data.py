"""
Raw mpMRI + 5-zone masks -> model input: each modality resampled to --spacing and cropped/zero-padded
to --size around the prostate centroid, the zone mask resampled onto the cropped T2 grid, and a
2-zone mask derived from it.

Per case (--csv row):
  1. crop center = centroid, as a physical point, of
       - the prostate mask {--prostate_dir}/{case_id}.nii.gz (voxels == --prostate_label, or every
         nonzero voxel), when --prostate_dir is given; it may be on any grid in the same physical space;
       - otherwise (or when that case has no prostate mask) the nonzero voxels of the zone mask
         {--zone_dir}/{case_id}.nii.gz (5 zones on the raw T2 grid: 1 PZ, 2 CZ, 3 TZ, 4 AFS, 5 urethra);
     dataset.csv records which one was used (center_source);
  2. T2, DWI, ADC: B-spline resampling to --spacing (origin and direction kept), then a --size
     crop centered on that point (zero padding where the crop leaves the image);
  3. zone mask: nearest-neighbor resampling onto the cropped T2 grid -> zones5/; then
     1 = TZ (TZ+CZ+AFS+urethra), 2 = PZ -> zones2/.
Volumes are not reoriented (--orientation to change that): keep one convention across every dataset
you train, fine-tune and run inference on, and check the --save_fig QC figures.

Input  --csv: case_id, t2, dwi, adc [, split, labels]  (paths relative to --raw_dir; blank DWI/ADC allowed)
Output --out_dir/
    dataset.csv                         case_id[, split], t2_path, dwi_path, adc_path, center_source, center_*_mm, note
    images/<case_id>/{t2,dwi,adc}.nii.gz
    zones5/<case_id>.nii.gz             1 PZ, 2 CZ, 3 TZ, 4 AFS, 5 urethra
    zones2/<case_id>.nii.gz             1 TZ, 2 PZ
    qc/<case_id>.png                    with --save_fig

    python preprocess_data.py --csv data/csv/internal.csv --raw_dir data/raw/internal \
        --zone_dir data/zones_raw/internal [--prostate_dir data/prostate_raw/internal] \
        --out_dir data/processed/internal --num_workers 8
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagai.utils import ID_COL, read_table, parse_tuple, parse_float_tuple  # noqa: E402

MODALITIES = ("t2", "dwi", "adc")


def to_2zones(zone):
    """5-zone labels (1 PZ, 2 CZ, 3 TZ, 4 AFS, 5 urethra) -> 1 = TZ (TZ+CZ+AFS+urethra), 2 = PZ."""
    out = np.zeros(zone.shape, dtype=np.uint8)
    out[np.isin(zone, (2, 3, 4, 5))] = 1
    out[zone == 1] = 2
    return out


def read_image(path, orientation):
    img = sitk.ReadImage(path)
    if img.GetDimension() == 4:  # occasionally a singleton 4th dimension
        img = img[:, :, :, 0]
    if orientation:
        img = sitk.DICOMOrient(img, orientation)
    return img


def resample(img, spacing):
    if all(abs(a - b) < 1e-2 for a, b in zip(img.GetSpacing(), spacing)):
        return img
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), img.GetSpacing(), spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing)
    f.SetSize(size)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkBSpline)
    f.SetDefaultPixelValue(0)
    return f.Execute(img)


def crop_around(img, center_index, size):
    """--size crop centered on center_index (voxel), zero-padded where it leaves the image."""
    index, crop, lower, upper = [0] * 3, [0] * 3, [0] * 3, [0] * 3
    for i in range(3):
        lo = center_index[i] - size[i] // 2
        hi = lo + size[i]
        if lo < 0:
            lower[i], lo = -lo, 0
        if hi > img.GetSize()[i]:
            upper[i], hi = hi - img.GetSize()[i], img.GetSize()[i]
        if hi <= lo:
            raise ValueError(f"crop center {tuple(center_index)} is outside the image {img.GetSize()}")
        index[i], crop[i] = lo, hi - lo
    return sitk.ConstantPad(sitk.RegionOfInterest(img, crop, index), lower, upper, 0)


def mask_center(path, label, orientation):
    """Physical point of the centroid of a mask's voxels == label (any nonzero voxel if label is None)."""
    mask = read_image(path, orientation)
    arr = sitk.GetArrayFromImage(mask)
    voxels = np.argwhere(arr > 0 if label is None else arr == label)  # (z, y, x)
    if len(voxels) == 0:
        raise RuntimeError(f"empty mask {path}")
    return mask.TransformIndexToPhysicalPoint([int(round(c)) for c in voxels.mean(axis=0)[::-1]])


def save_qc_figure(t2, zones, path, case_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = sitk.GetArrayFromImage(t2)  # [Z, Y, X]
    seg = sitk.GetArrayFromImage(zones)
    zs = np.where(seg.any(axis=(1, 2)))[0]
    zs = np.unique(np.linspace(zs[0], zs[-1], 5).round().astype(int)) if len(zs) else [img.shape[0] // 2]
    fig, axes = plt.subplots(1, len(zs), figsize=(3 * len(zs), 3.3), squeeze=False)
    colors = {1: "red", 2: "yellow", 3: "blue", 4: "lime", 5: "magenta"}
    vmin, vmax = np.percentile(img, [1, 99])
    for ax, z in zip(axes[0], zs):
        ax.imshow(img[z], cmap="gray", vmin=vmin, vmax=vmax)
        for label, color in colors.items():
            if (seg[z] == label).any():
                ax.contour(seg[z] == label, levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{case_id}  (PZ red, CZ yellow, TZ blue, AFS green, urethra magenta)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def process_case(job):
    case, args = job
    case_id = case[ID_COL]
    row = {ID_COL: case_id}
    if "split" in case:
        row["split"] = case["split"]
    try:
        zone_path = os.path.join(args.zone_dir, f"{case_id}.nii.gz")
        if not os.path.isfile(zone_path):
            raise RuntimeError("no zone mask")
        zone = read_image(zone_path, args.orientation)
        prostate_path = os.path.join(args.prostate_dir, f"{case_id}.nii.gz") if args.prostate_dir else None
        if prostate_path and os.path.isfile(prostate_path):
            center_phys = mask_center(prostate_path, args.prostate_label, args.orientation)
            row["center_source"] = "prostate"
        else:
            if prostate_path:
                print(f"[{case_id}] no prostate mask, cropping at the zone-mask centroid", flush=True)
            center_phys = mask_center(zone_path, None, args.orientation)
            row["center_source"] = "zone"
        row.update({f"center_{a}_mm": round(float(v), 2) for a, v in zip("xyz", center_phys)})

        cropped = {}
        for m in MODALITIES:
            rel = case.get(m)
            row[f"{m}_path"] = ""
            if not isinstance(rel, str) or not rel.strip():
                if m == "t2":
                    raise RuntimeError("no T2 path in --csv")
                continue
            img = resample(read_image(os.path.join(args.raw_dir, rel), args.orientation), args.spacing)
            out = crop_around(img, img.TransformPhysicalPointToIndex(center_phys), args.size)
            out_rel = os.path.join("images", case_id, f"{m}.nii.gz")
            os.makedirs(os.path.join(args.out_dir, "images", case_id), exist_ok=True)
            sitk.WriteImage(out, os.path.join(args.out_dir, out_rel), useCompression=True)
            row[f"{m}_path"] = out_rel
            cropped[m] = out

        zones5 = sitk.Resample(zone, cropped["t2"], sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        zones2 = sitk.GetImageFromArray(to_2zones(sitk.GetArrayFromImage(zones5)))
        zones2.CopyInformation(zones5)
        sitk.WriteImage(zones5, os.path.join(args.out_dir, "zones5", f"{case_id}.nii.gz"), useCompression=True)
        sitk.WriteImage(zones2, os.path.join(args.out_dir, "zones2", f"{case_id}.nii.gz"), useCompression=True)
        if args.save_fig:
            save_qc_figure(cropped["t2"], zones5, os.path.join(args.out_dir, "qc", f"{case_id}.png"), case_id)
        row["note"] = "ok"
    except Exception as e:  # one broken case shouldn't stop the dataset; blank paths keep it out of training
        row.update({f"{m}_path": "" for m in MODALITIES})
        row["note"] = f"error: {e}"
        print(f"[{case_id}] {row['note']}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Case CSV: case_id, t2, dwi, adc [, split, label columns].")
    p.add_argument("--raw_dir", default="", help="Root of the --csv image paths.")
    p.add_argument("--zone_dir", required=True, help="{case_id}.nii.gz 5-zone masks on the raw T2 grid.")
    p.add_argument("--prostate_dir", default=None,
                   help="Optional {case_id}.nii.gz prostate masks whose centroid is the crop center. "
                        "Default (and for cases without one): the zone-mask centroid.")
    p.add_argument("--prostate_label", type=int, default=None,
                   help="Prostate label in the --prostate_dir masks (e.g. 17 for a TotalSegmentator-MRI "
                        "multi-organ mask). Default: every nonzero voxel.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--spacing", type=parse_float_tuple, default=(1.0, 1.0, 3.0), help="Target spacing X,Y,Z in mm.")
    p.add_argument("--size", type=parse_tuple, default=(112, 112, 24), help="Crop size X,Y,Z in voxels (= --img_size).")
    p.add_argument("--orientation", default="", help="Reorient every volume first, e.g. LPS. Default: keep as is.")
    p.add_argument("--save_fig", action="store_true", help="Write QC figures (cropped T2 + zone contours).")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    cases = read_table(args.csv, required=["t2"])
    for m in ("dwi", "adc"):
        if m not in cases.columns:
            cases[m] = ""
    for sub in ("images", "zones5", "zones2") + (("qc",) if args.save_fig else ()):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    jobs = [(case, args) for case in cases.to_dict("records")]
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        rows = []
        for i, row in enumerate(pool.map(process_case, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} cases", flush=True)

    df = pd.DataFrame(rows)
    order = [ID_COL, "split"] + [f"{m}_path" for m in MODALITIES] \
        + ["center_source", "center_x_mm", "center_y_mm", "center_z_mm", "note"]
    df = df.reindex(columns=[c for c in order if c in df.columns])
    df.to_csv(os.path.join(args.out_dir, "dataset.csv"), index=False)
    n_ok = int((df["note"] == "ok").sum())
    print(f"{n_ok}/{len(df)} cases ok -> {os.path.join(args.out_dir, 'dataset.csv')}")
    if args.prostate_dir:
        print(f"crop center from: {df['center_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
