"""
Writes a small synthetic dataset in the raw layout of data/README.md, to check the pipeline end to end
(preprocess_data.py -> train.py -> inference.py) without real data. Images are random
ellipsoids, so any trained model is meaningless.

    python tools/make_synthetic_data.py --root data_synthetic --name internal --n_cases 24

Writes, under --root:
    raw/<name>/<case_id>/{t2,dwi,adc}.nii.gz   T2 0.6x0.6x3 mm, DWI/ADC 1x1x3 mm (some cases without ADC)
    seg_raw/<name>/<case_id>.nii.gz            segmentation mask on the T2 grid, labels 1-5 (--seg_labels 0,1,2,3,4,5)
    prostate_raw/<name>/<case_id>.nii.gz       prostate mask (label 17) on the DWI grid, for --prostate_dir
    csv/<name>.csv                             case_id, split, t2, dwi, adc, pirads (1-5), cspca (0/1)
    splits/<name>/{train,val,test}.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk


def grid_coords(size, spacing, origin):
    """Physical (x, y, z) coordinates of every voxel, each [Z, Y, X] (identity direction)."""
    axes = [origin[i] + spacing[i] * np.arange(size[i]) for i in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return x, y, z


def to_image(arr, spacing, origin, pixel=sitk.sitkFloat32):
    img = sitk.Cast(sitk.GetImageFromArray(arr), pixel)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def make_case(rng, pirads, has_adc):
    origin = tuple(float(v) for v in rng.uniform(-5, 5, size=3))
    center = np.array([48.0, 48.0, 45.0]) + rng.uniform(-8, 8, size=3) + np.array(origin)
    radii = np.array([20.0, 16.0, 12.0]) * rng.uniform(0.85, 1.15, size=3)
    lesion = center + rng.uniform(-0.5, 0.5, size=3) * radii
    strength = (pirads - 1) / 4.0

    def anatomy(size, spacing):
        x, y, z = grid_coords(size, spacing, origin)
        d = [(x - center[0]) / radii[0], (y - center[1]) / radii[1], (z - center[2]) / radii[2]]
        r = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        les = np.sqrt((x - lesion[0]) ** 2 + (y - lesion[1]) ** 2 + (z - lesion[2]) ** 2) < 6.0
        return d, r, les

    images = {}
    t2_size, t2_spacing = (160, 160, 30), (0.6, 0.6, 3.0)
    d, r, les = anatomy(t2_size, t2_spacing)
    t2 = 100 + 50 * (r < 1) - 40 * strength * (les & (r < 1)) + rng.normal(0, 8, r.shape)
    images["t2"] = to_image(t2, t2_spacing, origin)

    seg = np.zeros(r.shape, dtype=np.uint8)  # five arbitrary sub-regions of the "prostate"
    inside = r < 1
    seg[inside] = 3
    seg[inside & (d[2] > 0.5) & (r < 0.8)] = 2
    seg[inside & (d[1] > 0.6)] = 4
    seg[inside & (d[1] < -0.2) & (r > 0.5)] = 1
    seg[inside & (np.sqrt((d[0] * radii[0]) ** 2 + (d[1] * radii[1]) ** 2) < 2.5)] = 5
    seg_img = to_image(seg, t2_spacing, origin, sitk.sitkUInt8)

    dwi_size, dwi_spacing = (96, 96, 30), (1.0, 1.0, 3.0)
    d, r, les = anatomy(dwi_size, dwi_spacing)
    dwi = 50 + 20 * (r < 1) + 60 * strength * (les & (r < 1)) + rng.normal(0, 5, r.shape)
    images["dwi"] = to_image(dwi, dwi_spacing, origin)
    prostate_img = to_image(np.where(r < 1, 17, 0).astype(np.uint8), dwi_spacing, origin, sitk.sitkUInt8)
    if has_adc:
        adc = 1500 - 200 * (r < 1) - 600 * strength * (les & (r < 1)) + rng.normal(0, 50, r.shape)
        images["adc"] = to_image(adc, dwi_spacing, origin)
    return images, seg_img, prostate_img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data_synthetic")
    p.add_argument("--name", default="internal")
    p.add_argument("--n_cases", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_dir = os.path.join(args.root, "raw", args.name)
    seg_dir = os.path.join(args.root, "seg_raw", args.name)
    prostate_dir = os.path.join(args.root, "prostate_raw", args.name)
    split_dir = os.path.join(args.root, "splits", args.name)
    for d in (raw_dir, seg_dir, prostate_dir, split_dir, os.path.join(args.root, "csv")):
        os.makedirs(d, exist_ok=True)

    splits = np.array(["train"] * args.n_cases, dtype=object)
    n_eval = max(2, args.n_cases // 5)
    splits[-2 * n_eval:-n_eval] = "val"
    splits[-n_eval:] = "test"

    rows = []
    for i in range(args.n_cases):
        case_id = f"{args.name}_{i:04d}"
        pirads = int(rng.integers(1, 6))
        cspca = int(rng.random() < (0.8 if pirads >= 4 else 0.1))
        has_adc = rng.random() > 0.1
        images, seg, prostate = make_case(rng, pirads, has_adc)

        os.makedirs(os.path.join(raw_dir, case_id), exist_ok=True)
        row = {"case_id": case_id, "split": splits[i], "t2": "", "dwi": "", "adc": "", "pirads": pirads, "cspca": cspca}
        for m, img in images.items():
            rel = os.path.join(case_id, f"{m}.nii.gz")
            sitk.WriteImage(img, os.path.join(raw_dir, rel), useCompression=True)
            row[m] = rel
        sitk.WriteImage(seg, os.path.join(seg_dir, f"{case_id}.nii.gz"), useCompression=True)
        if i % 4:  # every 4th case has no prostate mask, to exercise the segmentation-mask fallback
            sitk.WriteImage(prostate, os.path.join(prostate_dir, f"{case_id}.nii.gz"), useCompression=True)
        rows.append(row)

    cases = pd.DataFrame(rows)
    cases.to_csv(os.path.join(args.root, "csv", f"{args.name}.csv"), index=False)
    for split in ("train", "val", "test"):
        cases.loc[cases["split"] == split, ["case_id"]].to_csv(os.path.join(split_dir, f"{split}.csv"), index=False)
    print(f"{args.n_cases} synthetic cases -> {args.root} ({args.name})")


if __name__ == "__main__":
    main()
