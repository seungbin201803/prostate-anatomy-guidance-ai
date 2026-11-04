# Data organization

No data is distributed with this repository. Put your own data here following the layout below; the
example commands in the main README assume it (every path is an argument, so any other layout works
too). `<dataset>` is a short name
per cohort, e.g. `internal` (used for training from scratch) and `external` (the external cohort for
fine-tuning / testing).

```
data/
├── raw/<dataset>/...                        original NIfTI volumes, any sub-layout
├── seg_raw/<dataset>/<case_id>.nii.gz       optional segmentation masks on the original T2 grid (you provide)
├── prostate_raw/<dataset>/<case_id>.nii.gz  optional whole-prostate masks for the crop center   (you provide)
├── csv/<dataset>.csv                        one row per case: raw volume paths + labels         (you provide)
├── splits/<dataset>/{train,val,test}.csv    optional; one case_id column each                   (you provide)
└── processed/<dataset>/                     written by preprocess_data.py = model input
    ├── dataset.csv
    ├── images/<case_id>/{t2,dwi,adc}.nii.gz
    ├── seg/<case_id>.nii.gz                 with --seg_dir
    └── qc/<case_id>.png                     with --save_fig
```

Templates of every CSV are in [`templates/`](templates/), in the same layout (`templates/csv/internal.csv`, `templates/splits/`).

## Rules

**Case IDs.** `case_id` is a unique string per MRI exam. It names the mask files
(`<case_id>.nii.gz`) and joins the case CSV, masks and splits, so it must be identical in all of them.
Avoid `/` and spaces.

**Images.** One 3D NIfTI (`.nii` / `.nii.gz`) per sequence:

| column | sequence | required |
|---|---|---|
| `t2` | axial T2-weighted | yes; cases without T2 are dropped |
| `dwi` | high-b-value diffusion-weighted image | no, a blank cell becomes an all-zero channel |
| `adc` | apparent diffusion coefficient map | no, a blank cell becomes an all-zero channel |

All three must share one physical space (same scanner coordinates). They may have different voxel
grids, because each is resampled separately. Intensities are z-scored per volume at load time, so no
intensity normalization is needed beforehand. Volumes are **not** reoriented by default. Keep one
orientation convention across every dataset you train, fine-tune and run inference on, and check the QC
figures. In our experiments one external cohort needed an axis flip to match the internal data;
`preprocess_data.py --orientation LPS` reorients everything to a common convention.

**Segmentation masks (optional).** These are the optional 4th input channel, for models trained with
a mask. You provide them: this repository neither distributes masks nor creates them. Use one 3D NIfTI
per case, `seg_raw/<dataset>/<case_id>.nii.gz`, **on the same voxel grid as that case's raw T2** (same
size, spacing, origin and direction), with integer labels: 0 for background, and any labels you like
for the structures (e.g. prostate zones). Pass `--seg_dir data/seg_raw/<dataset>` to
`preprocess_data.py`, which writes them to `processed/<dataset>/seg/` with the labels unchanged.

You list the labels when training (`train.py --seg_labels 0,1,2,3,4,5`). The channel then holds
background 0 as 0, and spreads the other labels, in the order given, evenly over (0, 1]
(`1,2,3,4,5` → 0.2, 0.4, 0.6, 0.8, 1.0; `0,1,2` → 0, 0.5, 1.0). A mask containing a label that is
not in the list is an error. Use the same labels for every dataset a model is fine-tuned or run on.
The label list is stored in the checkpoint, so fine-tuning and inference only need the mask directory.
Models trained without a mask need no masks at all. With `--seg_dir`, cases without a mask are
dropped.

**Prostate masks (optional).** The crop is centered on a prostate mask when there is one, and on the
segmentation mask otherwise. At least one of the two is needed per case. For a dedicated crop-center
mask (in our experiments, the prostate label of a separate segmentation), put one 3D NIfTI per case in
`prostate_raw/<dataset>/<case_id>.nii.gz` and pass `--prostate_dir` (plus `--prostate_label N` if the
file holds several labels; default: every nonzero voxel). It may be on any voxel grid in the same
physical space as the T2. It only sets the crop center and is never a model input. Cases without one
fall back to the segmentation-mask centroid, and `dataset.csv` records which was used
(`center_source`: `prostate` or `seg`).

**Case CSV** (`csv/<dataset>.csv`): one row per exam with `case_id`, `t2`, `dwi`, `adc`, optionally
`split`, and the label columns. Image paths are relative to `raw/<dataset>/` (or absolute).
`preprocess_data.py` reads the paths. `train.py` and `inference.py` read the labels from it
(`--label_csv data/csv/<dataset>.csv`), so labels can be corrected without preprocessing again.

**Labels** are columns of the case CSV, one per task. The column name is the task name (`--label_col`):

| column | values | loss | meaning |
|---|---|---|---|
| `pirads` | 1-5 (external cohorts: 3-5) | `coral` (ordinal) | highest PI-RADS category of the exam |
| `cspca` | 0 / 1 | `ce` | clinically significant prostate cancer, as defined for your cohort |

Blank cells are fine: those cases are not used for that task. Values outside `--classes` are dropped
from training and evaluation. A dataset used only for inference needs no label columns at all.

**Splits.** Give either a `split` column (`train` / `val` / `test`) in the case CSV, which is copied
into `dataset.csv`, or `splits/<dataset>/{train,val,test}.csv`. When both exist, the split files take
precedence. Split by patient, not by exam, if patients have several exams. A dataset used only for
inference needs no split (`inference.py --splits all`).

## Processed format

`preprocess_data.py` writes `processed/<dataset>/`. This is what `train.py` and
`inference.py` read. If you preprocess another way, reproduce this format:

- `images/<case_id>/{t2,dwi,adc}.nii.gz`: resampled to 1 x 1 x 3 mm and cropped to 112 x 112 x 24
  voxels around the prostate centroid (zero padding where the crop leaves the image).
- `seg/`: segmentation masks on exactly the cropped T2 grid (only needed for models with a mask).
- `dataset.csv`: `case_id, [split,] t2_path, dwi_path, adc_path` (relative to `processed/<dataset>/`;
  blank = missing). `center_source`, `center_{x,y,z}_mm` and `note` are informational. Rows whose
  `note` is not `ok` have blank paths and are skipped.
