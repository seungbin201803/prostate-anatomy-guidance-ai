# Prostate Anatomy Guidance AI

Code for training, fine-tuning and running 3D deep-learning classifiers of prostate multiparametric MRI
(mpMRI) that take prostate **zone anatomy** as an extra input. The pipeline:

- predicts **PI-RADS** (ordinal classification, CORAL loss) or **clinically significant prostate cancer** (csPCa; binary classification, cross-entropy loss) per exam;
- uses T2-weighted, DWI and ADC volumes, stacked as input channels;
- optionally adds a zone-mask channel: **2 zones** (transition / peripheral zone) or **5 zones**
  (PZ, CZ, TZ, anterior fibromuscular stroma, urethra);
- offers three backbones, all trained from scratch: 3D **ViT**, 3D **ResNet-18**, 3D **ResNet-50**.

```
T2   ┐
DWI  ├─ resample 1×1×3 mm, crop 112×112×24 at the prostate ─▶ channel stack ─▶ ViT / ResNet ─▶ mean-pool ─▶ head ─▶ PI-RADS / csPCa
ADC  ┤
zone ┘  (none | 2zone | 5zone)
```

The data used in our experiments is not public. This repository contains code only, and
[`data/README.md`](data/README.md) explains how to organize your own data.

## Contents

```
train.py                    training from scratch, fine-tuning, resuming
inference.py                predictions (+ metrics if labels exist) for any checkpoint and dataset
preprocess_data.py          raw NIfTI + zone masks -> model input (resample, crop, 2-/5-zone masks)
pagai/                      library: data loading, models, heads/losses/metrics, checkpoints, train/eval loops
scripts/                    the experiment grids (train_scratch.sh, finetune.sh, inference.sh) and smoke_test.sh
tools/make_synthetic_data.py  synthetic data for testing the pipeline
data/                       data layout guide and CSV templates (no data)
```

## Installation

Python ≥ 3.10 and a CUDA GPU are recommended (bf16 mixed precision is used on CUDA; the CPU works for
testing).

```bash
git clone <this repository> && cd prostate-anatomy-guidance-ai
conda create -n pagai python=3.11 && conda activate pagai
pip install torch --index-url https://download.pytorch.org/whl/cu121   # matching your CUDA
pip install -r requirements.txt
```

## Synthetic data (testing without real data)

The real data is not public, but a small synthetic dataset in the same layout can be generated to check
the whole pipeline or try the scripts. The images are random ellipsoids with a "prostate", zone masks
and labels, so the trained models are meaningless.

```bash
bash scripts/smoke_test.sh
```

This takes about 15 minutes on a CPU, with 1-epoch ResNet-18 runs. It runs every step on synthetic
data, and its outputs are git-ignored:

| folder | contents | written by |
|---|---|---|
| `data_synthetic/` | `internal` (20 cases) and `external` (15 cases), in the [`data/`](data/README.md) layout: `raw/`, `zones_raw/`, `prostate_raw/`, `csv/` (with labels), `splits/`, and `processed/` | `tools/make_synthetic_data.py`, then `preprocess_data.py` |
| `runs_synthetic/scratch/` | from-scratch runs on `internal`, both tasks × no mask / 2zone / 5zone | `scripts/train_scratch.sh` |
| `runs_synthetic/finetune/external/` | the four fine-tuning variants on `external`, both tasks | `scripts/finetune.sh` |
| `results_synthetic/external/` | predictions and scores of every fine-tuned run | `scripts/inference.sh` |

To generate only the synthetic data, without running the pipeline:

```bash
python tools/make_synthetic_data.py --root data_synthetic --name internal --n_cases 20 --seed 0
python tools/make_synthetic_data.py --root data_synthetic --name external --n_cases 15 --seed 1
for name in internal external; do
  python preprocess_data.py --csv data_synthetic/csv/$name.csv --raw_dir data_synthetic/raw/$name \
      --zone_dir data_synthetic/zones_raw/$name --prostate_dir data_synthetic/prostate_raw/$name \
      --prostate_label 17 --out_dir data_synthetic/processed/$name --save_fig
done
```

Every other script then works on it with `DATA_ROOT=data_synthetic`, e.g.
`DATA_ROOT=data_synthetic MODELS=resnet18 TASKS=pirads MASKS=5zone bash scripts/train_scratch.sh`.

In the synthetic data, about 10% of cases have no ADC, and every 4th case has no prostate mask, so it
falls back to the zone mask for the crop center.

## 1. Data

Organize each cohort as described in [`data/README.md`](data/README.md):

```
data/raw/<dataset>/...                       original T2 / DWI / ADC NIfTI
data/zones_raw/<dataset>/<case_id>.nii.gz    5-zone masks on the original T2 grid (1 PZ, 2 CZ, 3 TZ, 4 AFS, 5 urethra)
data/prostate_raw/<dataset>/<case_id>.nii.gz optional whole-prostate masks, used only to center the crop
data/csv/<dataset>.csv                       case_id, split, t2, dwi, adc, pirads, cspca (paths + labels)
data/splits/<dataset>/{train,val,test}.csv   optional, instead of the split column
```

**Zone masks are an input to this pipeline.** Every case needs one. The crop is centered on the centroid of a separate prostate mask when
you provide one (`--prostate_dir`), and on the zone-mask centroid otherwise.

## 2. Preprocessing

```bash
python preprocess_data.py \
    --csv data/csv/internal.csv --raw_dir data/raw/internal \
    --zone_dir data/zones_raw/internal --out_dir data/processed/internal \
    --save_fig --num_workers 8
# optional: center the crop on separate prostate masks (e.g. label 17 of a multi-organ segmentation)
#   --prostate_dir data/prostate_raw/internal --prostate_label 17
# optional: reorient every volume first, e.g. --orientation LPS
```

For each case, this script:

1. takes the prostate centroid from the prostate mask (`--prostate_dir`), or from the zone mask when
   no prostate mask is given or a case has none. `dataset.csv` records which (`center_source`);
2. B-spline resamples T2, DWI and ADC to 1 × 1 × 3 mm;
3. crops (or zero-pads) each sequence to 112 × 112 × 24 voxels around that centroid;
4. resamples the zone mask onto the cropped T2 grid (`zones5/`) and derives the 2-zone mask
   (`zones2/`: 1 = TZ+CZ+AFS+urethra, 2 = PZ);
5. writes `dataset.csv`, which the training and inference scripts read, and optional QC figures
   (`qc/`).

Run it once per cohort (`internal`, `external`), and look through the QC figures.

## 3. Training from scratch (internal dataset)

The scratch grid trains each model with each mask setting:

| model | no mask | 2 zones | 5 zones |
|---|---|---|---|
| ViT | `vit` | `vit+2zones` | `vit+5zones` |
| ResNet-18 | `resnet18` | `resnet18+2zones` | `resnet18+5zones` |
| ResNet-50 | `resnet50` | `resnet50+2zones` | `resnet50+5zones` |

Each is trained for both tasks (PI-RADS 1–5 with CORAL, csPCa 0/1 with cross-entropy):

```bash
bash scripts/train_scratch.sh                                   # all 18 runs, one after another
TASKS=pirads MODELS=resnet18 MASKS=5zone bash scripts/train_scratch.sh   # one run (e.g. one cluster job)
DRY_RUN=1 bash scripts/train_scratch.sh                         # print the commands
```

Runs are written to `runs/scratch/<task>_<model>_<mask>/`. A single run, written out in full:

```bash
python train.py \
    --data_csv data/processed/internal/dataset.csv --label_csv data/csv/internal.csv \
    --label_col pirads --classes 1,2,3,4,5 --loss coral \
    --model resnet18 --mask 5zone \
    --batch_size 4 --grad_accum_steps 16 --epochs 200 --lr_head 1e-3 --lr_backbone 2e-4 \
    --weight_decay 1e-5 --warmup_epochs 0 --early_stopping_patience 15 --early_stopping_min_delta 0.005 \
    --out_dir runs/scratch --run_name pirads_resnet18_5zone
```

The recipe is as follows:

- **Optimizer:** AdamW, effective batch size 64 (4 × 16 accumulation steps). Learning rate 1e-3 for
  the head and 2e-4 for the backbone, cosine-decayed to 0 over 200 epochs. bf16 autocast.
- **Early stopping:** patience 15 epochs on the validation score. The score is QWK for multi-class
  tasks and AUC for binary ones.
- **Mask channel values:** 2-zone TZ/PZ → 0.5/1.0. 5-zone PZ/CZ/TZ/AFS/urethra → 0.2/0.4/0.6/0.8/1.0.
  Every zone is nonzero, so it is distinguishable from background.

## 4. Fine-tuning (external dataset)

On each external cohort there are four variants per model and task:

| variant | initialization | input | name |
|---|---|---|---|
| `scratch` | random (baseline), from-scratch recipe | T2/DWI/ADC | `{model}-scratch` |
| `nomask` | `runs/scratch/<task>_<model>_none` | T2/DWI/ADC | `{model}` |
| `2zone` | `runs/scratch/<task>_<model>_2zone` | + 2-zone mask | `{model}+2zones` |
| `5zone` | `runs/scratch/<task>_<model>_5zone` | + 5-zone mask | `{model}+5zones` |

```bash
DATASET=external bash scripts/finetune.sh
DATASET=external TASKS=cspca MODELS="resnet18 resnet50" VARIANTS="scratch 5zone" bash scripts/finetune.sh
```

Runs are written to `runs/finetune/<dataset>/<task>_<model>_<variant>/`. Fine-tuning loads the
backbone weights of the source run's best-validation checkpoint (`--finetune_ckpt max_val_score`).
The architecture, input size and mask channel come from that checkpoint, so `--mask` has to match it.
The classification head is always newly initialized, because the label space can differ: the external
cohorts use PI-RADS 3–5, set with `PIRADS_CLASSES`.

The recipe uses smaller learning rates for the smaller cohorts:

- **Optimizer:** backbone LR 2e-5, head LR 1e-3, weight decay 1e-2. 2 warmup epochs, then cosine decay
  over 50 epochs.
- **Early stopping:** patience 15, minimum improvement 0.01, never before epoch 10.

A single run:

```bash
python train.py --finetune_from runs/scratch/pirads_resnet18_5zone --mask 5zone \
    --data_csv data/processed/external/dataset.csv --split_dir data/splits/external \
    --label_csv data/csv/external.csv --label_col pirads --classes 3,4,5 --loss coral \
    --epochs 50 --lr_backbone 2e-5 --weight_decay 1e-2 --warmup_epochs 2 \
    --early_stopping_min_delta 0.01 --early_stopping_min_epochs 10 \
    --out_dir runs/finetune/external --run_name pirads_resnet18_5zone
```

## 5. Inference

```bash
# one checkpoint: a run directory (uses heads_max_val_score.pt) or a heads_*.pt file
python inference.py --checkpoint runs/finetune/external/pirads_resnet18_5zone \
    --data_csv data/processed/external/dataset.csv \
    --label_csv data/csv/external.csv \
    --out_dir results/external/pirads_resnet18_5zone

# every run in a directory
DATASET=external RUNS_DIR=runs/finetune/external SPLITS=test bash scripts/inference.sh
```

Inference behaves as follows:

- **Labels are optional.** Without them you get predictions only. Label values outside the
  checkpoint's classes are reported as unlabeled. `--label_col` defaults to the checkpoint's task
  name.
- **Splits:** `--splits all` (the default) predicts every case in `dataset.csv`; `--splits test`
  together with `--split_dir` (or a split column) restricts it.
- **Zone masks:** a checkpoint trained with a mask channel reads the matching masks from
  `processed/<dataset>/zones2` or `zones5`, or from `--mask_dir`.

It writes:

- `predictions_<split>.csv`: `case_id`, `<task>_gt`, `<task>_pred`, plus class probabilities.
  CE models give `<task>_prob_<c>` = P(y = c). CORAL models give `<task>_prob_gt_<c>` = P(y > c), the
  cumulative probabilities.
- With labels, also `scores.json` (n, accuracy, MAE, MSE, quadratic weighted kappa, and AUC for binary
  tasks) and `confusion_<split>.csv/.png`.

## Outputs of a training run

```
runs/<...>/<run_name>/
├── log.txt, args.json         full log and arguments
├── metrics.jsonl              per epoch: train/val loss, accuracy, mae, mse, qwk (+auc), val_score
├── tensorboard/               tensorboard --logdir runs
├── {backbone,heads}_max_val_score.pt   best validation score (used for fine-tuning and evaluation)
├── {backbone,heads}_min_val.pt         lowest validation loss
├── {backbone,heads}_last.pt, train_state_last.pt   latest epoch, for --resume
├── val_predictions_max_val_score.csv
├── final_metrics.json         completed / early_stopped, best epochs
└── eval_max_val_score/        predictions_{val,test}.csv + scores.json of the best checkpoint
```

Behaviour to know about:

- **Resuming:** `python train.py --resume <run_dir> [--epochs N]` continues an interrupted run. All
  other arguments are taken from the run itself.
- **Overwriting:** `train.py` refuses to write into an existing run directory unless `--overwrite` is
  given.
- **Test set:** the test split is evaluated only once, after training, with the best-validation
  checkpoint. It is never used for model selection.
- **Missing sequences:** a missing DWI or ADC is fed as an all-zero channel. Cases without T2 are
  excluded (`--require_t2 false` keeps them).

## Notes

- **Checkpoint format:** each checkpoint stores its full argument list, and the architecture is
  rebuilt from it. Checkpoints from the original research code (`--ve_name 3dvit_channel` /
  `resnet18_channel` / `resnet50_channel`, `--seg_dir`) load in `inference.py` and as
  `--finetune_from` sources without conversion.
- **Memory:** the default batch size is 4 at 112 × 112 × 24. If GPU memory is short, lower
  `--batch_size` and raise `--grad_accum_steps` so their product stays 64.
- **Docker:** DataLoader workers need shared memory. Start the container with a larger `--shm-size`
  (e.g. `--shm-size 8g`), or use `--num_workers 0`.

## Citation

If you use this code, please cite our paper (citation to be added on publication).

## Acknowledgements

Built on [MONAI](https://monai.io) (3D ViT / ResNet) and
[coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) (Cao et al., *Rank consistent
ordinal regression for neural networks with application to age estimation*, Pattern Recognition
Letters 2020).
