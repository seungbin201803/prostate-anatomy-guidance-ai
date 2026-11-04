# Prostate Anatomy Guidance AI

Code for training, fine-tuning and running 3D deep-learning classifiers of prostate multiparametric MRI
(mpMRI) that take **prostate zonal segmentation masks** as an extra input channel. 

The pipeline:

- preprocess images (resamplling and cropping centered at given prostate masks).
- uses T2-weighted, DWI and ADC volumes, stacked as input channels.
- optionally adds a **segmentation-mask channel**. You provide the masks (e.g. prostate zones) with any
  integer labels, and the labels are normalized internally.
- predicts **PI-RADS** (ordinal classification, CORAL loss) or **clinically significant prostate cancer** (csPCa; binary classification, cross-entropy loss) per exam.
- offers three backbones: 3D **ViT**, 3D **ResNet-18**, 3D **ResNet-50**.

```
T2   ┐
DWI  ├─ resample 1×1×3 mm, crop 112×112×24 at the prostate ─▶ channel stack ─▶ ViT / ResNet ─▶ mean-pool ─▶ head ─▶ PI-RADS / csPCa
ADC  ┤
seg  ┘  (optional: no mask | mask)
```

The data used in our experiments is not public. This repository contains code only, and
[`data/README.md`](data/README.md) explains how to organize your own data.

## Paper

**Improving PI-RADS Prediction and Prostate Cancer Detection from Multiparametric MRI with Anatomical Guidance**<br>
Seungbin Park, Thibault Heintz, Rachit Saluja, Francesca Khani, Timothy McClure, Daniel Margolis,
Mert R. Sabuncu, Heejong Kim<br>
arXiv:2611.04155, 2026 · [paper](https://arxiv.org/abs/2611.04155) · [PDF](https://arxiv.org/pdf/2611.04155)

The paper tests whether adding prostate zonal masks as anatomical guidance improves PI-RADS prediction
and csPCa detection, using 3D ResNet-18, ResNet-50 and ViT models. Anatomical guidance generally
improved performance across the three architectures. See [Citation](#citation) to cite it.

## Contents

```
train.py                    training from scratch, fine-tuning, resuming
inference.py                predictions (+ metrics if labels exist) for any checkpoint and dataset
preprocess_data.py          raw NIfTI [+ segmentation masks] -> model input (resample, crop)
pagai/                      library: data loading, models, heads/losses/metrics, checkpoints, train/eval loops
tools/                      make_synthetic_data.py and smoke_test.sh, for testing the pipeline without real data
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
the whole pipeline or try the commands below. The images are random ellipsoids with a "prostate", a
5-label segmentation mask and labels, so the trained models are meaningless.

```bash
bash tools/smoke_test.sh
```

This runs every step below on synthetic data with 1-epoch ResNet-18 runs. It takes about 10 minutes on a CPU,
and its outputs are git-ignored (and removed at the start of each run):

| folder | contents |
|---|---|
| `data_synthetic/` | `internal` (20 cases) and `external` (15 cases), in the [`data/`](data/README.md) layout: `raw/`, `seg_raw/`, `prostate_raw/`, `csv/` (with labels), `splits/`, and `processed/` |
| `runs_synthetic/scratch/` | from-scratch runs on `internal`, both tasks × no mask / mask |
| `runs_synthetic/finetune/external/` | the three fine-tuning variants on `external`, both tasks |
| `results_synthetic/external/` | predictions and scores of every fine-tuned run |

To generate only the synthetic data, without running the pipeline:

```bash
python tools/make_synthetic_data.py --root data_synthetic --name internal --n_cases 20 --seed 0
python tools/make_synthetic_data.py --root data_synthetic --name external --n_cases 15 --seed 1
for name in internal external; do
  python preprocess_data.py --csv data_synthetic/csv/$name.csv --raw_dir data_synthetic/raw/$name \
      --seg_dir data_synthetic/seg_raw/$name --prostate_dir data_synthetic/prostate_raw/$name \
      --prostate_label 17 --out_dir data_synthetic/processed/$name --save_fig
done
```

The commands below then work on it with `data/` replaced by `data_synthetic/` (its masks have labels
`0,1,2,3,4,5`). In the synthetic data, about 10% of cases have no ADC, and every 4th case has no prostate mask, so it
falls back to the segmentation mask for the crop center.

## 1. Data

Organize each cohort as described in [`data/README.md`](data/README.md):

```
data/raw/<dataset>/...                       original T2 / DWI / ADC NIfTI
data/seg_raw/<dataset>/<case_id>.nii.gz      optional segmentation masks on the original T2 grid (0 background, your labels)
data/prostate_raw/<dataset>/<case_id>.nii.gz optional whole-prostate masks, used only to center the crop
data/csv/<dataset>.csv                       case_id, split, t2, dwi, adc, pirads, cspca (paths + labels)
data/splits/<dataset>/{train,val,test}.csv   optional, instead of the split column
```

**Segmentation masks are not distributed.** They are only needed for models trained with a mask
channel, and you create them yourself, with any integer labels. The crop is centered on the centroid of
a prostate mask when you provide one (`--prostate_dir`), and on the segmentation-mask centroid otherwise,
so every case needs at least one of the two.

## 2. Preprocessing

```bash
python preprocess_data.py \
    --csv data/csv/internal.csv --raw_dir data/raw/internal \
    --seg_dir data/seg_raw/internal --out_dir data/processed/internal \
    --save_fig --num_workers 8
# optional: center the crop on separate prostate masks (e.g. label 17 of a multi-organ segmentation)
#   --prostate_dir data/prostate_raw/internal --prostate_label 17
# without segmentation masks (no-mask models only), drop --seg_dir and give --prostate_dir
# optional: reorient every volume first, e.g. --orientation LPS
```

For each case, this script:

1. takes the prostate centroid from the prostate mask (`--prostate_dir`), or from the segmentation
   mask when no prostate mask is given or a case has none. `dataset.csv` records which
   (`center_source`);
2. B-spline resamples T2, DWI and ADC to 1 × 1 × 3 mm;
3. crops (or zero-pads) each sequence to 112 × 112 × 24 voxels around that centroid;
4. resamples the segmentation mask onto the cropped T2 grid (`seg/`), nearest neighbor, keeping its
   labels;
5. writes `dataset.csv`, which the training and inference scripts read, and optional QC figures
   (`qc/`).

Run it once per cohort (`internal`, `external`), and look through the QC figures.

## 3. Training from scratch (internal dataset)

Every run is one `train.py` call. Its defaults are the from-scratch recipe, so a run only needs the
data, the task, the model and, optionally, the mask.

**Without a mask** (T2/DWI/ADC):

```bash
python train.py \
    --data_csv data/processed/internal/dataset.csv --label_csv data/csv/internal.csv \
    --label_col pirads --classes 1,2,3,4,5 --loss coral \
    --model resnet18 \
    --out_dir runs/scratch --run_name pirads_resnet18_nomask
```

**With a mask:** add `--seg_dir` (the preprocessed masks) and `--seg_labels` (the labels in your
masks, background 0 first):

```bash
python train.py \
    --data_csv data/processed/internal/dataset.csv --label_csv data/csv/internal.csv \
    --label_col pirads --classes 1,2,3,4,5 --loss coral \
    --model resnet18 --seg_dir data/processed/internal/seg --seg_labels 0,1,2,3,4,5 \
    --out_dir runs/scratch --run_name pirads_resnet18_mask
```

Each run is written to `--out_dir/--run_name`. Choose the task, model and mask with:

| setting | arguments |
|---|---|
| PI-RADS task | `--label_col pirads --classes 1,2,3,4,5 --loss coral` |
| csPCa task | `--label_col cspca --classes 0,1 --loss ce` |
| model | `--model vit`, `resnet18` or `resnet50` |
| no mask | (nothing) |
| mask | `--seg_dir data/processed/<dataset>/seg --seg_labels <your labels>` |
| splits | the `split` column of `dataset.csv`, or `--split_dir data/splits/<dataset>` |

To train every combination in our experiments (2 tasks × 3 models × no mask / mask = 12 runs), loop
over them, or submit each command as its own cluster job:

```bash
for model in vit resnet18 resnet50; do
  for task in pirads cspca; do
    if [ $task = pirads ]; then t=(--label_col pirads --classes 1,2,3,4,5 --loss coral)
    else t=(--label_col cspca --classes 0,1 --loss ce); fi
    common=(--data_csv data/processed/internal/dataset.csv --label_csv data/csv/internal.csv
            "${t[@]}" --model $model --out_dir runs/scratch)
    python train.py "${common[@]}" --run_name ${task}_${model}_nomask
    python train.py "${common[@]}" --seg_dir data/processed/internal/seg --seg_labels 0,1,2,3,4,5 \
        --run_name ${task}_${model}_mask
  done
done
```

The from-scratch recipe (the defaults; written out, these are the flags):

| | value | flags |
|---|---|---|
| optimizer | AdamW, weight decay 1e-5 | `--weight_decay 1e-5` |
| learning rate | head 1e-3, backbone 2e-4, cosine decay to 0, no warmup | `--lr_head 1e-3 --lr_backbone 2e-4 --warmup_epochs 0` |
| batch | 4 × 16 accumulation steps = 64 | `--batch_size 4 --grad_accum_steps 16` |
| epochs | at most 200 | `--epochs 200` |
| early stopping | patience 15 on the validation score (QWK multi-class, AUC binary), min. improvement 0.005 | `--early_stopping_patience 15 --early_stopping_min_delta 0.005` |
| precision | bf16 autocast on CUDA | `--precision bf16` |

**Mask channel values.** Background 0 stays 0, and the other `--seg_labels`, in the order given, become
k/K: `0,1,2,3,4,5` → 0.2/0.4/0.6/0.8/1.0, and `0,1,2` → 0.5/1.0. Every label is nonzero, so it is
distinguishable from background. A mask containing a label that is not in `--seg_labels` is an error,
and with `--seg_dir`, cases without a mask are dropped.

## 4. Fine-tuning (external dataset)

On each external cohort there are three variants per model and task:

| variant | initialization | input | name in our experiments |
|---|---|---|---|
| scratch | random (baseline), from-scratch recipe | T2/DWI/ADC | `{model}-scratch` |
| no mask | `runs/scratch/<task>_<model>_nomask` | T2/DWI/ADC | `{model}` |
| mask | `runs/scratch/<task>_<model>_mask` | + segmentation mask | `{model}+mask` |

`--finetune_from` loads the backbone of a run's best-validation checkpoint (`--finetune_ckpt
max_val_score`). The architecture, input size and mask channel come from that checkpoint, so
`--model` and `--seg_labels` are not needed:

- **mask:** a source trained with a mask needs `--seg_dir`, and reuses its labels, so the external
  masks must have the same labels;
- **no mask:** a source trained without one must not get `--seg_dir`;
- **head:** the classification head is always newly initialized, because the label space can differ.
  Our external cohorts use PI-RADS 3–5 (`--classes 3,4,5`).

The fine-tuning recipe uses smaller learning rates for the smaller cohorts. These flags are the same
for every fine-tuning run:

```bash
FT=(--epochs 50 --lr_head 1e-3 --lr_backbone 2e-5 --weight_decay 1e-2 --warmup_epochs 2
    --early_stopping_patience 15 --early_stopping_min_delta 0.01 --early_stopping_min_epochs 10)
EXT=(--data_csv data/processed/external/dataset.csv --split_dir data/splits/external
     --label_csv data/csv/external.csv --label_col pirads --classes 3,4,5 --loss coral
     --out_dir runs/finetune/external)

# scratch baseline: from-scratch recipe (the defaults) on the external data
python train.py "${EXT[@]}" --model resnet18 --run_name pirads_resnet18_scratch
# no mask
python train.py "${EXT[@]}" "${FT[@]}" --finetune_from runs/scratch/pirads_resnet18_nomask \
    --run_name pirads_resnet18_nomask
# mask
python train.py "${EXT[@]}" "${FT[@]}" --finetune_from runs/scratch/pirads_resnet18_mask \
    --seg_dir data/processed/external/seg --run_name pirads_resnet18_mask
```

(backbone LR 2e-5, head LR 1e-3, weight decay 1e-2, 2 warmup epochs then cosine decay over 50 epochs;
early stopping with patience 15, minimum improvement 0.01, never before epoch 10.)

## 5. Inference

```bash
# --checkpoint: a run directory (uses heads_max_val_score.pt) or a heads_*.pt file
python inference.py --checkpoint runs/finetune/external/pirads_resnet18_mask \
    --data_csv data/processed/external/dataset.csv \
    --label_csv data/csv/external.csv \
    --out_dir results/external/pirads_resnet18_mask

# every run in a directory, test split only
for run in runs/finetune/external/*/; do
  python inference.py --checkpoint $run --data_csv data/processed/external/dataset.csv \
      --label_csv data/csv/external.csv --split_dir data/splits/external --splits test \
      --out_dir results/external/$(basename $run)
done
```

Inference behaves as follows:

- **Labels are optional.** Without them you get predictions only. Label values outside the
  checkpoint's classes are reported as unlabeled. `--label_col` defaults to the checkpoint's task
  name.
- **Splits:** `--splits all` (the default) predicts every case in `dataset.csv`; `--splits test`
  together with `--split_dir` (or a split column) restricts it.
- **Segmentation masks:** a checkpoint trained with a mask channel reads the masks from
  `processed/<dataset>/seg`, or from `--seg_dir`. The label encoding comes from the checkpoint. A
  checkpoint without a mask channel ignores them.

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
  `resnet18_channel` / `resnet50_channel`, `--seg_dir` / `--seg_label_values`) load in `inference.py`
  and as `--finetune_from` sources without conversion, and so do runs from earlier versions of this
  repository (`--mask 2zone` / `5zone`).
- **Memory:** the default batch size is 4 at 112 × 112 × 24. If GPU memory is short, lower
  `--batch_size` and raise `--grad_accum_steps` so their product stays 64.
- **Docker:** DataLoader workers need shared memory. Start the container with a larger `--shm-size`
  (e.g. `--shm-size 8g`), or use `--num_workers 0`.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{park2026anatomical,
  title   = {Improving {PI-RADS} Prediction and Prostate Cancer Detection from Multiparametric {MRI}
             with Anatomical Guidance},
  author  = {Park, Seungbin and Heintz, Thibault and Saluja, Rachit and Khani, Francesca and
             McClure, Timothy and Margolis, Daniel and Sabuncu, Mert R. and Kim, Heejong},
  journal = {arXiv preprint arXiv:2611.04155},
  year    = {2026}
}
```

## Acknowledgements

Built on [MONAI](https://monai.io) (3D ViT / ResNet) and
[coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) (Cao et al., *Rank consistent
ordinal regression for neural networks with application to age estimation*, Pattern Recognition
Letters 2020).
