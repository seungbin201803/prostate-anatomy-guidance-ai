#!/usr/bin/env bash
# Train on an external dataset, four variants per task x model:
#   scratch  from scratch, no mask, the from-scratch recipe          (baseline: {model}-scratch)
#   nomask   fine-tuned from runs/scratch/<task>_<model>_none        ({model})
#   2zone    fine-tuned from runs/scratch/<task>_<model>_2zone       ({model}+2zones)
#   5zone    fine-tuned from runs/scratch/<task>_<model>_5zone       ({model}+5zones)
# Fine-tuning loads the backbone only (new head) with smaller learning rates, 2 warmup epochs and a cosine
# schedule over 50 epochs.
# -> runs/finetune/<dataset>/<task>_<model>_<variant>
#
#   DATASET=external bash scripts/finetune.sh
#   DATASET=external TASKS=cspca MODELS=resnet18 VARIANTS="scratch 5zone" bash scripts/finetune.sh
#   DRY_RUN=1 DATASET=external bash scripts/finetune.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data}
DATASET=${DATASET:-external}
PRETRAINED_DIR=${PRETRAINED_DIR:-runs/scratch}
OUT_DIR=${OUT_DIR:-runs/finetune/$DATASET}
TASKS=${TASKS:-"pirads cspca"}
MODELS=${MODELS:-"vit resnet18 resnet50"}
VARIANTS=${VARIANTS:-"scratch nomask 2zone 5zone"}
PIRADS_CLASSES=${PIRADS_CLASSES:-3,4,5}   # the external cohorts were restricted to PI-RADS >= 3
EXTRA_ARGS=${EXTRA_ARGS:-}

SPLIT_ARGS=()
[ -d "$DATA_ROOT/splits/$DATASET" ] && SPLIT_ARGS=(--split_dir "$DATA_ROOT/splits/$DATASET")

SCRATCH_RECIPE=(--epochs 200 --lr_head 1e-3 --lr_backbone 2e-4 --weight_decay 1e-5 --warmup_epochs 0
  --early_stopping_patience 15 --early_stopping_metric val_score --early_stopping_min_delta 0.005)
FINETUNE_RECIPE=(--finetune_ckpt max_val_score
  --epochs 50 --lr_head 1e-3 --lr_backbone 2e-5 --weight_decay 1e-2 --warmup_epochs 2
  --early_stopping_patience 15 --early_stopping_metric val_score --early_stopping_min_delta 0.01
  --early_stopping_min_epochs 10)

for task in $TASKS; do
  case $task in
    pirads) task_args=(--label_col pirads --classes "$PIRADS_CLASSES" --loss coral) ;;
    cspca)  task_args=(--label_col cspca --classes 0,1 --loss ce) ;;
    *) echo "unknown task $task" >&2; exit 1 ;;
  esac
  for model in $MODELS; do
    for variant in $VARIANTS; do
      case $variant in
        scratch) variant_args=(--model "$model" --mask none "${SCRATCH_RECIPE[@]}") ;;
        nomask)  variant_args=(--finetune_from "$PRETRAINED_DIR/${task}_${model}_none" --mask none "${FINETUNE_RECIPE[@]}") ;;
        2zone|5zone) variant_args=(--finetune_from "$PRETRAINED_DIR/${task}_${model}_${variant}" --mask "$variant" "${FINETUNE_RECIPE[@]}") ;;
        *) echo "unknown variant $variant" >&2; exit 1 ;;
      esac
      cmd=(python train.py
        --data_csv "$DATA_ROOT/processed/$DATASET/dataset.csv" "${SPLIT_ARGS[@]}"
        --label_csv "$DATA_ROOT/csv/$DATASET.csv" "${task_args[@]}"
        "${variant_args[@]}"
        --batch_size 4 --grad_accum_steps 16 --precision bf16
        --out_dir "$OUT_DIR" --run_name "${task}_${model}_${variant}" $EXTRA_ARGS)
      echo "${cmd[*]}"
      [ -n "${DRY_RUN:-}" ] || "${cmd[@]}"
    done
  done
done
