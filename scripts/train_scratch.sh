#!/usr/bin/env bash
# Train from scratch on the internal dataset:
#   task (pirads: PI-RADS 1-5, CORAL | cspca: csPCa 0/1, cross-entropy)
#   x model (vit | resnet18 | resnet50)
#   x mask (none | 2zone | 5zone)
# -> runs/scratch/<task>_<model>_<mask>   (e.g. runs/scratch/pirads_resnet18_5zone)
#
# Runs one after another; narrow the grid with environment variables, e.g. to submit each run as its
# own cluster job:
#   TASKS=pirads MODELS=resnet18 MASKS=5zone bash scripts/train_scratch.sh
#   DRY_RUN=1 bash scripts/train_scratch.sh          # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data}
DATASET=${DATASET:-internal}
OUT_DIR=${OUT_DIR:-runs/scratch}
TASKS=${TASKS:-"pirads cspca"}
MODELS=${MODELS:-"vit resnet18 resnet50"}
MASKS=${MASKS:-"none 2zone 5zone"}
PIRADS_CLASSES=${PIRADS_CLASSES:-1,2,3,4,5}
EXTRA_ARGS=${EXTRA_ARGS:-}   # appended to every command, e.g. "--num_workers 8"

SPLIT_ARGS=()
[ -d "$DATA_ROOT/splits/$DATASET" ] && SPLIT_ARGS=(--split_dir "$DATA_ROOT/splits/$DATASET")

for task in $TASKS; do
  case $task in
    pirads) task_args=(--label_col pirads --classes "$PIRADS_CLASSES" --loss coral) ;;
    cspca)  task_args=(--label_col cspca --classes 0,1 --loss ce) ;;
    *) echo "unknown task $task" >&2; exit 1 ;;
  esac
  for model in $MODELS; do
    for mask in $MASKS; do
      cmd=(python train.py
        --data_csv "$DATA_ROOT/processed/$DATASET/dataset.csv" "${SPLIT_ARGS[@]}"
        --label_csv "$DATA_ROOT/csv/$DATASET.csv" "${task_args[@]}"
        --model "$model" --mask "$mask"
        --img_size 112,112,24 --batch_size 4 --grad_accum_steps 16
        --epochs 200 --lr_head 1e-3 --lr_backbone 2e-4 --weight_decay 1e-5 --warmup_epochs 0
        --early_stopping_patience 15 --early_stopping_metric val_score --early_stopping_min_delta 0.005
        --precision bf16
        --out_dir "$OUT_DIR" --run_name "${task}_${model}_${mask}" $EXTRA_ARGS)
      echo "${cmd[*]}"
      [ -n "${DRY_RUN:-}" ] || "${cmd[@]}"
    done
  done
done
