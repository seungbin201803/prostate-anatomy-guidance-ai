#!/usr/bin/env bash
# Run every trained run under RUNS_DIR on a dataset (labels optional).
# -> results/<dataset>/<run name>/predictions_all.csv (+ scores.json, confusion_all.* with labels)
#
#   DATASET=external RUNS_DIR=runs/scratch bash scripts/inference.sh
#   DATASET=external RUNS_DIR=runs/finetune/external SPLITS=test bash scripts/inference.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data}
DATASET=${DATASET:-external}
RUNS_DIR=${RUNS_DIR:-runs/scratch}
OUT_DIR=${OUT_DIR:-results/$DATASET}
CKPT=${CKPT:-max_val_score}
SPLITS=${SPLITS:-all}
EXTRA_ARGS=${EXTRA_ARGS:-}

LABEL_ARGS=()
[ -f "$DATA_ROOT/csv/$DATASET.csv" ] && LABEL_ARGS=(--label_csv "$DATA_ROOT/csv/$DATASET.csv")
SPLIT_ARGS=()
[ -d "$DATA_ROOT/splits/$DATASET" ] && SPLIT_ARGS=(--split_dir "$DATA_ROOT/splits/$DATASET")

for run in "$RUNS_DIR"/*/; do
  run=${run%/}
  [ -f "$run/heads_$CKPT.pt" ] || [ -f "$run/heads_min_val.pt" ] || continue
  cmd=(python inference.py --checkpoint "$run" --ckpt "$CKPT"
    --data_csv "$DATA_ROOT/processed/$DATASET/dataset.csv" --splits "$SPLITS" "${SPLIT_ARGS[@]}" "${LABEL_ARGS[@]}"
    --out_dir "$OUT_DIR/$(basename "$run")" $EXTRA_ARGS)
  echo "${cmd[*]}"
  [ -n "${DRY_RUN:-}" ] || "${cmd[@]}"
done
