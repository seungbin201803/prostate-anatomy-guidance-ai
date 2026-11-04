#!/usr/bin/env bash
# End-to-end check on synthetic data (CPU is fine, ~10 minutes), with 1-epoch ResNet-18 runs:
#   synthetic data -> preprocessing -> scratch training (no mask / mask, both tasks)
#   -> fine-tuning on "external" (scratch / nomask / mask) -> resume -> inference.
# Writes data_synthetic/, runs_synthetic/ and results_synthetic/ (git-ignored), removing them first.
#
#   bash tools/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

D=data_synthetic
R=runs_synthetic
O=results_synthetic
rm -rf "$D" "$R" "$O"
QUICK=(--epochs 1 --batch_size 2 --grad_accum_steps 1 --precision fp32 --num_workers 0)

python tools/make_synthetic_data.py --root $D --name internal --n_cases 20 --seed 0
python tools/make_synthetic_data.py --root $D --name external --n_cases 15 --seed 1
for name in internal external; do
  python preprocess_data.py --csv $D/csv/$name.csv --raw_dir $D/raw/$name --seg_dir $D/seg_raw/$name \
    --prostate_dir $D/prostate_raw/$name --prostate_label 17 --out_dir $D/processed/$name --save_fig --num_workers 2
done

for task in pirads cspca; do
  if [ $task = pirads ]; then
    int_task=(--label_col pirads --classes 1,2,3,4,5 --loss coral)
    ext_task=(--label_col pirads --classes 3,4,5 --loss coral)
  else
    int_task=(--label_col cspca --classes 0,1 --loss ce)
    ext_task=("${int_task[@]}")
  fi

  # from scratch on internal: without and with the segmentation mask
  internal=(python train.py --data_csv $D/processed/internal/dataset.csv --split_dir $D/splits/internal
    --label_csv $D/csv/internal.csv "${int_task[@]}" --model resnet18 --out_dir $R/scratch "${QUICK[@]}")
  "${internal[@]}" --run_name ${task}_resnet18_nomask
  "${internal[@]}" --seg_dir $D/processed/internal/seg --seg_labels 0,1,2,3,4,5 --run_name ${task}_resnet18_mask

  # external: scratch baseline, and fine-tuning from each scratch run (labels come from the source run)
  external=(python train.py --data_csv $D/processed/external/dataset.csv --split_dir $D/splits/external
    --label_csv $D/csv/external.csv "${ext_task[@]}" --out_dir $R/finetune/external "${QUICK[@]}")
  "${external[@]}" --model resnet18 --run_name ${task}_resnet18_scratch
  "${external[@]}" --finetune_from $R/scratch/${task}_resnet18_nomask --run_name ${task}_resnet18_nomask
  "${external[@]}" --finetune_from $R/scratch/${task}_resnet18_mask --seg_dir $D/processed/external/seg \
    --run_name ${task}_resnet18_mask
done

python train.py --resume $R/scratch/pirads_resnet18_mask --epochs 2 --num_workers 0

for run in $R/finetune/external/*/; do
  run=${run%/}
  python inference.py --checkpoint "$run" --data_csv $D/processed/external/dataset.csv \
    --label_csv $D/csv/external.csv --out_dir "$O/external/$(basename "$run")" --num_workers 0 --precision fp32
done

echo "smoke test passed: see $R/ and $O/"
