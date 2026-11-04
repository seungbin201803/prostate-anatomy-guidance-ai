#!/usr/bin/env bash
# End-to-end check on synthetic data (CPU is fine, a few minutes): preprocessing, scratch training,
# fine-tuning (all four variants), resume and inference, with 1-epoch runs of resnet18.
# Everything is written under data_synthetic/, runs_synthetic/ and results_synthetic/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=data_synthetic
QUICK="--epochs 1 --batch_size 2 --grad_accum_steps 1 --precision fp32 --num_workers 0 --overwrite"
export DATA_ROOT=$ROOT MODELS=resnet18 EXTRA_ARGS="$QUICK"

python tools/make_synthetic_data.py --root $ROOT --name internal --n_cases 20 --seed 0
python tools/make_synthetic_data.py --root $ROOT --name external --n_cases 15 --seed 1
for name in internal external; do
  python preprocess_data.py --csv $ROOT/csv/$name.csv --raw_dir $ROOT/raw/$name --zone_dir $ROOT/zones_raw/$name \
    --prostate_dir $ROOT/prostate_raw/$name --prostate_label 17 --out_dir $ROOT/processed/$name --save_fig --num_workers 2
done

OUT_DIR=runs_synthetic/scratch bash scripts/train_scratch.sh
DATASET=external PRETRAINED_DIR=runs_synthetic/scratch OUT_DIR=runs_synthetic/finetune/external bash scripts/finetune.sh
python train.py --resume runs_synthetic/scratch/pirads_resnet18_5zone --epochs 2 --num_workers 0
EXTRA_ARGS="--num_workers 0 --precision fp32" DATASET=external RUNS_DIR=runs_synthetic/finetune/external \
  OUT_DIR=results_synthetic/external bash scripts/inference.sh

echo "smoke test passed: see runs_synthetic/ and results_synthetic/"
