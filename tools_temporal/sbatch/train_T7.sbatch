#!/bin/bash
#SBATCH --job-name=temporal_train_T7
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Main training run (TEMPORAL_README.md 5.6, 8): T7 = adapter, gated on c_t,
# mask token + frame masking, synthetic occlusion (replay, p = 0.5), all three
# corpora. Long run with validation every VAL_EVERY steps on held-out signers
# and, with the same metrics, on 45 training clips (overfitting check).
# STEPS / VAL_EVERY / RUN / VARIANTS / EXTRA override.
set -euo pipefail
A=$PWD/data/temporal/all
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
RUN=${RUN:-T7_replay_long}
nvidia-smi --query-gpu=name --format=csv,noheader
$TPY -W ignore tools_temporal/train_temporal.py configs/temporal/T7.yaml out_dir=$PWD/runs/$RUN \
  data.cache_dir=$A/cache data.occ_dir=$A/occ data.variant_dir=$A/${VARIANTS:-variants_replay} \
  data.train_list=$A/lists/train.txt data.val_list=$A/lists/val.txt data.segments=$A/lists/segments.json \
  train.steps=${STEPS:-20000} train.val_every=${VAL_EVERY:-1000} train.log_every=100 train.batch_size=64 \
  train.eval_train_clips=${EVAL_TRAIN:-45} ${EXTRA:-}
