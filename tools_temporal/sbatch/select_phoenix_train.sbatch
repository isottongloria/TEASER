#!/bin/bash
#SBATCH --job-name=temporal_select_phoenix
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#SBATCH --error=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.err
#
# Score a random sample of PHOENIX train clips for hand-face occlusion and write
# the Pixel3DMM clip lists (TEMPORAL_README.md 3): round 1 = 100, round 2 = +200.
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX OMP_NUM_THREADS=8
~/miniconda3/envs/rgb2smplx/bin/python tools_temporal/select_train_clips.py \
  --corpus /leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/train \
  --out tools_temporal/lists/phoenix_train --rounds 100 200 --clean-frac 0.75 \
  --sample ${SAMPLE:-1200} --seed 0
