#!/bin/bash
#SBATCH --job-name=temporal_test_step1
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
# Step 1 checks (TEMPORAL_README.md 7): split forward == TeaserEncoder.forward
# (CPU + CUDA), and the feature cache == RGB2SMPLX's teaser.npz on a real clip.
set -euo pipefail
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
CLIP=${CLIP:-/leonardo_work/IscrC_SLPSCALE/csl/test/S005996_P0006_T00}
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTHONPATH=$PWD
nvidia-smi --query-gpu=name --format=csv,noheader
$TPY -m unittest discover -s tests/temporal -p 'test_*.py' -v
$TPY tests/temporal/check_cache_vs_stage.py "$CLIP" --out-dir "$PWD/logs/cache_vs_stage.$SLURM_JOB_ID"
