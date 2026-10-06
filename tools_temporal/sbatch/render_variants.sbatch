#!/bin/bash
#SBATCH --job-name=temporal_render_variants
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=120G
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Every synthetic-occlusion variant (train, val, test; param and replay) written
# out as frame folders under data/temporal/occluded_videos (render_variants.py),
# 32 processes. Resubmittable: finished variants are skipped.
set -uo pipefail
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
N=${N:-11}
for C in phoenix csl_daily how2sign; do for ((i = 0; i < N; i++)); do echo "$C $i/$N"; done; done |
  xargs -P 32 -L 1 bash -c 'PYTHONPATH=$PWD '$TPY' tools_temporal/render_variants.py --corpus $0 --shard $1' \
  > logs/render_variants_$SLURM_JOB_ID.txt 2>&1
grep -c "^\[render\].*pasted" logs/render_variants_$SLURM_JOB_ID.txt
grep -c "MISMATCH" logs/render_variants_$SLURM_JOB_ID.txt
grep -i "error\|Traceback" logs/render_variants_$SLURM_JOB_ID.txt | head
grep "shard" logs/render_variants_$SLURM_JOB_ID.txt
du -sh data/temporal/occluded_videos
