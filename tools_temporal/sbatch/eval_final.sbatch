#!/bin/bash
#SBATCH --job-name=temporal_eval_final
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Final evaluation (TEMPORAL_README.md 6-7): the final configuration (3 seeds) vs
# TEASER, SG9, the current pipeline and SmoothNet, on the TEST set (held-out
# signers, hands and dynamics; replay and parametric fakes) and on the 100
# PHOENIX clips with real occlusion (all, and the held-out signers only).
set -uo pipefail
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
E="$TPY -W ignore tools_temporal/eval_temporal.py --method T0=teaser --method T1=sg9 --method T2=sg9+interp --method T3_s0+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P2_T3_smoothnet/best.pt+sg9 --method T3_s1+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P3_T3_s1/best.pt+sg9 --method T3_s2+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P3_T3_s2/best.pt+sg9 --method final_s0=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s0/best.pt --method final_s0+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s0/best.pt+sg9 --method final_s1=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s1/best.pt --method final_s1+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s1/best.pt+sg9 --method final_s2=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s2/best.pt --method final_s2+sg9=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/P4_final_s2/best.pt+sg9"
$E --cache /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/cache --occ /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/occ --variants /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/variants_replay --list /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/lists/test.txt --out /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/eval_final_test_replay.json
$E --cache /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/cache --occ /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/occ --variants /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/variants --list /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/all/lists/test.txt --out /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/eval_final_test_param.json
$E --cache /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/cache --occ /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/occ --list /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/lists/all.txt --out /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/eval_final_real100.json
$E --cache /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/cache --occ /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/occ --list /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/data/temporal/phoenix_eval100/lists/heldout_signers.txt --out /leonardo_work/IscrC_SLPSCALE/TEASER_temporal/runs/eval_final_real100_heldout.json
