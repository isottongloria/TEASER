#!/bin/bash
#SBATCH --job-name=temporal_pilot_stats
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Dataset statistics and review page (TEMPORAL_README.md 4.2, 5.4), all automatic:
# mouth/nose/chin regions for every pilot clip, occlusion statistics per corpus
# (occluded-frame shares, severity classes, episode severity, real vs fake;
# corpora in parallel, each with its own worker pool), report JSON, HTML page.
# No GPU needed (CPU work on a booster node). MAX_VARIANTS caps the variants
# measured per mode and corpus (default all).
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
echo "== regions (mouth/nose/chin) $(date +%T)"
for C in phoenix csl_daily how2sign; do
  PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/export_face_regions.py \
    --fits_of $D/$C/lists/fits.tsv --clips $D/$C/lists/pilot_all.txt --out $D/$C/regions &
done
wait
echo "== statistics $(date +%T)"
declare -A W=([phoenix]=8 [csl_daily]=16 [how2sign]=8)
for C in phoenix csl_daily how2sign; do
  PYTHONPATH=$PWD $TPY -W ignore tools_temporal/occlusion_dataset_stats.py --corpus $C --workers ${W[$C]} \
    --max_variants ${MAX_VARIANTS:-0} --out $D/report/stats_$C.json 2>&1 | grep "^\[stats\]" &
done
wait
echo "== report and page $(date +%T)"
PYTHONPATH=$PWD $TPY -W ignore tools_temporal/report_dataset.py --out $D/report/report.json | tail -3
$TPY tools_temporal/build_review_page.py --report $D/report/report.json --notes $D/report/notes.json \
  --out $D/report/temporal_teaser_pilot_dataset.html
echo "== done $(date +%T)"
