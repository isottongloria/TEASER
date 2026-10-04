#!/bin/bash
#SBATCH --job-name=temporal_v2_preview
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Synthetic occlusion v2: face regions from the fits for every pilot clip, the
# statistics again (eyes-only rate), and a preview of the new variants on 2
# training clips per corpus (variants_preview/), to look at before regenerating.
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
declare -A FPS=([phoenix]=25 [csl_daily]=30 [how2sign]=30)
for C in phoenix csl_daily how2sign; do
  cat $D/$C/lists/{train,val,test}.txt > $D/$C/lists/pilot_all.txt
  PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/export_face_regions.py \
    --fits_of $D/$C/lists/fits.tsv --clips $D/$C/lists/pilot_all.txt --out $D/$C/regions &
done
wait
for C in phoenix csl_daily how2sign; do
  $TPY tools_temporal/occlusion_stats.py --occ $D/$C/occ --clips $D/$C/lists/train_signer_candidates.txt \
    --out $D/$C/lists/occlusion_stats_train.json --fps ${FPS[$C]}
  head -2 $D/$C/lists/train.txt | cut -f1 > $D/$C/lists/preview.txt
  PYTHONPATH=$PWD $TPY tools_temporal/make_synthetic_variants.py --clips_root $D/$C/clips --cache $D/$C/cache \
    --out $D/$C/variants_preview --hands $D/hand_bank/train --hand_list $D/$C/lists/hands_train.txt \
    --stats $D/$C/lists/occlusion_stats_train.json --segments $D/$C/lists/segments.json --regions $D/$C/regions \
    --variants 2 --names $D/$C/lists/preview.txt 2>/dev/null | grep "^\[variants\]" &
done
wait
