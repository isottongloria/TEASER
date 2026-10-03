#!/bin/bash
#SBATCH --job-name=temporal_phoenix_smoke
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
# A few PHOENIX test samples to run the whole Track 1 pipeline end to end
# (smoke test, not a dataset): real occlusion for CANDIDATES clips (not the
# 100 eval clips) from Chang's TEASER-face SMPL-X fits, PICK of them linked
# (half with a real occlusion episode), and their TEASER feature caches.
set -euo pipefail
D=/leonardo_work/IscrC_SLPSCALE/temporal_teaser/smoke
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
FITS=/leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/test
EVAL=/leonardo_work/IscrC_SLPSCALE/temporal_teaser/phoenix_eval100.txt
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
mkdir -p $D/occ $D/cache

ls $FITS | grep -vxFf $EVAL | head -${CANDIDATES:-30} > $D/candidates.txt
PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/compute_real_occlusion.py \
  --fits $FITS --out $D/occ --clips $D/candidates.txt
$TPY tools_temporal/occlusion_stats.py --occ $D/occ --out $D/occlusion_stats.json
$TPY tools_temporal/make_phoenix_clips.py --split test --out $D/clips --occ $D/occ \
  --pick ${PICK:-6} --exclude $EVAL
export PYTHONPATH=$PWD
for clip in $(cut -f1 $D/clips/signers.tsv); do
  $TPY tools_temporal/extract_features.py $D/clips/$clip $D/cache/$clip.npz \
    --temporal_feats expr+pose --batch-size 1 2>/dev/null | grep extract_features || echo "[FAILED] $clip"
done
