#!/bin/bash
#SBATCH --job-name=temporal_pilot_cache
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#SBATCH --error=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.err
#
# Pilot steps 5-6 (TEMPORAL_README.md 4.2) for PHOENIX and CSL-Daily: frames of
# the selected clips (train/val/test lists), checked against the fits, then the
# TEASER feature caches (expr+pose, batch 1), 4 processes on one GPU.
# Re-runnable: existing frames/caches are skipped.
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
R=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

echo "== PHOENIX frames (links to the extracted frames)"
for split in train val test; do cut -f1 $D/phoenix/lists/$split.txt; done > $D/phoenix/lists/pilot.txt
while read -r clip; do
  src=$(awk -v c=$clip '$1==c{print $2}' $D/phoenix/lists/source_split.tsv)
  dst=$D/phoenix/clips/$clip; mkdir -p $dst
  [ -e $dst/frames ] || ln -s /leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/$src/$clip/frames $dst/frames
  [ -f $dst/video.json ] || echo '{"output_fps": 25.0, "source": "PHOENIX-2014-T"}' > $dst/video.json
done < $D/phoenix/lists/pilot.txt

echo "== CSL-Daily frames (RGB2SMPLX prepare --target-fps 30, 8 parallel)"
for split in train val test; do cut -f1 $D/csl_daily/lists/$split.txt; done > $D/csl_daily/lists/pilot.txt
prep() {
  clip=$1; dst=$D/csl_daily/clips/$clip
  [ -f $dst/video.json ] && return
  (cd $R && PYTHONPATH=$R $RPY -m rgb2smplx.stages.prepare /leonardo_work/IscrC_SIGMA/csl_daily_crop/videos/$clip.mp4 \
     $dst --target-fps 30 > /dev/null 2>&1) || echo "[prepare FAILED] $clip"
}
export -f prep; export D RPY R
xargs -P 8 -I{} bash -c 'prep {}' < $D/csl_daily/lists/pilot.txt
$RPY - <<PY
import numpy as np, os
D="$D/csl_daily"; bad = 0
for clip in open(f"{D}/lists/pilot.txt").read().split():
    n_frames = len(os.listdir(f"{D}/clips/{clip}/frames")) if os.path.isdir(f"{D}/clips/{clip}/frames") else -1
    n_fit = len(np.load(f"/leonardo_work/IscrC_SIGMA/rgb2smplx/csl_daily/csldaily/{clip}/fit_gvhmr_face_teaser_upper.npz")["smplx_root_pose"])
    if n_frames != n_fit:
        bad += 1; print("[MISALIGNED]", clip, n_frames, n_fit)
print(f"[csl_daily] frame count == fit frame count on {150 - bad}/150 clips")
PY

echo "== TEASER feature caches (4 processes)"
cache() {
  corpus=$1; clip=$2; out=$D/$corpus/cache/$clip.npz
  [ -f $out ] && return
  PYTHONPATH=$PWD $TPY tools_temporal/extract_features.py $D/$corpus/clips/$clip $out \
    --temporal_feats expr+pose --batch-size 1 2>/dev/null | grep extract_features || echo "[cache FAILED] $corpus $clip"
}
export -f cache; export TPY
{ sed 's/^/phoenix /' $D/phoenix/lists/pilot.txt; sed 's/^/csl_daily /' $D/csl_daily/lists/pilot.txt; } \
  | xargs -P 4 -L 1 bash -c 'cache $0 $1' > $D/cache_log.txt
grep -c extract_features $D/cache_log.txt; grep FAILED $D/cache_log.txt
echo "phoenix caches: $(ls $D/phoenix/cache | wc -l)  csl_daily caches: $(ls $D/csl_daily/cache | wc -l)"
