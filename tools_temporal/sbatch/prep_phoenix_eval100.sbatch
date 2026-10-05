#!/bin/bash
#SBATCH --job-name=temporal_prep_eval100
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# The 100 PHOENIX test clips of RGB2SMPLX's occlusion evaluation (never in our
# training): frames linked, real occlusion from the TEASER-face SMPL-X fits,
# TEASER feature caches. For the real-occlusion test (no ground truth: stability
# and continuity on really occluded frames).
set -uo pipefail
D=$PWD/data/temporal/phoenix_eval100
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
F=/leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/test
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
mkdir -p $D/clips $D/occ $D/cache
cut -f1 $D/lists/all.txt > $D/lists/names.txt
while read -r c; do
  mkdir -p $D/clips/$c; [ -e $D/clips/$c/frames ] || ln -s $F/$c/frames $D/clips/$c/frames
  [ -f $D/clips/$c/video.json ] || echo '{"output_fps": 25.0, "source": "PHOENIX-2014-T"}' > $D/clips/$c/video.json
done < $D/lists/names.txt
split -n l/8 -d $D/lists/names.txt $D/lists/names.part_
for p in $D/lists/names.part_*; do
  PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/compute_real_occlusion.py --fits $F --clips $p --out $D/occ --fps 25 > $p.log 2>&1 &
done
wait; cat $D/lists/names.part_*.log | grep -v "^\[occlusion\] [0-9]*/"; rm -f $D/lists/names.part_*
cache() { c=$1; [ -f $D/cache/$c.npz ] && return
  PYTHONPATH=$PWD $TPY tools_temporal/extract_features.py $D/clips/$c $D/cache/$c.npz --temporal_feats expr+pose \
    --batch-size 1 2>/dev/null | grep -q extract_features || echo "[cache FAILED] $c"; }
export -f cache; export D TPY
xargs -P 6 -I{} bash -c 'cache {}' < $D/lists/names.txt
echo "caches: $(ls $D/cache | wc -l)  occ: $(ls $D/occ | wc -l)"
