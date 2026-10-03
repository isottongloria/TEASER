#!/bin/bash
#SBATCH --job-name=temporal_h2s_post
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
# How2Sign after the pipeline (TEMPORAL_README.md 4.2d): occlusion from the
# fits, clean-clip eligibility (signer = source video for now; real signers are
# assigned afterwards by face identity), TEASER caches of the eligible chunks.
set -uo pipefail
D=$PWD/data/temporal/how2sign
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
ls $D/clips/*/fit_gvhmr_upper.npz 2>/dev/null | while read f; do c=$(basename $(dirname $f)); echo -e "$c\t$f"; done > $D/lists/fits.tsv
cut -f1 $D/lists/fits.tsv > $D/lists/fitted.txt
echo "fitted chunks: $(wc -l < $D/lists/fitted.txt)"
split -n l/8 -d $D/lists/fitted.txt $D/lists/fitted_shard_
for s in $D/lists/fitted_shard_*; do
  PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/compute_real_occlusion.py --fits $D/clips \
    --fit-name fit_gvhmr_upper.npz --clips $s --out $D/occ --fps 30 > $s.log 2>&1 &
done
wait; cat $D/lists/fitted_shard_*.log | grep -v "^\[occlusion\] [0-9]*/"; rm -f $D/lists/fitted_shard_*
# provisional signer = source video id
awk -F'\t' '{split($1,a,"_c"); print $1"\t"a[1]}' $D/lists/fits.tsv > $D/lists/candidates_by_video.tsv
videos=$(cut -f2 $D/lists/candidates_by_video.tsv | sort -u | paste -sd,)
$TPY tools_temporal/select_clean_clips.py --occ $D/occ --signers $D/lists/candidates_by_video.tsv \
  --split all=$videos --pick all=10000 --max_per_signer 10000 --out $D/lists/eligibility
cache() { c=$1; [ -f $D/cache/$c.npz ] && return
  PYTHONPATH=$PWD $TPY tools_temporal/extract_features.py $D/clips/$c $D/cache/$c.npz --temporal_feats expr+pose \
    --batch-size 1 2>/dev/null | grep -q extract_features || echo "[cache FAILED] $c"; }
export -f cache; export D TPY
cut -f1 $D/lists/eligibility/all.txt | xargs -P 6 -I{} bash -c 'cache {}'
echo "how2sign caches: $(ls $D/cache | wc -l)"
