#!/bin/bash
#SBATCH --job-name=temporal_pilot_variants
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Pilot steps after the caches (TEMPORAL_README.md 4.2, 5.4), per corpus given in
# CORPORA (default "phoenix csl_daily"):
#  1. hand bank: train signers -> bank/train, val+test signers -> bank/heldout;
#  2. occlusion statistics of the training signers' candidates (sizing the fakes);
#  3. synthetic variants inside the clean segments: 3 per training clip with
#     training hands, 2 per val/test clip with held-out hands. 8 processes.
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
CORPORA=${CORPORA:-"phoenix csl_daily"}
declare -A MINPX=([phoenix]=18 [csl_daily]=60 [how2sign]=40)
declare -A FPS=([phoenix]=25 [csl_daily]=30 [how2sign]=30)

for C in $CORPORA; do
  echo "== $C: hand bank"
  cat $D/$C/lists/val.txt $D/$C/lists/test.txt > $D/$C/lists/heldout.txt
  PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/build_hand_bank.py --corpus $C \
    --clips_root $D/$C/clips --fits_of $D/$C/lists/fits.tsv --split train=$D/$C/lists/train.txt \
    --split heldout=$D/$C/lists/heldout.txt --out $D/hand_bank --min_px ${MINPX[$C]} 2>&1 | grep "^\[hands\]"
  for s in train heldout; do ls $D/hand_bank/$s | grep "^${C}__" | sed 's/\.png$//' > $D/$C/lists/hands_$s.txt; done
  echo "hands: train $(wc -l < $D/$C/lists/hands_train.txt), heldout $(wc -l < $D/$C/lists/hands_heldout.txt)"

  echo "== $C: occlusion statistics of the training signers"
  awk -F'\t' 'NR==FNR {keep[$2]=1; next} ($2 in keep) {print $1}' $D/$C/lists/train.txt \
    $D/$C/lists/candidates.tsv > $D/$C/lists/train_signer_candidates.txt
  $TPY tools_temporal/occlusion_stats.py --occ $D/$C/occ --clips $D/$C/lists/train_signer_candidates.txt \
    --out $D/$C/lists/occlusion_stats_train.json --fps ${FPS[$C]}
done

echo "== synthetic variants (8 processes)"
variants() {
  C=$1; clip=$2; k=$3; hands=$4
  echo "$clip" > /tmp/v_$$_$clip.txt
  PYTHONPATH=$PWD $TPY tools_temporal/make_synthetic_variants.py --clips_root $D/$C/clips --cache $D/$C/cache \
    --out $D/$C/variants --hands $D/hand_bank/$hands --hand_list $D/$C/lists/hands_$hands.txt \
    --stats $D/$C/lists/occlusion_stats_train.json --segments $D/$C/lists/segments.json --variants $k \
    --names /tmp/v_$$_$clip.txt 2>/dev/null | grep "^\[variants\]" || echo "[variants FAILED] $C $clip"
  rm -f /tmp/v_$$_$clip.txt
}
export -f variants; export D TPY
for C in $CORPORA; do
  cut -f1 $D/$C/lists/train.txt | sed "s/^/$C /; s/$/ 3 train/"
  cut -f1 $D/$C/lists/heldout.txt | sed "s/^/$C /; s/$/ 2 heldout/"
done | xargs -P 8 -L 1 bash -c 'variants $0 $1 $2 $3' > $D/variants_log_$SLURM_JOB_ID.txt
grep -c "^\[variants\]" $D/variants_log_$SLURM_JOB_ID.txt; grep FAILED $D/variants_log_$SLURM_JOB_ID.txt | head
for C in $CORPORA; do echo "$C variants: $(ls $D/$C/variants | wc -l)"; done
