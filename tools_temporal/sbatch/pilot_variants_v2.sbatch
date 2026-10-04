#!/bin/bash
#SBATCH --job-name=temporal_pilot_variants_v2
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
# Synthetic variants v2 (TEMPORAL_README.md 5.4) for all pilot clips, MODE=param
# (calibrated parametric episodes, -> variants/) or MODE=replay (real occlusion
# dynamics, -> variants_replay/): 3 per training clip with training hands and
# training-signer dynamics, 2 per val/test clip with held-out ones. 8 processes.
set -uo pipefail
D=$PWD/data/temporal
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
MODE=${MODE:?MODE=param|replay}
OUT=variants; [ $MODE = replay ] && OUT=variants_replay
variants() {
  C=$1; clip=$2; k=$3; split=$4
  extra=""
  [ $MODE = replay ] && extra="--replay $D/replay_bank/$C/$split/$C.jsonl --hand_index $D/hand_bank/$C/index.tsv"
  echo "$clip" > /tmp/v_$$_$clip.txt
  PYTHONPATH=$PWD $TPY tools_temporal/make_synthetic_variants.py --clips_root $D/$C/clips --cache $D/$C/cache \
    --out $D/$C/$OUT --hands $D/hand_bank/$C/$split --hand_list $D/$C/lists/hands_$split.txt \
    --stats $D/$C/lists/occlusion_stats_train.json --segments $D/$C/lists/segments.json --regions $D/$C/regions \
    --variants $k --names /tmp/v_$$_$clip.txt $extra 2>/dev/null | grep "^\[variants\]" || echo "[variants FAILED] $C $clip"
  rm -f /tmp/v_$$_$clip.txt
}
export -f variants; export D TPY MODE OUT
for C in phoenix csl_daily how2sign; do
  cut -f1 $D/$C/lists/train.txt | sed "s/^/$C /; s/$/ 3 train/"
  cut -f1 $D/$C/lists/heldout.txt | sed "s/^/$C /; s/$/ 2 heldout/"
done | xargs -P 8 -L 1 bash -c 'variants $0 $1 $2 $3' > $D/variants_${MODE}_log_$SLURM_JOB_ID.txt
grep -c "^\[variants\]" $D/variants_${MODE}_log_$SLURM_JOB_ID.txt; grep FAILED $D/variants_${MODE}_log_$SLURM_JOB_ID.txt | head
for C in phoenix csl_daily how2sign; do echo "$C $OUT: $(ls $D/$C/$OUT | wc -l)"; done
