#!/bin/bash
#SBATCH --job-name=temporal_pilot_banks
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# Pilot banks (TEMPORAL_README.md 5.4): per corpus, the hand bank (training
# signers -> train, val+test signers -> heldout) and the replay bank of real
# occlusion dynamics (same split), all corpora in parallel.
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX
declare -A MINPX=([phoenix]=18 [csl_daily]=60 [how2sign]=40)
for C in phoenix csl_daily how2sign; do
  cat $D/$C/lists/val.txt $D/$C/lists/test.txt > $D/$C/lists/heldout.txt
  train_s=$(cut -f2 $D/$C/lists/train.txt | sort -u | paste -sd,)
  held_s=$(cut -f2 $D/$C/lists/heldout.txt | sort -u | paste -sd,)
  ( $RPY tools_temporal/build_hand_bank.py --corpus $C --clips_root $D/$C/clips --fits_of $D/$C/lists/fits.tsv \
      --split train=$D/$C/lists/train.txt --split heldout=$D/$C/lists/heldout.txt --out $D/hand_bank/$C \
      --min_px ${MINPX[$C]} 2>&1 | grep "^\[hands\]"
    for s in train heldout; do ls $D/hand_bank/$C/$s | sed 's/\.png$//' > $D/$C/lists/hands_$s.txt; done ) &
  $RPY tools_temporal/build_replay_bank.py --corpus $C --occ $D/$C/occ --fits_of $D/$C/lists/fits.tsv \
    --signers $D/$C/lists/candidates.tsv --split train=$train_s --split heldout=$held_s \
    --out $D/replay_bank/$C 2>&1 | grep "^\[replay\]" &
done
wait
for C in phoenix csl_daily how2sign; do
  echo "$C hands: train $(wc -l < $D/$C/lists/hands_train.txt), heldout $(wc -l < $D/$C/lists/hands_heldout.txt);" \
       "replay: train $(wc -l < $D/replay_bank/$C/train/$C.jsonl), heldout $(wc -l < $D/replay_bank/$C/heldout/$C.jsonl)"
done
