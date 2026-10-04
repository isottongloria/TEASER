#!/bin/bash
#SBATCH --job-name=temporal_prep_v3
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
# Before the v3 variants: mouth/nose/chin regions for every pilot clip, real
# occlusion statistics of the training signers with per-episode (mnc peak, lip
# peak, frames) triples, and the pooled replay bank for How2Sign (its training
# signer has only ~30 real episodes; PHOENIX and CSL-Daily dynamics are added,
# training with training, held-out with held-out).
set -uo pipefail
D=$PWD/data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
declare -A FPS=([phoenix]=25 [csl_daily]=30 [how2sign]=30)
for C in phoenix csl_daily how2sign; do
  split -n l/4 -d $D/$C/lists/pilot_all.txt $D/$C/lists/pilot_all.part_
  for p in $D/$C/lists/pilot_all.part_*; do
    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX $RPY tools_temporal/export_face_regions.py \
      --fits_of $D/$C/lists/fits.tsv --clips $p --out $D/$C/regions &
  done
done
wait
rm -f $D/*/lists/pilot_all.part_*
for C in phoenix csl_daily how2sign; do
  $TPY tools_temporal/occlusion_stats.py --occ $D/$C/occ --clips $D/$C/lists/train_signer_candidates.txt \
    --out $D/$C/lists/occlusion_stats_train.json --fps ${FPS[$C]}
done
for s in train heldout; do
  mkdir -p $D/replay_bank/how2sign_pooled/$s
  cat $D/replay_bank/{how2sign,phoenix,csl_daily}/$s/*.jsonl > $D/replay_bank/how2sign_pooled/$s/how2sign.jsonl
  echo "how2sign pooled replay $s: $(wc -l < $D/replay_bank/how2sign_pooled/$s/how2sign.jsonl)"
done
$TPY - <<PY
import numpy as np, glob
for C in ("phoenix", "csl_daily", "how2sign"):
    n = sum("mnc" in np.load(f).files for f in glob.glob(f"$D/{C}/regions/*.npz"))
    print(C, "regions with mnc:", n)
PY
