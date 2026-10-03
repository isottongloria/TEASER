#!/bin/bash
#SBATCH --job-name=temporal_pilot_occlusion
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
#SBATCH --error=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.err
#
# Pilot step 2 (TEMPORAL_README.md 4.2): real occlusion of the candidate clips,
# from their SMPL-X fits only (no frames). 16 parallel shards. Re-runnable:
# clips that already have an .occ.npz are skipped.
set -uo pipefail
D=data/temporal
RPY=$HOME/miniconda3/envs/rgb2smplx/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX
run() {  # fits_root list out fps
  split -n l/8 -d "$2" "$2.shard_"
  for s in "$2".shard_*; do
    $RPY tools_temporal/compute_real_occlusion.py --fits "$1" --clips "$s" --out "$3" --fps "$4" > "$s.log" 2>&1 &
  done
}
cut -f1 $D/csl_daily/lists/candidates.tsv > $D/csl_daily/lists/candidates.txt
run /leonardo_work/IscrC_SIGMA/rgb2smplx/csl_daily/csldaily $D/csl_daily/lists/candidates.txt $D/csl_daily/occ 30
for sp in train dev test; do
  grep "^$sp/" $D/phoenix/lists/candidates_paths.txt | cut -d/ -f2 > $D/phoenix/lists/candidates_$sp.txt
done
# PHOENIX: one process per split directory, 8 shards over the train list
run /leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/train $D/phoenix/lists/candidates_train.txt $D/phoenix/occ 25
for sp in dev test; do
  $RPY tools_temporal/compute_real_occlusion.py --fits /leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/$sp \
    --clips $D/phoenix/lists/candidates_$sp.txt --out $D/phoenix/occ --fps 25 > $D/phoenix/lists/candidates_$sp.log 2>&1 &
done
wait
cat $D/*/lists/*.log | grep -v "^\[occlusion\] [0-9]*/" | sort | uniq -c
rm -f $D/*/lists/*.shard_*
echo "csl_daily occ: $(ls $D/csl_daily/occ | wc -l)  phoenix occ: $(ls $D/phoenix/occ | wc -l)"
