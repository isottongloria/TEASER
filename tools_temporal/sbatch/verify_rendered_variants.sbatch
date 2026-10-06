#!/bin/bash
#SBATCH --job-name=temporal_verify_rendered
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
#
# 10% of the rendered variants, re-extracted with TEASER and compared with the
# caches training / evaluation used (verify_rendered_variants.py); 8 processes
# on 4 GPUs, plus 2 clean clips each as the re-run noise floor.
set -uo pipefail
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
OUT=data/temporal/report/verify_rendered; mkdir -p $OUT
for ((i = 0; i < 8; i++)); do
  CUDA_VISIBLE_DEVICES=$((i % 4)) PYTHONPATH=$PWD $TPY tools_temporal/verify_rendered_variants.py \
    --fraction 0.1 --shard $i/8 --clean 2 --out $OUT/check_$i.json > $OUT/log_$i.txt 2>&1 &
done
wait
$TPY - <<'PY'
import json, glob
import numpy as np
V, C = {}, {}
for f in sorted(glob.glob("data/temporal/report/verify_rendered/check_*.json")):
    d = json.load(open(f)); V.update(d["variants"]); C.update(d["clean"]); pop = d["population"]
print("population", pop, "sum", sum(pop.values()))
print("checked variants", len(V), "frames", sum(v["frames"] for v in V.values()),
      "pasted", sum(v["pasted_frames"] for v in V.values()))
for part in ("all", "pasted"):
    for k in ("feat_expr", "feat_pose", "expression", "jaw_pose", "eyelid", "landmarks"):
        m = [v[part][k]["max"] for v in V.values()]
        print(f"variants {part:6s} {k:10s} max {max(m):.3e}  median-of-max {np.median(m):.3e}  "
              f"exact {sum(x == 0 for x in m)}/{len(m)}  scale {np.mean([v[part][k]['ref_scale'] for v in V.values()]):.3e}")
print("valid equal", sum(v["all"]["valid_equal"] for v in V.values()), "/", len(V))
for k in ("feat_expr", "feat_pose", "expression", "landmarks"):
    m = [c[k]["max"] for c in C.values()]
    print(f"clean re-run {k:10s} max {max(m):.3e}  exact {sum(x == 0 for x in m)}/{len(m)}")
PY
