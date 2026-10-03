#!/bin/bash
#SBATCH --job-name=temporal_smoke_pipeline
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
# End-to-end smoke test of the Track 1 pipeline on the few PHOENIX test samples
# of phoenix_smoke.sbatch: synthetic variants (procedural hands), tiny T3/T4/T7
# trainings, evaluation of T0-T7, inference to teaser_temporal.npz, the demo
# with --temporalize_teaser, and the flag-OFF check of the demo (same output
# as the production TEASER checkout). Numbers are meaningless (4 train clips,
# 200 steps); only that every step runs and the checks hold.
set -euo pipefail
D=/leonardo_work/IscrC_SLPSCALE/temporal_teaser/smoke
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
PROD=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/third_party/TEASER
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTHONPATH=$PWD
STEPS=${STEPS:-200}

echo "== unit tests"
$TPY -W ignore -m unittest discover -s tests/temporal -p 'test_*.py' 2>&1 | tail -3

echo "== splits (one signer per clip: train/val are signer-disjoint)"
cut -f1 $D/clips/signers.tsv | head -4 > $D/train.txt
cut -f1 $D/clips/signers.tsv | tail -2 > $D/val.txt

echo "== synthetic variants (procedural hands: smoke only)"
$TPY tools_temporal/make_synthetic_variants.py --clips_root $D/clips --cache $D/cache --out $D/variants \
  --hands procedural --stats $D/occlusion_stats.json --variants 2 2>/dev/null | grep -v "No face detected"

COMMON="data.cache_dir=$D/cache data.occ_dir=$D/occ data.variant_dir=$D/variants data.train_list=$D/train.txt \
  data.val_list=$D/val.txt train.steps=$STEPS train.val_every=100 train.log_every=50 train.batch_size=32"
for run in T3 T4 T7; do
  echo "== train $run"
  $TPY tools_temporal/train_temporal.py configs/temporal/$run.yaml out_dir=$D/runs/$run $COMMON \
    data.feature_set=expr+pose 2>&1 | grep -E "^\[train\]" | grep -vE "step=[0-9]*[1-9]0 " || true
done

echo "== eval"
$TPY tools_temporal/eval_temporal.py --cache $D/cache --occ $D/occ --variants $D/variants --list $D/val.txt \
  --method T0=teaser --method T1=sg9 --method T2=sg9+interp --method T3=$D/runs/T3/last.pt \
  --method T4=$D/runs/T4/last.pt --method T7=$D/runs/T7/last.pt --out $D/eval_val.json 2>/dev/null

echo "== infer_clip -> teaser_temporal.npz"
CLIP=$(head -1 $D/val.txt)
$TPY tools_temporal/infer_clip.py $D/clips/$CLIP $D/teaser_temporal_$CLIP.npz --temporal_ckpt $D/runs/T7/last.pt \
  --cache $D/cache/$CLIP.npz --occ $D/occ/$CLIP.occ.npz 2>/dev/null
$TPY - <<PY
import numpy as np
t = np.load("$D/teaser_temporal_$CLIP.npz"); c = np.load("$D/cache/$CLIP.npz")
keys = ("frame_index", "valid", "pose_params", "jaw_pose", "expression", "eyelid", "cam", "shape_params")
print("teaser.npz keys present:", all(k in t.files for k in keys),
      "| shapes match:", all(t[k].shape == c[k].shape for k in keys),
      "| expression moved from TEASER by", float(np.abs(t["expression"] - c["expression"]).mean()))
PY

echo "== demo: video from a clip's frames"
mkdir -p $D/demo_in $D/demo_out_prod $D/demo_out_off $D/demo_out_on
$TPY - <<PY
import cv2, glob
files = sorted(glob.glob("$D/clips/$CLIP/frames/*.png"))
first = cv2.imread(files[0]); h, w = first.shape[:2]
out = cv2.VideoWriter("$D/demo_in/$CLIP.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
for f in files: out.write(cv2.imread(f))
out.release()
PY
echo "== demo flag OFF: production checkout vs this branch"
(cd $PROD && PYTHONPATH=$PROD $TPY main/demo_video.py --input_path $D/demo_in --out_path $D/demo_out_prod \
   --checkpoint pretrained_models/TEASER.pt --crop > /dev/null 2>&1)
$TPY main/demo_video.py --input_path $D/demo_in --out_path $D/demo_out_off \
   --checkpoint pretrained_models/TEASER.pt --crop > /dev/null 2>&1
$TPY - <<PY
import glob, imageio, numpy as np
a = sorted(glob.glob("$D/demo_out_prod/**/*.mp4", recursive=True))
b = sorted(glob.glob("$D/demo_out_off/**/*.mp4", recursive=True))
fa = [np.asarray(x) for x in imageio.mimread(a[0], memtest=False)]
fb = [np.asarray(x) for x in imageio.mimread(b[0], memtest=False)]
same = len(fa) == len(fb) and all(np.array_equal(x, y) for x, y in zip(fa, fb))
print(f"flag OFF: {len(fa)} vs {len(fb)} frames, identical output: {same}")
PY
echo "== demo flag ON"
$TPY main/demo_video.py --input_path $D/demo_in --out_path $D/demo_out_on --checkpoint pretrained_models/TEASER.pt \
  --crop --temporalize_teaser --temporal_ckpt $D/runs/T7/last.pt 2>&1 | grep -E "temporal demo|Error|Traceback"
echo "== done"
