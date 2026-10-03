#!/bin/bash
#SBATCH --job-name=temporal_h2s_pipeline
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%A_%a.out
#
# How2Sign pilot (TEMPORAL_README.md 4.2d): the full RGB2SMPLX pipeline, code
# unchanged, over the chunks of make_how2sign_chunks.py -- prepare (30 fps cap),
# RTMDet, Sapiens2, WiLoR, GVHMR, fit (--solve-joints upper, like the PHOENIX fits).
# No face stage: the occlusion meter needs only the body/hand fit and its camera,
# and TEASER runs later in our own cache. Budget: 4 GPUs x 1 hour (array 0-3,
# hard time limit). Each task takes a quarter of the chunks in two batches, so
# the first batch is complete even if the second one runs out of time.
set -uo pipefail
R=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX
E=$HOME/miniconda3/envs
H=$PWD/data/temporal/how2sign
W=$H/clips
T=${SLURM_ARRAY_TASK_ID:-0}
export PYTHONNOUSERSITE=1 PYTHONPATH=$R TQDM_DISABLE=1 HF_HUB_OFFLINE=1
mkdir -p $W $W/videos $W/gvhmr_batch $H/lists/tasks
cut -f1 $H/chunks/chunks.tsv | awk -v t=$T 'NR % 4 == t' > $H/lists/tasks/task$T.txt
split -n l/2 -d $H/lists/tasks/task$T.txt $H/lists/tasks/task${T}_batch
cd $R
for B in $H/lists/tasks/task${T}_batch*; do
  echo "== task $T $(basename $B): $(wc -l < $B) chunks, $(date +%T)"
  while read -r c; do
    [ -f $W/$c/video.json ] || $E/rgb2smplx/bin/python -m rgb2smplx.stages.prepare $H/chunks/$c.mp4 $W/$c --target-fps 30 \
      > /dev/null 2>&1 || echo "[prepare FAILED] $c"
  done < $B
  echo "prepare done $(date +%T)"
  $E/rgb2smplx-rtmdet/bin/python scripts/run_corpus_stage.py rtmdet --work-root $W --clips $B --device cuda 2>&1 | tail -1
  $E/rgb2smplx-sapiens2/bin/python scripts/run_corpus_stage.py sapiens2 --work-root $W --clips $B --dtype float16 --batch-size 16 2>&1 | tail -1
  $E/rgb2smplx-wilor/bin/python scripts/run_corpus_stage.py wilor --work-root $W --clips $B 2>&1 | tail -1
  echo "rtmdet/sapiens2/wilor done $(date +%T)"
  while read -r c; do
    [ -f $W/videos/$c.mp4 ] && continue
    FPS=$($E/rgb2smplx/bin/python -c "import json;print(json.load(open('$W/$c/video.json'))['output_fps'])")
    $E/rgb2smplx-vid2smplx/bin/ffmpeg -nostdin -loglevel error -y -framerate $FPS -i $W/$c/frames/%06d.jpg \
      -c:v libx264 -crf 12 -pix_fmt yuv420p -r $FPS $W/videos/$c.mp4
  done < $B
  (cd third_party/vid2smplx/GVHMR && PYTHONPATH= $E/rgb2smplx-vid2smplx/bin/python $R/scripts/gvhmr_batch.py \
      --video-dir $W/videos --output-root $W/gvhmr_batch --clips $B > $H/lists/tasks/gvhmr_task$T.log 2>&1)
  while read -r c; do
    [ -f $W/$c/gvhmr.npz ] && continue
    $E/rgb2smplx/bin/python scripts/gvhmr_to_body_npz.py $W/$c $W/$c/gvhmr.npz \
      --result $W/gvhmr_batch/$c/hmr4d_results.pt --frames $(ls $W/$c/frames | wc -l) > /dev/null 2>&1 || echo "[gvhmr FAILED] $c"
  done < $B
  echo "gvhmr done $(date +%T)"
  $E/rgb2smplx/bin/python scripts/run_corpus_stage.py fit --work-root $W --clips $B --solve-joints upper 2>&1 | tail -1
  echo "fit done $(date +%T): $(ls $W/*/fit_gvhmr*.npz 2>/dev/null | grep -vc raw) fits so far"
done
