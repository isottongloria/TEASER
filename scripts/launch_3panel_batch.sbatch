#!/bin/bash
#SBATCH --job-name=teaser_3panel_batch
#SBATCH --account=IscrC_SLPSCALE
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER/logs/3panel_batch.%A_%a.out
#
# Uses the 'normal' QOS rather than 'boost_qos_dbg' because the debug QOS
# caps a user at 2 concurrently-submitted jobs (fine for the single-video
# test jobs used during development, too restrictive for a 7-task array).
#
# Slurm array launcher for scripts/reconstruct_video_3panel.py: one video per
# array task, each on its own GPU allocation.
#
# --- Video list (index -> input, output filename) ------------------------
#   0: smirk/samples/05May_2011_Thursday_tagesschau-26.mp4
#   1: smirk/samples/05May_2011_Thursday_tagesschau-27.mp4
#   2: RGB2SMPLX/.../CSY/front/0.MP4
#   3: RGB2SMPLX/.../CSY/front/100.MP4
#   4: RGB2SMPLX/.../CSY/front/3D.MP4
#   5: RGB2SMPLX/.../CSY/front/3.MP4
#   6: RGB2SMPLX/.../CSY/front/40.MP4
#
# All 7 use 'pose_roi' (ported from the sibling SMIRK repo's own fix for
# this exact problem, utils/pose_roi.py -- SMIRK's own launcher also applies
# it to every video, not just the hard cases). It first locates the face
# with MediaPipe Pose, then runs TEASER's own unmodified FaceLandmarker on a
# zoomed-in crop of it; any residual per-frame miss (Pose or FaceLandmarker)
# is filled by linear interpolation over time from the nearest valid
# detections, so no frame is ever left without a reconstruction. Verified on
# CSY/front/0.MP4: 2/120 direct detections -> 120/120 real (no interpolation
# needed) with pose_roi. Everything downstream of face localization
# (TEASER's own crop_face + encoder + FLAME + renderer) is untouched.
#
#SBATCH --array=0-6

set -e
source /leonardo/home/userexternal/gisotto1/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser

cd /leonardo_work/IscrC_SLPSCALE/TEASER
export PYTHONPATH=.
export TORCH_HOME=$PWD/.cache/torch
export HF_HOME=$PWD/.cache/huggingface
export HF_HUB_OFFLINE=1

INPUTS=(
  "/leonardo_work/IscrC_SLPSCALE/smirk/samples/05May_2011_Thursday_tagesschau-26.mp4"
  "/leonardo_work/IscrC_SLPSCALE/smirk/samples/05May_2011_Thursday_tagesschau-27.mp4"
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/0.MP4"
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/100.MP4"
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/3D.MP4"
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/3.MP4"
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/40.MP4"
)

OUTPUT_NAMES=(
  "05May_2011_Thursday_tagesschau-26_teaser_comparison.mp4"
  "05May_2011_Thursday_tagesschau-27_teaser_comparison.mp4"
  "0_teaser_comparison.mp4"
  "100_teaser_comparison.mp4"
  "3D_teaser_comparison.mp4"
  "3_teaser_comparison.mp4"
  "40_teaser_comparison.mp4"
)

FACE_MODES=(
  "pose_roi"
  "pose_roi"
  "pose_roi"
  "pose_roi"
  "pose_roi"
  "pose_roi"
  "pose_roi"
)

OUT_DIR="/leonardo_work/IscrC_SLPSCALE/TEASER/results/sign_language_visual_eval"
mkdir -p "$OUT_DIR"

IDX=$SLURM_ARRAY_TASK_ID
INPUT="${INPUTS[$IDX]}"
OUTPUT="${OUT_DIR}/${OUTPUT_NAMES[$IDX]}"
FACE_MODE="${FACE_MODES[$IDX]}"

echo "Array task $IDX : $INPUT -> $OUTPUT (face_detection_mode=$FACE_MODE)"

python scripts/reconstruct_video_3panel.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint pretrained_models/TEASER.pt \
    --device cuda \
    --face_detection_mode "$FACE_MODE"
