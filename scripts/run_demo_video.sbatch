#!/bin/bash
#SBATCH --job-name=teaser_demo_video
#SBATCH --account=IscrC_SLPSCALE
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER/logs/demo_video.%j.out

set -e
source /leonardo/home/userexternal/gisotto1/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser

cd /leonardo_work/IscrC_SLPSCALE/TEASER
export PYTHONPATH=.
export TORCH_HOME=/leonardo_work/IscrC_SLPSCALE/TEASER/.cache/torch
export HF_HOME=/leonardo_work/IscrC_SLPSCALE/TEASER/.cache/huggingface
export HF_HUB_OFFLINE=1

python main/demo_video.py \
    --input_path "${DEMO_VIDEO_INPUT:-samples/demo_videos}" \
    --out_path results/reconstruct_videos \
    --checkpoint pretrained_models/TEASER_v1.pt \
    --crop --use_teaser_generator
