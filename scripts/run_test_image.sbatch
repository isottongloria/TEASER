#!/bin/bash
#SBATCH --job-name=teaser_test_image
#SBATCH --account=IscrC_SLPSCALE
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER/logs/test_image.%j.out

set -e
source /leonardo/home/userexternal/gisotto1/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser

cd /leonardo_work/IscrC_SLPSCALE/TEASER
export PYTHONPATH=.
# repo-local caches so no internet is needed on the (often air-gapped) GPU compute node
export TORCH_HOME=/leonardo_work/IscrC_SLPSCALE/TEASER/.cache/torch
export HF_HOME=/leonardo_work/IscrC_SLPSCALE/TEASER/.cache/huggingface
export HF_HUB_OFFLINE=1

python main/test_image.py \
    --input_path samples/1.jpg \
    --out_path results/ \
    --checkpoint pretrained_models/TEASER.pt \
    --crop --use_teaser_generator
