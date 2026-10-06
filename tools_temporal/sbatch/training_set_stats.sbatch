#!/bin/bash
#SBATCH --job-name=temporal_trainset_stats
#SBATCH --account=iscrc_slpscale
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --chdir=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER_temporal/logs/%x.%j.out
/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python tools_temporal/training_set_stats.py \
    --root data/temporal/all --out data/temporal/report/training_set_stats.json
