#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
##SBATCH --qos=mp-med
##SBATCH --partition=posa-compute
##SBATCH --account mp-account
#SBATCH --time=12:00:00
#SBATCH --output=/home/bibit/dair_pll/logs/generate_toss_predictions.txt

source /mnt/kostas-graid/sw/envs/bibit/pll_env/bin/activate;
export PYTHONPATH=/mnt/kostas-graid/sw/envs/bibit:/home/bibit/dair_pll;

cd /home/bibit/dair_pll/helpers
PYTHONUNBUFFERED=1 python generate_toss_predictions.py

