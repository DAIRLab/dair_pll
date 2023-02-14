#!/bin/bash
#SBATCH --qos=viz
#SBATCH --partition=viz
#SBATCH --cores=1


TB_FOLDER=$1
TB_TITLE=$2

tensorboard --samples_per_plugin="images=0" --bind_all --logdir $TB_FOLDER  --window_title $TB_TITLE
