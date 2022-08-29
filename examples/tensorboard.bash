#!/bin/bash
#SBATCH --qos=viz
#SBATCH --partition=viz
#SBATCH --cores=1

PORT_MAP=/tmp/tensorboard_port_map

TB_PORT=$(cat $PORT_MAP | grep "$SLURM_JOBID," | cut -d',' -f2)
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=$1
TB_TITLE=$2

echo "http://$IP_ADDRESS:$TB_PORT"

tensorboard --samples_per_plugin="images=0" --bind_all --logdir $TB_FOLDER  --window_title $TB_TITLE --port $TB_PORT &
