#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --qos=low
#SBATCH --job-name=pll_t0
#SBATCH --output=/mnt/beegfs/scratch/bibit/pll_env/dair_pll/logs/slurm_t0.txt
#SBATCH --exclude=node-3090-1,node-3090-2,node-3090-3,node-1080ti-0,node-2080ti-7,node-v100-0

echo "display" >> /mnt/beegfs/scratch/bibit/pll_env/dair_pll/logs/start_t0.txt
#Xvfb :6 -screen 0 800x600x24 &
source /mnt/beegfs/scratch/bibit/pll_env/dair_pll/../bin/activate;
export PYTHONPATH=/mnt/beegfs/scratch/bibit/pll_env/dair_pll;
export DISPLAY=:5;
# export PLL_EXPERIMENT=t0;


echo "meshcat server"
xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" meshcat-server &

echo "train" >> /mnt/beegfs/scratch/bibit/pll_env/dair_pll/logs/start_t0.txt
python /mnt/beegfs/scratch/bibit/pll_env/dair_pll/examples/contactnets_simple.py t0 --source real &> /mnt/beegfs/scratch/bibit/pll_env/dair_pll/logs/train_t0.txt
