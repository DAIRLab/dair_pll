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
export XDG_RUNTIME_DIR=/mnt/beegfs/scratch/bibit/tmp;


echo "meshcat server"
meshcat-server &

echo "open meshcat browswer in screen"
xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" /mnt/beegfs/scratch/bibit/firefox/firefox http://127.0.0.1:7000/static/ &


echo "train"
python /mnt/beegfs/scratch/bibit/pll_env/dair_pll/examples/contactnets_simple.py t0 --source real &> /mnt/beegfs/scratch/bibit/pll_env/dair_pll/logs/train_t0.txt

echo "killing meshcat server and firefox"
kill %%
