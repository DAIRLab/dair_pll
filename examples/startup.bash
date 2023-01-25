#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --job-name=pll_{name}
#SBATCH --output={pll_dir}/logs/slurm_{name}.txt
#SBATCH --exclude=node-3090-1,node-3090-2,node-3090-3,node-1080ti-0,node-2080ti-7,node-v100-0

echo "display" >> {pll_dir}/logs/start_{name}.txt
#Xvfb :6 -screen 0 800x600x24 &
source {pll_dir}/../bin/activate;
export PYTHONPATH={pll_dir};
export PLL_EXPERIMENT={name};
export DISPLAY=:5;


echo "repo at hash {hash}" >> {pll_dir}/logs/start_{name}.txt

echo "train" >> {pll_dir}/logs/start_{name}.txt
PYTHONUNBUFFERED=1 python {pll_dir}/examples/contactnets_simple.py {name} {train_args} >> {pll_dir}/logs/train_{name}.txt

