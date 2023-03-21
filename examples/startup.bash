#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --qos=mp-med
#SBATCH --partition=posa-compute
#SBATCH --account mp-account
#SBATCH --time=12:00:00
#SBATCH --job-name=pll_{name}
#SBATCH --output={pll_dir}/logs/slurm_{name}.txt

echo "display" >> {pll_dir}/logs/start_{name}.txt
source {pll_dir}/pll_env/bin/activate;
export PYTHONPATH={pll_dir};
export PLL_EXPERIMENT={name};


echo "repo at hash {hash}" >> {pll_dir}/logs/start_{name}.txt

echo "train" >> {pll_dir}/logs/start_{name}.txt
PYTHONUNBUFFERED=1 xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" python {pll_dir}/examples/contactnets_simple.py {name} {train_args} >> {pll_dir}/logs/train_{name}.txt

