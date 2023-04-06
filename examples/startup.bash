#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --qos=mp-med
#SBATCH --partition=posa-compute
#SBATCH --account mp-account
#SBATCH --time=12:00:00
#SBATCH --job-name=pll_{run_name}
#SBATCH --output={pll_dir}/logs/slurm_{run_name}.txt

echo "display" >> {pll_dir}/logs/start_{run_name}.txt
source {pll_dir}/pll_env/bin/activate;
export PYTHONPATH={pll_dir};
export PLL_EXPERIMENT={run_name};


echo "repo at hash {hash}" >> {pll_dir}/logs/start_{run_name}.txt

echo "train" >> {pll_dir}/logs/start_{run_name}.txt
PYTHONUNBUFFERED=1 xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" python {pll_dir}/examples/contactnets_simple.py {storage_folder_name} {run_name} {train_args} >> {pll_dir}/logs/train_{run_name}.txt

