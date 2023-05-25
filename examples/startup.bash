#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
##SBATCH --qos=mp-med
##SBATCH --partition=posa-compute
##SBATCH --account mp-account
#SBATCH --time=12:00:00
#SBATCH --job-name=pll_{run_name}
#SBATCH --output={pll_dir}/logs/slurm_{run_name}.txt

echo "display" >> {pll_dir}/logs/start_{run_name}.txt
source /mnt/kostas-graid/sw/envs/bibit/pll_env/bin/activate;
export PYTHONPATH=/mnt/kostas-graid/sw/envs/bibit:{pll_dir};
export PLL_EXPERIMENT={run_name};


echo "repo at hash {hash}" >> {pll_dir}/logs/start_{run_name}.txt

if {restart}; then
	echo "restarting" >> {pll_dir}/logs/start_{run_name}.txt
	WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" python {pll_dir}/examples/restart_run.py {storage_folder_name} {run_name} >> {pll_dir}/logs/train_{run_name}.txt
else
	export WANDB_RUN_GROUP={wandb_group_id};
	echo "setting wandb run group to {wandb_group_id}" >> {pll_dir}/logs/start_{run_name}.txt

	echo "train" >> {pll_dir}/logs/start_{run_name}.txt
	WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" python {pll_dir}/examples/contactnets_simple.py {storage_folder_name} {run_name} {train_args} >> {pll_dir}/logs/train_{run_name}.txt
fi
