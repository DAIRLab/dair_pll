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
export XDG_RUNTIME_DIR=/mnt/beegfs/scratch/bibit/tmp;


echo "repo at hash {hash}" >> {pll_dir}/logs/start_{name}.txt

if {gen_videos}; then
	echo "meshcat server" >> {pll_dir}/logs/start_{name}.txt
	PYTHONUNBUFFERED=1 meshcat-server >> {pll_dir}/logs/meshcat_{name}.txt &

	echo "delay to let server start up" >> {pll_dir}/logs/start_{name}.txt
	sleep 3s

	echo "write meshcat html"
	python {pll_dir}/examples/dynamic_meshcat.py

	echo "open meshcat browser in screen" >> {pll_dir}/logs/start_{name}.txt
	xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" {firefox_dir}/firefox {pll_dir}/results/{name}/static.html &
else
	echo "skip video visualizations" >> {pll_dir}/logs/start_{name}.txt
fi

echo "train" >> {pll_dir}/logs/start_{name}.txt
python {pll_dir}/examples/contactnets_simple.py {name} {train_args} &> {pll_dir}/logs/train_{name}.txt

if {gen_videos}; then
	echo "kill meshcat server and firefox" >> {pll_dir}/logs/start_{name}.txt
	kill %%
fi
