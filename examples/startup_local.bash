#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --job-name=pll_{name}

echo "display"
source {pll_dir}/../bin/activate;
export PYTHONPATH={pll_dir};
export PLL_EXPERIMENT={name};


echo "repo at hash {hash}"

if {gen_videos}; then
	echo "meshcat server"
	PYTHONUNBUFFERED=1 meshcat-server >> {pll_dir}/logs/meshcat_{name}.txt &

	echo "delay to let server start up"
	sleep 3s

	echo "write meshcat html"
	python {pll_dir}/examples/dynamic_meshcat.py

	echo "open meshcat browser in screen"
	open -a "Google Chrome" {pll_dir}/results/{name}/static.html &
else
	echo "skipping video visualizations"
fi

echo "train"
python {pll_dir}/examples/contactnets_simple.py {name} {train_args}

if {gen_videos}; then
	echo "killing meshcat server and firefox"
	kill %%
fi
