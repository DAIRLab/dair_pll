#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --job-name=pll_{name}

echo "display"
source {pll_dir}/cnets_env/bin/activate;
export PYTHONPATH={sophter_dir};
export CONTACTNETS_EXPERIMENT={name};

echo "display"
source {pll_dir}/../bin/activate;
export PYTHONPATH={pll_dir};


echo "repo at hash {hash}"

echo "meshcat server and automatic open"
meshcat-server --open &

echo "train"
python {pll_dir}/examples/contactnets_simple.py {name} {train_args}

echo "killing meshcat server and firefox"
kill %%
