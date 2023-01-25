#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --job-name=pll_{name}

echo "display"
source {pll_dir}/pll_env/bin/activate;
# export PYTHONPATH={pll_dir};  # commented out bc using drake PR build which
								# installed drake at /opt/drake
export PLL_EXPERIMENT={name};


echo "repo at hash {hash}"

echo "train"
python {pll_dir}/examples/contactnets_simple.py {name} {train_args}
