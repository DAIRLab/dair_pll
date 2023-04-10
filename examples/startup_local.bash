#!/bin/bash

echo "display"
source {pll_dir}/pll_env/bin/activate;
# export PYTHONPATH={pll_dir};  # commented out bc added to pll_env/bin/activate
export PLL_EXPERIMENT={run_name};


echo "repo at hash {hash}"

if {restart}; then
	echo "restarting"
	PYTHONFAULTHANDLER=1 python {pll_dir}/examples/restart_run.py {storage_folder_name} {run_name} {train_args}
else
	echo "train"
	PYTHONFAULTHANDLER=1 python {pll_dir}/examples/contactnets_simple.py {storage_folder_name} {run_name} {train_args}
fi
