"""Explore the results of a failed experiment run."""

import pdb
import torch
import wandb
import csv

from dair_pll.file_utils import *


WANDB_PROJECT_CLUSTER = 'dair_pll-cluster'
WANDB_PROJECT_LOCAL = 'dair_pll-dev'


CLUSTER_PROJECT = ''


run_name = 'te12'  #'he0333'
storage_name = '/home/bibit/dair_pll/results/test_elbow'  #'hyperparam_elbow'

# checkpoint.pt
# config.pkl
# statistics.pkl

try:
	statistics = load_evaluation(storage_name, run_name)
except FileNotFoundError:
	statistics = None
	print('No statistics file found.')

config = load_configuration(storage_name, run_name)
checkpoint_filename = get_model_filename(storage_name, run_name)
checkpoint_dict = torch.load(checkpoint_filename)

wandb_id = checkpoint_dict['wandb_run_id']

api = wandb.Api()
run = api.run(f'ebianchi/{WANDB_PROJECT_LOCAL}/{wandb_id}')
run_history = run.history(pandas=False)
pdb.set_trace()

