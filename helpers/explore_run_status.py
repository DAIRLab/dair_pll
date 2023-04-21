"""Explore the results of a failed experiment run."""

import pdb
import torch
import wandb
import csv

from dair_pll.file_utils import *


WANDB_PROJECT_CLUSTER = 'dair_pll-cluster'
WANDB_PROJECT_LOCAL = 'dair_pll-dev'

VALID_MSE = 'valid_model_trajectory_mse_mean'

storage_name = '/home/bibit/dair_pll/results/hyperparam_elbow'  #'test_elbow'


lookup_by_run_name = {}
lookup_by_wandb_id = {}


with open('hyperparameter_lookup_from_cluster.csv', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        lookup_by_run_name[row['run name']] = row
        lookup_by_wandb_id[row['wandb_id']] = row


for run_name in lookup_by_run_name.keys():
    run_dict = lookup_by_run_name[run_name]
    wandb_id = run_dict['wandb_id']

    try:
        statistics = load_evaluation(storage_name, run_name)
        run_dict['best valid MSE'] = statistics[VALID_MSE]

    except FileNotFoundError:
        print('No statistics file found; searching wandb logs.')

        # config = load_configuration(storage_name, run_name)
        # checkpoint_filename = get_model_filename(storage_name, run_name)
        # checkpoint_dict = torch.load(checkpoint_filename)

        # wandb_id = checkpoint_dict['wandb_run_id']

        api = wandb.Api()
        run = api.run(f'ebianchi/{WANDB_PROJECT_CLUSTER}/{wandb_id}')
        run_history = run.history(pandas=False)

        best_valid_mse = run_history[0][VALID_MSE]

        for epoch_dict in run_history:
            new_valid_mse = epoch_dict[VALID_MSE]
            if new_valid_mse < best_valid_mse:
                best_valid_mse = new_valid_mse

        run_dict['best valid MSE'] = best_valid_mse


    lookup_by_run_name[run_name] = run_dict
    lookup_by_wandb_id[wandb_id] = run_dict



pdb.set_trace()

