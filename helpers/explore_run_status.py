"""Explore the results of a failed experiment run.

Note:  File gather_hyperparam_results.py now gets all of this information
individually.
"""

import pdb
import torch
import wandb
import csv

from dair_pll.file_utils import *


WANDB_PROJECT_CLUSTER = "dair_pll-cluster"
WANDB_PROJECT_LOCAL = "dair_pll-dev"

VALID_MODEL_MSE = "valid_model_trajectory_mse_mean"
POS_MODEL_ERROR = "valid_model_pos_int_traj_mean"
ROT_MODEL_ERROR = "valid_model_angle_int_traj_mean"
PENETRATION_MODEL = "valid_model_penetration_int_traj_mean"

VALID_MSE = "valid_trajectory_mse"
POS_ERROR = "valid_pos_int_traj"
ROT_ERROR = "valid_angle_int_traj"
PENETRATION = "valid_penetration_int_traj"

storage_name = "/home/bibit/dair_pll/results/hpreal_elbow"  #'test_elbow'


lookup_by_run_name = {}
lookup_by_wandb_id = {}


with open("hyperparameter_real_lookup.csv", newline="") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        lookup_by_run_name[row["run name"]] = row
        lookup_by_wandb_id[row["wandb_id"]] = row


for run_name in lookup_by_run_name.keys():
    run_dict = lookup_by_run_name[run_name]
    wandb_id = run_dict["wandb_id"]

    try:
        statistics = load_evaluation(storage_name, run_name)
        run_dict["best valid MSE"] = statistics[VALID_MODEL_MSE]
        run_dict["best position MSE"] = statistics[POS_MODEL_ERROR]
        run_dict["best angular MSE"] = statistics[ROT_MODEL_ERROR]
        run_dict["best penetration"] = statistics[PENETRATION_MODEL]
        print("Found statistics file.")

    except FileNotFoundError:
        print("No statistics file found; searching wandb logs.")

        # config = load_configuration(storage_name, run_name)
        checkpoint_filename = get_model_filename(storage_name, run_name)
        checkpoint_dict = torch.load(checkpoint_filename)

        wandb_run_id = checkpoint_dict["wandb_run_id"]

        api = wandb.Api()
        run = api.run(f"ebianchi/{WANDB_PROJECT_CLUSTER}/{wandb_run_id}")
        run_history = run.history(pandas=False)

        best_valid_mse = run_history[0][VALID_MSE]
        best_pos_error = run_history[0][POS_ERROR]
        best_rot_error = run_history[0][ROT_ERROR]
        best_penetration = run_history[0][PENETRATION]

        for epoch_dict in run_history:
            new_valid_mse = epoch_dict[VALID_MSE]
            if new_valid_mse < best_valid_mse:
                best_valid_mse = new_valid_mse

            new_pos_error = epoch_dict[POS_ERROR]
            if new_pos_error < best_pos_error:
                best_pos_error = new_pos_error

            new_rot_error = epoch_dict[ROT_ERROR]
            if new_rot_error < best_rot_error:
                best_rot_error = new_rot_error

            new_penetration = epoch_dict[PENETRATION]
            if new_penetration < best_penetration:
                best_penetration = new_penetration

        run_dict["best valid MSE"] = best_valid_mse
        run_dict["best position MSE"] = best_pos_error
        run_dict["best angular MSE"] = best_rot_error
        run_dict["best penetration"] = best_penetration

    lookup_by_run_name[run_name] = run_dict
    lookup_by_wandb_id[wandb_id] = run_dict


with open("hyperparameter_real_performance.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=run_dict.keys())

    writer.writeheader()
    for run_info in lookup_by_wandb_id.keys():
        writer.writerow(lookup_by_wandb_id[run_info])

pdb.set_trace()
