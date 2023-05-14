"""Helper script to determine the run parameters from the W&B group ID."""

import os
import os.path as op
import git
import fnmatch
import pdb
import csv
import wandb
import torch

from dair_pll.file_utils import *


# For results that generated hp_search_2.csv:
# ELBOW_HP_SCRIPT_PATTERN = 'startup_hpreal_elbow_ie????.bash'
# storage_name = '/home/bibit/dair_pll/results/hpreal_elbow'
# Set a minimum run number since multiple hyperparameter searches were conducted
# in the same results folder.
# MINIMUM_RUN_NUM = 372

# For results that generated hp_search_3.csv:
ELBOW_HP_SCRIPT_PATTERN = 'startup_elbow_real_re??-?.bash'
storage_name = '/home/bibit/dair_pll/results/elbow_real'
# Set a minimum run number since multiple hyperparameter searches were conducted
# in the same results folder.
MINIMUM_RUN_NUM = 13   # the maximum for hp_search_3.csv is 33

WANDB_PROJECT_CLUSTER = 'dair_pll-cluster'
WANDB_PROJECT_LOCAL = 'dair_pll-dev'

VALID_MODEL_MSE = 'valid_model_trajectory_mse_mean'
POS_MODEL_ERROR = 'valid_model_pos_int_traj_mean'
ROT_MODEL_ERROR = 'valid_model_angle_int_traj_mean'
PENETRATION_MODEL = 'valid_model_penetration_int_traj_mean'

VALID_MSE = 'valid_trajectory_mse'
POS_ERROR = 'valid_pos_int_traj'
ROT_ERROR = 'valid_angle_int_traj'
PENETRATION = 'valid_penetration_int_traj'



repo = git.Repo(search_parent_directories=True)
git_folder = repo.git.rev_parse("--show-toplevel")
git_folder = op.normpath(git_folder)

startup_scripts_folder = op.join(git_folder, 'examples')

startup_scripts_list = sorted(os.listdir(startup_scripts_folder))

def get_parameter_from_string(param: str, long_string: str):
    return long_string.split(f'{param}=')[-1].split('\n')[0].split(' ')[0].split(';')[0]


def get_params_from_bash_script(script_name):
    full_script_path = f'{startup_scripts_folder}/{script_name}'
    script = open(full_script_path, 'r').read()

    wandb_id = get_parameter_from_string('WANDB_RUN_GROUP', script)
    loss_variation = get_parameter_from_string('loss-variation', script)
    w_pred = get_parameter_from_string('w-pred', script)
    w_comp = get_parameter_from_string('w-comp', script)
    w_diss = get_parameter_from_string('w-diss', script)
    w_pen = get_parameter_from_string('w-pen', script)
    w_res = get_parameter_from_string('w-res', script)

    return {'wandb_id': wandb_id,
            'loss variation': loss_variation,
            'w_pred': w_pred,
            'w_comp': w_comp,
            'w_diss': w_diss,
            'w_pen': w_pen,
            'w_res': w_res,}


lookup_by_wandb_id = {}
lookup_by_run_name = {}

for script in startup_scripts_list:
    if fnmatch.fnmatch(script, ELBOW_HP_SCRIPT_PATTERN):
        run_name = script.split('_')[-1].split('.')[0]
        try:
            run_num = int(run_name.split('-')[0][2:])
        except:
            continue

        if run_num >= MINIMUM_RUN_NUM:
            params_dict = get_params_from_bash_script(script)
            params_dict['run name'] = run_name
            wandb_id = params_dict['wandb_id']

            print(f'{run_name}, {wandb_id}:', end=' ')

            try:
                statistics = load_evaluation(storage_name, run_name)
                params_dict['best valid MSE'] = statistics[VALID_MODEL_MSE]
                params_dict['best position MSE'] = statistics[POS_MODEL_ERROR]
                params_dict['best angular MSE'] = statistics[ROT_MODEL_ERROR]
                params_dict['best penetration'] = statistics[PENETRATION_MODEL]
                print('Found statistics file.')

            except FileNotFoundError:
                print('No statistics file found; searching wandb logs.', end='')

                checkpoint_filename = get_model_filename(storage_name, run_name)
                checkpoint_dict = torch.load(checkpoint_filename)

                wandb_run_id = checkpoint_dict['wandb_run_id']

                api = wandb.Api()
                try:
                    run = api.run(
                        f'ebianchi/{WANDB_PROJECT_CLUSTER}/{wandb_run_id}')
                    print('')
                except:
                    print(' --> Could not find W&B run, skipping.')

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

                params_dict['best valid MSE'] = best_valid_mse
                params_dict['best position MSE'] = best_pos_error
                params_dict['best angular MSE'] = best_rot_error
                params_dict['best penetration'] = best_penetration


            lookup_by_run_name[run_name] = params_dict
            lookup_by_wandb_id[wandb_id] = params_dict



with open('hp_search_3.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=params_dict.keys())

    writer.writeheader()
    for run_info in lookup_by_wandb_id.keys():
        writer.writerow(lookup_by_wandb_id[run_info])


pdb.set_trace()

