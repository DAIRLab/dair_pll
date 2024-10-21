"""Helper script to determine the run parameters from the W&B group ID.

Note:  File gather_hyperparam_results.py now gets all of this information
individually.
"""

import os
import os.path as op
import git
import fnmatch
import pdb
import csv


ELBOW_HP_SCRIPT_PATTERN = "startup_hpreal_elbow_ie????.bash"

FIELDNAMES = ["wandb_id", "run name", "loss variation", "w_comp", "w_diss", "w_pen"]


repo = git.Repo(search_parent_directories=True)
git_folder = repo.git.rev_parse("--show-toplevel")
git_folder = op.normpath(git_folder)

startup_scripts_folder = op.join(git_folder, "examples")

startup_scripts_list = sorted(os.listdir(startup_scripts_folder))


def get_params_from_bash_script(script_name):
    full_script_path = f"{startup_scripts_folder}/{script_name}"
    script = open(full_script_path, "r").read()
    wandb_id = script.split("WANDB_RUN_GROUP=")[-1].split(";")[0]
    loss_variation = script.split("loss-variation=")[-1].split(" ")[0]
    w_comp = script.split("w-comp=")[-1].split(" ")[0]
    w_diss = script.split("w-diss=")[-1].split(" ")[0]
    w_pen = script.split("w-pen=")[-1].split("\n")[0].split(" ")[0]

    return {
        "wandb_id": wandb_id,
        "loss variation": loss_variation,
        "w_comp": w_comp,
        "w_diss": w_diss,
        "w_pen": w_pen,
    }


lookup_by_wandb_id = {}
lookup_by_run_name = {}

for script in startup_scripts_list:
    if fnmatch.fnmatch(script, ELBOW_HP_SCRIPT_PATTERN):
        run_name = script.split("_")[-1].split(".")[0]
        params_dict = get_params_from_bash_script(script)
        params_dict["run name"] = run_name
        wandb_id = params_dict["wandb_id"]

        print(run_name, wandb_id)

        lookup_by_wandb_id[wandb_id] = params_dict
        lookup_by_run_name[run_name] = params_dict


with open("hyperparameter_real_lookup.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

    writer.writeheader()
    for run_info in lookup_by_wandb_id.keys():
        writer.writerow(lookup_by_wandb_id[run_info])


pdb.set_trace()
