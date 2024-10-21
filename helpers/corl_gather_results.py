"""Script to help generate plots for CoRL 2023 submission.

First, this script compiles all results into a json file.  Then, the
corl_plot.py script can gather the results from the json file and generate plots
from them.

The json file has the following format:
{
    experiment_1: {
        system:  cube/elbow/asymmetric
        prefix:  e.g. sc for 'cube' experiments
        data_sweep: {
            dataset_size_1: {
                run_1: {
                    structured:  bool
                    contactnets:  bool
                    loss_variation:  int
                    residual:  bool
                    result_set:  test/validation
                    results: {
                        metric_1:  float
                        metric_2:  float
                        ...
                    }
                    initial_params: {
                        body_1: {
                            param_1:  float
                            param_2:  float
                            ...
                        }
                        body_2: {...}
                    }
                    learned_params: {
                        body_1: {
                            param_1:  float
                            param_2:  float
                            ...
                        }
                        body_2: {...}
                    }
                    post_results: {
                        post_metric_1:  float
                        post_metric_2:  float
                        ...
                    }
                    fixed_horizon_post_results: {
                        fixed_horizon_metric_1:  float
                        fixed_horizon_metric_2:  float
                        ...
                    }
                    target_trajs: []      <-- excluded from json due to datatype
                    prediction_trajs: []  <-- excluded from json due to datatype
                }
                run_2: {...}
                run_3: {...}
                ...
            }
            dataset_size_2: {...}
            dataset_size_3: {...}
            ...
        }
    }
    experiment_2: {...}
    experiment_3: {...}
    ...
}
...where `experiment_1` might be 'cube', corresponding to the real data sweep
results.
"""

import json
import os
import os.path as op
import pdb
import pickle
import torch
import wandb
from copy import deepcopy

import numpy as np

from dair_pll.deep_learnable_system import DeepLearnableSystemConfig
from dair_pll.drake_experiment import MultibodyLosses
from dair_pll.geometry import _NOMINAL_HALF_LENGTH
from dair_pll.inertia import InertialParameterConverter


"""Note:  might need the below in drake_experiment.py for backwards
compatibility:

@dataclass
class DrakeMultibodyLearnableExperimentConfig(SupervisedLearningExperimentConfig
                                             ):
    visualize_learned_geometry: bool = True
    \"""Whether to use learned geometry in trajectory overlay visualization.\"""
"""


# Directory management.
RESULTS_DIR = op.join(op.dirname(__file__), "..", "results")
OUTPUT_DIR = op.join(op.dirname(__file__), "..", "plots")
JSON_OUTPUT_FILE = op.join(op.dirname(__file__), "results.json")

ROLLOUT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 120]

WANDB_PROJECT_CLUSTER = "dair_pll-cluster"

BODY_NAMES_BY_SYSTEM = {
    "cube": ["body"],
    "elbow": ["elbow_1", "elbow_2"],
    "asymmetric": ["body"],
}
VERTEX_NUMBERS_PER_BODY_BY_SYSTEM = {"cube": 8, "elbow": 8, "asymmetric": 6}
BODY_PARAMETERS = {
    "m": "Mass",
    "com_x": "CoM x",
    "com_y": "CoM y",
    "com_z": "CoM z",
    "I_xx": "I_xx",
    "I_yy": "I_yy",
    "I_zz": "I_zz",
    "I_xy": "I_xy",
    "I_xz": "I_xz",
    "I_yz": "I_yz",
    "mu": "Friction coefficient",
    "center_x": "Geometry center x",
    "center_y": "Geometry center y",
    "center_z": "Geometry center z",
    "diameter_x": "Geometry diameter x",
    "diameter_y": "Geometry diameter x",
    "diameter_z": "Geometry diameter x",
}
POLYGON_GEOMETRY_PARAMETERS = [
    "center_x",
    "center_y",
    "center_z",
    "diameter_x",
    "diameter_y",
    "diameter_z",
]

INERTIA_KEY = "multibody_terms.lagrangian_terms.inertial_parameters"
FRICTION_KEY = "multibody_terms.contact_terms.friction_params"
GEOMETRY_PREFIX = "multibody_terms.contact_terms.geometries"
GEOMETRY_KEY_BODY_1 = f"{GEOMETRY_PREFIX}.2.vertices_parameter"
GEOMETRY_KEY_BODY_2 = f"{GEOMETRY_PREFIX}.0.vertices_parameter"
# GEOMETRY_KEY2 = 'multibody_terms.contact_terms.geometries.0.length_params'

FRICTION_INDEX_BY_BODY_NAME = {"body": 0, "elbow_2": 0, "elbow_1": 2}

PERFORMANCE_METRICS = [
    "delta_v_squared_mean",
    "v_plus_squared_mean",
    "model_loss_mean",
    "oracle_loss_mean",
    "model_trajectory_mse_mean",
    "oracle_trajectory_mse_mean",
    "model_pos_int_traj",
    "oracle_pos_int_traj",
    "model_angle_int_traj",
    "oracle_angle_int_traj",
    "model_penetration_int_traj",
    "oracle_penetration_int_traj",
]
POST_PERFORMANCE_METRICS = [f"pos_error_w_horizon_{i}" for i in ROLLOUT_LENGTHS] + [
    f"rot_error_w_horizon_{i}" for i in ROLLOUT_LENGTHS
]
FIXED_HORIZON = 16
FIXED_HORIZON_METRICS = [
    f"pos_error_w_horizon_{FIXED_HORIZON}",
    f"rot_error_w_horizon_{FIXED_HORIZON}",
]

DATASET_EXPONENTS = [2, 3, 4, 5, 6, 7, 8, 9]
SYSTEMS = ["cube", "elbow", "asymmetric"]
ORDERED_INERTIA_PARAMS = [
    "m",
    "px",
    "py",
    "pz",
    "I_xx",
    "I_yy",
    "I_zz",
    "I_xy",
    "I_xz",
    "I_yz",
]
WANDB_INERTIA_PARAM_KEYS = [
    "m",
    "com_x",
    "com_y",
    "com_z",
    "I_xx",
    "I_yy",
    "I_zz",
    "I_xy",
    "I_xz",
    "I_yz",
]
TARGET_SAMPLE_KEY = "model_target_sample"
PREDICTION_SAMPLE_KEY = "model_prediction_sample"

ORIGINAL_KIND = "original"
POST_KIND = "post"
FIXED_HORIZON_KIND = "fixed_horizon"
KINDS = [ORIGINAL_KIND, POST_KIND, FIXED_HORIZON_KIND]

# Template dictionaries, from low- to high-level.
RUN_DICT = {
    "structured": None,
    "contactnets": None,
    "loss_variation": None,
    "residual": None,
    "result_set": None,
    "results": None,
    "initial_params": None,
    "learned_params": None,
    "post_results": None,
    "fixed_horizon_post_results": None,
}
EXPERIMENT_DICT = {"system": None, "prefix": None, "data_sweep": None}

BAD_RUN_NUMBERS = {
    "elbow": [i for i in range(24)]
    + [i for i in range(25, 32)]
    + [35, 36, 37, 38, 39, 40, 41, 42],
    "cube": [i for i in range(24)] + [i for i in range(25, 32)],
    "asymmetric_vortex": [i for i in range(24)]
    + [i for i in range(25, 32)]
    + [33, 35, 36, 37, 39, 40],
    "elbow_vortex": [i for i in range(24)] + [i for i in range(25, 30)] + [31, 33, 35],
    "asymmetric_viscous": [i for i in range(24)]
    + [i for i in range(25, 30)]
    + [31, 33, 35],
    "elbow_viscous": [i for i in range(24)] + [i for i in range(25, 30)] + [31, 33, 35],
}

# Prepend the below with 'sweep_' and postpend with '-#' to get the folders.
EXPERIMENTS = {
    "cube": {"system": "cube", "prefix": "sc"},
    "elbow": {"system": "elbow", "prefix": "se"},
    "asymmetric_vortex": {"system": "asymmetric", "prefix": "va"},
}
#'elbow_vortex': {'system': 'elbow', 'prefix': 've'},
#'asymmetric_viscous': {'system': 'asymmetric', 'prefix': 'ba'},
#'elbow_viscous': {'system': 'elbow', 'prefix': 'be'}}


# ============================= Helper functions ============================= #
# Return an empty data sweep dictionary, to prevent unintended data retention.
def make_empty_data_sweep_dict():
    new_dict = {}
    for exp in DATASET_EXPONENTS:
        new_dict.update({exp: {}})
    return new_dict


# Extract information out of a configuration object.
def get_run_info_from_config(config):
    run_dict = deepcopy(RUN_DICT)

    run_dict["structured"] = (
        False
        if isinstance(config.learnable_config, DeepLearnableSystemConfig)
        else True
    )
    run_dict["contactnets"] = (
        False
        if not run_dict["structured"]
        else (
            True
            if config.learnable_config.loss == MultibodyLosses.CONTACTNETS_LOSS
            else False
        )
    )
    run_dict["loss_variation"] = (
        0 if not run_dict["structured"] else config.learnable_config.loss_variation
    )
    run_dict["residual"] = (
        False if not run_dict["structured"] else config.learnable_config.do_residual
    )
    run_dict["result_set"] = "test"
    run_name = config.run_name

    return run_name, run_dict


# Calculate geometry measurements from a set of polygon vertices.
def get_geometry_metrics_from_params(geom_params):
    # First, convert the parameters to meters.
    vertices = geom_params * _NOMINAL_HALF_LENGTH

    # Extract diameters and centers.
    mins = vertices.min(axis=0).values
    maxs = vertices.max(axis=0).values

    diameters = maxs - mins
    centers = (maxs + mins) / 2

    geom_dict = {
        "diameter_x": diameters[0].item(),
        "diameter_y": diameters[1].item(),
        "diameter_z": diameters[2].item(),
        "center_x": centers[0].item(),
        "center_y": centers[1].item(),
        "center_z": centers[2].item(),
        "vertices": vertices.tolist(),
    }
    return geom_dict


def geometry_keys_by_sys_and_bodies(system, body_name):
    if system == "cube" or system == "asymmetric":
        return {"body": GEOMETRY_KEY_BODY_2}
    return {"elbow_1": GEOMETRY_KEY_BODY_1, "elbow_2": GEOMETRY_KEY_BODY_2}


# Get individual physical parameters from best learned system state.
def get_physical_parameters(system, body_names, best_system_state):
    physical_params_dict = {}

    theta = best_system_state[INERTIA_KEY]
    friction_params = best_system_state[FRICTION_KEY]
    if GEOMETRY_KEY_BODY_2 in best_system_state.keys():
        geometry_keys = geometry_keys_by_sys_and_bodies(system, body_names)
    else:
        geometry_keys = {}
        print(f"\t\tFound non-polygon; won't gather geometry results.")

    inertia_pi_cm_params = InertialParameterConverter.theta_to_pi_cm(theta)

    # Loop over each body.
    for i in range(len(body_names)):
        body = body_names[i]

        # First, get the inertial parameters.
        i_params = inertia_pi_cm_params[i, :]
        i_params[1:4] /= i_params[0].item()  # Divide out the mass.

        body_params = {}

        for j in range(10):
            body_params.update({ORDERED_INERTIA_PARAMS[j]: i_params[j].item()})

        # Second, get the friction parameters.
        mu_index = FRICTION_INDEX_BY_BODY_NAME[body]
        body_params.update({"mu": friction_params[mu_index].item()})

        # Third, get the geometry parameters.
        try:
            geometry_params = best_system_state[geometry_keys[body]]
            geom_dict = get_geometry_metrics_from_params(geometry_params)
            body_params.update(geom_dict)
        except:
            pass

        # Store the results.
        physical_params_dict.update({body: body_params})

    return physical_params_dict


# Get the initial physical parameters.
def get_init_physical_parameters(system, body_names, checkpoint, wandb_api):
    physical_params_dict = {}

    # Get the W&B stored initial parameters.
    wandb_run_id = checkpoint["wandb_run_id"]
    wandb_run_key = f"ebianchi/{WANDB_PROJECT_CLUSTER}/{wandb_run_id}"
    wandb_run = wandb_api.run(wandb_run_key)
    init_params = wandb_run.history(pandas=False)[0]

    # Iterate over each body in the system.
    for body in body_names:
        # prefix = f'{system}_{body}'
        body_params = {}

        # First, get the inertial parameters.
        for i in range(len(WANDB_INERTIA_PARAM_KEYS)):
            param_key = WANDB_INERTIA_PARAM_KEYS[i]
            inertia_param = init_params[f"{system}_{body}_{param_key}"]
            body_params.update({ORDERED_INERTIA_PARAMS[i]: inertia_param})

        # Second, get the friction parameters.
        init_mu = init_params[f"{system}_{body}_mu"]
        body_params.update({"mu": init_mu})

        # Third, get the geometry parameters.
        for geom_key in POLYGON_GEOMETRY_PARAMETERS:
            geom_value = init_params[f"{system}_{body}_{geom_key}"]
            body_params.update({geom_key: geom_value})

        # Fourth, get the individual vertex locations.
        n_expected_vertices = VERTEX_NUMBERS_PER_BODY_BY_SYSTEM[system]
        vertices = [[0, 0, 0]] * n_expected_vertices
        for vertex_i in range(n_expected_vertices):
            pre_key = f"{system}_{body}_v{vertex_i}"
            vertices[vertex_i][0] = init_params[f"{pre_key}_x"]
            vertices[vertex_i][1] = init_params[f"{pre_key}_y"]
            vertices[vertex_i][2] = init_params[f"{pre_key}_z"]
        body_params.update({"vertices": vertices})

        # Store the results.
        physical_params_dict.update({body: body_params})

    return physical_params_dict


# Extract the desired statistics from the larger stats file.  Will convert
# numpy arrays into averages.
def get_performance_from_stats(stats, set_name, kind=ORIGINAL_KIND):
    metrics = (
        PERFORMANCE_METRICS
        if kind == ORIGINAL_KIND
        else POST_PERFORMANCE_METRICS if kind == POST_KIND else FIXED_HORIZON_METRICS
    )

    performance_dict = {}
    for metric in metrics:
        key = f"{set_name}_{metric}"
        try:
            if type(stats[key]) == np.ndarray:
                performance_dict.update({key: np.average(stats[key])})
            else:
                performance_dict.update({key: stats[key]})
        except:
            print(f"\t\tDidn't find {key} in stats...")
    return performance_dict


# Extract the target and prediction trajectories from the larger stats file.
# This isn't called since the datatype isn't json serializable, but keeping this
# function here for future reference.
def get_sample_trajectories_from_stats(stats, set_name):
    targets, predictions = [], []

    target_key = f"{set_name}_{TARGET_SAMPLE_KEY}"
    try:
        targets = stats[target_key]
    except:
        print(f"\t\tDidn't find {target_key} in stats...")

    prediction_key = f"{set_name}_{PREDICTION_SAMPLE_KEY}"
    try:
        predictions = stats[prediction_key]
    except:
        print(f"\t\tDidn't find {prediction_key} in stats...")

    return targets, predictions


# Get run configuration, statistics, and checkpoint objects.  Returns None for
# any that don't exist.
def get_config_stats_checkpoint(runs_path, run):
    config, stats, checkpoint = None, None, None

    config_file = op.join(runs_path, run, "config.pkl")
    if op.exists(config_file):
        with open(config_file, "rb") as file:
            config = pickle.load(file)

    stats_file = op.join(runs_path, run, "statistics.pkl")
    if op.exists(stats_file):
        with open(stats_file, "rb") as file:
            stats = pickle.load(file)

    checkpoint_file = op.join(runs_path, run, "checkpoint.pt")
    if op.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)

    return config, stats, checkpoint


def get_post_processed_stats_file(runs_path, run):
    stats = None
    stats_file = op.join(runs_path, run, "post_processing", "post_statistics.pkl")
    if op.exists(stats_file):
        with open(stats_file, "rb") as file:
            stats = pickle.load(file)
    return stats


def get_post_processed_fixed_horizon_stats_file(runs_path, run):
    stats = None
    stats_file = op.join(runs_path, run, "traj_sweep_statistics.pkl")
    if op.exists(stats_file):
        with open(stats_file, "rb") as file:
            stats = pickle.load(file)
    return stats


# =============================== Gather data ================================ #
# Loop over dataset categories, then dataset size, then individual runs.
runs_needing_statistics = []
finished_runs_needing_post_statistics = []
results = {}

sent_warning = {"elbow": False, "cube": False}

wandb_api = wandb.Api()

for experiment in EXPERIMENTS.keys():
    print(f"\n\n============== Starting {experiment} ==============")
    exp_dict = deepcopy(EXPERIMENT_DICT)
    system = EXPERIMENTS[experiment]["system"]
    exp_dict["system"] = system
    exp_dict["prefix"] = EXPERIMENTS[experiment]["prefix"]
    exp_dict["data_sweep"] = make_empty_data_sweep_dict()

    body_names = BODY_NAMES_BY_SYSTEM[system]

    for exponent in DATASET_EXPONENTS:
        results_folder_name = f"sweep_{experiment}-{exponent}"
        runs_path = op.join(RESULTS_DIR, results_folder_name, "runs")
        if not op.isdir(runs_path):
            print(f"Could not find {results_folder_name} runs; skipping.")
            continue

        print(f"\nFound {results_folder_name}.")

        for run in os.listdir(runs_path):
            if int(run[2:4]) in BAD_RUN_NUMBERS[experiment]:
                continue
                if not sent_warning[experiment]:
                    print(
                        f"WARNING: Skipping run numbers "
                        + f"{BAD_RUN_NUMBERS[experiment]}"
                    )
                    sent_warning[experiment] = True

            config, stats, checkpoint = get_config_stats_checkpoint(runs_path, run)

            if stats == None:
                print(f"\tNo stats file for {run}; skipping.")
                runs_needing_statistics.append(
                    op.join(runs_path, run).split("results/")[-1]
                )
                continue

            assert config != None and checkpoint != None
            print(f"\tFound statistics for {run}.", end="")

            run_key, run_dict = get_run_info_from_config(config)

            performance_dict = get_performance_from_stats(stats, run_dict["result_set"])
            run_dict["results"] = performance_dict

            # Check for post-processed statistics.
            post_stats = get_post_processed_stats_file(runs_path, run)
            if post_stats == None:
                print(f"  No post-processed statistics found.", end="")
                finished_runs_needing_post_statistics.append(
                    op.join(runs_path, run).split("results/")[-1]
                )

            else:
                print(f"  Found post-processed stats, too.", end="")
                post_performance_dict = get_performance_from_stats(
                    post_stats, "test", kind=POST_KIND
                )
                run_dict["post_results"] = post_performance_dict

            # Check for post-processed fixed horizon statistics.
            fixed_horizon_stats = get_post_processed_fixed_horizon_stats_file(
                runs_path, run
            )
            if fixed_horizon_stats == None:
                print(f"  No fixed horizon stats.")
            else:
                print(f"  Also fixed horizon stats!")
                fixed_horizon_dict = get_performance_from_stats(
                    fixed_horizon_stats, "test", kind=FIXED_HORIZON_KIND
                )
                run_dict["fixed_horizon_post_results"] = fixed_horizon_dict

            # If structured, save initial and learned physical parameters.
            if run_dict["structured"]:
                best_system_state = checkpoint["best_learned_system_state"]
                params_dict = get_physical_parameters(
                    system, body_names, best_system_state
                )
                run_dict["learned_params"] = params_dict

                init_params_dict = get_init_physical_parameters(
                    system, body_names, checkpoint, wandb_api
                )
                run_dict["initial_params"] = init_params_dict

            # Store everything in larger dictionary.
            exp_dict["data_sweep"][exponent].update({run_key: run_dict})

    results.update({experiment: exp_dict})

print(f"\n\nSaving results to json file.")
with open(JSON_OUTPUT_FILE, "w") as file:
    json.dump(results, file, indent=2)

pdb.set_trace()

print(f"\n\nRuns needing statistics: {runs_needing_statistics}")
