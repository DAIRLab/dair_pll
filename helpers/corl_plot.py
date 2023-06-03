"""This script is meant to be run after `corl_gather_results.py`, whose output
is a json file that this script accesses to generate plots.

experiment/
    method/
        n_runs:  dataset size --> number of experiments
        metric/
            dataset size/
                [list of values]
        parameter/
            dataset size/
                [list of values]   <-- empty if end-to-end
"""

from collections import defaultdict
import sys
from copy import deepcopy

import json
import math
import os
import os.path as op
import pdb
import re
from typing import Any, DefaultDict, List, Tuple

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np



RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
JSON_OUTPUT_FILE = op.join(op.dirname(__file__), 'results_cluster.json')
JSON_GRAVITY_FILE = op.join(op.dirname(__file__), 'gravity_results.json')


CN_METHODS_ONLY = ['VimpI', 'VimpI RP',
                   'Vimp', 'Vimp RP']
METHOD_RESULTS = {'VimpI': '#01256e',
                  'VimpI RP': '#398537',
                  'Vimp': '#1111ff',
                  'Vimp RP': '#11ff11',
                  'DiffSim': '#95001a',
                  'DiffSim RP': '#92668d',
                  'End-to-End': '#4a0042',}
METRICS = {'model_loss_mean': {
                'label': 'Loss', 'scaling': 1.0,
                'yformat': {'elbow': "%.0f", 'cube': "%.0f",
                            'asymmetric': "%.0f"},
                'ylims': {'elbow': [None, None], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'}},
           # 'oracle_loss_mean': {
           #      'label': 'Loss',
           #      'yformat': "%.0f", 'scaling': 1.0,
           #      'ylims': {'elbow': [None, None], 'cube': [None, None],
           #                'asymmetric': [None, None]},
           #      'legend_loc': 'best'},
           'model_trajectory_mse_mean': {
                'label': 'Accumulated trajectory error', 'scaling': 1.0,
                'yformat': {'elbow': "%.0f", 'cube': "%.0f",
                            'asymmetric': "%.0f"},
                'ylims': {'elbow': [None, None], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'}},
           'model_pos_int_traj': {
                'label': 'Trajectory positional error [m]', 'scaling': 1.0,
                'yformat': {'elbow': "%.2f", 'cube': "%.1f",
                            'asymmetric': "%.2f"},
                'ylims': {'elbow': [0.0, 0.45], 'cube': [-0.01, 0.45],
                          'asymmetric': [-0.01, 0.4]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'}},
           'model_angle_int_traj': {
                'label': 'Trajectory rotational error [deg]',
                'scaling': 180/np.pi,
                'yformat': {'elbow': "%.0f", 'cube': "%.0f",
                            'asymmetric': "%.0f"},
                'ylims': {'elbow': [None, None], 'cube': [None, 100],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'}},
           'model_penetration_int_traj': {
                'label': 'Trajectory penetration [m]', 'scaling': 1.0,
                'yformat': {'elbow': "%.3f", 'cube': "%.3f",
                            'asymmetric': "%.3f"},
                'ylims': {'elbow': [-0.005, 0.03], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'}}
            }

PARAMETER_VALUES = ["m", "px", "py", "pz", "I_xx", "I_yy", "I_zz", "I_xy",
                   "I_xz", "I_yz", "mu", "diameter_x", "diameter_y",
                   "diameter_z", "center_x", "center_y", "center_z"]

GEOMETRY_PARAMETER_ERROR = 'geometry_parameter_error'
FRICTION_PARAMETER_ERROR = 'friction_error'
PARAMETER_ERRORS = {
    GEOMETRY_PARAMETER_ERROR: {'label': 'Geometry parameter error [m]',
                               'scaling': 1.0,
                               'yformat': {'elbow': "%.2f", 'cube': "%.2f",
                                           'asymmetric': "%.2f"},
                               'ylims': {'elbow': [0.0, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': {'elbow': 'best', 'cube': 'best',
                                              'asymmetric': 'best'}},
    FRICTION_PARAMETER_ERROR: {'label': 'Friction error',
                               'scaling': 1.0,
                               'yformat': {'elbow': "%.2f", 'cube': "%.2f",
                                           'asymmetric': "%.2f"},
                               'ylims': {'elbow': [None, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': {'elbow': 'best', 'cube': 'best',
                                              'asymmetric': 'best'}}
}

PARAMETER_METRICS_BY_EXPERIMENT = {
    'cube': [GEOMETRY_PARAMETER_ERROR],
    'elbow': [GEOMETRY_PARAMETER_ERROR],
    'asymmetric_vortex': [GEOMETRY_PARAMETER_ERROR, FRICTION_PARAMETER_ERROR],
    'asymmetric_viscous': [GEOMETRY_PARAMETER_ERROR, FRICTION_PARAMETER_ERROR],
    'elbow_vortex': [GEOMETRY_PARAMETER_ERROR, FRICTION_PARAMETER_ERROR],
    'elbow_viscous': [GEOMETRY_PARAMETER_ERROR, FRICTION_PARAMETER_ERROR]}

CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY = {
    'cube': {
        'body': {
            'diameter_x': 0.1048, 'diameter_y': 0.1048, 'diameter_z': 0.1048,
            'center_x': 0., 'center_y': 0., 'center_z': 0.,
            'mu': 0.15}
    },
    'elbow': {
        'elbow_1': {
            'diameter_x': 0.1, 'diameter_y': 0.05, 'diameter_z': 0.05,
            'center_x': 0., 'center_y': 0., 'center_z': 0.,
            'mu': 0.3},
        'elbow_2': {
            'diameter_x': 0.1, 'diameter_y': 0.05, 'diameter_z': 0.05,
            'center_x': 0.035, 'center_y': 0., 'center_z': 0.,
            'mu': 0.3}
    },
    'asymmetric': {
        'body': {
            'diameter_x': 0.10000000149011612,
            'diameter_y': 0.07500000298023224,
            'diameter_z': 0.07500000298023224,
            'center_x': 0.02500000223517418,
            'center_y': 0.012500000186264515,
            'center_z': -0.012500000186264515,
            'mu': 0.15}
    }
}
N_RUNS = 'n_runs'

SYSTEM_BY_EXPERIMENT = {
    'cube': 'cube',
    'elbow': 'elbow',
    'asymmetric_vortex': 'asymmetric',
    'asymmetric_viscous': 'asymmetric',
    'elbow_vortex': 'elbow',
    'elbow_viscous': 'elbow'}

DATASET_SIZE_DICT = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
GRAVITY_FRACTION_DICT = {0.: [], 0.5: [], 1.: [], 1.5: [], 2.0: []}
    
# The following are t values for 95% confidence interval.
T_SCORE_PER_DOF = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776,
                   5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                   9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
                   13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
                   17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
                   21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
                   25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
                   29: 2.045, 30: 2.042}

RUN_NUMBERS_TO_SKIP = [str(i).zfill(2) for i in range(20)]
GRAVITY_RUN_NUMBERS_TO_SKIP = [str(i).zfill(2) for i in range(3)]

XS = [2**(key-1) for key in DATASET_SIZE_DICT.keys()]

# Some settings on the plot generation.
rc('legend', fontsize=30)
plt.rc('axes', titlesize=40)    # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels


# ============================= Helper functions ============================= #
def get_empty_experiment_dict_by_experiment(experiment):
    # First get a list of bodies in the system.
    system = SYSTEM_BY_EXPERIMENT[experiment]
    bodies = CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys()

    # Then build structure.
    empty_dict_per_experiment = deepcopy(METHOD_RESULTS)
    for method in empty_dict_per_experiment.keys():
        empty_dict_per_experiment[method] = deepcopy(METRICS)
        empty_dict_per_experiment[method].update(
            {N_RUNS: deepcopy(DATASET_SIZE_DICT)})
        for metric in METRICS.keys():
            empty_dict_per_experiment[method][metric] = \
                deepcopy(DATASET_SIZE_DICT)
        for param_metric in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
            empty_dict_per_experiment[method].update(
                {param_metric: deepcopy(DATASET_SIZE_DICT)})
        for exponent in DATASET_SIZE_DICT.keys():
            empty_dict_per_experiment[method][N_RUNS][exponent] = 0

    return empty_dict_per_experiment

def get_empty_gravity_experiment_dict_by_experiment(experiment):
    # First get a list of bodies in the system.
    system = SYSTEM_BY_EXPERIMENT[experiment]
    bodies = CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys()

    # Then build structure.
    empty_dict_per_experiment = deepcopy(METHOD_RESULTS)
    for method in empty_dict_per_experiment.keys():
        empty_dict_per_experiment[method] = deepcopy(METRICS)
        empty_dict_per_experiment[method].update(
            {N_RUNS: deepcopy(GRAVITY_FRACTION_DICT)})
        for metric in METRICS.keys():
            empty_dict_per_experiment[method][metric] = \
                deepcopy(GRAVITY_FRACTION_DICT)
        for param_metric in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
            empty_dict_per_experiment[method].update(
                {param_metric: deepcopy(GRAVITY_FRACTION_DICT)})
        for gravity_frac in GRAVITY_FRACTION_DICT.keys():
            empty_dict_per_experiment[method][N_RUNS][gravity_frac] = 0

    return empty_dict_per_experiment

def set_of_vals_to_t_confidence_interval(ys):
    if len(ys) <= 1:
        return None, None, None

    dof = len(ys) - 1

    ys_np = np.array(ys)

    mean = np.mean(ys)
    lower = mean - T_SCORE_PER_DOF[dof]*np.std(ys)/np.sqrt(dof+1)
    upper = mean + T_SCORE_PER_DOF[dof]*np.std(ys)/np.sqrt(dof+1)

    return mean, lower, upper

def get_method_name_by_run_dict(run_dict):
    if not run_dict['structured']:
        return 'End-to-End'
    elif not run_dict['contactnets'] and run_dict['residual']:
        return 'DiffSim RP'
    elif not run_dict['contactnets'] and not run_dict['residual']:
        return 'DiffSim'
    elif run_dict['loss_variation'] == 3:
        if run_dict['contactnets'] and run_dict['residual']:
            return 'VimpI RP'
        elif run_dict['contactnets'] and not run_dict['residual']:
            return 'VimpI'
    elif run_dict['loss_variation'] == 1:
        if run_dict['contactnets'] and run_dict['residual']:
            return 'Vimp RP'
        elif run_dict['contactnets'] and not run_dict['residual']:
            return 'Vimp'

    raise RuntimeError(f"Unknown method with run_dict: {run_dict}")

def fill_exp_dict_with_single_run_data(run_dict, sweep_instance, exp_dict):
    method = get_method_name_by_run_dict(run_dict)

    for result_metric in run_dict['results'].keys():
        new_key = result_metric[5:] if result_metric[:5] == 'test_' else \
            result_metric

        if new_key in METRICS:    
            exp_dict[method][new_key][sweep_instance].append(
                run_dict['results'][result_metric])
        elif new_key in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
            exp_dict[method][new_key][sweep_instance].append(
                run_dict['results'][result_metric])

    return exp_dict

def convert_lists_to_t_conf_dict(exp_dict, sweep_instance):
    # Iterate over methods then metrics and parameters.
    for method in METHOD_RESULTS.keys():
        # Here "quantity" can be a metric or parameter.
        for quantity in exp_dict[method].keys():
            if quantity == N_RUNS:
                continue

            vals = exp_dict[method][quantity][sweep_instance]

            mean, lower, upper = set_of_vals_to_t_confidence_interval(vals)

            exp_dict[method][quantity][sweep_instance] = {
                'mean': mean, 'lower': lower, 'upper': upper
            }
            exp_dict[method][N_RUNS][sweep_instance] = \
                max(len(vals), exp_dict[method][N_RUNS][sweep_instance])

    return exp_dict

def get_plottable_values(exp_dict, metric, method, metric_lookup, gravity=False):
    try:
        data_dict = exp_dict[method][metric]
    except:
        return [None], [None], [None], [None]

    xs, ys, lowers, uppers = [], [], [], []

    scaling = metric_lookup[metric]['scaling']

    for x in data_dict.keys():
        if gravity:
            xs.append(x)
        else:
            xs.append(2**(x-1))
        ys.append(data_dict[x]['mean'])
        lowers.append(data_dict[x]['lower'])
        uppers.append(data_dict[x]['upper'])

    if None not in ys:
        ys = [y*scaling for y in ys]
        lowers = [lower*scaling for lower in lowers]
        uppers = [upper*scaling for upper in uppers]

    return xs, ys, lowers, uppers

def get_plottable_run_counts(exp_dict, method, gravity=False):
    data_dict = exp_dict[method][N_RUNS]

    xs, ys = [], []

    for x in data_dict.keys():
        if not gravity:
            xs.append(2**(x-1))
        else:
            xs.append(x)
        ys.append(data_dict[x])

    return xs, ys

def convert_parameters_to_errors(run_dict, experiment):
    params_dict = run_dict['learned_params']
    if params_dict == None:
        return run_dict

    for param_metric in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
        if param_metric == GEOMETRY_PARAMETER_ERROR:
            run_dict = calculate_geometry_error(run_dict, experiment)
        elif param_metric == FRICTION_PARAMETER_ERROR:
            run_dict = calculate_friction_error(run_dict, experiment)
        else:
            raise RuntimeError(f"Can't handle {param_metric} type.")

    return run_dict

def format_plot(ax, fig, metric, metric_lookup, system, gravity=False):
    if not gravity:
        ax.set_xscale('log')
        ax.set_xlim(min(XS), max(XS))
        x_markers = [round(x, 1) for x in XS]
    else:
        ax.set_xlim(0, 2)
        x_markers = [0, 0.5, 1, 1.5, 2]

    ax.set_ylim(bottom=metric_lookup[metric]['ylims'][system][0],
                   top=metric_lookup[metric]['ylims'][system][1])

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xticks(x_markers)
    ax.set_xticklabels(x_markers)

    ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
    ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)

    ax.tick_params(axis='y', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)

    ax.yaxis.set_major_formatter(
        FormatStrFormatter(metric_lookup[metric]['yformat'][system]))
    ax.yaxis.set_minor_formatter(
        FormatStrFormatter(metric_lookup[metric]['yformat'][system]))

    if not gravity:
        plt.xlabel('Training tosses')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    else:
        plt.xlabel('Gravity fraction')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.ylabel(metric_lookup[metric]['label'])

    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='major')

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(loc=metric_lookup[metric]['legend_loc'][system],
               prop=dict(weight='bold'))

    fig.set_size_inches(13, 13)

def get_single_body_correct_geometry_array(system, body):
    # In order of diameters then centers x y z, get the correct parameters.
    params = CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]
    ground_truth = np.array([params['diameter_x'], params['diameter_y'],
        params['diameter_z'], params['center_x'], params['center_y'],
        params['center_z']])
    return ground_truth

def calculate_geometry_error(run_dict, experiment):
    system = SYSTEM_BY_EXPERIMENT[experiment]

    # Start an empty numpy array to store true and learned values.
    true_vals = np.array([])
    learned_vals = np.array([])

    # Iterate over bodies in the system.
    for body in CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys():
        body_dict = run_dict['learned_params'][body]

        ground_truth = get_single_body_correct_geometry_array(system, body)

        learned = np.array([body_dict['diameter_x'], body_dict['diameter_y'],
                            body_dict['diameter_z'], body_dict['center_x'],
                            body_dict['center_y'], body_dict['center_z']])
        
        true_vals = np.concatenate((true_vals, ground_truth))
        learned_vals = np.concatenate((learned_vals, learned))
    
    # Calculate geometry error as norm of the difference between learned and
    # true values.
    geometry_error = np.linalg.norm(true_vals - learned_vals)

    # Insert this error into the results dictionary.
    run_dict['results'].update({GEOMETRY_PARAMETER_ERROR: geometry_error})
    return run_dict

def calculate_friction_error(run_dict, experiment):
    system = SYSTEM_BY_EXPERIMENT[experiment]

    # Start an empty numpy array to store true and learned values.
    true_vals = np.array([])
    learned_vals = np.array([])

    # Iterate over bodies in the system.
    for body in CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys():
        body_dict = run_dict['learned_params'][body]

        ground_truth = np.array([
            CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]['mu']])
        learned = np.array([body_dict['mu']])
        
        true_vals = np.concatenate((true_vals, ground_truth))
        learned_vals = np.concatenate((learned_vals, learned))
    
    # Calculate friction error as norm of the difference between learned and
    # true values.
    friction_error = np.linalg.norm(true_vals - learned_vals)

    # Insert this error into the results dictionary.
    run_dict['results'].update({FRICTION_PARAMETER_ERROR: friction_error})
    return run_dict

def do_run_num_plot(exp_dict, experiment, gravity=False):
    # Start a plot.
    fig = plt.figure()
    ax = plt.gca()

    for method in METHOD_RESULTS.keys():
        xs, ys = get_plottable_run_counts(exp_dict, method, gravity=gravity)

        # Plot the run numbers.
        ax.plot(xs, ys, label=method, linewidth=5,
                color=METHOD_RESULTS[method])

    if not gravity:
        ax.set_xscale('log')
        ax.set_xlim(min(XS), max(XS))
        x_markers = [round(x, 1) for x in XS]
    else:
        ax.set_xlim(0, 2)
        x_markers = [0, 0.5, 1, 1.5, 2]

    ax.set_ylim(0, 10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xticks(x_markers)
    ax.set_xticklabels(x_markers)

    ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
    ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)

    ax.tick_params(axis='y', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

    if not gravity:
        plt.xlabel('Training tosses')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    else:
        plt.xlabel('Gravity fraction')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.ylabel('Number of runs')

    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='major')

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(prop=dict(weight='bold'))

    fig.set_size_inches(13, 13)

    plt.title(experiment)
    fig_name = 'gravity_' if gravity else ''
    fig_name += f'{experiment}_run_nums.png'
    fig_path = op.join(OUTPUT_DIR, fig_name)
    fig.savefig(fig_path, dpi=100)
    plt.close()

# =============================== Plot results =============================== #
# Load the results from the json file.
with open(JSON_OUTPUT_FILE) as file:
    results = json.load(file)

sent_warning = False

# Iterate over experiments.
for experiment in results.keys():
    system = SYSTEM_BY_EXPERIMENT[experiment]
    exp_dict = get_empty_experiment_dict_by_experiment(experiment)

    data_sweep = results[experiment]['data_sweep']

    # Iterate over dataset sizes to collect all the data.
    for exponent_str in data_sweep.keys():
        exponent = int(exponent_str)

        # Iterate over runs.
        for run_name, run_dict in data_sweep[exponent_str].items():
            if run_name[2:4] in RUN_NUMBERS_TO_SKIP:
                if not sent_warning:
                    print(f'WARNING: Skipping any run numbers in ' + \
                          f'{RUN_NUMBERS_TO_SKIP}.')
                    sent_warning = True
                continue

            run_dict = convert_parameters_to_errors(run_dict, experiment)
            exp_dict = fill_exp_dict_with_single_run_data(run_dict, exponent,
                                                          exp_dict)

        # Convert lists to dictionary with keys average, upper, and lower.
        exp_dict = convert_lists_to_t_conf_dict(exp_dict, exponent)

    # Iterate over the metrics to do plots of each.
    for metric in METRICS.keys():
        # Start a plot.
        fig = plt.figure()
        ax = plt.gca()

        for method in METHOD_RESULTS.keys():
            xs, ys, lowers, uppers = get_plottable_values(exp_dict, metric,
                                                          method, METRICS)
            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, metric, METRICS, system)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'{experiment}_{metric}.png')
        fig.savefig(fig_path, dpi=100)
        plt.close()

    # Iterate over parameter metrics to do plots of each.
    for parameter_metric in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
        # Start a plot.
        fig = plt.figure()
        ax = plt.gca()

        for method in METHOD_RESULTS.keys():
            xs, ys, lowers, uppers = get_plottable_values(
                exp_dict, parameter_metric, method, PARAMETER_ERRORS)

            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, parameter_metric, PARAMETER_ERRORS, system)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'{experiment}_{parameter_metric}.png')
        fig.savefig(fig_path, dpi=100)
        plt.close()

    # Add in a test plot of the number of experiments.
    do_run_num_plot(exp_dict, experiment)
        


# =========================== Plot gravity results =========================== #
# Load the results from the gravity json file.
with open(JSON_GRAVITY_FILE) as file:
    results = json.load(file)

send_warning = False

# Iterate over gravity experiments.
for experiment in results.keys():
    system = SYSTEM_BY_EXPERIMENT[experiment]
    exp_dict = get_empty_gravity_experiment_dict_by_experiment(experiment)

    gravity_sweep = results[experiment]['gravity_sweep']

    # Iterate over dataset sizes to collect all the data.
    for grav_frac in gravity_sweep.keys():
        grav_frac = float(grav_frac)

        # Iterate over runs.
        for run_name, run_dict in gravity_sweep[str(grav_frac)].items():
            if run_name[2:4] in GRAVITY_RUN_NUMBERS_TO_SKIP:
                if not sent_warning:
                    print(f'WARNING: Skipping any run numbers in ' + \
                          f'{GRAVITY_RUN_NUMBERS_TO_SKIP}.')
                    sent_warning = True
                continue

            run_dict = convert_parameters_to_errors(run_dict, experiment)
            exp_dict = fill_exp_dict_with_single_run_data(run_dict, grav_frac,
                                                          exp_dict)

        # Convert lists to dictionary with keys average, upper, and lower.
        exp_dict = convert_lists_to_t_conf_dict(exp_dict, grav_frac)

    # Iterate over the metrics to do plots of each.
    for metric in METRICS.keys():
        # Start a plot.
        fig = plt.figure()
        ax = plt.gca()

        for method in METHOD_RESULTS.keys():
            xs, ys, lowers, uppers = get_plottable_values(
                exp_dict, metric, method, METRICS, gravity=True)

            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, metric, METRICS, system, gravity=True)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'gravity_{experiment}_{metric}.png')
        fig.savefig(fig_path, dpi=100)
        plt.close()

    # Iterate over parameter metrics to do plots of each.
    for parameter_metric in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
        # Start a plot.
        fig = plt.figure()
        ax = plt.gca()

        for method in METHOD_RESULTS.keys():
            xs, ys, lowers, uppers = get_plottable_values(
                exp_dict, parameter_metric, method, PARAMETER_ERRORS)

            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, parameter_metric, PARAMETER_ERRORS, system)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'{experiment}_{parameter_metric}.png')
        fig.savefig(fig_path, dpi=100)
        plt.close()

    # Add in a test plot of the number of experiments.
    do_run_num_plot(exp_dict, experiment, gravity=True)





