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


CN_METHODS_ONLY = ['ContactNetsI', 'ContactNetsI + Residual',
                   'ContactNets', 'ContactNets + Residual']
METHOD_RESULTS = {'ContactNetsI': '#01256e',
                  'ContactNetsI + Residual': '#398537',
                  'Prediction': '#95001a',
                  'Prediction + Residual': '#92668d',
                  'End-to-End': '#4a0042',
                  'ContactNets': '#1111ff',
                  'ContactNets + Residual': '#11ff11',}
METRICS = {#'model_loss_mean': {
           #      'label': 'Loss',
           #      'yformat': "%.0f", 'scaling': 1.0,
           #      'ylims': {'elbow': [None, None], 'cube': [None, None],
           #                'asymmetric': [None, None]},
           #      'legend_loc': 'best'},
           # 'oracle_loss_mean': {
           #      'label': 'Loss',
           #      'yformat': "%.0f", 'scaling': 1.0,
           #      'ylims': {'elbow': [None, None], 'cube': [None, None],
           #                'asymmetric': [None, None]},
           #      'legend_loc': 'best'},
           'model_trajectory_mse_mean': {
                'label': 'Accumulated trajectory error',
                'yformat': "%.0f", 'scaling': 1.0,
                'ylims': {'elbow': [None, None], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': 'best'},
           'model_pos_int_traj': {
                'label': 'Trajectory positional error [m]',
                'yformat': "%.2f", 'scaling': 1.0,
                'ylims': {'elbow': [-0.01, 0.4], 'cube': [-0.01, 0.4],
                          'asymmetric': [-0.01, 0.4]},
                'legend_loc': 'best'},
           'model_angle_int_traj': {
                'label': 'Trajectory rotational error [deg]',
                'yformat': "%.0f", 'scaling': 180/np.pi,
                'ylims': {'elbow': [None, None], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': 'lower right'},
           'model_penetration_int_traj': {
                'label': 'Trajectory penetration [m]',
                'yformat': "%.3f", 'scaling': 1.0,
                'ylims': {'elbow': [None, 0.02], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': 'best'}
            }

PARAMETER_VALUES = ["m", "px", "py", "pz", "I_xx", "I_yy", "I_zz", "I_xy",
                   "I_xz", "I_yz", "mu", "diameter_x", "diameter_y",
                   "diameter_z", "center_x", "center_y", "center_z"]

GEOMETRY_PARAMETER_ERROR = 'geometry_parameter_error'
FRICTION_PARAMETER_ERROR = 'friction_error'
PARAMETER_ERRORS = {
    GEOMETRY_PARAMETER_ERROR: {'label': 'Geometry parameter error [m]',
                               'yformat': "%.2f", 'scaling': 1.0,
                               'ylims': {'elbow': [0.0, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': 'upper left'},
    FRICTION_PARAMETER_ERROR: {'label': 'Friction error',
                               'yformat': "%.2f", 'scaling': 1.0,
                               'ylims': {'elbow': [None, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': 'best'}
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
        return 'Prediction + Residual'
    elif not run_dict['contactnets'] and not run_dict['residual']:
        return 'Prediction'
    elif run_dict['loss_variation'] == 3:
        if run_dict['contactnets'] and run_dict['residual']:
            return 'ContactNetsI + Residual'
        elif run_dict['contactnets'] and not run_dict['residual']:
            return 'ContactNetsI'
    elif run_dict['loss_variation'] == 1:
        if run_dict['contactnets'] and run_dict['residual']:
            return 'ContactNets + Residual'
        elif run_dict['contactnets'] and not run_dict['residual']:
            return 'ContactNets'

    raise RuntimeError(f"Unknown method with run_dict: {run_dict}")

def fill_exp_dict_with_single_run_data(run_dict, exponent, exp_dict):
    method = get_method_name_by_run_dict(run_dict)

    for result_metric in run_dict['results'].keys():
        new_key = result_metric[5:] if result_metric[:5] == 'test_' else \
            result_metric

        if new_key in METRICS:    
            exp_dict[method][new_key][exponent].append(
                run_dict['results'][result_metric])
        elif new_key in PARAMETER_METRICS_BY_EXPERIMENT[experiment]:
            exp_dict[method][new_key][exponent].append(
                run_dict['results'][result_metric])

    return exp_dict

def convert_lists_to_t_conf_dict(exp_dict, exponent):
    # Iterate over methods then metrics and parameters.
    for method in METHOD_RESULTS.keys():
        # Here "quantity" can be a metric or parameter.
        for quantity in exp_dict[method].keys():
            if quantity == N_RUNS:
                continue

            vals = exp_dict[method][quantity][exponent]

            mean, lower, upper = set_of_vals_to_t_confidence_interval(vals)

            exp_dict[method][quantity][exponent] = {
                'mean': mean, 'lower': lower, 'upper': upper
            }
            exp_dict[method][N_RUNS][exponent] = \
                max(len(vals), exp_dict[method][N_RUNS][exponent])

    return exp_dict

def get_plottable_values(exp_dict, metric, method, metric_lookup):
    try:
        data_dict = exp_dict[method][metric]
    except:
        return [None], [None], [None], [None]

    xs, ys, lowers, uppers = [], [], [], []

    scaling = metric_lookup[metric]['scaling']

    for x in data_dict.keys():
        xs.append(2**(x-1))
        ys.append(data_dict[x]['mean'])
        lowers.append(data_dict[x]['lower'])
        uppers.append(data_dict[x]['upper'])

    if None not in ys:
        ys = [y*scaling for y in ys]
        lowers = [lower*scaling for lower in lowers]
        uppers = [upper*scaling for upper in uppers]

    return xs, ys, lowers, uppers

def get_plottable_run_counts(exp_dict, method):
    data_dict = exp_dict[method][N_RUNS]

    xs, ys = [], []

    for x in data_dict.keys():
        xs.append(2**(x-1))
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

def format_plot(ax, fig, metric, metric_lookup, system):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(XS), max(XS))

    ax.set_ylim(bottom=metric_lookup[metric]['ylims'][system][0],
                   top=metric_lookup[metric]['ylims'][system][1])

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    xs_rounded = [round(x, 1) for x in XS]
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xticks(xs_rounded)
    ax.set_xticklabels(xs_rounded)

    ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
    ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.tick_params(axis='y', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)

    ax.yaxis.set_major_formatter(
        FormatStrFormatter(metric_lookup[metric]['yformat']))
    ax.yaxis.set_minor_formatter(
        FormatStrFormatter(metric_lookup[metric]['yformat']))

    plt.xlabel('Training tosses')
    plt.ylabel(metric_lookup[metric]['label'])

    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='major')

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(loc=metric_lookup[metric]['legend_loc'],
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

def do_run_num_plot(exp_dict, experiment):
    # Start a plot.
    fig = plt.figure()
    ax = plt.gca()

    for method in METHOD_RESULTS.keys():
        xs, ys = get_plottable_run_counts(exp_dict, method)

        # Plot the run numbers.
        ax.plot(xs, ys, label=method, linewidth=5,
                color=METHOD_RESULTS[method])

    ax.set_xscale('log')
    ax.set_xlim(min(XS), max(XS))

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    xs_rounded = [round(x, 1) for x in XS]
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xticks(xs_rounded)
    ax.set_xticklabels(xs_rounded)

    ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
    ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.tick_params(axis='y', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

    plt.xlabel('Training tosses')
    plt.ylabel('Number of runs')

    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='major')

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(prop=dict(weight='bold'))

    fig.set_size_inches(13, 13)

    plt.title(experiment)
    fig_path = op.join(OUTPUT_DIR, f'{experiment}_run_nums.png')
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
        

pdb.set_trace()









