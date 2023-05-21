"""This script is meant to be run after `corl_gather_results.py`, whose output
is a json file that this script accesses to generate plots.

experiment/
    method/
        metric/
            dataset size/
                [list of values]
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


METHOD_RESULTS = {'ContactNets': '#01256e',
                  'ContactNets + Residual': '#398537',
                  'Prediction': '#95001a',
                  'Prediction + Residual': '#92668d',
                  'End-to-End': '#4a0042'}
METRICS = {'model_loss_mean': {
                'label': 'Loss',
                'yformat': "%.0f", 'scaling': 1.0},
           'oracle_loss_mean': {
                'label': 'Oracle loss',
                'yformat': "%.0f", 'scaling': 1.0},
           'model_trajectory_mse_mean': {
                'label': 'Accumulated trajectory error',
                'yformat': "%.0f", 'scaling': 1.0},
           'oracle_trajectory_mse_mean': {
                'label': 'Oracle accumulated trajectory error',
                'yformat': "%.0f", 'scaling': 1.0},
           'model_pos_int_traj': {
                'label': 'Trajectory positional error [m]',
                'yformat': "%.2f", 'scaling': 1.0},
           'oracle_pos_int_traj': {
                'label': 'Oracle trajectory positional error [m]',
                'yformat': "%.2f", 'scaling': 1.0},
           'model_angle_int_traj': {
                'label': 'Trajectory rotational error [deg]',
                'yformat': "%.0f", 'scaling': 180/np.pi},
           'oracle_angle_int_traj': {
                'label': 'Oracle trajectory rotational error [deg]',
                'yformat': "%.0f", 'scaling': 180/np.pi},
           'model_penetration_int_traj': {
                'label': 'Trajectory penetration [m]',
                'yformat': "%.3f", 'scaling': 1.0},
           'oracle_penetration_int_traj': {
                'label': 'Oracle trajectory penetration [m]',
                'yformat': "%.3f", 'scaling': 1.0}
            }

DATASET_SIZE_DICT = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
EMPTY_METHOD_DICT_PER_EXPERIMENT = deepcopy(METHOD_RESULTS)
for key in EMPTY_METHOD_DICT_PER_EXPERIMENT.keys():
    EMPTY_METHOD_DICT_PER_EXPERIMENT[key] = deepcopy(METRICS)
    for inner_key in METRICS.keys():
        EMPTY_METHOD_DICT_PER_EXPERIMENT[key][inner_key] = \
            deepcopy(DATASET_SIZE_DICT)
    
# The following are t values for 95% confidence interval.
T_SCORE_PER_DOF = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776,
                   5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                   9: 2.262, 10: 2.228}

XS = [2**(key-1) for key in DATASET_SIZE_DICT.keys()]

# Some settings on the plot generation.
rc('legend', fontsize=30)
plt.rc('axes', titlesize=40)    # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels


# ============================= Helper functions ============================= #
def set_of_vals_to_t_confidence_interval(ys):
    if len(ys) == 0:
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
    if run_dict['contactnets'] and run_dict['residual']:
        return 'ContactNets + Residual'
    elif run_dict['contactnets'] and not run_dict['residual']:
        return 'ContactNets'
    elif not run_dict['contactnets'] and run_dict['residual']:
        return 'Prediction + Residual'
    elif not run_dict['contactnets'] and not run_dict['residual']:
        return 'Prediction'

    raise RuntimeError(f"Unknown method with run_dict: {run_dict}")

def fill_exp_dict_with_single_run_data(run_dict, exponent, exp_dict):
    method = get_method_name_by_run_dict(run_dict)

    for metric in METRICS.keys():
        try:
            exp_dict[method][metric][exponent].append(
                run_dict['results'][f'test_{metric}'])
        except:
            pass

    return exp_dict

def convert_lists_to_t_conf_dict(exp_dict, exponent):
    # Iterate over methods then metrics.
    for method in METHOD_RESULTS.keys():
        for metric in METRICS.keys():
            vals = exp_dict[method][metric][exponent]
            if len(vals) > 9:
                pdb.set_trace()
            mean, lower, upper = set_of_vals_to_t_confidence_interval(vals)

            exp_dict[method][metric][exponent] = {
                'mean': mean, 'lower': lower, 'upper': upper
            }
    return exp_dict

def get_plottable_values(exp_dict, metric, method):
    data_dict = exp_dict[method][metric]
    xs, ys, lowers, uppers = [], [], [], []

    scaling = METRICS[metric]['scaling']

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

def format_plot(ax, fig, metric):
    ax.set_xscale('log')
    ax.set_yscale('log')
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

    ax.yaxis.set_major_formatter(FormatStrFormatter(METRICS[metric]['yformat']))
    ax.yaxis.set_minor_formatter(FormatStrFormatter(METRICS[metric]['yformat']))

    plt.xlabel('Training tosses')
    plt.ylabel(METRICS[metric]['label'])

    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='major')

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(loc=1, prop=dict(weight='bold'))

    fig.set_size_inches(13, 13)

# =============================== Plot results =============================== #
# Load the results from the json file.
with open(JSON_OUTPUT_FILE) as file:
    results = json.load(file)

sent_warning = False

# Iterate over experiments.
for experiment in results.keys():
    exp_dict = deepcopy(EMPTY_METHOD_DICT_PER_EXPERIMENT)
    data_sweep = results[experiment]['data_sweep']

    # Iterate over dataset sizes to collect all the data.
    for exponent_str in data_sweep.keys():
        exponent = int(exponent_str)

        # Iterate over runs.
        for run_name, run_dict in data_sweep[exponent_str].items():
            if '00' in run_name:
                if not sent_warning:
                    print(f'WARNING: Skipping initial runs, e.g. {run_name}.')
                    sent_warning = True
                continue

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
                                                          method)
            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, metric)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'{experiment}_{metric}.png')
        fig.savefig(fig_path, dpi=100)

pdb.set_trace()










