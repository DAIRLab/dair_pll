#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1:00:00
#SBATCH --qos=posa-high
#SBATCH --partition=posa-compute

import json
import math
import os
import os.path as op
import pdb  # noqa
import re

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import FormatStrFormatter

from dair_pll import file_utils


rc('legend', fontsize=24)
plt.rc('axes', titlesize=30)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


# NAME_TO_EXPERIMENT = {'s00-.+': 'Simulation Inertia Mode 0'} #,
                      # 's01-.+': 'Simulation Inertia Mode 1',
                      # 's02-.+': 'Simulation Inertia Mode 2',
                      # 's03-.+': 'Simulation Inertia Mode 3',
                      # 's04-.+': 'Simulation Inertia Mode 4'}

SWEEP_NAMES = ['s05-.+', 's06-.+', 's07-.+', 's08-.+', 's09-.+']

VALIDATION_LOSS = 'valid_model_loss_mean'

RESULTS_FOLDER = file_utils.RESULTS_DIR
PLOTS_FOLDER = file_utils.PLOTS_DIR
LOG_FOLDER = file_utils.LOG_DIR

SYSTEMS = ['elbow', 'cube']
SOURCES = ['real', 'simulation']
TRAIN_LOSSES = ['ContactNets', 'prediction']
GEOMETRY_TYPE = ['box', 'mesh']
INERTIA_MODES = ['none', 'masses', 'CoMs', 'CoMsandmasses', 'all']
INITIAL_URDF = ['correct', 'wrong']
DATASET_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
CONFIG_KEYS = ['system', 'source', 'loss_type', 'geometry_type',
               'inertia_mode', 'initial_urdf', 'dataset_size', 'timestep']
RESULTS_KEYS = ['experiment_config', 'datasets', 'scalars_list', 'stats_list',
                'success']

"""Get the scalars and statistics per epoch, the experiment settings from the
params.txt file of the experiment name, a dictionary of the dataset indices
for each train/valid/test set, and whether the training process went to
completion, returned as a dictionary of these five elements (exp_config,
datasets, scalars_list, stats_list, success)."""
def load_results_from_experiment(exp_name):
    txt_file = f'{RESULTS_FOLDER}/{exp_name}/params.txt'

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # do an initial test that the text file format is as expected.
    start_line = lines.index('Epoch 0:\n')
    assert lines[start_line+3] == 'Epoch 1:\n'

    def compile_experiment_config_dict(beginning_lines):
        lns = [item for item in beginning_lines if item != '\n']

        # first grab experiment configuration settings
        experiment_config = {}
        experiment_config['system'] = lns[1].split(': ')[1][:-1].replace(
                                                    ' ', '').replace('.','')
        experiment_config['source'] = lns[2].split(': ')[1][:-1].replace(
                                                    ' ', '').replace('.','')
        experiment_config['inertia_mode'] = lns[9].split(': ')[1][:-1].replace(
                                                    ' ', '').replace('.','')
        experiment_config['loss_type'] = 'ContactNets' if \
            'True' in lns[3] else 'prediction'
        experiment_config['geometry_type'] = 'box' if \
            'True' in lns[4] else 'mesh'
        experiment_config['initial_urdf'] = 'correct' if \
            'True' in lns[10] else 'wrong'

        assert experiment_config['system'] in SYSTEMS
        assert experiment_config['source'] in SOURCES
        assert experiment_config['inertia_mode'] in INERTIA_MODES

        config_line = 11
        while 'n_pop' not in lns[config_line]:  config_line += 1

        # get the timestep
        experiment_config['timestep'] = float(lns[config_line].split(
                                            'dt=')[1].split(',')[0])

        if experiment_config['source'] == 'simulation':
            # get the dataset size from n_pop
            dataset_str = lns[config_line].split('n_pop=')[1].split(',')[0]
            experiment_config['dataset_size'] = int(dataset_str)
        else:
            # get the dataset size from n_import
            while 'n_import' not in lns[config_line]:  config_line += 1
            dataset_str = lns[config_line].split('n_import=')[1].split(',')[0]
            experiment_config['dataset_size'] = int(dataset_str)

        assert experiment_config['dataset_size'] in DATASET_SIZES

        return experiment_config

    def convert_scalars_line_to_dict(line):
        dict_str = line.split('\tscalars: ')[1][:-1]
        dict_str = dict_str.replace('\'', '\"')
        return json.loads(dict_str)

    def convert_stats_line_to_dict(line):
        dict_str = line.split('\tstatistics: ')[1][:-1]
        dict_str = dict_str.split(']), ')[-1]
        dict_str = '{' + dict_str.replace('\'', '\"')
        return json.loads(dict_str)

    def compile_datasets(beginning_lines):
        datasets = {}

        # Get the start and end of the training set indices.
        i = 11
        while 'indices' not in beginning_lines[i]: i += 1
        line = beginning_lines[i]
        j = i+1
        while 'indices' not in beginning_lines[j]:
            line += beginning_lines[j]
            j += 1

        train_indices = line.split('[')[1].split(']')[0]
        datasets['train'] = [int(k) for k in train_indices.split(',') if k != '\n']

        # Get the start and end of the validation set indices.
        i = j
        j = i+1
        line = beginning_lines[i]
        while 'indices' not in beginning_lines[j]:
            line += beginning_lines[j]
            j += 1

        valid_indices = line.split('[')[1].split(']')[0]
        datasets['valid'] = [int(k) for k in valid_indices.split(',') if k != '\n']

        # Get the start and end of the test set indices.
        i = j
        j = i+1
        line = beginning_lines[i]
        while j < len(beginning_lines):
            line += beginning_lines[j]
            j += 1

        test_indices = line.split('[')[1].split(']')[0]
        datasets['test'] = [int(k) for k in test_indices.split(',') if k != '\n']

        return datasets

    experiment_config = compile_experiment_config_dict(lines[:start_line])

    datasets = compile_datasets(lines[:start_line])

    scalars_list = []
    stats_list = [None]

    scalars_list.append(convert_scalars_line_to_dict(lines[start_line+1]))

    # Collect all the other epoch scalars and statistics.
    # The text file should have epoch numbers listed on a line followed
    # by one line of scalars, possibly multiple lines of statistics,
    # one line of the training loss, then a blank line.
    i = start_line + 4
    while i+1 < len(lines):
        if 'scalars' not in lines[i]:
            i += 1
            continue

        scalars_list.append(convert_scalars_line_to_dict(lines[i]))

        i += 1
        stats_line = lines[i]
        while 'train_loss' not in lines[i+1]:
            stats_line += lines[i+1]
            i += 1

        stats_list.append(convert_stats_line_to_dict(stats_line))

    log_file = f'{LOG_FOLDER}/train_{exp_name}.txt'
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    success = True if 'Saving the final' in lines[-1] else False

    return {'experiment_config': experiment_config,
            'datasets': datasets,
            'scalars_list': scalars_list,
            'stats_list': stats_list,
            'success': success}

"""Make a dictionary with keys for experiment names and entries as another
dictionary of experiment configs, dataset indices, scalar list, statistics list,
and training completion success."""
def load_results(instance_regex):
    pattern = re.compile(instance_regex + '\Z')
    results = {}
    for instance_name in os.listdir(RESULTS_FOLDER):
        if pattern.match(instance_name):
            print(f'Found {instance_name}...')

            # get the results
            results[instance_name] = load_results_from_experiment(instance_name)
            
    return results

"""Get the scalars and statistics corresponding to the best validation
loss and the initial scalars upon initialization, returned as a 3-tuple
of (initial_scalars, best_scalars, best_stats)."""
def get_initial_and_learned_model(scalars_list, stats_list):
    lowest_valid_loss = 1e6
    lowest_valid_idx = 1

    for i in range(1, len(scalars_list)):
        valid_loss = stats_list[i][VALIDATION_LOSS]
        if valid_loss < lowest_valid_loss:
            lowest_valid_loss = valid_loss
            lowest_valid_idx = i

    return (scalars_list[0], scalars_list[lowest_valid_idx],
           stats_list[lowest_valid_idx])

"""Make a plot of some statistic given the results dictionary."""
def plot_statistics_key_over_epochs(results_dict, key):
    fig = plt.figure()
    ax = plt.gca()

    # for every experiment
    for exp in results_dict.keys():
        # check if went to completion or not
        if not results_dict[exp]['success']:
            print(f'Experiment {exp} did not complete -- skipping')
            continue

        # grab the right data
        stats_list = results_dict[exp]['stats_list']

        n_epochs = len(stats_list)
        stats = []

        for i in range(1, n_epochs):
            stats.append(stats_list[i][key])

        ax.plot(range(1, n_epochs), stats, linewidth=3, label=exp)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend(prop=dict(weight='bold'))
    fig.set_size_inches(13, 13)
    fig.savefig(f'{PLOTS_FOLDER}/{key}.png', dpi=100)
    print(f'Saved {PLOTS_FOLDER}/{key}.png')


if __name__ == "__main__":
    all_results = {}
    for sweep_name in SWEEP_NAMES:
        results = load_results(sweep_name)
        all_results = {**all_results, **results}

    plot_statistics_key_over_epochs(all_results, VALIDATION_LOSS)

    exp_config, dataset_indices, scalar_list, stat_list = \
              load_results_from_experiment('t19')
    scalars, stats = get_learned_model(scalar_list, stat_list)
    pdb.set_trace()


 
