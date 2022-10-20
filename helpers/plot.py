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

EXPERIMENT_NAMES = ['s00-.+', 's01-.+', 's02-.+', 's03-.+', 's04-.+']

VALIDATION_LOSS = 'valid_model_loss_mean'

RESULTS_FOLDER = file_utils.RESULTS_DIR

SYSTEMS = ['elbow', 'cube']
SOURCES = ['real', 'simulation']
TRAIN_LOSSES = ['ContactNets', 'prediction']
GEOMETRY_TYPE = ['box', 'mesh']
INERTIA_MODES = ['none', 'masses', 'CoMs', 'CoMsandmasses', 'all']
INITIAL_URDF = ['correct', 'wrong']
DATASET_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
CONFIG_KEYS = ['system', 'source', 'loss_type', 'geometry_type',
               'inertia_mode', 'initial_urdf', 'dataset_size']

"""Get the scalars and statistics per epoch, and the experiment settings
from the params.txt file of the experiment name."""
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

    experiment_config = compile_experiment_config_dict(lines[:start_line])

    scalars_list = []
    stats_list = [None]

    scalars_list.append(convert_scalars_line_to_dict(lines[start_line+1]))

    # Collect all the other epoch scalars and statistics.
    # The text file should have epoch numbers listed at every 5th line
    # followed by scalars, statistics, the training loss, then a blank line.
    i = start_line + 4
    while i+1 < len(lines):
        scalars_list.append(convert_scalars_line_to_dict(lines[i]))
        stats_list.append(convert_stats_line_to_dict(lines[i+1]))
        i += 5

    return experiment_config, scalars_list, stats_list

"""Get the scalars and statistics corresponding to the best validation
loss."""
def get_learned_model(scalars_list, stats_list):
    lowest_valid_loss = 1e6
    lowest_valid_idx = 1

    for i in range(1, len(scalars_list)):
        valid_loss = stats_list[i][VALIDATION_LOSS]
        if valid_loss < lowest_valid_loss:
            lowest_valid_loss = valid_loss
            lowest_valid_idx = i

    return scalars_list[lowest_valid_idx], stats_list[lowest_valid_idx]

"""Make a dictionary with keys for experiment names and entries as a
3-tuple of experiment configs, scalar list, and statistics list."""
def load_results(instance_regex):
    pattern = re.compile(instance_regex + '\Z')
    results = {}
    for instance_name in os.listdir(RESULTS_FOLDER):
        if pattern.match(instance_name):
            print(f'Found {instance_name}...')

            # get the results
            exp_config, scalar_list, stat_list = \
                load_results_from_experiment(instance_name)
            results[instance_name] = (exp_config, scalar_list, stat_list)
            
    return results

all_results = {}
for sweep_name in EXPERIMENT_NAMES:
    results = load_results(sweep_name)
    all_results = {**all_results, **results}


# exp_config, scalar_list, stat_list = load_results_from_experiment('t19')
# scalars, stats = get_learned_model(scalar_list, stat_list)
pdb.set_trace()


 
