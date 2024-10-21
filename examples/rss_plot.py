from collections import defaultdict
import sys

import json
import math
import os
import os.path as op
import pdb  # noqa
import re
from typing import Any, DefaultDict, List, Tuple

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np

from dair_pll import file_utils


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "storage")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


rc("legend", fontsize=30)
plt.rc("axes", titlesize=40)  # fontsize of the axes title
plt.rc("axes", labelsize=40)  # fontsize of the x and y labels

yscale = 1
use_logs = [True, True, True, False, False, False, False]
plot_points = False

params_to_track = {
    "cube_body_len_x": 0.1048,
    "cube_body_len_y": 0.1048,
    "cube_body_len_z": 0.1048,
    "cube_body_mu": 0.15,
}
yfields = [
    "train_model_trajectory_mse_mean",
    "valid_model_trajectory_mse_mean",
    "train_loss",
    "cube_body_len_x",
    "cube_body_len_y",
    "cube_body_len_z",
    "cube_body_mu",
]
ylabels = [
    "Trajectory state space error (training)",
    "Trajectory state space error",
    "Training loss",
    "Cube x length (normalized)",
    "Cube y length (normalized)",
    "Cube z length (normalized)",
    "Friction coefficient (normalized)",
]
val_scales = [1.0, 1.0, 1.0, 1.0 / 0.1048, 1.0 / 0.1048, 1.0 / 0.1048, 1.0 / 0.15]

for yfield, ylabel, val_scale, use_log in zip(yfields, ylabels, val_scales, use_logs):
    models = {
        "ContactNets, L": ["cn_.+-0", "cn_.+-2", "cn_.+-4", "cn_.+-6"],
        "ContactNets, S": ["cn_.+-1", "cn_.+-3", "cn_.+-5", "cn_.+-7"],
        "DiffSim, L": ["ds_.+-0", "ds_.+-2", "ds_.+-4", "ds_.+-6"],
        "DiffSim, S": ["ds_.+-1", "ds_.+-3", "ds_.+-5", "ds_.+-7"],
    }
    label_lookup = {
        "cn_.+-0": "ContactNets, L",
        "cn_.+-1": "ContactNets, S",
        "cn_.+-2": "ContactNets, L",
        "cn_.+-3": "ContactNets, S",
        "cn_.+-4": "ContactNets, L",
        "cn_.+-5": "ContactNets, S",
        "cn_.+-6": "ContactNets, L",
        "cn_.+-7": "ContactNets, S",
        "ds_.+-0": "DiffSim, L",
        "ds_.+-1": "DiffSim, S",
        "ds_.+-2": "DiffSim, L",
        "ds_.+-3": "DiffSim, S",
        "ds_.+-4": "DiffSim, L",
        "ds_.+-5": "DiffSim, S",
        "ds_.+-6": "DiffSim, L",
        "ds_.+-7": "DiffSim, S",
    }
    color_lookup = {
        "DiffSim, L": "#95001a",
        "ContactNets, L": "#01256e",
        "DiffSim, S": "#92668d",
        "ContactNets, S": "#398537",
    }  # 4a0042

    print(f"\n\n========== Starting {yfield} ==========")

    def num(s: str):
        try:
            return int(s)
        except ValueError:
            return float(s)

    def load_results(instance_regex: str) -> Tuple[DefaultDict[int, List[Any]], bool]:
        pattern = re.compile(instance_regex + "\Z")
        results = defaultdict(list)

        # load results from previous tests
        for instance_name in os.listdir(RESULTS_DIR):
            if (pattern.match(instance_name)) and "64" not in instance_name:
                # print(f'\tFound {instance_name} folder...')

                params_file = op.join(RESULTS_DIR, instance_name, "params.txt")

                if not os.path.isfile(params_file):
                    print(f"\t\t--> did not find params_file in {instance_name}")
                    continue

                data_size = int(instance_name.split("_")[-1].split("-")[0])

                stats = read_params_file(params_file)
                results[int(data_size)].append(stats)

        return results

    def read_params_file(file_name):
        file = open(file_name, "r")

        filestr = file.read().replace("'", "")

        stats = {}
        for key in yfields:
            stats[key] = float(
                filestr.split(f"{key}: ")[-1].split(",")[0].split("}")[0]
            )

        return stats

    def extract_xys(results, y_field):
        extracted = defaultdict(list)
        for i in results.keys():
            for result in results[i]:
                extracted[i].append(float(result[y_field] * val_scale))
        return extracted

    def extract_points(results, y_field):
        extracted = extract_xys(results, y_field)
        xs, ys = [], []
        for x in extracted.keys():
            for y in extracted[x]:
                xs.append(x)
                ys.append(y)
        return xs, ys

    def scatter_to_t_conf_int_plot(extracted):
        # the following are t values for 95% confidence interval
        t_per_dof = {
            1: 12.71,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            0: 0.5,
        }

        means, lowers, uppers = {}, {}, {}

        for k, v in extracted.items():
            dof = len(v) - 1
            means[k] = np.mean(v)
            lowers[k] = np.mean(v) - t_per_dof[dof] * np.std(v) / np.sqrt(dof + 1)
            uppers[k] = np.mean(v) + t_per_dof[dof] * np.std(v) / np.sqrt(dof + 1)

        xs = list(means.keys())
        ys, y_lowers, y_uppers = [], [], []

        for x in xs:
            ys.append(means[x])
            y_lowers.append(lowers[x])
            y_uppers.append(uppers[x])

        xs, ys, y_lowers, y_uppers = zip(*sorted(zip(xs, ys, y_lowers, y_uppers)))

        return xs, ys, y_lowers, y_uppers

    def get_data_counts(extracted):
        return {k: len(v) for k, v in extracted.items()}

    fig = plt.figure()
    ax = plt.gca()

    for model in models.keys():
        print(f"Working on {model}:", end="")

        dicts = []
        for mod in models[model]:
            results = load_results(mod)
            dicts.append(results)

        combined_results = {}
        for k in dicts[0].keys():
            combined_results[k] = []
            for d in dicts:
                for item in d[k]:
                    combined_results[k].append(item)

        results = combined_results
        prefix = ""

        if plot_points:
            xs, ys = extract_points(results, prefix + yfield)
            xs = [x / 2 for x in xs]
            plt.scatter(
                xs,
                ys,
                s=200,
                c=color_lookup[model],
                label=label_lookup[model],
                alpha=0.5,
            )
        else:
            extracted = extract_xys(results, prefix + yfield)
            print(f" with counts {get_data_counts(extracted)}")
            xs, ys, y_lowers, y_uppers = scatter_to_t_conf_int_plot(extracted)
            xs = [x / 2 for x in xs]
            ax.plot(xs, ys, label=model, linewidth=5, color=color_lookup[model])
            ax.fill_between(
                xs, y_lowers, y_uppers, alpha=0.3, color=color_lookup[model]
            )

    ax.set_xscale("log")
    if use_log:
        ax.set_yscale("log")
    elif yfield == "cube_body_mu":
        ax.set_ylim(0, 3.5)
    else:
        ax.set_ylim(0, 1.5)

    xs = [2 * 2**j for j in range(0, 4)]
    ax.set_xlim(min(xs), max(xs))

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    xs_rounded = [round(x, 1) for x in xs]
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xticks(xs_rounded)
    ax.set_xticklabels(xs_rounded)

    ax.tick_params(axis="x", which="minor", bottom=False, labelsize=20)
    ax.tick_params(axis="x", which="major", bottom=False, labelsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.tick_params(axis="y", which="minor", labelsize=20)
    ax.tick_params(axis="y", which="major", labelsize=20)
    if ("body_len" in yfield) or ("body_mu" in yfield):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

    plt.xlabel("Training tosses")
    plt.ylabel(ylabel)

    ax.yaxis.grid(True, which="both")
    ax.xaxis.grid(True, which="major")

    lines = ax.get_lines()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles, labels)
    plt.legend(loc=1, prop=dict(weight="bold"))

    fig.set_size_inches(13, 13)

    fig.savefig(f"{OUTPUT_DIR}/{yfield}.png", dpi=100)
    # fig.savefig(f'{OUTPUT_DIR}/tp_{yfield}.png', transparent=True, dpi=100)
