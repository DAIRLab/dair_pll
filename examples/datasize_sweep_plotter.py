from typing import List, Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from dair_pll import file_utils
from plot_styler import PlotStyler
from real_data_study import STUDY_PARAMS, study_name_from_params, \
    STUDY_NAME_PREFIX, STORAGE
from sweep_study_plotter import PLOT_DIR, find_sweep_instances

STYLER = PlotStyler()
STYLER.set_default_styling(directory=PLOT_DIR)
STORAGE_NAME = STUDY_NAME_PREFIX
# plot config: (evaluation key, plot label, scale factor, filename)
STUDY_COLORS = [STYLER.blue]
STUDY_DISPLAY_NAMES = ['End-to-end DNN']
ROT_PLOT = ('test_model_rot_err_1',
            'Rotation Error [Degrees]',
            180. / 3.14159,
            'rot_err.png')
POS_PLOT = ('test_model_pos_err_1',
            'Position Error [% Block Width]',
            1000.,
            'pos_err.png')
PLOT_CONFIGS = [
    ROT_PLOT,
    POS_PLOT
]

def get_ylabels(yim: Tuple[float, float]) -> Tuple[List[float], List[str]]:
    """construct y tick labels as powers of ten, displayed as decimals,
    clipped to the orders of magnitude of the data."""
    min_mag = int(np.floor(np.log10(yim[0])))
    max_mag = int(np.ceil(np.log10(yim[1])))
    yticks = [10 ** i for i in range(min_mag, max_mag + 1)]
    ytick_labels = []
    for mag in range(min_mag, max_mag + 1):
        if mag >= 0:
            ytick_labels.append('1' + '0' * mag)
        else:
            ytick_labels.append('0.' + '0' * (-mag - 1) + '1')
    return yticks, ytick_labels

YTICKS = [0.01, 0.1, 1, 10, 100]
YTICK_LABELS = ['0.01', '0.1', '1', '10', '100']


def get_quantity_from_sweep_instance(sweep_instance: str,
                                     quantity_name: str,
                                     scale: float) -> float:
    evaluation = file_utils.load_evaluation(STORAGE, sweep_instance)
    quantities = evaluation[quantity_name]
    return scale * sum(quantities) / len(quantities)


ConfidenceInterval = Tuple[float, float, float]

def log_gaussian_confidence_interval(
        quantity_list: List[float],
        z_score: float = 2.576) -> ConfidenceInterval:
    """Construct a log-gaussian confidence interval via Cox's method."""
    quantities = torch.tensor(quantity_list)
    Y = quantities.log()
    Y_bar = Y.mean()
    n = len(quantity_list)
    S_squared = ((Y- Y_bar) ** 2).sum() / (n - 1)

    log_mean_estimate = Y_bar + S_squared/2
    log_interval_width = z_score * (
        S_squared / n + (S_squared ** 2) / (2 * (n - 1))
    ) ** 0.5
    return (
        torch.exp(log_mean_estimate - log_interval_width).item(),
        torch.exp(log_mean_estimate).item(),
        torch.exp(log_mean_estimate + log_interval_width).item()
    )


def get_confidence_interval_from_instances(
        sweep_instances: List[str], quantity_name: str, scale: float) -> \
        ConfidenceInterval:
    quantities = []
    for sweep_instance in sweep_instances:
        try:
            quantities.append(get_quantity_from_sweep_instance(
                sweep_instance, quantity_name, scale))
        except FileNotFoundError:
            continue
    return log_gaussian_confidence_interval(quantities)


def get_sweep_confidence_intervals(
        sweep_instance_map: Dict[int, List[str]],
        quantity_name: str,
        scale: float) -> Dict[int, ConfidenceInterval]:
    sweep_values = {}
    for sweep_value, sweep_instances in sweep_instance_map.items():
        sweep_values[sweep_value] = get_confidence_interval_from_instances(
            sweep_instances, quantity_name, scale)
    return sweep_values


def datasize_comparison():
    for plot_config in PLOT_CONFIGS:
        plt.figure()
        for study_params, color in zip(STUDY_PARAMS, STUDY_COLORS):
            study_name = study_name_from_params(study_params)
            sweep_instances = find_sweep_instances(STORAGE_NAME, study_name)
            confidence_intervals = get_sweep_confidence_intervals(sweep_instances,
                plot_config[0], plot_config[2])
            sweep_values = np.array(sorted(confidence_intervals.keys()))
            bounds = []
            for i in range(3):
                bounds.append(np.array([confidence_intervals[sweep_value][i]
                               for sweep_value in sweep_values]))
            #pdb.set_trace()
            STYLER.plot(sweep_values, bounds[1], color=color)
            STYLER.plot_bands(sweep_values, sweep_values, bounds[0], bounds[2],
                              color=color)
        plt.xlabel('Number of Training Trajectories')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(plot_config[1])
        ax = plt.gca()
        ax.minorticks_off()
        #ax.majorticks_off()
        ylim = plt.ylim()
        yticks, ylabels = get_ylabels(ylim)
        #pdb.set_trace()
        plt.setp(ax,
                 xticks=sweep_values,
                 xticklabels=[str(int(s)) for s in sweep_values],
                 yticks=yticks,
                 yticklabels=ylabels
                 )
        #ax.set_xticks(sweep_values)
        plt.xlim(sweep_values[0], sweep_values[-1])
        plt.legend(STUDY_DISPLAY_NAMES)
        STYLER.save_fig(plot_config[3])



if __name__ == '__main__':
    datasize_comparison()
