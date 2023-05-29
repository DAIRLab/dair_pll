import pdb
from typing import List, Dict, Tuple

import torch
from matplotlib import pyplot as plt

from dair_pll import file_utils
from plot_styler import PlotStyler
from real_data_study import STUDY_PARAMS, study_name_from_params, \
    STUDY_NAME_PREFIX, STORAGE
from sweep_study_plotter import PLOT_DIR, find_sweep_instances

STORAGE_NAME = STUDY_NAME_PREFIX

PLOT_QUANTITIES = ['test_model_rot_err_1', 'test_model_pos_err_1']


def get_quantity_from_sweep_instance(sweep_instance: str,
                                     quantity_name: str) -> float:
    evaluation = file_utils.load_evaluation(STORAGE, sweep_instance)
    pdb.set_trace()
    return evaluation[f'{quantity_name}_mean']


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
        sweep_instances: List[str], quantity_name: str) -> ConfidenceInterval:
    quantities = [
        get_quantity_from_sweep_instance(sweep_instance, quantity_name)
        for sweep_instance in sweep_instances
    ]
    return log_gaussian_confidence_interval(quantities)


def get_sweep_confidence_intervals(
        sweep_instance_map: Dict[int, List[str]],
        quantity_name: str) -> Dict[int, ConfidenceInterval]:
    sweep_values = {}
    for sweep_value, sweep_instances in sweep_instance_map.items():
        sweep_values[sweep_value] = get_confidence_interval_from_instances(
            sweep_instances, quantity_name)
    return sweep_values


def datasize_comparison():
    styler = PlotStyler()
    styler.set_default_styling(directory=PLOT_DIR)
    plt.figure()
    for plot_quantity in PLOT_QUANTITIES:
        for study_params in STUDY_PARAMS:
            study_name = study_name_from_params(study_params)
            sweep_instances = find_sweep_instances(STORAGE_NAME, study_name)
            confidence_intervals = get_sweep_confidence_intervals(sweep_instances,
                plot_quantity)
            pdb.set_trace()


if __name__ == '__main__':
    datasize_comparison()
