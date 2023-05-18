import pickle
from typing import List, Tuple, Dict, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from dair_pll import file_utils
from examples.plot_styler import PlotStyler


def average(list: List) -> float:
    return torch.tensor(list).mean().item()


tt = lambda x: torch.tensor(x)
na = lambda x: np.array(x, dtype=np.float32)


def load_sweep(study_name: str) -> Dict:
    datasizes = file_utils.sweep_data_sizes(study_name)
    sweep = {}
    for N_train in datasizes:
        N_runs = file_utils.get_sweep_summary_count(study_name, N_train)
        if N_runs > 0:
            sweep[N_train] = []
            for N_run in range(N_runs):
                runfile = file_utils.sweep_summary_file(study_name, N_train,
                                                        N_run)
                with open(runfile, 'rb') as f:
                    sweep[N_train].append(pickle.load(f))
    # pdb.set_trace()
    return sweep


def log_gaussian_band_values(runs: List[Dict], param: str,
                             param2: Optional[str] = None) -> Tuple[
    float, float, float]:
    T = 2.02
    # pdb.set_trace()
    if param2 is None:
        v = torch.tensor([tt(d[param]).mean() for d in runs])
    else:
        v = torch.tensor(
            [tt(d[param]).mean() - tt(d[param2]).mean() for d in runs])

    if v.min() <= 0.:
        print("WARNING: fallback to gaussian interval")
        M = v.mean()
        S = v.std()
        return ((M - T * S).item(), M.item(), (M + T * S).item())
    log_v = torch.log(v)
    M = torch.mean(log_v)
    V = torch.var(log_v)
    N = v.nelement()
    l = M + (V / 2)
    # pdb.set_trace()
    r = T * torch.sqrt((V / N) + ((V ** 2) / (2 * (N - 1))))
    return (
        torch.exp(l - r).item(), torch.exp(l).item(), torch.exp(l + r).item())


def query_sweep(sweep: Dict[int, List], func: Callable, param: str,
                param2: Optional[str] = None) -> Dict[
    int, Tuple[float, float, float]]:
    return {k: func(v, param, param2) for (k, v) in sweep.items()}


def plot_sweep_comparison(plot_name: str, study_names: List[str]) -> None:
    study_bands = []
    metrics = ['loss', 'traj_mse', 'rot_err', 'pos_err']
    distributions = ['train_oracle', 'train_model', 'test_model']
    comps = ['oracle', 'train', 'test', 'suboptimality', 'generalization']
    comparison_chain = []
    comparison_names = []
    for metric in metrics:
        fields = [f'{dist}_{metric}' for dist in distributions]
        metric_names = [f'{metric}_{comp}' for comp in comps]
        comparison_names += metric_names
        comparison_chain += [(f, None) for f in fields]
        comparison_chain += [(fields[1], fields[0]), (fields[2], fields[1])]

    # fields = ['train_oracle_loss', 'train_model_loss', 'test_model_loss', 'test_model_traj_mse', 'test_model_rot_err', 'test_model_pos_err']
    # comparison_names = fields + ['model_suboptimality', 'generalization_error']
    # comparison_chain = [(f,None) for f in fields] + [(fields[1], fields[0]), (fields[2], fields[1])]
    colors = ['r', 'g', 'b']
    for study_name in study_names:
        sweep = load_sweep(study_name)
        bands = []
        for i in range(len(comparison_chain)):
            query = query_sweep(sweep, log_gaussian_band_values,
                                comparison_chain[i][0], comparison_chain[i][1])
            bands.append(query)
        study_bands.append(bands)

        for band in bands:
            print(band)

    # pdb.set_trace()
    for i in range(len(comparison_chain)):
        sub_opt_loss = [b[i] for b in study_bands]
        x_lows = [na(list(b.keys())) for b in sub_opt_loss]
        xs = x_lows
        x_highs = x_lows
        y_lows = [na([b[k][0] for k in b.keys()]) for b in sub_opt_loss]
        ys = [na([b[k][1] for k in b.keys()]) for b in sub_opt_loss]
        y_highs = [na([b[k][2] for k in b.keys()]) for b in sub_opt_loss]
        plt.figure()
        ps = PlotStyler()
        j = 0
        for x_low, x, x_high, y_low, y, y_high in zip(x_lows, xs, x_highs,
                                                      y_lows, ys, y_highs):
            ps.plot(x, y, color=colors[j])
            # pdb.set_trace()
            ps.plot_bands(x_low, x_high, y_low, y_high, color=colors[j])
            j += 1
        ps.set_default_styling(directory=file_utils.plots_dir())
        ps.save_fig(f'{plot_name}_{comparison_names[i]}')
        # ps.show_fig()


if __name__ == '__main__':
    TEST = False

    if TEST:
        plot_name = 'mujoco_cube_test'
        study_names = [f'mujoco_cube_{stiffness}_sweep_test' for stiffness in
                       [300]]
    else:
        plot_name = 'mujoco_cube'
        study_names = [f'mujoco_cube_{stiffness}' for stiffness in
                       [100, 300, 2500]]
    plot_sweep_comparison(plot_name, study_names)
