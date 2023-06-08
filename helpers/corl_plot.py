"""This script is meant to be run after `corl_gather_results.py`, whose output
is a json file that this script accesses to generate plots.

experiment/
    method/
        n_runs:  dataset size      <-- number of experiments
        metric/
            dataset size/
                [list of values]
        parameter/
            dataset size/
                [list of values]   <-- empty if end-to-end
        post_metric/
            dataset_size/
                [list of values]   <-- empty if not calculated
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
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import torch
from torch import Tensor

from dair_pll.system import MeshSummary
from dair_pll.deep_support_function import extract_outward_normal_hyperplanes



RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
JSON_OUTPUT_FILE = op.join(op.dirname(__file__), 'results_cluster.json')
JSON_GRAVITY_FILE = op.join(op.dirname(__file__), 'gravity_results.json')


CN_METHODS_ONLY = [#'VimpI', 'VimpI RP',
                   'Vimp', 'Vimp RP']
METHOD_RESULTS = {#'VimpI': '#01256e',
                  #'VimpI RP': '#398537',
                  'CCN': '#01256e',  #'#1111ff',
                  'CCN-R': '#398537',  #'#11ff11',
                  'DiffSim': '#95001a',
                  'DiffSim-R': '#92668d',
                  'End-to-End': '#4a0042',}
METRICS = {'model_loss_mean': {
                'label': 'Loss', 'scaling': 1.0,
                'yformat': {'elbow': "%.0f", 'cube': "%.0f",
                            'asymmetric': "%.0f"},
                'ylims': {'elbow': [None, None], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'},
                'log': False},
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
                               'asymmetric': 'best'},
                'log': False},
           'model_pos_int_traj': {
                'label': 'Trajectory positional error [m]', 'scaling': 1.0,
                'yformat': {'elbow': "%.2f", 'cube': "%.2f",
                            'asymmetric': "%.2f"},
                'ylims': {'elbow': [0.0, None], 'cube': [-0.01, None],
                          'asymmetric': [-0.01, 0.4]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'},
                'log': False},
           'model_angle_int_traj': {
                'label': 'Trajectory rotational error [deg]',
                'scaling': 180/np.pi,
                'yformat': {'elbow': "%.0f", 'cube': "%.0f",
                            'asymmetric': "%.0f"},
                'ylims': {'elbow': [0.0, None], 'cube': [0.0, None],
                          'asymmetric': [0.0, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'},
                'log': False},
           'model_penetration_int_traj': {
                'label': 'Trajectory penetration [m]', 'scaling': 1.0,
                'yformat': {'elbow': "%.3f", 'cube': "%.3f",
                            'asymmetric': "%.3f"},
                'ylims': {'elbow': [-0.005, 0.03], 'cube': [None, None],
                          'asymmetric': [None, None]},
                'legend_loc': {'elbow': 'best', 'cube': 'best',
                               'asymmetric': 'best'},
                'log':  False}
            }

ROLLOUT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 120]
POST_METRICS_BY_EXPERIMENT = {
    'cube': [],
    'elbow': [f'test_pos_error_w_horizon_{i}' for i in ROLLOUT_LENGTHS] + \
               [f'test_rot_error_w_horizon_{i}' for i in ROLLOUT_LENGTHS],
    'cube_gravity': [],
    'asymmetric_vortex': [],
    'asymmetric_viscous': [],
    'asymmetric_gravity': [],
    'elbow_vortex': [],
    'elbow_viscous': [],
    'elbow_gravity': []
}

PARAMETER_VALUES = ["m", "px", "py", "pz", "I_xx", "I_yy", "I_zz", "I_xy",
                   "I_xz", "I_yz", "mu", "diameter_x", "diameter_y",
                   "diameter_z", "center_x", "center_y", "center_z"]

GEOMETRY_PARAMETER_ERROR = 'geometry_parameter_error'
FRICTION_PARAMETER_ERROR = 'friction_error'
INERTIA_PARAMETER_ERROR = 'inertia_error'
PARAMETER_ERRORS = {
    GEOMETRY_PARAMETER_ERROR: {'label': 'Average vertex location error [m]',
                               'scaling': 1.0,
                               'yformat': {'elbow': "%.3f", 'cube': "%.3f",
                                           'asymmetric': "%.3f"},
                               'ylims': {'elbow': [0.0, None],
                                         'cube': [0.0, None],
                                         'asymmetric': [0.0, None]},
                               'legend_loc': {'elbow': 'best', 'cube': 'best',
                                              'asymmetric': 'best'},
                                'log': False},
    FRICTION_PARAMETER_ERROR: {'label': 'Friction error',
                               'scaling': 1.0,
                               'yformat': {'elbow': "%.1f", 'cube': "%.2f",
                                           'asymmetric': "%.2f"},
                               'ylims': {'elbow': [None, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': {'elbow': 'best', 'cube': 'best',
                                              'asymmetric': 'best'},
                                'log': False},
    INERTIA_PARAMETER_ERROR: {'label': 'Inertia parameter error',
                               'scaling': 1.0,
                               'yformat': {'elbow': "%.0f", 'cube': "%.2f",
                                           'asymmetric': "%.2f"},
                               'ylims': {'elbow': [None, None],
                                         'cube': [None, None],
                                         'asymmetric': [None, None]},
                               'legend_loc': {'elbow': 'best', 'cube': 'best',
                                              'asymmetric': 'best'},
                               'log':  True},
}

ALL_PARAMETER_METRICS = [GEOMETRY_PARAMETER_ERROR, FRICTION_PARAMETER_ERROR,
                         INERTIA_PARAMETER_ERROR]
PARAMETER_METRICS_BY_EXPERIMENT = {
    'cube': [GEOMETRY_PARAMETER_ERROR],
    'elbow': [GEOMETRY_PARAMETER_ERROR],
    'cube_gravity': ALL_PARAMETER_METRICS,
    'asymmetric_vortex': ALL_PARAMETER_METRICS,
    'asymmetric_viscous': ALL_PARAMETER_METRICS,
    'asymmetric_gravity': ALL_PARAMETER_METRICS,
    'elbow_vortex': ALL_PARAMETER_METRICS,
    'elbow_viscous': ALL_PARAMETER_METRICS,
    'elbow_gravity': ALL_PARAMETER_METRICS}

ELBOW_HALF_VERTICES = Tensor([
    [-0.0500, -0.02500, 0.02500],
    [0.0500, -0.02500, 0.02500],
    [-0.0500, 0.02500, 0.02500],
    [0.0500, 0.02500, 0.02500],
    [-0.0500, 0.02500, -0.02500],
    [0.0500, 0.02500, -0.02500],
    [-0.0500, -0.02500, -0.02500],
    [0.0500, -0.02500, -0.02500]])
ASYMMETRIC_VERTICES = Tensor([
    [ 0.0, -0.02500000037252903, -0.05000000074505806],
    [ 0.07500000298023224, 0.0, 0.0],
    [ 0.0, 0.05000000074505806, -0.02500000037252903],
    [ -0.02500000037252903, 0.02500000037252903, -0.02500000037252903],
    [ 0.02500000037252903, 0.02500000037252903, 0.02500000037252903],
    [ 0.05000000074505806, -0.02500000037252903, 0.02500000037252903]])
CUBE_VERTICES = Tensor([
    [ -0.052400, -0.052400, 0.052400],
    [ 0.052400, -0.052400, 0.052400],
    [ -0.052400, 0.052400, 0.052400],
    [ 0.052400, 0.052400, 0.052400],
    [ -0.052400, 0.052400, -0.052400],
    [ 0.052400, 0.052400, -0.052400],
    [ -0.052400, -0.052400, -0.052400],
    [ 0.052400, -0.052400, -0.052400]])

CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY = {
    'cube': {
        'body': {
            'diameter_x': 0.1048, 'diameter_y': 0.1048, 'diameter_z': 0.1048,
            'center_x': 0., 'center_y': 0., 'center_z': 0.,
            'mu': 0.15, 'm': 0.37, 'px': 0.0, 'py': 0.0, 'pz': 0.0,
            'I_xx': 0.00081, 'I_yy': 0.00081, 'I_zz': 0.00081,
            'I_xy': 0.0, 'I_xz': 0.0, 'I_yz': 0.0,
            'scaling_vector': 1.0 / np.array([
            0.37, 0.035, 0.035, 0.035, 0.00081, 0.00081, 0.00081, 0.00081,
            0.00081, 0.00081]), 'vertices': CUBE_VERTICES
        }
    },
    'elbow': {
        'elbow_1': {
            'diameter_x': 0.1, 'diameter_y': 0.05, 'diameter_z': 0.05,
            'center_x': 0., 'center_y': 0., 'center_z': 0.,
            'mu': 0.3, 'm': 0.37, 'px': 0.0, 'py': 0.0, 'pz': 0.0,
            'I_xx': 0.0006167, 'I_yy': 0.0006167, 'I_zz': 0.0006167,
            'I_xy': 0.0, 'I_xz': 0.0, 'I_yz': 0.0,
            'scaling_vector' :1.0 / np.array([
            0.37, 0.035, 0.035, 0.035, 0.0006167, 0.0006167, 0.0006167,
            0.0006167, 0.0006167, 0.0006167]), 'vertices': ELBOW_HALF_VERTICES
        },
        'elbow_2': {
            'diameter_x': 0.1, 'diameter_y': 0.05, 'diameter_z': 0.05,
            'center_x': 0.035, 'center_y': 0., 'center_z': 0.,
            'mu': 0.3, 'm': 0.37, 'px': 0.035, 'py': 0.0, 'pz': 0.0,
            'I_xx': 0.0006167, 'I_yy': 0.0006167, 'I_zz': 0.0006167,
            'I_xy': 0.0, 'I_xz': 0.0, 'I_yz': 0.0,
            'scaling_vector' :1.0 / np.array([
            0.37, 0.035, 0.035, 0.035, 0.0006167, 0.0006167, 0.0006167,
            0.0006167, 0.0006167, 0.0006167]), 'vertices': ELBOW_HALF_VERTICES
        }
    },
    'asymmetric': {
        'body': {
            'diameter_x': 0.10000000149011612,
            'diameter_y': 0.07500000298023224,
            'diameter_z': 0.07500000298023224,
            'center_x': 0.02500000223517418,
            'center_y': 0.012500000186264515,
            'center_z': -0.012500000186264515,
            'mu': 0.15, 'm': 0.25, 'px': 0.0, 'py': 0.0, 'pz': 0.0,
            'I_xx': 0.00081, 'I_yy': 0.00081, 'I_zz': 0.00081,
            'I_xy': 0.0, 'I_xz': 0.0, 'I_yz': 0.0,
            'scaling_vector': 1.0 / np.array([
            0.25, 0.035, 0.035, 0.035, 0.00081, 0.00081, 0.00081, 0.00081,
            0.00081, 0.00081]), 'vertices': ASYMMETRIC_VERTICES
        }
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
def extract_mesh_from_support_points(support_points: Tensor) -> MeshSummary:
    """Given a set of convex polytope vertices, extracts a vertex/face mesh.

    Args:
        support_points: ``(*, 3)`` polytope vertices.

    Returns:
        Object vertices and face indices.
    """
    support_point_hashes = set()
    unique_support_points = []

    # remove duplicate vertices
    for vertex in support_points:
        vertex_hash = hash(vertex.numpy().tobytes())
        if vertex_hash in support_point_hashes:
            continue
        support_point_hashes.add(vertex_hash)
        unique_support_points.append(vertex)

    vertices = torch.stack(unique_support_points)
    hull = ConvexHull(vertices.numpy())
    faces = Tensor(hull.simplices).to(torch.long)  # type: ignore

    _, backwards, _ = extract_outward_normal_hyperplanes(
        vertices.unsqueeze(0), faces.unsqueeze(0))
    backwards = backwards.squeeze(0)
    faces[backwards] = faces[backwards].flip(-1)

    return MeshSummary(vertices=support_points, faces=faces)

def _get_mesh_interior_point(halfspaces: np.ndarray) -> Tuple[np.ndarray, float]:
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))
    objective_coefficients = np.zeros((halfspaces.shape[1],))
    objective_coefficients[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(objective_coefficients, A_ub=A, b_ub=b, bounds=(None, None))
    interior_point = res.x[:-1]
    interior_point_gap = res.x[-1]
    return interior_point, interior_point_gap

def calculate_error_vertices(vertices_learned: Tensor,
                             vertices_true: Tensor) -> Tensor:
    """Relative error between two convex hulls of provided vertices.

    use the identity that the area of the non-overlapping region is the
    sum of the areas of the two polygons minus twice the area of their
    intersection.

    Args:
        vertices_learned: (N, 3) tensor of vertices of the learned geometry.
        vertices_true: (N, 3) tensor of vertices of the true geometry.
    """
    # pylint: disable=too-many-locals
    true_volume = ConvexHull(vertices_true.numpy()).volume
    sum_volume = ConvexHull(vertices_learned.numpy()).volume + true_volume

    mesh_learned = extract_mesh_from_support_points(vertices_learned)
    mesh_true = extract_mesh_from_support_points(vertices_true)

    normal_learned, _, extent_learned = extract_outward_normal_hyperplanes(
        mesh_learned.vertices.unsqueeze(0), mesh_learned.faces.unsqueeze(0))
    normal_true, _, extent_true = extract_outward_normal_hyperplanes(
        mesh_true.vertices.unsqueeze(0), mesh_true.faces.unsqueeze(0))

    halfspaces_true = torch.cat(
        [normal_true.squeeze(), -extent_true.squeeze().unsqueeze(-1)],
        dim=1)

    halfspaces_learned = torch.cat(
        [normal_learned.squeeze(), -extent_learned.squeeze().unsqueeze(-1)],
        dim=1)

    intersection_halfspaces = torch.cat(
        [halfspaces_true, halfspaces_learned], dim=0).numpy()

    # find interior point of intersection
    interior_point, interior_point_gap = _get_mesh_interior_point(
        intersection_halfspaces)

    intersection_volume = 0.

    if interior_point_gap > 0.:
        # intersection is non-empty
        intersection_halfspace_convex = HalfspaceIntersection(
            intersection_halfspaces, interior_point)

        intersection_volume = ConvexHull(
            intersection_halfspace_convex.intersections).volume

    learned_volume = ConvexHull(vertices_learned.numpy()).volume
    print(f'learned, true, intersection volumes: {learned_volume}, {true_volume}, {intersection_volume}')

    return Tensor([sum_volume - 2 * intersection_volume
                  ]).abs() / true_volume

def calculate_vertex_position_error(true_vertices, learned_vertices):
    true_vertices = Tensor(true_vertices)
    learned_vertices = Tensor(learned_vertices)
    assert true_vertices.shape == learned_vertices.shape
    assert true_vertices.shape[1] == 3

    vert_displacement = true_vertices - learned_vertices
    vert_dists = torch.linalg.norm(vert_displacement, dim=1)

    return vert_dists.sum().item()

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
        for param_metric in PARAMETER_METRICS_BY_EXPERIMENT[f'{experiment}_gravity']:
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
        return 'DiffSim-R'
    elif not run_dict['contactnets'] and not run_dict['residual']:
        return 'DiffSim'
    elif run_dict['loss_variation'] == 3:
        return 'dummy'
        # if run_dict['contactnets'] and run_dict['residual']:
        #     return 'VimpI RP'
        # elif run_dict['contactnets'] and not run_dict['residual']:
        #     return 'VimpI'
    elif run_dict['loss_variation'] == 1:
        if run_dict['contactnets'] and run_dict['residual']:
            return 'CCN-R'
        elif run_dict['contactnets'] and not run_dict['residual']:
            return 'CCN'

    raise RuntimeError(f"Unknown method with run_dict: {run_dict}")

def fill_exp_dict_with_single_run_data(run_dict, sweep_instance, exp_dict, gravity=False):
    method = get_method_name_by_run_dict(run_dict)

    exp_key = f'{experiment}_gravity' if gravity else experiment

    for result_metric in run_dict['results'].keys():
        new_key = result_metric[5:] if result_metric[:5] == 'test_' else \
            result_metric

        if new_key in METRICS:    
            exp_dict[method][new_key][sweep_instance].append(
                run_dict['results'][result_metric])
        elif new_key in PARAMETER_METRICS_BY_EXPERIMENT[exp_key]:
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

def convert_parameters_to_errors(run_dict, experiment, gravity=False):
    params_dict = run_dict['learned_params']
    if params_dict == None:
        return run_dict

    exp_key = f'{experiment}_gravity' if gravity else experiment

    for param_metric in PARAMETER_METRICS_BY_EXPERIMENT[exp_key]:
        if param_metric == GEOMETRY_PARAMETER_ERROR:
            run_dict = calculate_geometry_error(run_dict, experiment)
        elif param_metric == FRICTION_PARAMETER_ERROR:
            run_dict = calculate_friction_error(run_dict, experiment)
        elif param_metric == INERTIA_PARAMETER_ERROR:
            run_dict = calculate_inertia_error(run_dict, experiment)
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

    if metric_lookup[metric]['log']:
        ax.set_yscale('log')

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
    # ax.yaxis.set_minor_formatter(
    #     FormatStrFormatter(metric_lookup[metric]['yformat'][system]))

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

    # plt.legend(handles, labels)
    # plt.legend(loc=metric_lookup[metric]['legend_loc'][system],
    #            prop=dict(weight='bold'))

    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    # # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #           prop=dict(weight='bold'))


    fig.set_size_inches(13, 13)

def get_single_body_correct_geometry_array(system, body):
    # In order of diameters then centers x y z, get the correct parameters.
    params = CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]
    ground_truth = np.array([params['diameter_x'], params['diameter_y'],
        params['diameter_z'], params['center_x'], params['center_y'],
        params['center_z']])
    return ground_truth

def get_single_body_correct_inertia_array(system, body):
    # In order of mass, CoM xyz, inertia xx yy zz xy xz yz, get the correct
    # parameters.
    params = CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]
    ground_truth = np.array([params['m'], params['px'], params['py'],
        params['pz'], params['I_xx'], params['I_yy'], params['I_zz'],
        params['I_xy'], params['I_xz'], params['I_yz']])
    return ground_truth

def calculate_geometry_error(run_dict, experiment):
    system = SYSTEM_BY_EXPERIMENT[experiment]

    # Start an empty numpy array to store true and learned values.
    true_vals = np.array([])
    learned_vals = np.array([])

    err = 0.

    # Iterate over bodies in the system.
    for body in CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys():
        body_dict = run_dict['learned_params'][body]

        ground_truth_verts = \
            CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]['vertices']
        learned_verts = Tensor(body_dict['vertices'])
        # err += calculate_error_vertices(learned_verts, ground_truth_verts).item()
        err += calculate_vertex_position_error(ground_truth_verts, learned_verts)

    #     ground_truth = get_single_body_correct_geometry_array(system, body)

    #     learned = np.array([body_dict['diameter_x'], body_dict['diameter_y'],
    #                         body_dict['diameter_z'], body_dict['center_x'],
    #                         body_dict['center_y'], body_dict['center_z']])
        
    #     true_vals = np.concatenate((true_vals, ground_truth))
    #     learned_vals = np.concatenate((learned_vals, learned))
    
    # # Calculate geometry error as norm of the difference between learned and
    # # true values.
    # geometry_error = np.linalg.norm(true_vals - learned_vals)

    # geometry_error = err / len(CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys())
    n_bodies = len(CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys())
    n_verts = len(CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]['vertices'])
    geometry_error = err / (n_bodies * n_verts)

    # Insert this error into the results dictionary.
    run_dict['results'].update({GEOMETRY_PARAMETER_ERROR: geometry_error})
    return run_dict

def calculate_inertia_error(run_dict, experiment):
    system = SYSTEM_BY_EXPERIMENT[experiment]

    # Start an empty numpy array to store true and learned values.
    true_vals = np.array([])
    learned_vals = np.array([])

    # Iterate over bodies in the system.
    for body in CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system].keys():
        body_dict = run_dict['learned_params'][body]

        ground_truth = get_single_body_correct_inertia_array(system, body)

        learned = np.array([body_dict['m'], body_dict['px'], body_dict['py'],
                            body_dict['pz'], body_dict['I_xx'],
                            body_dict['I_yy'], body_dict['I_zz'],
                            body_dict['I_xy'], body_dict['I_xz'],
                            body_dict['I_yz']])

        # Since inertia parameters can be such different sizes, scale all of
        # them to get on similar scale.
        ground_truth = np.multiply(
            ground_truth,
            CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]['scaling_vector']
        )
        learned = np.multiply(
            learned,
            CORRECT_PARAMETERS_BY_SYSTEM_AND_BODY[system][body]['scaling_vector']
        )
        
        true_vals = np.concatenate((true_vals, ground_truth))
        learned_vals = np.concatenate((learned_vals, learned))
    
    # Calculate inertia error as norm of the scaled difference between learned
    # and true values.
    inertia_error = np.linalg.norm(true_vals - learned_vals)

    # Insert this error into the results dictionary.
    run_dict['results'].update({INERTIA_PARAMETER_ERROR: inertia_error})
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

    ax.set_ylim(0, None)

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
            if get_method_name_by_run_dict(run_dict) == 'dummy':  continue
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

    # Iterate over post-processing metrics to do plots of each.
    for 

    # Add in a test plot of the number of experiments.
    do_run_num_plot(exp_dict, experiment)
        

# =========================== Plot gravity results =========================== #
# Load the results from the gravity json file.
with open(JSON_GRAVITY_FILE) as file:
    results = json.load(file)

sent_warning = False

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

            run_dict = convert_parameters_to_errors(run_dict, experiment,
                                                    gravity=True)
            if get_method_name_by_run_dict(run_dict) == 'dummy':  continue
            exp_dict = fill_exp_dict_with_single_run_data(
                run_dict, grav_frac, exp_dict, gravity=True)

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
    exp_key = f'{experiment}_gravity'
    for parameter_metric in PARAMETER_METRICS_BY_EXPERIMENT[exp_key]:
        # Start a plot.
        fig = plt.figure()
        ax = plt.gca()

        for method in METHOD_RESULTS.keys():
            xs, ys, lowers, uppers = get_plottable_values(
                exp_dict, parameter_metric, method, PARAMETER_ERRORS,
                gravity=True)

            # Plot the method unless there are any None objects.
            if None in ys or None in lowers or None in lowers:
                continue

            ax.plot(xs, ys, label=method, linewidth=5,
                    color=METHOD_RESULTS[method])
            ax.fill_between(xs, lowers, uppers, alpha=0.3,
                            color=METHOD_RESULTS[method])

        format_plot(ax, fig, parameter_metric, PARAMETER_ERRORS, system,
                    gravity=True)
        plt.title(experiment)
        fig_path = op.join(OUTPUT_DIR, f'gravity_{experiment}_{parameter_metric}.png')
        fig.savefig(fig_path, dpi=100)
        plt.close()

    # Add in a test plot of the number of experiments.
    do_run_num_plot(exp_dict, experiment, gravity=True)







