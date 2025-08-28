#!/usr/bin/env python3

"""Utility functions for running the experiment"""
import time
from typing import Optional

import numpy as np
from pydrake.geometry import HalfSpace, Box, Mesh, Shape
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
import trimesh
import torch

from dair_pll import action_utils, dataset_management
from dair_pll.action_utils import Action, ActionWorkspaceParams, action_to_knots
from dair_pll.trifinger_utils import TrifingerLCMService
from dair_pll.drake_system import DrakeSystem
from dair_pll.hack_utils import extract_robot_trajectory
from dair_pll.geometry import PydrakeToCollisionGeometryFactory, GeometryRepresentation
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem
from dair_pll.dataset_management import TrajectorySet


### Evaluation Functions
def get_true_geometry_and_mesh(sample_count: int = 1000) -> tuple[Shape, np.ndarray]:
    """Get True Geometry from configured base system"""
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    system = DrakeSystem()
    inspector = system.plant_diagram.scene_graph.model_inspector()
    all_geom_ids = inspector.GetAllGeometryIds()
    trimesh_mesh = None
    for geom_id in all_geom_ids:
        true_geom = inspector.GetShape(geom_id)
        if isinstance(true_geom, HalfSpace):
            continue

        if isinstance(true_geom, Box):
            trimesh_mesh = trimesh.primitives.Box(extents=true_geom.size())
        elif isinstance(true_geom, Mesh):
            vertices = np.stack(
                [
                    true_geom.GetConvexHull().vertex(idx)
                    for idx in range(true_geom.GetConvexHull().num_vertices())
                ]
            )
            trimesh_mesh = trimesh.Trimesh(vertices=vertices).convex_hull
            trimesh_mesh.process()

        if trimesh_mesh is None:
            continue

        return true_geom, np.array(
            trimesh.sample.sample_surface(trimesh_mesh, sample_count)[0]
        )
    raise Exception("True geometry not found")


def chamfer_metric(
    learned_system: MultibodyLearnableTactileSystem,
    true_mesh: np.ndarray,
    true_pose: np.ndarray,
) -> float:
    """Chamfer distance between true object and learned object"""
    learned_mesh = learned_system.get_learned_geometry(surface_sample=True)
    learned_pose = learned_system.get_learned_pose().detach().cpu().numpy()

    assert len(learned_mesh.shape) == 2
    assert len(true_mesh.shape) == 2
    assert learned_mesh.shape[-1] == 3
    assert true_mesh.shape[-1] == 3
    assert true_pose.shape == (7,)
    assert learned_pose.shape == (7,)

    # Transform mesh into world frame
    true_mesh_world = (
        Rotation.from_quat(true_pose[:4], scalar_first=True).apply(true_mesh)
        + true_pose[4:]
    )
    learned_mesh_world = (
        Rotation.from_quat(learned_pose[:4], scalar_first=True).apply(learned_mesh)
        + learned_pose[4:]
    )

    return chamfer_distance(true_mesh_world, learned_mesh_world)


# From https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def chamfer_distance(
    x: np.ndarray, y: np.ndarray, metric: str = "l2", direction: str = "bi"
) -> float:
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist


### Scoring Functions
def score_random(actions: list[Action]) -> list[float]:
    """Random actions have equal score"""
    return [1.0] * len(actions)


def score_eig(
    params: ActionWorkspaceParams,
    learned_system: MultibodyLearnableTactileSystem,
    data_trajectories: TrajectorySet,
    trifinger_lcm: TrifingerLCMService,
    force_finger: int,
    actions: list[Action],
) -> list[float]:
    """Score is EIG"""
    start = time.time()
    action_knots = action_to_knots(
        params,
        actions,
        learned_system.get_learned_centroid(),
        force_finger=force_finger,
    )
    traj_x, traj_time = action_utils.interpolate_sampled_action(
        data=torch.tensor(action_knots)
    )
    robot_traj = extract_robot_trajectory(
        traj_x,
        learned_system,
        trifinger_lcm,
    )

    ## Weight by observed info, ignore object qw
    if dataset_management.obs_info_cache is None:
        print("New Obs Info")
        obs_info = (
            torch.zeros_like(fisher[0])
            if len(data_trajectories.trajectories) == 0
            else learned_system.observed_info(data_trajectories)[..., 1:, 1:]
        )
        dataset_management.obs_info_cache = obs_info
    else:
        print("Cached Obs Info")
        obs_info = dataset_management.obs_info_cache
    obs_info_inv = torch.linalg.inv(obs_info + 1e-1 * torch.eye(obs_info.size()[0]))

    # ignore object qw
    fisher = learned_system.expected_fisher_info(
        ctrl_desired=robot_traj,
        timestamps=traj_time,
    )[..., 1:, 1:]

    # breakpoint()

    fisher_obs_weighted = fisher @ obs_info_inv
    fisher_traces = torch.vmap(torch.trace)(fisher_obs_weighted)
    print(f"Evaluated {len(fisher_traces)} actions in {(time.time()-start):.3f}s")
    return fisher_traces.detach().cpu().numpy().tolist()
