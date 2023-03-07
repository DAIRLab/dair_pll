"""Utility functions for visualizing trajectories.

Visualization of Drake systems can be done with Drake's VideoWriter. This allows
for a relatively thin implementation of visualization for very complex
geometries.

The main contents of this file are as follows:

    * A method to generate a dummy ``DrakeSystem`` which simultaneously
      visualizes two trajectories of the same system.
    * A method which takes a ``DrakeSystem`` and corresponding trajectory,
      captures a visualization video, and outputs it as a numpy ndarray.
"""
from copy import deepcopy
from typing import Tuple

import numpy as np
import time

from pydrake.geometry import Role, RoleAssign, Rgba  # type: ignore
from pydrake.math import RigidTransform  # type: ignore
from pydrake.visualization import VideoWriter  # type: ignore

from torch import Tensor
from PIL import Image

from dair_pll.drake_system import DrakeSystem
from dair_pll import file_utils

RESOLUTION = [640, 480]
RED = Rgba(0.6, 0.0, 0.0, 0.5)
BLUE = Rgba(0.0, 0.0, 0.6, 0.7)
BASE_SYSTEM_DEFAULT_COLOR = RED
LEARNED_SYSTEM_DEFAULT_COLOR = BLUE
PERCEPTION_COLOR_GROUP = 'phong'
PERCEPTION_COLOR_PROPERTY = 'diffuse'
LEARNED_TAG = '__learned__'


def generate_visualization_system(
        base_system: DrakeSystem,
        learned_system: DrakeSystem = None,
        base_system_color: Rgba = BASE_SYSTEM_DEFAULT_COLOR,
        learned_system_color: Rgba = LEARNED_SYSTEM_DEFAULT_COLOR
) -> DrakeSystem:
    """Generate a dummy ``DrakeSystem`` for visualizing comparisons between two
    trajectories of ``base_system``.

    Does so by generating a new ``DrakeSystem`` in which every model in the
    base system has a copy. Each illustration geometry element in these two
    copies is uniformly colored to be visually distinguishable.

    The copy of the base system can optionally be rendered in its learned
    geometry.

    Args:
        base_system: System to be visualized.
        learned_system: Optionally, the learned system so the predicted
          trajectory is rendered with the learned geometry.
        base_system_color: Color to repaint every thing in base system.
        learned_system_color: Color to repaint every thing in duplicated system.

    Returns:
        New ``DrakeSystem`` with doubled state and repainted elements.
    """
    # Start with true base system.
    double_urdfs = deepcopy(base_system.urdfs)
    double_urdfs.update({
        k: file_utils.get_geometrically_accurate_urdf(v) for k, v in \
        double_urdfs.items()
    })

    # Either copy the base system's geometry or optionally use the learned
    # geometry.
    if learned_system is None:
        double_urdfs.update({
            (k + LEARNED_TAG): file_utils.get_geometrically_accurate_urdf(v) \
            for k, v in double_urdfs.items()
        })

    else:
        double_urdfs.update({
            (k + LEARNED_TAG): v for k, v in learned_system.urdfs.items()
        })

    visualization_system = DrakeSystem(double_urdfs,
                                       base_system.dt,
                                       enable_visualizer=True)

    # Recolors every perception geometry to default colors
    plant_diagram = visualization_system.plant_diagram
    plant = plant_diagram.plant
    scene_graph = plant_diagram.scene_graph
    scene_graph_context = scene_graph.GetMyContextFromRoot(
                            plant_diagram.sim.get_mutable_context())
    inspector = scene_graph.model_inspector()
    for model_id in plant_diagram.model_ids:
        model_name = plant.GetModelInstanceName(model_id)
        for body_index in plant.GetBodyIndices(model_id):
            body_frame = plant.GetBodyFrameIdOrThrow(body_index)
            for geometry_id in inspector.GetGeometries(body_frame,
                                                       Role.kPerception):
                props = inspector.GetPerceptionProperties(geometry_id)
                # phong.diffuse is the name of property controlling perception
                # color.
                if props and \
                   props.HasProperty(PERCEPTION_COLOR_GROUP, \
                                     PERCEPTION_COLOR_PROPERTY):
                    # Sets color in properties.
                    props.UpdateProperty(
                        PERCEPTION_COLOR_GROUP, PERCEPTION_COLOR_PROPERTY,
                        learned_system_color
                        if LEARNED_TAG in model_name else base_system_color)
                    # Tells ``scene_graph`` to update the color.
                    scene_graph.RemoveRole(scene_graph_context,
                                           plant.get_source_id(), geometry_id,
                                           Role.kPerception)
                    scene_graph.AssignRole(scene_graph_context,
                                           plant.get_source_id(), geometry_id,
                                           props, RoleAssign.kNew)

    # Changing perception properties requires the ``Simulator`` to be
    # re-initialized.
    plant_diagram.sim.Initialize()

    return visualization_system


def visualize_trajectory(drake_system: DrakeSystem,
                         x_trajectory: Tensor,
                         framerate: int = 30
) -> Tuple[np.ndarray, int]:
    """Visualizes trajectory of system.

    Specifies a ``framerate`` for output video, though should be noted that
    this framerate is only approximately represented by homogeneous integer
    downsampling of the state trajectory. For example, ``if drake_system.dt ==
    1/60`` and ``framerate == 11``, the true video framerate will be::

        max(round(60/11), 1) == 5.

    Args:
        drake_system: System associated with provided trajectory.
        x_trajectory: (T, drake_system.space.n_x) state trajectory.
        framerate: desired frames per second of output video.

    Returns:
        (1, T, 3, H, W) ndarray video capture of trajectory with resolution
        H x W, which are set to 480x640 in :py:mod:`dair_pll.drake_utils`.
        The true framerate, rounded to an integer.
    """
    assert drake_system.plant_diagram.visualizer is not None
    assert x_trajectory.dim() == 2

    vis = drake_system.plant_diagram.visualizer
    sim = drake_system.plant_diagram.sim

    # Downsample trajectory to approximate framerate.
    temporal_downsample = max(round((1 / drake_system.dt) / framerate), 1)
    actual_framerate = round((1 / drake_system.dt) / temporal_downsample)
    x_trajectory = x_trajectory[::temporal_downsample, :]

    # Simulate the system according to the provided data.
    _, carry = drake_system.sample_initial_condition()
    for x_current in x_trajectory:
        drake_system.preprocess_initial_condition(x_current.unsqueeze(0), carry)

        # Force publish video frame.
        sim_context = sim.get_mutable_context()
        video_context = vis.GetMyContextFromRoot(sim_context)
        vis._publish(video_context)

    # Compose a video ndarray of shape (T, H, W, 4[rgba]).
    video = np.stack([np.asarray(frame) for frame in vis._pil_images])
    vis.Save()

    # Since Drake's VideoWriter defaults to not looping gifs, re-load and re-
    # save the gif to ensure it loops.  This gif is only for debugging purposes,
    # as the gif gets overwritten with every trajectory.  The actual output of
    # this function is a numpy array.
    im = Image.open(vis._filename)
    new_name = vis._filename.split('.')[0] + '_.gif'
    im.save(new_name, save_all=True, loop=0)
    im.close()

    # Remove alpha channel and reorder axes to output type.
    height = video.shape[1]
    video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
    video = video[:, :, :3, :, :]
    return video, actual_framerate

