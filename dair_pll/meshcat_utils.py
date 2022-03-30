"""Utility functions for visualizing trajectories with meshcat.

Visualization of Drake systems is managed by meshcat. This allows for a
relatively thin implementation of visualization for very complex geometries.

The main contents of this file are as follows:

    * A method to generate a dummy ``DrakeSystem`` which simultaneously
      visualizes two trajectories of the same system.
    * A method which takes a ``DrakeSystem`` and corresponding
      trajectory, captures a visualization video, and outputs it as a numpy
      ndarray.

Meshcat visualization must be externally set up with the following steps:

    * Run ``meshcat-server``, which will put up a meshcat server at the
      default address.
    * Open the visualizer ``assets/static.html`` in a browser.

Occasionally, the visualization step may block indefinitely. This is usually
caused by the visualizer browser window loosing connect to the server,
and can be fixed by refreshing the browser page.
"""
from copy import deepcopy
from typing import Tuple

import meshcat  # type: ignore
import numpy as np
from pydrake.geometry import Role, RoleAssign, Rgba  # type: ignore
from pydrake.math import RigidTransform  # type: ignore
from torch import Tensor

from dair_pll.drake_system import DrakeSystem

CAM_FOV = np.pi / 3
RESOLUTION = [640, 480]
RED = Rgba(0.6, 0.0, 0.0, 0.7)
BLUE = Rgba(0.0, 0.0, 0.6, 0.7)
BASE_SYSTEM_DEFAULT_COLOR = RED
LEARNED_SYSTEM_DEFAULT_COLOR = BLUE
ILLUSTRATION_COLOR_GROUP = 'phong'
ILLUSTRATION_COLOR_PROPERTY = 'diffuse'
LEARNED_TAG = '__learned__'


def generate_visualization_system(
        base_system: DrakeSystem,
        base_system_color: Rgba = BASE_SYSTEM_DEFAULT_COLOR,
        learned_system_color: Rgba = LEARNED_SYSTEM_DEFAULT_COLOR
) -> DrakeSystem:
    """Generate a dummy ``DrakeSystem`` for visualizing comparisons between two
    trajectories of ``base_system``.

    Does so by generating a new ``DrakeSystem`` in which every model in the
    base system has a copy. Each illustration geometry element in these two
    copies is uniformly colored to be visually distinguishable.

    Args:
        base_system: System to be visualized.
        base_system_color: Color to repaint every thing in base system.
        learned_system_color: Color to repaint every thing in duplicated system.

    Returns:
        New ``DrakeSystem`` with doubled state and repainted elements.
    """
    # Creates duplicated system
    double_urdfs = deepcopy(base_system.urdfs)
    double_urdfs.update({
        (k + LEARNED_TAG): v for k, v in base_system.urdfs.items()
    })
    visualization_system = DrakeSystem(double_urdfs,
                                       base_system.dt,
                                       enable_visualizer=True)

    # Recolors every illustration geometry to default colors
    plant_diagram = visualization_system.plant_diagram
    plant = plant_diagram.plant
    scene_graph = plant_diagram.scene_graph
    inspector = scene_graph.model_inspector()
    for model_id in plant_diagram.model_ids:
        model_name = plant.GetModelInstanceName(model_id)
        for body_index in plant.GetBodyIndices(model_id):
            body_frame = plant.GetBodyFrameIdOrThrow(body_index)
            for geometry_id in inspector.GetGeometries(body_frame,
                                                       Role.kIllustration):
                props = inspector.GetIllustrationProperties(geometry_id)
                # phong.diffuse is the name of property controlling
                # illustration color.
                if props and props.HasProperty(ILLUSTRATION_COLOR_GROUP,
                                               ILLUSTRATION_COLOR_PROPERTY):
                    # Sets color in properties.
                    props.UpdateProperty(
                        ILLUSTRATION_COLOR_GROUP, ILLUSTRATION_COLOR_PROPERTY,
                        learned_system_color
                        if LEARNED_TAG in model_name else base_system_color)
                    # Tells ``scene_graph`` to update the color.
                    scene_graph.AssignRole(plant.get_source_id(), geometry_id,
                                           props, RoleAssign.kReplace)

    # Changing illustration properties requires the ``SceneGraph`` to have
    # its context reset in order. For these  changes to then be loaded by the
    # ``MeshcatVisualizer``, the ``Simulator`` must then be re-initialized to
    # trigger ``MeshcatVisualizer.load()``.
    scene_graph.SetDefaultContext(
        scene_graph.GetMyContextFromRoot(
            plant_diagram.sim.get_mutable_context()))
    plant_diagram.sim.Initialize()

    return visualization_system


def visualize_trajectory(drake_system: DrakeSystem,
                         x_trajectory: Tensor,
                         framerate: int = 30) -> Tuple[np.ndarray, int]:
    """Visualizes trajectory of system.

    Specifies a ``framerate`` for output video, though should be noted that
    this framerate is only approximately represented by homogeneous integer
    downsampling of the state trajectory. For example, ``if drake_system.dt ==
    1/60`` and ``framerate == 11``, the true video framerate will be::

        max(round(60/11), 1) == 5.

    Code mostly sourced from Greg Izatt (@gizatt) in this Gist::

        https://gist.github.com/gizatt/d96e6b0bcbe6c7ec331efad53a6f11bc

    Args:
        drake_system: System associated with provided trajectory.
        x_trajectory: (T, drake_system.space.n_x) state trajectory.
        framerate: desired frames per second of output video.

    Returns:
        (1, T, 3, H, W) ndarray video capture of trajectory with resolution
        H x W, approximately 640 x 480.
        The true framerate, rounded to an integer
    """
    assert drake_system.plant_diagram.visualizer is not None
    assert x_trajectory.dim() == 2

    # downsample trajectory to approximate framerate.
    temporal_downsample = max(round((1 / drake_system.dt) / framerate), 1)
    actual_framerate = round((1 / drake_system.dt) / temporal_downsample)
    x_trajectory = x_trajectory[::temporal_downsample, :]

    # Set camera to look at origin from specified point ``camera_position``
    camera_position = np.array([0., 20., 5.])
    meshcat_visualizer = drake_system.plant_diagram.visualizer.vis
    meshcat_visualizer["/Cameras/default/rotated"].set_object(
        meshcat.geometry.PerspectiveCamera(fov=CAM_FOV, zoom=0.4))
    meshcat_visualizer['/Cameras/default'].set_transform(
        RigidTransform(p=camera_position).GetAsMatrix4())

    # Not sure if this part is necessary
    meshcat_visualizer['/Cameras/default/rotated/<object>'].set_property(
        'position', [0, 0, 0])

    # Capture frames
    frames = []
    _, carry = drake_system.sample_initial_condition()
    for x_current in x_trajectory:
        drake_system.preprocess_initial_condition(x_current.unsqueeze(0), carry)
        frame = meshcat_visualizer.get_image()
        frames.append(frame)

    # composes video ndarray of shape (T, H, W, 4[rgba])
    video = np.stack([np.array(frame) for frame in frames])

    # remove alpha channel, downsample, and reorder axes to output type
    video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
    resolution_downsample = max(frames[0].size[0] // RESOLUTION[0], 1)
    video = video[:, :, :3, 0::resolution_downsample, 0::resolution_downsample]
    return video, actual_framerate
