"""Cross-example utility functions

TODO: HACK Put in a better place
"""

import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from tensordict import TensorDictBase
import torch
from torch import Tensor

from dair_pll.trifinger_utils import TrifingerLCMService
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem


def finger_idx_from_body_name(
    plant: MultibodyPlant, robot_id: ModelInstanceIndex, body_names: list[str]
) -> list[int]:
    """Get plant state index from the fingertip body names, return -1 if not in list"""
    state_names = plant.GetStateNames(robot_id)
    ret = [-1] * len(body_names)
    for idx in range(len(state_names) // 2):  # ignore velocity
        for body_idx, body_name in enumerate(body_names):
            if body_name in state_names[idx]:
                ret[body_idx] = idx // 3
                break
    assert np.all(np.array(ret) < len(body_names))
    return ret


def extract_robot_trajectory(
    data: TensorDictBase,
    system: MultibodyLearnableTactileSystem,
    trifinger: TrifingerLCMService,
) -> Tensor:
    """
    Params:
            data: batched TensorDict from trifinger_utils.execute_trajectory()
            system: used to get robot_model_name
            trifinger: used to get fingertip_body_names

    Returns:
            (batch, robot_space_nx)
    """

    assert len(data.size()) >= 1
    batch_dims = data.size()
    plant = system.plant
    robot_space = system.controlled_space
    fingertip_body_names = trifinger.fingertip_body_names
    assert len(system.controlled_model_names) == 1
    robot_model_name = system.controlled_model_names[0]
    ret = torch.zeros(batch_dims + (robot_space.n_x,))

    for finger_name, state_idx in zip(
        fingertip_body_names,
        finger_idx_from_body_name(
            plant, plant.GetModelInstanceByName(robot_model_name), fingertip_body_names
        ),
    ):
        if state_idx < 0:
            continue
        pos_idx = 3 * state_idx
        vel_idx = robot_space.n_x // 2 + pos_idx
        ret[..., pos_idx : pos_idx + 3] = data[finger_name]["position"]
        ret[..., vel_idx : vel_idx + 3] = data[finger_name]["velocity"]

    return ret
