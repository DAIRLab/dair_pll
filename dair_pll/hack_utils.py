"""Cross-example utility functions

TODO: HACK Put in a better place
"""

from typing import List

import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex

def finger_idx_from_body_name(plant: MultibodyPlant, robot_id: ModelInstanceIndex, body_names: List[str]) -> List[int]:
  state_names = plant.GetStateNames(robot_id)
  ret = [-1] * len(body_names)
  for idx in range(len(state_names)//2): # ignore velocity
    for body_idx, body_name in enumerate(body_names):
      if body_name in state_names[idx]:
        ret[body_idx] = idx // 3
        break
  assert np.all(np.array(ret) >= 0) and np.all(np.array(ret) < len(body_names))
  return ret

