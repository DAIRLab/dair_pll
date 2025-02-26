import torch
from chamferdist import ChamferDistance
import numpy as np
from trimesh.sample import sample_surface_even
from trimesh.creation import box

from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere  # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore

def get_chamfer_distance(learned_geom, true_geom, learned_pose, true_pose):
    """Get Chamfer Distance between learned and true geometry"""

    if isinstance(learned_geom, DrakeBox):
        learned_geom = box(extents=[learned_geom.width(), 
                                    learned_geom.depth(), 
                                    learned_geom.height()])
        true_geom = box(extents=[true_geom.width(), 
                                true_geom.depth(), 
                                true_geom.height()])
        
        learned_pc,_ = sample_surface_even(learned_geom, count = 100)
        true_pc,_ = sample_surface_even(true_geom, count = 100)

        cD = ChamferDistance()
        dist = cD(torch.from_numpy(learned_pc.astype(np.float32)[np.newaxis, :]), 
                torch.from_numpy(true_pc.astype(np.float32)[np.newaxis,:]), 
                bidirectional=True)
        
    print("Learned Geometry: ", learned_geom)

    return dist.detach().cpu().numpy()