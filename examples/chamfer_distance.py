from chamferdist import ChamferDistance
import numpy as np
import torch

from trimesh import Trimesh
from trimesh.convex import convex_hull
from trimesh.sample import sample_surface_even
from trimesh.primitives import Box as TrimeshBox 
from trimesh.primitives import Sphere as TrimeshSphere

from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere  # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore

from pydrake.all import Shape


def get_chamfer_distance(learned_geom: Shape, 
                         learned_trans: np.ndarray, 
                         true_geom: Shape, 
                         true_trans: np.ndarray,
                         n: int = 1000,
                         ) -> float:
    """Get Chamfer Distance between learned and true geometry"""
    
    learned_pc, _ = sample_surface_even(
            DraketoTrimeshFactory.convert(learned_geom), 
            count = n)
    true_pc, _ = sample_surface_even(
            DraketoTrimeshFactory.convert(true_geom), 
            count = n)

    learned_pc_in_origin = learned_pc @ learned_trans[:3, :3].T + learned_trans[:3, 3]
    true_pc_in_origin = true_pc @ true_trans[:3, :3].T + true_trans[:3, 3]

    cD = ChamferDistance()
    dist = cD(torch.from_numpy(learned_pc_in_origin.astype(np.float32)[np.newaxis, :]), 
            torch.from_numpy(true_pc_in_origin.astype(np.float32)[np.newaxis,:]), 
            bidirectional=True,
            point_reduction="mean",
            batch_reduction= None,)

    return dist.detach().cpu().numpy()

class DraketoTrimeshFactory:
    @staticmethod
    def convert(shape: Shape,
                ) -> Trimesh:
        """Convert Drake geometry to Trimesh"""
        if isinstance(shape, DrakeBox):
            return TrimeshBox(extents=[shape.width(), shape.depth(), shape.height()])
        if isinstance(shape, DrakeMesh):
            convex_surface_mesh = shape.GetConvexHull()
            vertices = np.stack([convex_surface_mesh.vertex(i) 
                                 for i in range(convex_surface_mesh.num_vertices())])
            return convex_hull(vertices)
        if isinstance(shape, DrakeSphere):
            return TrimeshSphere(radius=shape.radius())
        NotImplementedError(f"Can't convert {shape} to Trimesh") 