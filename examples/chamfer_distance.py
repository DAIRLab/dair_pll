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

from dair_pll.drake_system import DrakeSystem

from pydrake.all import Shape


class ChamferDistanceMetric:
    def __init__(self):
        self.chamfer_dist = ChamferDistance()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self,
                 learned_geom,
                 learned_trans,
                 true_geom,
                 true_trans,
                 n_sample_points: int = 500,
                 ) -> float:
                 
        """Get Chamfer Distance between learned and true objects"""
        with torch.no_grad():
            learned_pc,_ = sample_surface_even(
                DraketoTrimeshFactory.convert(learned_geom), count = n_sample_points)
            true_pc,_ = sample_surface_even(
                DraketoTrimeshFactory.convert(true_geom), count = n_sample_points)    
            
            # point cloud representations in world frame
            learned_pc_in_origin = learned_pc @ learned_trans[:3, :3].T + learned_trans[:3, 3]
            true_pc_in_origin = true_pc @ true_trans[:3, :3].T + true_trans[:3, 3]

            # CRITICAL: using pytorch's implementation for cfd
            # $CD(A, B)=\sum \limits_{x_{i}\in A} ||x_{i}-N N(x_{i},B)||^{2}_{2} + \sum \limits_{x_{j}\in B} ||x_{j}-N N(x_{j}, A)||^{2}_{2}$

            learned_tensor = torch.from_numpy(learned_pc_in_origin.astype(np.float32)[np.newaxis, :])
            true_tensor = torch.from_numpy(true_pc_in_origin.astype(np.float32)[np.newaxis, :])
                            
            dist = self.chamfer_dist(learned_tensor, 
                                    true_tensor, 
                                    bidirectional=True,
                                    point_reduction="mean",
                                    batch_reduction= None,)
            
            # Clean up GPU memory if needed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
            assert dist > 0

            return dist.detach().cpu().squeeze()


class DraketoTrimeshFactory:
    @staticmethod
    def convert(shape: Shape) -> Trimesh:
        if isinstance(shape, DrakeBox):
            return TrimeshBox(extents=[shape.width(), shape.depth(), shape.height()])
        if isinstance(shape, DrakeMesh):
            convex_surface_mesh = shape.GetConvexHull()
            vertices = np.stack([convex_surface_mesh.vertex(i) 
                                for i in range(convex_surface_mesh.num_vertices())])
            return convex_hull(vertices)
        if isinstance(shape, DrakeSphere):
            return TrimeshSphere(radius=shape.radius())
        else:
            NotImplementedError(f"Can't convert {shape} to Trimesh") 