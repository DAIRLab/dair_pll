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


def get_chamfer_distance(learned_geom, 
                         learned_trans, 
                         true_geom, 
                         true_trans,
                         n: int = 500) -> float:
    """Get Chamfer Distance between learned and true geometry"""
    
    learned_pc,_ = sample_surface_even(
            DraketoTrimeshFactory.convert(learned_geom), count = n)
    true_pc,_ = sample_surface_even(
            DraketoTrimeshFactory.convert(true_geom), count = n)

    # point cloud representations in world frame
    learned_pc_in_origin = learned_pc @ learned_trans[:3, :3].T + learned_trans[:3, 3]
    true_pc_in_origin = true_pc @ true_trans[:3, :3].T + true_trans[:3, 3]

    # CRITICAL: using pytorch's implementation for cfd
    # $CD(A, B)=\sum \limits_{x_{i}\in A} ||x_{i}-N N(x_{i},B)||^{2}_{2} + \sum \limits_{x_{j}\in B} ||x_{j}-N N(x_{j}, A)||^{2}_{2}$
    cD = ChamferDistance()

    dist = cD(torch.from_numpy(learned_pc_in_origin.astype(np.float32)[np.newaxis, :]), 
            torch.from_numpy(true_pc_in_origin.astype(np.float32)[np.newaxis,:]), 
            bidirectional=True,
            point_reduction="mean",
            batch_reduction= None,)
    
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

# def get_true_geometry(path: str = "assets/contactnets_cube.urdf.xacro") -> Shape:
#     """Get True Geometry from configured base system"""

#     CUBE_DATA_ASSET = "contactnets_cube"
#     BOX_URDF_ASSET = "contactnets_cube.urdf"
#     CUBE_MODEL = "cube"

#     #urdfs = file_utils.get_urdf_asset_contents(BOX_URDF_ASSET, mappings={CUBE_MODEL: })
#     m = {
# 	"length_x": 0.065, # m
# 	"length_y": 0.065, # m
# 	"length_z": 0.065, # m
# 	"mu": 0.1,
# 	"planar_xz": "false",
#     }

#     urdf_contents = file_utils.get_urdf_asset_contents(
#     urdf_file_basename="contactnets_cube.urdf.xacro",
#     mappings=m,
#     )

#     urdfs = {CUBE_MODEL: urdf_contents}
#     base_config = DrakeSystem(urdfs=urdfs)

#     inspector = base_config.plant_diagram.scene_graph.model_inspector()
#     all_geom_ids = inspector.GetAllGeometryIds()

#     for geom_id in all_geom_ids:
#         true_geom = inspector.GetShape(geom_id)
#         if isinstance(true_geom, DrakeHalfSpace):
#             continue
#         return true_geom
#     assert False, "Could not find true geometry"

# class get_chamfer_distance(Simulation):


#     def get_true_geometry() -> Shape:
#         """Get True Geometry from configured base system"""
#         system = DrakeSystem()
#         inspector = system.plant_diagram.scene_graph.model_inspector()
#         all_geom_ids = inspector.GetAllGeometryIds()
#         for geom_id in all_geom_ids:
#             true_geom = inspector.GetShape(geom_id)
#             if isinstance(true_geom, DrakeHalfSpace):
#                 continue
#             return true_geom
#         assert False, "Could not find true geometry"
#         return None