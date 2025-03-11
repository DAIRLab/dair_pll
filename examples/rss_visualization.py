import torch
from chamferdist import ChamferDistance
import numpy as np

from abc import ABC, abstractmethod
from trimesh import Trimesh
from trimesh.convex import convex_hull
from trimesh.sample import sample_surface_even

from trimesh.primitives import Box as TrimeshBox 
from trimesh.primitives import Sphere as TrimeshSphere

from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere  # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore

#from pydrake.geometry import GetConvexHull  # type: ignore

from pydrake.all import Rgba, Shape
from dair_pll.drake_system import DrakeSystem
from scipy.spatial.transform import Rotation as R


from dair_pll.geometry import (
    GeometryCollider,
    PydrakeToCollisionGeometryFactory,
    CollisionGeometry,
    DeepSupportConvex,
    Polygon,
    Box,
    Plane,
    _NOMINAL_HALF_LENGTH,
)

def visualize_geometries(meshcat,
                         system, 
                         true_geom,
                         true_pose):
    
    """ Visualize the learned and true geometries """

    learned_geom = system.get_learned_geometry()
    learned_pose = system.get_learned_pose().cpu().numpy()

    def display_geom(name: str, 
                     geom: Shape, 
                     pose: list, 
                     color: Rgba) -> None:
        """Display Geometry in Meshcat"""
        assert len(pose) == 7, "Only Free Floating State Accepted"
        transform = np.eye(4)
        #first 4 elements are quaternion
        transform[:3, :3] = R.from_quat(pose[:4], scalar_first=True).as_matrix()
        #last 3 elements are translation
        transform[:3, 3] = pose[4:]

        meshcat.SetObject(name, geom, color)
        meshcat.SetTransform(name, transform)
  
        return transform

    blue = Rgba(0.1, 0.1, 0.9, 0.5)
    red = Rgba(0.9, 0.1, 0.1, 1.0)

    #testing only
    learned_geom = DrakeBox(width = 0.00001, depth = 1, height = 1)
    true_geom = DrakeBox(width = 0.00001, depth = 1, height = 1)
    learned_pose = [1, 0, 0, 0, 0, 0, 0]
    true_pose = [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2, 0.5, 0, 0]
    true_pose = [1, 0, 0, 0, 1, 0, 0]


    learned_trans = display_geom("/learned", learned_geom, learned_pose, blue)
    true_trans = display_geom("/true", true_geom, true_pose, red)

    chamfer_metric = get_chamfer_distance(learned_geom, learned_trans, 
                                          true_geom, true_trans)
    
    print(f"Chamfer Distance: {chamfer_metric}")

def get_chamfer_distance(learned_geom, learned_trans, 
                         true_geom, true_trans,
                         n: int = 1000) -> float:
    """Get Chamfer Distance between learned and true geometry"""
    
    learned_pc,_ = sample_surface_even(
            DraketoTrimeshFactory.convert(learned_geom), count = n)
    true_pc,_ = sample_surface_even(
            DraketoTrimeshFactory.convert(true_geom), count = n)

    #learned_pose = object's pc in object frame
    #learned_pc_in_origin = object's pc in world frame = 
    #   object's pc in object frame * object's frame in world's frame
    import pdb; pdb.set_trace()

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

def get_true_geometry() -> Shape:
    """Get True Geometry from configured base system"""
    system = DrakeSystem()
    inspector = system.plant_diagram.scene_graph.model_inspector()
    all_geom_ids = inspector.GetAllGeometryIds()

    #import pdb; pdb.set_trace()

    for geom_id in all_geom_ids:
        true_geom = inspector.GetShape(geom_id)
        if isinstance(true_geom, DrakeHalfSpace):
            continue
        return true_geom
    assert False, "Could not find true geometry"