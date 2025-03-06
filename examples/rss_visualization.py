import torch
from chamferdist import ChamferDistance
import numpy as np
from trimesh.sample import sample_surface_even
from trimesh.creation import box

from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere  # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore

from pydrake.all import Rgba, Shape
from dair_pll.drake_system import DrakeSystem
from scipy.spatial.transform import Rotation as R

def visualize_geometries(meshcat, system, true_geometry, true_pose):
    """ Visualize the learned and true geometries """

    geom = system.get_learned_geometry()
    pose = system.get_learned_pose().cpu().numpy()
    assert len(pose) == 7, "Only Free Floating State Accepted"
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(pose[:4], scalar_first=True).as_matrix()
    transform[:3, 3] = pose[4:]
    #learned = blue
    meshcat.SetObject("/learned", geom, Rgba(0.1, 0.1, 0.9, 0.5))
    meshcat.SetTransform("/learned", transform)

    assert len(true_pose) == 7, "Only Free Floating State Accepted"
    true_transform = np.eye(4)
    true_transform[:3, :3] = R.from_quat(true_pose[:4], scalar_first=True).as_matrix()
    true_transform[:3, 3] = true_pose[4:]
    meshcat.SetObject("/true", true_geometry, Rgba(0.9, 0.1, 0.1, 1.0))
    meshcat.SetTransform("/true", true_transform)

    chamfer_metric = get_chamfer_distance(geom, true_geometry, pose, true_pose)
    print(f"Chamfer Distance: {chamfer_metric}")

def get_true_geometry() -> Shape:
    """Get True Geometry from configured base system"""
    system = DrakeSystem()
    inspector = system.plant_diagram.scene_graph.model_inspector()
    all_geom_ids = inspector.GetAllGeometryIds()
    for geom_id in all_geom_ids:
        true_geom = inspector.GetShape(geom_id)
        if isinstance(true_geom, DrakeHalfSpace):
            continue
        return true_geom
    assert False, "Could not find true geometry"
    return None

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