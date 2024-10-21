"""Helper script to generate a mesh file from custom vertex locations."""

import pdb

from torch import Tensor
from scipy.spatial import ConvexHull

from dair_pll import file_utils
from dair_pll.deep_support_function import (
    extract_obj_from_mesh_summary,
    get_mesh_summary_from_polygon,
)
from dair_pll.geometry import Polygon


MESH_PATH = "/home/bibit/dair_pll/assets/contactnets_asymmetric.obj"
VERTICES = (
    Tensor([[0.0, -1, -2], [3, 0, 0], [0, 2, -1], [-1, 1, -1], [1, 1, 1], [2, -1, 1]])
    * 0.025
)

polygon = Polygon(VERTICES)

mesh_summary = get_mesh_summary_from_polygon(polygon)
obj = extract_obj_from_mesh_summary(mesh_summary)
file_utils.save_string(MESH_PATH, obj)

pdb.set_trace()
