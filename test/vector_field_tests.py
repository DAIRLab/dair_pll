"""Test script for visualizing force vector fields."""

import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from typing import Tuple


ROTATION_PRIMITIVE = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
INWARD_PRIMITIVE = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 0]])

W_ROT = 1.
W_IN = 1.


def vortex_by_location(xyz_loc: Tensor) -> Tensor:
	xy_mag = torch.linalg.norm(xyz_loc[:2])
	rotation_mat = W_ROT * ROTATION_PRIMITIVE / xy_mag
	inward_mat = W_IN * INWARD_PRIMITIVE

	return (rotation_mat + inward_mat) @ xyz_loc



def vortex_by_coords(x: float, y: float, z: float) -> Tuple[float, float, float]:
	xy_mag = np.sqrt(x**2 + y**2) + 1e-4
	rotation_mat = W_ROT * ROTATION_PRIMITIVE / xy_mag
	inward_mat = W_IN * INWARD_PRIMITIVE / xy_mag

	force = (rotation_mat + inward_mat) @ np.array([x, y, z])

	return force[0], force[1], force[2]


vf_x = lambda x, y, z: vortex_by_coords(x, y, z)[0]
vf_y = lambda x, y, z: vortex_by_coords(x, y, z)[1]
vf_z = lambda x, y, z: vortex_by_coords(x, y, z)[2]


# # 1D arrays
# x_locs = np.arange(-0.5, 0.5, 0.01)
# y_locs = np.arange(-0.5, 0.5, 0.01)

# # Meshgrid
# X, Y = np.meshgrid(x_locs, y_locs, indexing='ij')
x = np.arange(-0.5, 0.5, 0.1)
y = np.arange(-0.5, 0.5, 0.1)
  
# Meshgrid
X,Y = np.meshgrid(x,y)

# Store U and V
U, V = np.zeros_like(X), np.zeros_like(X)

for i in range(X.shape[0]):
	for j in range(X.shape[0]):
		U[i,j] = vf_x(X[i,j], Y[i,j], 0)
		V[i,j] = vf_y(X[i,j], Y[i,j], 0)

# Depict illustration
plt.figure(figsize=(10, 10))
plt.quiver(X,Y,U,V, units='xy')
plt.streamplot(X,Y,U,V, density=1.4, linewidth=None, color='#A23BEC')
plt.title('Electromagnetic Field')

plt.grid()
plt.savefig('/home/bibit/Desktop/test.png')

pdb.set_trace()
