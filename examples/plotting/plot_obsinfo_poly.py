#!/usr/bin/env python3
"""
Quick script to plot the observed info painted onto the object
"""
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def main():
    # Load data
    file_name = input("Enter data pickle file: ")
    with open(file_name, "rb") as datafile:
        data_dict = pickle.load(datafile)
    learned_vertices_raw = data_dict["learned"]['_multibody_terms.contact_terms.geometries.3._vertices_param'].detach().cpu().numpy()
    obs_info_geometry = data_dict["obs_info"][7:, 7:] # Remove position-based
    # Get logdet for each vertex
    vertex_infos = []
    for idx in range(len(learned_vertices_raw)):
        obs_minor = obs_info_geometry[idx*3:(idx+1)*3, idx*3:(idx+1)*3]
        vertex_infos.append(np.log(np.linalg.det(obs_minor)))

    # Transform learned vertices into world frame
    key_base = '_learned_trajectory._trajectories_q0'
    key_idx = 0
    while (key_base + f".{key_idx}") in data_dict["learned"].keys():
        key_idx += 1
    key_idx -= 1
    learned_q = data_dict["learned"][key_base + f".{key_idx}"].detach().cpu().numpy()
    # Map to final location
    learned_vertices = Rotation.from_quat(learned_q[:4], scalar_first=True).apply(learned_vertices_raw) + learned_q[-3:]
    # Re-center instead
    #learned_vertices = Rotation.from_quat(learned_q[:4], scalar_first=True).apply(learned_vertices_raw) - learned_vertices_raw.mean(axis=0)

    # Get colormap for info
    cm = plt.get_cmap("coolwarm")
    # Manual Scale, can also use max/min of vertex_infos
    cNorm = matplotlib.colors.Normalize(vmin=-20, vmax=55)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # Plot Learned Surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    hull = ConvexHull(learned_vertices)
    for simp in hull.simplices:
        tri = Poly3DCollection([learned_vertices[simp]])
        tri.set_color(scalarMap.to_rgba(np.mean(np.array(vertex_infos)[simp]))[:3])
        tri.set_alpha(0.5)
        ax.add_collection3d(tri)

    # Plot Learned Vertices
    ax.scatter(learned_vertices[:, 0], learned_vertices[:, 1], learned_vertices[:, 2], marker='o', s=30, color=scalarMap.to_rgba(vertex_infos))
    scalarMap.set_array(vertex_infos)
    fig.colorbar(scalarMap, ax=ax)

    # Plot Ground Truth Object
    ground_vertex_raw = 0.02 * np.array([[1., -1., -1.],
                                    [1., -1., 1.],
                                    [1., 1., -1.],
                                    [1., 1., 1.],
                                    [-1., -1., -1.],
                                    [-1., -1., 1.],
                                    [-1., 1., -1.],
                                    [-1., 1., 1.],
                                    ])
    ground_q = data_dict["data"].trajectories[-1]["cube_groundtruth"][-1].detach().cpu().numpy()
    # Map to ground_q
    ground_vertices = Rotation.from_quat(ground_q[:4], scalar_first=True).apply(ground_vertex_raw) + ground_q[-3:]
    # Recenter instead
    #ground_vertices = ground_vertex_raw - ground_vertex_raw.mean(axis=0)
    ground_hull = ConvexHull(ground_vertices)
    for simp in ground_hull.simplices:
        tri = Poly3DCollection([ground_vertices[simp]])
        tri.set_color('gray')
        tri.set_alpha(0.1)
        ax.add_collection3d(tri)

    ax.set_axis_off()
    plt.show()

if __name__ == '__main__':
    main()