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
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import time
import sys

from pydrake.geometry import StartMeshcat, Meshcat, Rgba, Convex, Sphere

from dair_pll.gui_utils import transform_from_state_q
from dair_pll.tensor_utils import stable_inv

from selenium import webdriver
import base64

def main():
    if len(sys.argv) != 3:
        print("Usage: plot_data_*.py <data_file.pkl> <sample_file.pkl>")
        sys.exit(-1)
    # Read Data
    file_name = sys.argv[1]
    print(f"Opening data file: {file_name}")
    with open(file_name, "rb") as datafile:
        data_dict = pickle.load(datafile)
    file_name = sys.argv[2]
    print(f"Opening Sample file: {file_name}")
    with open(file_name, "rb") as datafile:
        sample_dict = pickle.load(datafile)

    learn_param = data_dict["learned"]['_multibody_terms.contact_terms.geometries.3._vertices_param'].detach().cpu().numpy().T
    key_base = '_learned_trajectory._trajectories_q0'
    key_idx = 0
    while (key_base + f".{key_idx}") in data_dict["learned"].keys():
        key_idx += 1
    key_idx -= 1
    learn_q = data_dict["learned"][key_base + f".{key_idx}"].detach().cpu().numpy()
    ground_q = data_dict["data"].trajectories[-1]["cube_groundtruth"][-1].detach().cpu().numpy()
    learned_vertices = Rotation.from_quat(learn_q[:4], scalar_first=True).apply(learn_param.T) + learn_q[-3:] - ground_q[-3:]

    # Setup Meshcat
    meshcat = StartMeshcat()
    meshcat.SetCameraPose(np.array([0.1, -0.1, 0.1]), np.array([0., 0., 0.]))
    input("Start Meshcat... [Enter]")
    radius = 0.01
    robot_geom = Sphere(radius)
    meshcat.SetObject("/robot/0", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))
    meshcat.SetObject("/robot/1", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))

    meshcat.SetObject("/learned", Convex(learn_param, "Learned"), Rgba(0.0, 0.0, 0.8, 0.7))
    meshcat.SetTransform(
                    "/learned",
                    transform_from_state_q(learn_q),
                )

    # Sort samples by EIG
    fishers = torch.tensor(sample_dict["expected_info"])
    obs_info = torch.tensor(data_dict["obs_info"])
    eigs_proxy = torch.log(torch.det(fishers + obs_info))
    sortedpairs = sorted([(value, index) for index, value in enumerate(eigs_proxy)], reverse=True)
    indices = np.array([index for (value, index) in sortedpairs])
    # TODO
    eig_idx = int(input("Choose index (0 = best, -1 = worst): "))
    sample_idx = indices[eig_idx]
    print(f"Selecting action index: {sample_idx}")

    # Setup matplotlib plot
    fisher_info_geometry = fishers[sample_idx, 7:, 7:] # Remove position-based
    # Get logdet for each vertex
    vertex_infos = []
    for idx in range(len(learned_vertices)):
        fisher_minor = fisher_info_geometry[idx*3:(idx+1)*3, idx*3:(idx+1)*3]
        vertex_infos.append(np.log(np.linalg.det(fisher_minor)))
    # Get colormap for info
    cm = plt.get_cmap("coolwarm")
    # Manual Scale, can also use max/min of vertex_infos
    cNorm = matplotlib.colors.Normalize(vmin=min(vertex_infos), vmax=max(vertex_infos))
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
    ax.set_axis_off()
    ax.set_xlim3d(-0.03, 0.03)
    ax.set_ylim3d(-0.03, 0.03)
    ax.set_zlim3d(-0.03, 0.03)
    ax.view_init(elev=30, azim=70, roll=0)
    plt.show()

    # Set up Chrome options to run in headless mode
    print("Set up Chrome Webdriver")
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox') 
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Initialize the Chrome WebDriver 
    driver = webdriver.Chrome(chrome_options)
    # Open the Meshcat viewer's URL
    print("Get Meshcat URL")
    driver.get(meshcat.web_url())

    # Loop through sample trajectory
    print("Set up Trajectories...")
    zero_rot = np.array([1.0, 0.0, 0.0, 0.0])
    robot_trajs = sample_dict["trajectories"][sample_idx]["robot_state"][:, :6].cpu().numpy()
    robot_0_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, 3:6]]
                )
    robot_1_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, :3]]
                )
    obj_traj = sample_dict["trajectories"][sample_idx]["cube_state"].cpu().numpy()[:, :7] # Remove velocity

    # Action Execution
    print("Action...")
    for idx in range(len(robot_trajs)):
        print(f"Timestep: {idx}/{len(robot_trajs)-1}")

        # Draw in Meshcat
        meshcat.SetTransform(
                    "/robot/0",
                    transform_from_state_q(robot_0_traj[idx, :]),
                )
        meshcat.SetTransform(
                    "/robot/1",
                    transform_from_state_q(robot_1_traj[idx, :]),
                )
        meshcat.SetTransform(
                    "/learned",
                    transform_from_state_q(obj_traj[idx, :]),
                )

        # Object+Robot Meshcat Figure
        meshcat.SetProperty("/robot", "visible", True)
        meshcat.SetProperty("/learned", "visible", True)
        # Save Meshcat
        image_data_url = driver.execute_script("""
            return viewer.capture_image(1920, 1080);  // Capture the image with width and height
        """)
        # Extract the base64 part of the data URL (after "data:image/png;base64,")
        image_base64 = image_data_url.split(",")[1]

        # Decode the base64 string and save it as an image file
        image_data = base64.b64decode(image_base64)
        # Write the image to a file
        with open(f"./output/meshcat_sample_{idx:04}.png", "wb") as f:
            f.write(image_data)
    
    input("Press [enter] to continue...")

if __name__ == '__main__':
    main()