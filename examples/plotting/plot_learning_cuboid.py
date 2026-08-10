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

from pydrake.geometry import StartMeshcat, Meshcat, Rgba, Box, Sphere

from dair_pll.gui_utils import transform_from_state_q

from selenium import webdriver
import base64

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_data_*.py <data_file.pkl> [data_file_init_learn.pkl]")
        sys.exit(-1)
    # Load Init Param
    nominal_scale = 2e-1
    if len(sys.argv) == 3:
        init_file_name = sys.argv[-1]
        print(f"Opening init learning file: {init_file_name}")
        with open(init_file_name, "rb") as datafile:
            init_data_dict = pickle.load(datafile)
        init_param = 2.0 * nominal_scale * init_data_dict["learned"]['_multibody_terms.contact_terms.geometries.3.length_params'].detach().cpu().numpy()[0]
        key_base = '_learned_trajectory._trajectories_q0'
        key_idx = 0
        while (key_base + f".{key_idx}") in init_data_dict["learned"].keys():
            key_idx += 1
        key_idx -= 1
        init_q = init_data_dict["learned"][key_base + f".{key_idx}"].detach().cpu().numpy()
    else:
        init_param = 0.001 * np.ones(3)
        init_q = np.array([1., 0., 0., 0., 0., 0., 0.])

    # Read Data
    file_name = sys.argv[1]
    print(f"Opening data file: {file_name}")
    with open(file_name, "rb") as datafile:
        data_dict = pickle.load(datafile)

    final_param = 2.0 * nominal_scale * data_dict["learned"]['_multibody_terms.contact_terms.geometries.3.length_params'].detach().cpu().numpy()[0]
    key_base = '_learned_trajectory._trajectories_q0'
    key_idx = 0
    while (key_base + f".{key_idx}") in data_dict["learned"].keys():
        key_idx += 1
    key_idx -= 1
    final_q = data_dict["learned"][key_base + f".{key_idx}"].detach().cpu().numpy()

    # Setup Meshcat
    meshcat = StartMeshcat()
    meshcat.SetCameraPose(np.array([0.1, -0.1, 0.1]), np.array([0., 0., 0.]))
    input("Start Meshcat... [Enter]")
    # sim Cuboid
    #true_geom = Box(0.04, 0.04, 0.04)
    # dynamixel box
    true_geom = Box(0.058, 0.059, 0.053)
    radius = 0.01
    robot_geom = Sphere(radius)
    meshcat.SetObject("/true", true_geom, Rgba(0.8, 0.0, 0.0, 0.3))
    meshcat.SetObject("/robot/0", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))
    meshcat.SetObject("/robot/1", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))

    meshcat.SetObject("/learned", Box(*init_param.tolist()), Rgba(0.0, 0.0, 0.8, 0.7))
    meshcat.SetTransform(
                    "/learned",
                    transform_from_state_q(init_q),
                )

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

    # Loop through all data trajectories
    print("Set up Trajectories...")
    zero_rot = np.array([1.0, 0.0, 0.0, 0.0])
    robot_trajs = data_dict["data"].trajectories[-1]["robot_state"][:, :6].cpu().numpy()
    true_traj = data_dict["data"].trajectories[-1]["cube_groundtruth"].cpu().numpy()
    robot_0_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, 3:6]]
                )
    robot_1_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, :3]]
                )
    finger_0_normal = data_dict["data"].trajectories[-1]["contact_normals"]["finger_0"].cpu().numpy()
    finger_1_normal = data_dict["data"].trajectories[-1]["contact_normals"]["finger_1"].cpu().numpy()

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
                    "/true",
                    transform_from_state_q(true_traj[idx, :]),
                )

        # Draw normal
        if np.linalg.norm(finger_0_normal[idx, :]) > 0.5:
            normal0_start = -radius * finger_0_normal[idx, :]
            normal0_end = -(radius * 1.5) * finger_0_normal[idx, :]
            meshcat.SetLine("/robot/0/normal", np.stack([normal0_start, normal0_end]).T, 3.0, Rgba(1., 0., 0., 1.))

        if np.linalg.norm(finger_1_normal[idx, :]) > 0.5:
            normal1_start = -radius * finger_1_normal[idx, :]
            normal1_end = -(radius * 1.5) * finger_1_normal[idx, :]
            meshcat.SetLine("/robot/1/normal", np.stack([normal1_start, normal1_end]).T, 3.0, Rgba(1., 0., 0., 1.))

        # Object+Robot Meshcat Figure
        meshcat.SetProperty("/true", "visible", True)
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
        with open(f"./output/meshcat_execution_{idx:04}.png", "wb") as f:
            f.write(image_data)

    # Learning
    print("Learning...")
    for idx in range(120):
        interp_param = float(idx) / (120.-1.)
        current_q = final_q * interp_param + init_q * (1.0-interp_param)
        current_param = final_param * interp_param + init_param * (1.0-interp_param)
        print(f"Learning Step: {idx}/119")

        meshcat.SetObject("/learned", Box(*current_param.tolist()), Rgba(0.0, 0.0, 0.8, 0.7))
        meshcat.SetTransform(
                        "/learned",
                        transform_from_state_q(current_q),
                    )


        # Save Meshcat
        image_data_url = driver.execute_script("""
            return viewer.capture_image(1920, 1080);  // Capture the image with width and height
        """)
        # Extract the base64 part of the data URL (after "data:image/png;base64,")
        image_base64 = image_data_url.split(",")[1]

        # Decode the base64 string and save it as an image file
        image_data = base64.b64decode(image_base64)
        # Write the image to a file
        with open(f"./output/meshcat_execution_{idx + len(robot_trajs):04}.png", "wb") as f:
            f.write(image_data)
    
    input("Press [enter] to continue...")

if __name__ == '__main__':
    main()