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
    # Load data
    if len(sys.argv) < 2:
        print("Usage: plot_data_*.py <data_file.pkl>")
        sys.exit(-1)
    file_name = sys.argv[-1]
    print(f"Opening file: {file_name}")
    with open(file_name, "rb") as datafile:
        data_dict = pickle.load(datafile)

    # Plot robot data
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    fig.set_size_inches(6., 3., forward=True)
    fig.set_dpi(300)

    # Draw 2 robots
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    radius = 0.01
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    plt.ion()
    plt.show()

    def reset_axes():
        ax1.clear()
        ax2.clear()
        ax1.plot_surface(x, y, z, color="green", alpha=0.2)
        ax2.plot_surface(x, y, z, color="green", alpha=0.2)
        ax1.set_box_aspect((1, 1, 1))
        ax2.set_box_aspect((1, 1, 1))
        ax1.set_axis_off()
        ax2.set_axis_off()
        plt.draw()
        plt.pause(0.001)
    reset_axes()


    # Setup Meshcat
    meshcat = StartMeshcat()
    meshcat.SetCameraPose(np.array([0.1, -0.1, 0.1]), np.array([0., 0., 0.]))
    input("Start Meshcat... [Enter]")
    true_geom = Box(0.04, 0.04, 0.04)
    robot_geom = Sphere(radius)
    meshcat.SetObject("/true", true_geom, Rgba(0.8, 0.0, 0.0, 0.3))
    meshcat.SetObject("/robot/0", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))
    meshcat.SetObject("/robot/1", robot_geom, Rgba(0.0, 0.8, 0.0, 0.8))

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
    robot_trajs = torch.cat(data_dict["data"].trajectories)["robot_state"][:, :6].cpu().numpy()
    true_traj = torch.cat(data_dict["data"].trajectories)["cube_groundtruth"].cpu().numpy()
    robot_0_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, :3]]
                )
    robot_1_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_trajs.shape[0], 4)), robot_trajs[:, 3:6]]
                )
    finger_0_normal = torch.cat(data_dict["data"].trajectories)["contact_normals"]["finger_0"].cpu().numpy()
    finger_1_normal = torch.cat(data_dict["data"].trajectories)["contact_normals"]["finger_1"].cpu().numpy()

    for idx in range(len(robot_trajs)):
        print(f"Timestep: {idx}/{len(robot_trajs)}")
        # Reset Matplotlib
        reset_axes()

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


        # Draw in MatplotLib
        if np.linalg.norm(finger_0_normal[idx, :]) > 0.5:
            normal0_start = -radius * finger_0_normal[idx, :]
            normal0_end = -(radius * 1.01) * finger_0_normal[idx, :]
            ax1.plot(normal0_start[0], normal0_start[1], normal0_start[2], 'o', markersize=10, color='red', alpha=0.8)
            ax1.quiver(normal0_start[0], normal0_start[1], normal0_start[2], normal0_end[0], normal0_end[1], normal0_end[2], arrow_length_ratio=0.2, color='red')

        if np.linalg.norm(finger_1_normal[idx, :]) > 0.5:
            normal1_start = -radius * finger_1_normal[idx, :]
            normal1_end = -(radius * 1.01) * finger_1_normal[idx, :]
            ax2.plot(normal1_start[0], normal1_start[1], normal1_start[2], 'o', markersize=10, color='red', alpha=0.8)
            ax2.quiver(normal1_start[0], normal1_start[1], normal1_start[2], normal1_end[0], normal1_end[1], normal1_end[2], arrow_length_ratio=0.2, color='red')



        plt.draw()
        plt.pause(0.001)
        time.sleep(0.033)
        fig.savefig(f"./output/contact_{idx:03}.png", dpi=300)

        # Robot-only Meshcat Figure
        meshcat.SetProperty("/true", "visible", False)
        meshcat.SetProperty("/robot", "visible", True)
        # Save Meshcat
        image_data_url = driver.execute_script("""
            return viewer.capture_image(1920, 1080);  // Capture the image with width and height
        """)
        # Extract the base64 part of the data URL (after "data:image/png;base64,")
        image_base64 = image_data_url.split(",")[1]

        # Decode the base64 string and save it as an image file
        image_data = base64.b64decode(image_base64)
        # Write the image to a file
        with open(f"./output/meshcat_robot_{idx:03}.png", "wb") as f:
            f.write(image_data)

        # Object+Robot Meshcat Figure
        meshcat.SetProperty("/true", "visible", True)
        meshcat.SetProperty("/robot", "visible", True)
        # Save Meshcat
        image_data_url = driver.execute_script("""
            return viewer.capture_image(1920, 1080);  // Capture the image with width and height
        """)
        # Extract the base64 part of the data URL (after "data:image/png;base64,")
        image_base64 = image_data_url.split(",")[1]

        # Decode the base64 string and save it as an image file
        image_data = base64.b64decode(image_base64)
        # Write the image to a file
        with open(f"./output/meshcat_{idx:03}.png", "wb") as f:
            f.write(image_data)
    
    input("Press [enter] to continue...")

if __name__ == '__main__':
    main()