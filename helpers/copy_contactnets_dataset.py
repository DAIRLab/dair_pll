"""
Helper script to access the ContactNets elbow dataset and copy into PLL in the expected format.

The ContactNets format is in:
	[ position  quaternion  artic_angle  linear_velocity  angular_velocity  artic_velocity ]
	where:
		- position:  		[x, y, z] in BLOCK_HALF_WIDTHS
		- quaternion: 		[qw, qx, qy, qz]
		- artic_angle: 		[theta] in rad
		- linear_velocity: 	[vx, vy, vz] in BLOCK_HALF_WIDTHS/second
		- angular_velocity:	[wx, wy, wz] in rad/second in body frame
		- artic_velocity:	[dtheta] in rad/second

The PLL format is in:
	[ quaternion  position  artic_angle  angular_velocity  linear_velocity  artic_velocity ]
	where:
		- position:  		[x, y, z] in meters
		- quaternion: 		[qw, qx, qy, qz]
		- artic_angle: 		[theta] in rad
		- linear_velocity: 	[vx, vy, vz] in meters/second
		- angular_velocity:	[wx, wy, wz] in rad/second in body frame
		- artic_velocity:	[dtheta] in rad/second
"""

import pdb
import os
import torch
from torch import Tensor


LAB_COMPUTER = False

COPY_DATASET_TO_PLL = False
ELIMINATE_EMPTY_TOSSES = True

if LAB_COMPUTER:
    REPO_DIR = "/home/bibit/SoPhTER/"
else:
    REPO_DIR = "/Users/bibit/Documents/College/Penn/Quals/repo/SoPhTER/"

PLL_DIR = "/Users/bibit/Documents/College/Penn/DAIRLab/pll_env/dair_pll/"

BLOCK_HALF_WIDTH = 0.050
METERS_PER_BHW = BLOCK_HALF_WIDTH


###############################################################################################
##########                         COPY DATASET TO PLL FORMAT                        ##########
###############################################################################################
def copy_dataset_to_pll_format():
    assert LAB_COMPUTER == False

    SOURCE_DIR = REPO_DIR + "data/rect_elbow3d/franka/"
    TARGET_DIR = PLL_DIR + "assets/contactnets_elbow/"

    # copy every real toss to TARGET_DIR after rearranging state vector to expected format.
    for i in range(601):
        # check if the toss exists in the source directory
        try:
            toss = torch.load(SOURCE_DIR + str(i) + ".pt")
        except:
            print(f"Skipping toss {i}.")
            continue

        print(f"Found toss {i}...", end="")

        # split the toss into its individual portions
        tsteps = toss.shape[0]

        pos = toss[:, :3]
        quat = toss[:, 3:7]
        artic = toss[:, 7].reshape(tsteps, 1)
        vels = toss[:, 8:11]
        ang_vels = toss[:, 11:14]
        artic_vel = toss[:, 14].reshape(tsteps, 1)

        # convert to correct units
        pos_pll = pos * METERS_PER_BHW
        vels_pll = vels * METERS_PER_BHW

        # combine in correct order for PLL convention
        pll_toss = torch.cat(
            (quat, pos_pll, artic, ang_vels, vels_pll, artic_vel), dim=1
        )

        # pdb.set_trace()
        torch.save(pll_toss, TARGET_DIR + str(i) + ".pt")
        print(f" converted and copied!")


def check_pll_cube_format():
    print(f"Checking PLL cube format:")
    assert LAB_COMPUTER == False

    CUBE_PLL_DIR = PLL_DIR + "assets/contactnets_cube/"

    file_name = CUBE_PLL_DIR + str(0) + ".pt"
    pll_data = torch.load(file_name)

    pdb.set_trace()


def check_cn_elbow_format():
    print(f"Checking ContactNets elbow format:")
    ELBOW_CN_DIR = REPO_DIR + "data/rect_elbow3d/franka/"

    file_name = ELBOW_CN_DIR + str(0) + ".pt"
    cn_data = torch.load(file_name)

    pdb.set_trace()


###############################################################################################
##########                           ELIMINATE EMPTY TOSSES                          ##########
###############################################################################################
def eliminate_empty_tosses():
    # goal is to search through the toss directory and eliminate any gaps between toss numbers,
    # e.g. to convert '0.pt, 2.pt, 3.pt' to '0.pt, 1.pt, 2.pt'.
    TARGET_DIR = PLL_DIR + "assets/contactnets_elbow/"

    move_to_i = 0
    for orig_i in range(601):
        # check if the toss exists already
        try:
            toss = torch.load(TARGET_DIR + str(orig_i) + ".pt")
        except:
            print(f"Skipping toss {orig_i}.")
            continue

        # here, we know that toss orig_i exists.
        print(f"Found toss {orig_i} --> moving it to toss {move_to_i}.")

        # delete the original file
        os.remove(TARGET_DIR + str(orig_i) + ".pt")

        # copy the trajectory data over to toss move_to_i, and increment move_to_i
        torch.save(toss, TARGET_DIR + str(move_to_i) + ".pt")
        move_to_i += 1


###############################################################################################
##########                                   TESTS                                   ##########
###############################################################################################

if COPY_DATASET_TO_PLL:
    # check_pll_cube_format()
    # check_cn_elbow_format()
    copy_dataset_to_pll_format()

if ELIMINATE_EMPTY_TOSSES:
    eliminate_empty_tosses()
