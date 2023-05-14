"""Visualization tests of data created with force vector fields.  The goal is to
be able to visualize a generated toss already saved to storage. possibly
overlaid with a toss from the same initial condition and no force vector field.
"""

import os
import os.path as op
import pdb

import shutil
import torch
from torch import Tensor

from dair_pll.drake_system import DrakeSystem
from dair_pll import vis_utils


pdb.set_trace()
SYSTEM = 'cube'
RESULTS_FOLDER_NAME = 'viscous_cube' #'vortex_cube'

TRUE_URDFS = {'cube': '/home/bibit/dair_pll/assets/contactnets_cube.urdf',
              'elbow': '/home/bibit/dair_pll/assets/contactnets_elbow.urdf'}
URDF = {SYSTEM: TRUE_URDFS[SYSTEM]}

TOSS_DIRS = {
    'cube': f'/home/bibit/dair_pll/results/{RESULTS_FOLDER_NAME}/data/ground_truth'}
TOSS_DIR = TOSS_DIRS[SYSTEM]

DT = 0.0068
DUMMY_CARRY = Tensor([0])

VIS_DIR = '/home/bibit/dair_pll/test'
VIS_FILE = op.join(VIS_DIR, 'test.gif')

DRAKE_SYSTEM = DrakeSystem(URDF, DT)
VIS_SYSTEM = vis_utils.generate_visualization_system(DRAKE_SYSTEM, VIS_FILE)


def get_non_augmented_traj(traj):
    x0 = traj[0, :].unsqueeze(0)
    steps = traj.shape[0] - 1

    na_traj, _ = DRAKE_SYSTEM.simulate(x0, DUMMY_CARRY, steps=steps)

    return na_traj


pdb.set_trace()

for toss in os.listdir(TOSS_DIR):
    print(f"Making comparison toss for {toss}...")
    toss_file = op.join(TOSS_DIR, toss)
    augmented_traj = torch.load(toss_file)

    no_augment_traj = get_non_augmented_traj(augmented_traj)

    space = DRAKE_SYSTEM.space
    vis_traj = torch.cat( \
        (space.q(augmented_traj), space.q(no_augment_traj),
         space.v(augmented_traj), space.v(no_augment_traj)), -1)

    video, framerate = vis_utils.visualize_trajectory(VIS_SYSTEM, vis_traj)

    toss_num = toss.split('.')[0]
    new_filename = f'toss_{toss_num}.gif'
    new_file = op.join(VIS_DIR, new_filename)

    test_filepath = op.join(VIS_DIR, 'test_.gif')

    shutil.copyfile(test_filepath, new_file)
