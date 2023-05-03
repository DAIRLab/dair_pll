"""Test script to help with inertia randomizations."""
import torch
from torch import Tensor

from dair_pll.geometry import _NOMINAL_HALF_LENGTH
from dair_pll.inertia import InertialParameterConverter


N_TESTS = 1000

bad_runs = 0

for _ in range(N_TESTS):
	pi_cm_params = torch.ones((2,10))
	theta_params = torch.ones((2,10))

	# Randomize the inertia.
	for idx in range(theta_params.shape[0]):
	    pi_cm = pi_cm_params[idx]

	    # Let the center of mass be anywhere within the inner half of a
	    # nominal geometry.
	    mass = pi_cm[0].item()
	    pi_cm[1:4] = mass * (torch.rand(3) - 0.5) * _NOMINAL_HALF_LENGTH

	    # Define the moments of inertia assuming a solid block of
	    # homogeneous density with random mass and random lengths.
	    rand_mass = mass * (torch.rand(1) + 0.5)
	    rand_lengths = (torch.rand(3) + 0.5) * _NOMINAL_HALF_LENGTH
	    Ixx = (rand_mass/12) * (rand_lengths[1]**2 + rand_lengths[2]**2)
	    Iyy = (rand_mass/12) * (rand_lengths[0]**2 + rand_lengths[2]**2)
	    Izz = (rand_mass/12) * (rand_lengths[0]**2 + rand_lengths[1]**2)
	    pi_cm[4:7] = Tensor([Ixx, Iyy, Izz])

	    # Define the products of inertia assuming the mass is concentrated
	    # at a point somewhere within the inner half of a nominal geometry.
	    rand_com = (torch.rand(3) - 0.5) * _NOMINAL_HALF_LENGTH * 0.3
	    Ixy = -rand_mass * rand_com[0] * rand_com[1]
	    Ixz = -rand_mass * rand_com[0] * rand_com[2]
	    Iyz = -rand_mass * rand_com[1] * rand_com[2]
	    pi_cm[7:10] = Tensor([Ixy, Ixz, Iyz])

	    pi_cm_params[idx] = pi_cm

	theta_params = InertialParameterConverter.pi_cm_to_theta(
	        pi_cm_params)

	if torch.any(torch.isnan(theta_params)):
		bad_runs += 1



print(f'Failure rate: {bad_runs/N_TESTS} \nFailures: {bad_runs}')