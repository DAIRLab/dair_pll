"""Test script for InertialParameterConverter.
"""
import torch
import pdb

from dair_pll.inertia import InertialParameterConverter as ipc



N_TESTS = 20
VERBOSE = True
THETA_SCALE = 1e0
N_BODIES_UP_TO = 6


# Check that theta_to_pi_o and pi_o_to_theta are inverses of each other.
def do_theta_pi_o_test():
    n_bodies = int(torch.rand(1).item()*(N_BODIES_UP_TO-1)+1)

    # Make a random set of theta parameters.
    theta = (torch.rand(n_bodies, 10) - 0.5) * THETA_SCALE

    # Calculate pi_o from this random theta.
    pi_o = ipc.theta_to_pi_o(theta)

    # Return the norm difference between theta and theta>pi_o>theta.
    return torch.norm(theta - ipc.pi_o_to_theta(pi_o))


# Check that pi_o_to_pi_cm and pi_cm_to_pi_o are inverses of each other.
def do_pi_o_cm_test(use_thetas = True):
    n_bodies = int(torch.rand(1).item()*(N_BODIES_UP_TO-1)+1)

    if use_thetas:
        # Make a random set of theta parameters.
        theta = (torch.rand(n_bodies, 10) - 0.5) * THETA_SCALE

        # Make pi_o parameters from this random theta.
        pi_o = ipc.theta_to_pi_o(theta)

    else:
        pi_o = torch.rand(n_bodies, 10)

    # Calculate pi_cm from this set.
    pi_cm = ipc.pi_o_to_pi_cm(pi_o)

    # Return the norm difference between theta and pi_o>pi_cm>pi_o.
    return torch.norm(pi_o - ipc.pi_cm_to_pi_o(pi_cm))


# Check that theta_to_pi_cm and pi_cm_to_theta are inverses of each other.
def do_theta_pi_cm_test():
    n_bodies = int(torch.rand(1).item()*(N_BODIES_UP_TO-1)+1)

    # Make a random set of theta parameters.
    theta = torch.rand(n_bodies, 10) * THETA_SCALE

    # Calculate pi_cm from this random theta.
    pi_cm = ipc.theta_to_pi_cm(theta)

    # Return the norm difference between theta and theta>pi_cm>theta.
    return torch.norm(theta - ipc.pi_cm_to_theta(pi_cm))


print(f'\n=== Doing theta to pi_o test ===')
cost = 0
for _ in range(N_TESTS):
    new_cost = do_theta_pi_o_test().item()
    cost += new_cost
    if VERBOSE:
        print(new_cost)
print(f'\tAverage difference:  {cost/N_TESTS}')

print(f'\n=== Doing pi_o to pi_cm test, starting from thetas ===')
cost = 0
for _ in range(N_TESTS):
  new_cost = do_pi_o_cm_test(use_thetas = True).item()
  cost += new_cost
  if VERBOSE:
      print(new_cost)
print(f'\tAverage difference:  {cost/N_TESTS}')

print(f'\n=== Doing pi_o to pi_cm test, not using thetas ===')
cost = 0
for _ in range(N_TESTS):
  new_cost = do_pi_o_cm_test(use_thetas = False).item()
  cost += new_cost
  if VERBOSE:
      print(new_cost)
print(f'\tAverage difference:  {cost/N_TESTS}')

print(f'\n=== Doing theta to pi_cm test ===')
cost = 0
for _ in range(N_TESTS):
  new_cost = do_theta_pi_cm_test().item()
  cost += new_cost
  if VERBOSE:
      print(new_cost)
print(f'\tAverage difference:  {cost/N_TESTS}')