import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from tensordict import TensorDict

import torch
from torch import Tensor

import gin

from tensordict import TensorDict, TensorDictBase

#from sim_experiment import load_configs

# import os
# import sys
# import git

# REPO_DIR = os.path.normpath(
# git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
# )
# DEFAULT_CONFIG = "rss_experiment.gin"

# config_file = DEFAULT_CONFIG
# if len(sys.argv) < 2:
#     print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
# else:
#     config_file = sys.argv[1]
# gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))

#from trifinger_lcm_service import TrifingerLCMService
@gin.configurable
class TrajectoryFactory():
    """
    Class for generating trajectories
    """
    def __init__(self,
                 fingertip_body_names,
                 object_rad = 0.1,
                 workspace_radius = 0.2,
                 traj_n_steps = 100,
                 traj_len_s = 2,
                 ):
        
        assert object_rad < workspace_radius
        self._object_rad = object_rad
        self._workspace_radius = workspace_radius
        
        self._traj_n_steps = traj_n_steps
        self._traj_len_s = traj_len_s
        self._fingertip_body_names = fingertip_body_names

    def cubic_spline_interpolation(
        self,
        data: Tensor,
        traj_n_steps: int = 60
        ) -> TensorDictBase:
        """
        Interpolates a start/end action using a cubic spline.

        Params:
            data: outputs of sample_action size (batch, 2, 18)

        Returns:
            TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
            Timestamps = Tensor size (traj_n_steps,)
        """
        traj_len_s = self._traj_len_s
        traj_n_steps = self._traj_n_steps
        fingertip_body_names = ['finger_0', 'finger_1']

        assert len(data.size()) >= 2
        batch_dims = data.size()[:-2]
        assert data.size() == batch_dims + (2, 18)

        ret = TensorDict({}, batch_size = batch_dims + (traj_n_steps,))

        ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps) # (traj_n_steps,)
        rel_timestamps = ((ret_timestamps - ret_timestamps[0]) / (ret_timestamps[-1] - ret_timestamps[0])).unsqueeze(0) # (1, traj_n_steps)
        
        samples = data[..., :, :9] # (batch, 2, 9)
        samples_dot = data[..., :, 9:] # (batch, 2, 9)
        spline_a = samples[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
        spline_b = samples_dot[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
        spline_c = (3.*(samples[..., 1, :]-samples[..., 0, :]) - 2.*samples_dot[..., 0, :] - samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)
        spline_d = (2.*(samples[..., 0, :]-samples[..., 1, :]) + samples_dot[..., 0, :] + samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)

        # (batch, traj_n_steps, 9)
        data_lerp = spline_a @ torch.pow(rel_timestamps, 0.) + spline_b @ torch.pow(rel_timestamps, 1.) + spline_c @ torch.pow(rel_timestamps, 2.) + spline_d @ torch.pow(rel_timestamps, 3.)
        data_lerp = torch.transpose(data_lerp, -1, -2)
        data_lerp_dot = spline_b @ torch.pow(rel_timestamps, 0.) + 2.*spline_c @ torch.pow(rel_timestamps, 1.) + 3.*spline_d @ torch.pow(rel_timestamps, 2.)
        data_lerp_dot = torch.transpose(data_lerp_dot, -1, -2)

        for fingertip in fingertip_body_names:
            ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
            ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))
        for finger_idx, fingertip in enumerate(fingertip_body_names):
            pos_idx = 3 * finger_idx
            ret[fingertip, "position"][..., :] = data_lerp[..., pos_idx : pos_idx + 3]
            ret[fingertip, "velocity"][..., :] = data_lerp_dot[..., pos_idx : pos_idx + 3]

        return ret, ret_timestamps
    
    def __state_data_to_traj(self, 
                             state_dict: Tensor,
                             ) -> TensorDict:
        fingertip_body_names = self._fingertip_body_names
        traj_n_steps = self._traj_n_steps
        batch_dims = state_dict.size()

        ret = TensorDict({}, batch_size = batch_dims + (traj_n_steps,))

        for fingertip in fingertip_body_names:
            ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
            ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))

        for finger_idx, fingertip in enumerate(fingertip_body_names):
            pos_idx = 3 * finger_idx
            ret[fingertip, "position"][..., :] = state_dict[0][..., pos_idx : pos_idx + 3]
            ret[fingertip, "velocity"][..., :] = state_dict[1][..., pos_idx : pos_idx + 3]

        return ret

    def object_avoidance(
        self,
        state_data: Tensor,
        ) -> TensorDict:

        """Generates trajectory avoiding object by removing a inner sphere from its workspace
        """
        object_rad = self._object_rad
        n = self._traj_n_steps

        target_dis = np.array([np.linalg.norm(i) for i in [np.split(state_data[0, :9], 3)]])
        init_dis = np.array([np.linalg.norm(i) for i in [np.split(state_data[1, :9], 3)]])
        
        # If the target or init is the inner sphere
        if np.any(target_dis < object_rad) or np.any(init_dis < object_rad):
            interoplated_state_data = self.cubic_spline_interpolation(state_data)
            return self.__state_data_to_traj(interoplated_state_data)
        
        else:
            # a cost function that minimizes distance in cartesian
            def cost_func(x):
                x_mat = x.reshape((n, 9))
                dx = np.gradient(x_mat, axis = 0)
                return np.sum(dx**2)

            traj_guess = np.random.uniform(0, 1, size = (n, 18))
            
            x0 = traj_guess.flatten()

            # constraint on start point
            start_eq = lambda x: x[:18]
            start_cons = NonlinearConstraint(start_eq, lb = state_data[0], ub = state_data[0])

            # constraint on end point
            end_eq = lambda x: x[-18:]
            end_cons = NonlinearConstraint(end_eq, lb = state_data[1], ub = state_data[1])

            radial_eq = lambda x: np.array([np.linalg.norm(arr) for arr in np.split(x, n * 3)])
            radial_cons = NonlinearConstraint(radial_eq, lb = object_rad, ub = object_rad)
            
            result = minimize(cost_func, x0, constraints={start_cons, end_cons, radial_cons})

            soln = result.x.reshape((n, 9))
            assert result.success, "Optimizer failed!"

            soln_dot = np.gradient(soln, axis = 0)

        # No velocity at ends
        soln_dot[0,:] = 0
        soln_dot[-1,:] = 0

# def collision_free_traj(
#     init_state: np.ndarray,
#     target_state: np.ndarray,
#     constraint_rad: float,
#     workspace_rad: float,
#     n: int = 100) -> np.ndarray:
#     """
#     Args:
#         sta
#     """
#     state_data = TensorDict({}, batch_size = 1 + (2, 18))

#     target_state = target_state[:9]
#     assert workspace_rad > constraint_rad 

#     target_dis = np.array([np.linalg.norm(i) for i in [target_state[:3], target_state[3:6], target_state[6:]]])
#     init_dis = np.array([np.linalg.norm(i) for i in [init_state[:3], init_state[3:6], init_state[6:]]])
    
#     # If the target or init is the inner sphere
    
#     if np.any(target_dis < constraint_rad) or np.any(init_dis < constraint_rad):



#     else:
#         # a cost function that minimizes distance in cartesian
#         def cost_func(x):
#             x_mat = x.reshape((n, 9))
#             dx = np.gradient(x_mat, axis = 0)
#             return np.sum(dx**2)

#         traj_guess = np.random.uniform(0, 1, size = (n, 9))
        
#         x0 = traj_guess.flatten()

#         # constraint on start point
#         start_eq = lambda x: x[:9]
#         start_cons = NonlinearConstraint(start_eq, lb = init_state, ub = init_state)

#         # constraint on end point
#         end_eq = lambda x: x[-9:]
#         end_cons = NonlinearConstraint(end_eq, lb = target_state, ub = target_state)

#         radial_eq = lambda x: np.array([np.linalg.norm(arr) for arr in np.split(x, n * 3)])
#         radial_cons = NonlinearConstraint(radial_eq, lb = constraint_rad, ub = workspace_rad)
        
#         result = minimize(cost_func, x0, constraints={start_cons, end_cons, radial_cons})

#         soln = result.x.reshape((n, 9))
#         assert result.success, "Optimizer failed!"

#         soln_dot = np.gradient(soln, axis = 0)

#     # No velocity at ends
#     soln_dot[0,:] = 0
#     soln_dot[-1,:] = 0

#     traj = TensorDict({
#         ('finger_0', 'position'): torch.zeros((n, 3)),
#         ('finger_0', 'velocity'): torch.zeros((n, 3)),
#         ('finger_1', 'position'): torch.zeros((n, 3)),
#         ('finger_1', 'velocity'): torch.zeros((n, 3))
#     }, batch_size=n)

#     # Fill the TensorDict with data
#     for i in range(n):
#         for j, name in enumerate(['finger_0', 'finger_1']):
#             pos = soln[i, 3*j:3*j+3]
#             vel = soln_dot[i, 3*j:3*j+3]
#             traj[name, "position"][i] = torch.from_numpy(pos)
#             traj[name, "velocity"][i] = torch.from_numpy(vel)

#     return traj


# @gin.configurable(denylist=["data"])
# def interpolate_sampled_action(
#     data: Tensor, fingertip_body_names: List[str], traj_len_s=2.0, traj_n_steps=61
# ) -> TensorDictBase:
#     """
#     Interpolates a start/end action using a cubic spline.

#     Params:
#         data: outputs of sample_action size (batch, 2, 18)

#     Returns:
#         TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
#         Timestamps = Tensor size (traj_n_steps,)
#     """
#     assert len(data.size()) >= 2
#     batch_dims = data.size()[:-2]
#     assert data.size() == batch_dims + (2, 18)
#     ret = TensorDict({}, batch_size=batch_dims + (traj_n_steps,))
#     ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps) # (traj_n_steps,)
#     rel_timestamps = ((ret_timestamps - ret_timestamps[0]) / (ret_timestamps[-1] - ret_timestamps[0])).unsqueeze(0) # (1, traj_n_steps)
#     samples = data[..., :, :9] # (batch, 2, 9)
#     samples_dot = data[..., :, 9:] # (batch, 2, 9)
#     spline_a = samples[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
#     spline_b = samples_dot[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
#     spline_c = (3.*(samples[..., 1, :]-samples[..., 0, :]) - 2.*samples_dot[..., 0, :] - samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)
#     spline_d = (2.*(samples[..., 0, :]-samples[..., 1, :]) + samples_dot[..., 0, :] + samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)

#     # (batch, traj_n_steps, 9)
#     data_lerp = spline_a @ torch.pow(rel_timestamps, 0.) + spline_b @ torch.pow(rel_timestamps, 1.) + spline_c @ torch.pow(rel_timestamps, 2.) + spline_d @ torch.pow(rel_timestamps, 3.)
#     data_lerp = torch.transpose(data_lerp, -1, -2)
#     data_lerp_dot = spline_b @ torch.pow(rel_timestamps, 0.) + 2.*spline_c @ torch.pow(rel_timestamps, 1.) + 3.*spline_d @ torch.pow(rel_timestamps, 2.)
#     data_lerp_dot = torch.transpose(data_lerp_dot, -1, -2)

#     for fingertip in fingertip_body_names:
#         ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
#         ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))
#     for finger_idx, fingertip in enumerate(fingertip_body_names):
#         pos_idx = 3 * finger_idx
#         ret[fingertip, "position"][..., :] = data_lerp[..., pos_idx : pos_idx + 3]
#         ret[fingertip, "velocity"][..., :] = data_lerp_dot[..., pos_idx : pos_idx + 3]

#     return ret, ret_timestamps

if __name__ == '__main__':

    f = TrajectoryFactory()
    f.object_avoidance(np.zeros((1,18), dtype=np.float32), np.ones((1,18), dtype=np.float32))
    #pass
    #collision_free_traj(-np.ones((9,)), np.ones((9,)), steps=10)