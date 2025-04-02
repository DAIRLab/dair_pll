import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from tensordict import TensorDict

import torch
from torch import Tensor
from tensordict import TensorDict, TensorDictBase
from typing import List

class TrajectoryFactory():
    """
    Class for generating trajectories
    """
    def __init__(
        self,
        fingertip_body_names: List,
        object_rad: float,
        workspace_rad: float,
        traj_n_steps:int,
        traj_time_len: float,
        ):
        
        assert object_rad < workspace_rad
        self._object_rad = object_rad
        self._workspace_rad = workspace_rad
        
        self._traj_n_steps = traj_n_steps
        self._traj_len_s = traj_time_len
        self._fingertip_body_names = fingertip_body_names

    def cubic_spline_interpolation(
        self,
        data: Tensor,
        ) -> TensorDictBase:
        """
        Interpolates a start/end action using a cubic spline.

        Params:
            data: outputs of sample_action size (batch, 2, 18)

        Returns:
            TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
            Timestamps = Tensor size (traj_n_steps,)
        """
        data = data.to('cuda')
        traj_len_s = self._traj_len_s
        traj_n_steps = self._traj_n_steps

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

        ret = self.__state_data_to_traj(ret_timestamps, data_lerp, data_lerp_dot)

        return ret
    
    def __state_data_to_traj(
        self, 
        timestamps: Tensor,
        data_lerp: Tensor,
        data_lerp_dot: Tensor,
        ) -> TensorDict:

        fingertip_body_names = self._fingertip_body_names
        batch_size = data_lerp.size()

        ret = TensorDict({})
        
        for fingertip in fingertip_body_names:
            ret[fingertip, "position"] = torch.broadcast_to(data_lerp[0], batch_size)
            ret[fingertip, "velocity"] = torch.zeros(batch_size)

        for finger_idx, fingertip in enumerate(fingertip_body_names):
            pos_idx = 3 * finger_idx
            ret[fingertip, "position"] = data_lerp[..., pos_idx : pos_idx + 3]
            ret[fingertip, "velocity"] = data_lerp_dot[..., pos_idx : pos_idx + 3]

        ret['timestamp'] = timestamps

        return ret

    def object_avoidance(
        self,
        state_data: Tensor,
        ) -> TensorDict:

        """Generates trajectory avoiding object by removing a inner sphere from its workspace
        """
        workspace_rad = self._workspace_rad
        object_rad = self._object_rad
        n = self._traj_n_steps

        target_dis = np.array([np.linalg.norm(i) for i in [np.split(state_data[0, :9], 3)]])
        init_dis = np.array([np.linalg.norm(i) for i in [np.split(state_data[1, :9], 3)]])
        
        # If the target or init is the inner sphere
        #if np.any(target_dis < object_rad) or np.any(init_dis < object_rad):
        if True:
            interoplated_traj = self.cubic_spline_interpolation(state_data)
            return interoplated_traj
        
        if False:
            # a cost function that minimizes distance in cartesian
            def cost_func(x):
                q = np.split(x, 2*n)
                pos = np.stack(q[::2])
                vel = np.stack(q[1::2])
                
                dx = np.diff(pos, axis = 0)
                pos_cost = np.sum(dx**2)

                dv = np.diff(vel, axis = 0)
                vel_cost = np.sum(dv**2)

                return pos_cost + vel_cost

            x0 = np.linspace(state_data[0], state_data[1], n).flatten()

            # constraint on start point
            start_eq = lambda x: x[:18]
            start_cons = NonlinearConstraint(start_eq, lb = state_data[0], ub = state_data[0])

            # constraint on end point
            end_eq = lambda x: x[-18:]
            end_cons = NonlinearConstraint(end_eq, lb = state_data[1], ub = state_data[1])

            radial_eq = lambda x: np.array([np.linalg.norm(arr) 
                                            for arr in 
                                            np.split(np.concatenate(np.split(x, 2*n)[::2]), n * 3)])
            
            radial_cons = NonlinearConstraint(radial_eq, lb = object_rad, ub = workspace_rad)
            
            result = minimize(cost_func, x0, constraints={start_cons, end_cons, radial_cons})

            soln = result.x.reshape((n, 18))
            assert result.success, "Optimizer failed!"
            pos, vel = np.array_split(soln, 2, axis = 1)

            return self.__state_data_to_traj(torch.linspace(0, self._traj_len_s, n),
                                             Tensor(pos).unsqueeze(0), 
                                             Tensor(np.gradient(pos, axis = 1)).unsqueeze(0))

if __name__ == '__main__':
    ###### TESTING

    f = TrajectoryFactory()

    state_data = torch.stack([torch.from_numpy(np.full((1,18), 0.05, dtype=np.float32)).squeeze(0),
                              torch.from_numpy(np.full((1,18), 0.01, dtype=np.float32)).squeeze(0)])
    #print(state_data)
    d = f.object_avoidance(state_data)
    print(d['finger_0', 'position'])
    #import pdb; pdb.set_trace()
    #pass
    #collision_free_traj(-np.ones((9,)), np.ones((9,)), steps=10)