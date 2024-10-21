import pdb
from typing import Tuple, Optional

import numpy as np
import optuna
import torch
from mujoco_py import load_model_from_xml, MjSim
from scipy.optimize import minimize
from torch import Tensor

from dair_pll import state_space
from dair_pll.integrator import StateIntegrator
from dair_pll.system import System
from dair_pll.ukf import UKF


class MuJoCoStateConverter:

    @staticmethod
    def mujoco_to_state(x_mujoco):
        # mujoco ordering: [p, q, j]
        t = lambda x: torch.tensor(x).clone()
        p = t(x_mujoco.qpos[:3])
        q = t(x_mujoco.qpos[3:7])
        j = t(x_mujoco.qpos[7:])
        v = t(x_mujoco.qvel[:3])
        omega = t(x_mujoco.qvel[3:6])
        vj = t(x_mujoco.qvel[6:])
        return torch.cat((q, p, j, omega, v, vj)).unsqueeze(0)

    @staticmethod
    def state_to_mujoco(x_mujoco, q: Tensor, v: Tensor):
        # mujoco ordering: [p, q, j]
        x_mujoco.qpos[3:7] = q[..., :4].squeeze()
        x_mujoco.qpos[:3] = q[..., 4:7].squeeze()
        x_mujoco.qpos[7:] = q[..., 7:].squeeze()

        x_mujoco.qvel[3:7] = v[..., :3].squeeze()
        x_mujoco.qvel[:3] = v[..., 3:6].squeeze()
        x_mujoco.qvel[7:] = v[..., 6:].squeeze()

        return x_mujoco


class MuJoCoSystem(System):
    sim: MjSim

    def __init__(
        self,
        mjcf: str,
        dt: float,
        stiffness: float,
        damping_rato: float,
        v200: bool = False,
    ) -> None:

        time_constant = 1.0 / (damping_rato * np.sqrt(stiffness))
        total_damping = damping_rato * 2 * np.sqrt(stiffness)
        sys_xml = ""
        with open(mjcf, "r") as sysfile:
            if v200:
                sys_xml = (
                    sysfile.read()
                    .replace("$solrefarg1", str(-stiffness))
                    .replace("$solrefarg2", str(-total_damping))
                    .replace("$dt", str(dt))
                )
            else:
                sys_xml = (
                    sysfile.read()
                    .replace("$solrefarg1", str(time_constant))
                    .replace("$solrefarg2", str(damping_rato))
                    .replace("$dt", str(dt))
                )
        # print(sys_xml)
        model = load_model_from_xml(sys_xml)
        sim = MjSim(model)
        sim_state = sim.get_state()
        # pdb.set_trace()
        n_joints = len(sim_state.qpos) - 7
        space = state_space.FloatingBaseSpace(n_joints)
        integrator = StateIntegrator(space, self.sim_step, dt)
        super().__init__(space, integrator)
        self.max_batch_dim = 0
        self.sim = sim
        self.set_carry_sampler(lambda: torch.tensor([[False]]))

    def preprocess_initial_condition(
        self, x_0: Tensor, carry_0: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # get dummy state
        while len(x_0.shape) >= 2:
            # x0 has (probably trivial) duration dimension;
            # select most recent precondition
            x_0 = x_0[-1, ...]

        while len(carry_0.shape) >= 2:
            # x0 has (probably trivial) duration dimension;
            # select most recent precondition
            carry_0 = carry_0[-1, ...]

        sim_state = self.sim.get_state()
        # pdb.set_trace()
        q0, v0 = self.space.q_v(x_0)
        # pdb.set_trace()
        state0 = MuJoCoStateConverter.state_to_mujoco(sim_state, q0, v0)
        self.sim.set_state(state0)
        self.sim.forward()
        # pdb.set_trace()
        return x_0, carry_0 or torch.tensor([True])

    def sim_step(self, x: Tensor, carry: Tensor) -> Tensor:
        # carry detects if first step has been taken yet
        # pdb.set_trace()
        self.sim.step()
        # pdb.set_trace()
        x_next = MuJoCoStateConverter.mujoco_to_state(self.sim.get_state())
        # print(self.sim.get_state())
        return x_next, carry


SENSE_VELOCITY = True

BIAS = False
BIAS_VEL = False
SENSE_BIAS = True


class MuJoCoUKFSystem(MuJoCoSystem):
    P0: Tensor
    R: Tensor

    @staticmethod
    def noise_stds_to_P0_R_stds(
        static_stds: Tensor, dynamic_std: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor]:
        if BIAS:
            M = 1.0
            nv = static_stds.nelement() // 2
            composite_noise_diag = dynamic_std[:nv]  # + static_stds[:nv]
            state0_diag = (
                torch.cat((composite_noise_diag, dynamic_std[:nv] * np.sqrt(2 / dt)))
                * M
            )
            bias0_diag = (static_stds + 1e-8) * M
            if BIAS_VEL:
                P0_diag = torch.cat((state0_diag, bias0_diag))
            else:
                P0_diag = torch.cat((state0_diag, bias0_diag[:nv]))
            if SENSE_VELOCITY:
                R_diag = state0_diag.clone()
                if SENSE_BIAS:
                    R_diag = P0_diag.clone()
            else:
                # hack to fix error detected??
                R_diag = composite_noise_diag.clone()
            # R_diag = composite_noise_diag.clone()

            return (P0_diag, R_diag)
        else:
            nv = static_stds.nelement() // 2
            config_noise_diag = torch.sqrt(static_stds**2 + dynamic_std**2)[:nv]
            P0_diag = torch.cat((config_noise_diag, dynamic_std[:nv] * np.sqrt(2 / dt)))
            if SENSE_VELOCITY:
                R_diag = P0_diag.clone()
            else:
                R_diag = config_noise_diag.clone()
            return (P0_diag, R_diag)

    def __init__(
        self,
        mjcf: str,
        dt: float,
        stiffness: float,
        damping_rato: float,
        v200: bool = False,
        P0: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
    ) -> None:
        super().__init__(mjcf, dt, stiffness, damping_rato, v200)
        if P0 is None:
            P0 = torch.eye((4 if BIAS else 2) * self.space.n_v)
        if R is None:
            R = torch.eye((2 if SENSE_VELOCITY else 1) * self.space.n_v) * 1e-2
        self.P0 = P0
        self.R = R

    def ukf_estimate(self, x0: Tensor) -> Tensor:

        # pdb.set_trace()

        numpify = lambda x: x.detach().numpy()
        torchify = lambda x: torch.tensor(x, dtype=torch.float64).clone()

        SIC = super().preprocess_initial_condition

        # reduce to 1 trajectory
        while len(x0.shape) >= 3:
            # x0 has (probably trivial) duration dimension;
            # select most recent precondition
            x0 = x0[-1, ...]

        def ukf_f(state, omega, w, dt):
            # omega is input; neglect for now
            # w is a tangent space noise vector
            # pdb.set_trace()
            state = torchify(state).unsqueeze(0)
            w = torchify(w).unsqueeze(0)
            x0 = self.space.shift_state(state, w)
            carry = self.carry_callback()
            SIC(x0, carry)
            return numpify(self.sim_step(x0, carry)[0].squeeze(0))

        def ukf_h(state):
            state = torchify(state).unsqueeze(0)
            zero = torchify(self.space.zero_state().unsqueeze(0))
            if SENSE_VELOCITY:
                return numpify(self.space.state_difference(zero, state).squeeze(0))
            # pdb.set_trace()
            else:
                zero_q = self.space.q(zero)
                state_q = self.space.q(state)
                return numpify(
                    self.space.configuration_difference(zero_q, state_q).squeeze(0)
                )

        def ukf_phi(state, dstate):
            state = torchify(state).unsqueeze(0)
            dstate = torchify(dstate).unsqueeze(0)
            return numpify(self.space.shift_state(state, dstate).squeeze(0))

        def ukf_phi_inv(x1, x2):
            x1 = torchify(x1).unsqueeze(0)
            x2 = torchify(x2).unsqueeze(0)
            return numpify(self.space.state_difference(x1, x2).squeeze(0))

        Q = numpify(1e-10 * torch.eye(2 * self.space.n_v))

        alpha = 1e-1 * np.array([1.0, 1.0, 1.0])

        start = numpify(torchify(x0[0, :]))

        R = numpify(self.R)
        # pdb.set_trace()

        P0 = numpify(self.P0.clone())

        ukf = UKF(ukf_f, ukf_h, ukf_phi, ukf_phi_inv, Q, R, alpha, start, P0)
        # pdb.set_trace()
        for x_i in x0[1:, :]:
            ukf.propagation(torch.tensor(0.0), self.integrator.dt)

            y_i = ukf_h(numpify(x_i))
            ukf.update(y_i)

        # pdb.set_trace()
        print("done!")
        return torchify(ukf.state).unsqueeze(0)

    def ukf_bias_estimate(self, x0: Tensor) -> Tensor:

        # pdb.set_trace()

        numpify = lambda x: x.detach().numpy()
        torchify = lambda x: torch.tensor(x, dtype=torch.float64).clone()

        SIC = super().preprocess_initial_condition

        # reduce to 1 trajectory
        while len(x0.shape) >= 3:
            # x0 has (probably trivial) duration dimension;
            # select most recent precondition
            x0 = x0[-1, ...]

        def ukf_f(state, omega, w, dt):
            # omega is input; neglect for now
            # w is a tangent space noise vector
            # pdb.set_trace()

            # state is actual state and a bias
            # pdb.set_trace()
            state = ukf_phi(state, w)
            state = torchify(state).unsqueeze(0)
            bias = state[:, self.space.n_x :]
            shift = bias
            if not BIAS_VEL:
                shift = torch.cat((bias, 0.0 * bias), dim=1)
            state = self.space.shift_state(state[:, : self.space.n_x], shift)
            # w = torchify(w).unsqueeze(0)
            # x0 = self.space.shift_state(state, w)
            carry = self.carry_callback()
            SIC(x0, carry)
            real_next_state = self.sim_step(x0, carry)[0]
            # pdb.set_trace()
            sensed_next_state = self.space.shift_state(real_next_state, -shift)
            return numpify(torch.cat((sensed_next_state.squeeze(0), bias.squeeze(0))))

        def ukf_h(state):
            # pdb.set_trace()
            state = torchify(state).unsqueeze(0)
            bias = state[:, self.space.n_x :]
            state = state[:, : self.space.n_x]
            zero = torchify(self.space.zero_state().unsqueeze(0))
            if SENSE_VELOCITY:
                ds = self.space.state_difference(zero, state)
                if SENSE_BIAS:
                    return numpify(torch.cat((ds.squeeze(0), bias.squeeze(0))))
                return numpify(ds.squeeze(0))

            else:
                zero_q = self.space.q(zero)
                state_q = self.space.q(state)
                return numpify(
                    self.space.configuration_difference(zero_q, state_q).squeeze(0)
                )

        def ukf_phi(state, delta):
            state = torchify(state).unsqueeze(0)
            bias = state[:, self.space.n_x :]
            state = state[:, : self.space.n_x]
            dstate = torchify(delta[: (2 * self.space.n_v)]).unsqueeze(0)
            dbias = torchify(delta[(2 * self.space.n_v) :]).unsqueeze(0)

            fullx = self.space.shift_state(state, dstate).squeeze(0)
            if dbias.nelement() == 0:
                pdb.set_trace()
            fullbias = (bias + dbias).squeeze(0)

            return numpify(torch.cat((fullx, fullbias)))

        def ukf_phi_inv(x1, x2):
            x1 = torchify(x1).unsqueeze(0)
            x2 = torchify(x2).unsqueeze(0)
            delta_bias = (x2[:, self.space.n_x :] - x1[:, self.space.n_x :]).squeeze(0)
            delta_state = self.space.state_difference(
                x1[:, : self.space.n_x], x2[:, : self.space.n_x]
            ).squeeze(0)

            return numpify(torch.cat((delta_state, delta_bias)))

        Q = numpify(1e-8 * torch.eye((4 if BIAS_VEL else 3) * self.space.n_v))

        alpha = 1e-1 * np.array([1.0, 1.0, 1.0])

        # start = numpify(torchify(x0[0, :]))

        NT = 2 * self.space.n_v
        start = numpify(
            torchify(torch.cat((x0[0, :], torch.zeros(NT if BIAS_VEL else NT // 2))))
        )

        R = numpify(self.R)
        # pdb.set_trace()

        # P0 = torch.eye(2 * NT)
        # P0[:NT, :NT] = self.P0.clone()
        # P0[NT:, NT:] = self.P0.clone()
        P0 = numpify(self.P0.clone())
        # pdb.set_trace()
        ukf = UKF(ukf_f, ukf_h, ukf_phi, ukf_phi_inv, Q, R, alpha, start, P0)
        # pdb.set_trace()
        for x_i in x0[1:, :]:
            ukf.propagation(torch.tensor(0.0), self.integrator.dt)

            y_i = ukf_h(numpify(torch.cat((x_i, 0.0 * x_i[(1 if BIAS_VEL else 7) :]))))
            ukf.update(y_i)

        # pdb.set_trace()
        state = ukf.state
        state = torchify(state).unsqueeze(0)
        bias = state[:, self.space.n_x :]
        shift = bias
        if not BIAS_VEL:
            shift = torch.cat((bias, 0.0 * bias), dim=1)
        state = self.space.shift_state(state[:, : self.space.n_x], shift)
        # pdb.set_trace()
        print("done", bias.norm())
        return state

    def mll_estimate(self, x0: Tensor) -> Tensor:
        # pdb.set_trace()
        x0 = x0.squeeze()
        torchify = lambda x: torch.tensor(x, dtype=torch.float64).clone()
        torchify32 = lambda x: torch.tensor(x, dtype=torch.float32).clone()
        # torchify = lambda x: torch.tensor(x).clone()
        T = x0.shape[0]

        base_x0 = torchify(x0[0, :])

        SIC = super().preprocess_initial_condition

        x064 = torchify(x0)
        OPTUNA = True
        LSQ = False

        def eval_ic(exp_ic: np.ndarray) -> float:
            # pdb.set_trace()

            exp_ic = torchify(exp_ic).unsqueeze(0)
            ic = self.space.shift_state(base_x0.unsqueeze(0), exp_ic)
            # pdb.set_trace()
            # ic_traj = SSIM(ic, self.carry_callback(), T)
            x, carry = SIC(ic, self.carry_callback())
            ic_traj, carrytraj = self.integrator.simulate(x, carry, T - 1)
            deltas = self.space.state_difference(x064, ic_traj)
            # NLL = (deltas ** 2) / torch.diag(self.R)
            # return NLL.sum().item()
            scd = deltas / torch.sqrt(torch.diag(self.R)) * 1e-4
            scd = scd[:, :]
            if LSQ:
                return torch.flatten(scd).detach().numpy()
            else:
                return (scd**2).sum()

        # fitted_x0 = minimize(eval_ic, np.zeros((12,)), method = 'Nelder-Mead')
        z_window = 1 * torch.sqrt(torch.diag(self.R)).numpy()

        def hp2state(ostate):
            vstate = np.zeros(12)
            for i in range(12):
                vstate[i] = ostate[f"x_{i}"]
            return vstate

        def optuna_shell(trial):
            ostate = {}
            for i in range(12):
                p = f"x_{i}"
                ostate[p] = trial.suggest_float(p, -z_window[i], z_window[i])
            vstate = hp2state(ostate)
            return eval_ic(vstate)

        if OPTUNA:
            study = optuna.create_study()
            optuna.logging.disable_default_handler()
            study.optimize(optuna_shell, n_trials=100)
            fitted_x0 = hp2state(study.best_params)

        else:

            LM = False

            method = "lm" if LM else "dogbox"
            if LSQ:
                bounds = (-np.inf, np.inf) if LM else (-z_window, z_window)
            else:
                bounds = [(-zi, zi) for zi in z_window]
            # fitted_x0 = least_squares(eval_ic, np.zeros((12,)), bounds=bounds, jac='3-point', verbose=1, method=method)
            fitted_x0 = minimize(
                eval_ic, np.zeros((12,)), method="Nelder-Mead", bounds=bounds
            ).x
            # pdb.set_trace()

        start = self.space.shift_state(
            x0[0, :].unsqueeze(0), torchify32(fitted_x0).unsqueeze(0)
        )
        # TODO
        start_traj, carrytraj = self.integrator.simulate(
            start, self.carry_callback(), T - 1
        )
        # pdb.set_trace()
        print("done")
        return self.space.shift_state(
            start_traj[-1, :].unsqueeze(0), torchify32(fitted_x0).unsqueeze(0)
        )

    def preprocess_initial_condition(
        self, x_0: Tensor, carry_0: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # pdb.set_trace()
        estimate = self.ukf_bias_estimate(x_0) if BIAS else self.ukf_estimate(x_0)
        return super().preprocess_initial_condition(estimate, carry_0)


if __name__ == "__main__":
    """
    mjcsys = MuJoCoSystem('assets/cube_mujoco.xml', 6.74e-3, 2500., 1.04)
    starting_state = mjcsys.space.zero_state()
    starting_state[6] += 0.07
    mjcsys.set_state_sampler(state_space.ConstantSampler(mjcsys.space, starting_state))

    xtraj, carry = mjcsys.sample_trajectory(20)

    ukfsys = MuJoCoUKFSystem('assets/cube_mujoco.xml', 6.74e-3, 2500., 1.04)

    #pdb.set_trace()

    ukfsys.set_initial_condition(xtraj, carry[0])

    #learned_system = DeepLearnableSystem(mjcsys, DeepLearnableSystemConfig())
    print(xtraj.shape)
    print(xtraj[-1, :])
    """
    """
    import matplotlib.pyplot as plt
    plt.plot(xtraj[:, 6])
    plt.plot(xtraj_chain[:, 6].detach())
    plt.legend(['drake', 'todorov'])
    plt.show()
    """
