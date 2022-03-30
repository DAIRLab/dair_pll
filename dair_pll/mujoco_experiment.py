import os
import pdb
from dataclasses import dataclass
from typing import cast, Callable

import torch

from dair_pll.deep_learnable_system import \
    DeepLearnableSystemConfig, DeepLearnableExperiment
from dair_pll.experiment import SupervisedLearningExperimentConfig, \
    OptimizerConfig, DataConfig, TrajectorySliceDataset
from dair_pll.mujoco_system import MuJoCoSystem, MuJoCoUKFSystem


@dataclass
class MuJoCoExperimentConfig(SupervisedLearningExperimentConfig):
    xml: str = 'assets/cube_mujoco.xml'
    stiffness: float = 100.
    damping_ratio: float = 1.00
    v200: bool = False


class MuJoCoExperiment(DeepLearnableExperiment):

    def __init__(self, config: MuJoCoExperimentConfig) -> None:
        super().__init__(config)

    def get_base_system(self) -> MuJoCoSystem:
        config = cast(MuJoCoExperimentConfig, self.config)
        dt = config.data_config.dt
        return MuJoCoSystem(config.xml, dt, config.stiffness,
                            config.damping_ratio, config.v200)

    def get_oracle_system(self) -> MuJoCoSystem:
        config = cast(MuJoCoExperimentConfig, self.config)
        data_config = config.data_config
        noiser = data_config.noiser_type(self.space)
        P0_diag, R_diag = MuJoCoUKFSystem.noise_stds_to_P0_R_stds(
            data_config.static_noise, data_config.dynamic_noise, data_config.dt)
        P0 = noiser.covariance(P0_diag)
        R = noiser.covariance(R_diag)
        return MuJoCoUKFSystem(config.xml, data_config.dt, config.stiffness,
                               config.damping_ratio, config.v200, P0, R)


if __name__ == "__main__":
    '''
    stiffness = 2500
    study_name = f'mujoco_cube_{stiffness}_experiment_test'
    if stiffness == 2500:
        optimizer_config = OptimizerConfig(
            lr = 1e-4,
            wd = 0.
            )

    elif stiffness == 100:
        optimizer_config = OptimizerConfig()
    experiment_config = MuJoCoDataExperimentConfig(
        optimizer_config = optimizer_config,
        stiffness = stiffness,
        study = study_name,
        N_pop = 1024
        )
    experiment = MuJoCoDataExperiment(experiment_config)
    experiment.train()
    '''

    eval_test = False
    ukf_test = False
    if eval_test:
        stiffness = 2500
        POP = 1024
        T_SKIP = 16
        V200 = False
        CUBE_XML = 'assets/cube_mujoco.xml'

        study_name = f'mujoco_cube_{stiffness}_eval_test'
        os.system(f'rm -r results/{study_name}')

        optimizer_config = OptimizerConfig(lr=1e-4, wd=0., patience=0)

        learnable_config = DeepLearnableSystemConfig()
        data_config = DataConfig(
            n_pop=POP,
            n_train=POP - 4,
            n_valid=2,
            n_test=2,
            t_skip=T_SKIP,
            t_history=1,
            storage=study_name  # ,
            # static_noise = torch.zeros(12),
            # dynamic_noise = torch.zeros(12)
        )

        experiment_config = MuJoCoExperimentConfig(
            xml=CUBE_XML,
            learnable_config=learnable_config,
            optimizer_config=optimizer_config,
            data_config=data_config,
            stiffness=stiffness,
            v200=V200)

        experiment = MuJoCoExperiment(experiment_config)
        _, best_valid_loss, learned_system, train_traj, valid_traj, test_traj = experiment.train(
        )
        stats = experiment.evaluation(learned_system, train_traj, valid_traj,
                                      test_traj,
                                      TrajectorySliceDataset(train_traj),
                                      TrajectorySliceDataset(valid_traj),
                                      TrajectorySliceDataset(test_traj))
        # print(stats['train_oracle_loss_mean'])
        oracle_tensor = torch.tensor(stats['train_oracle_loss'])
        print('oracle: loss ', oracle_tensor.mean())
        print('rot err degrees',
              torch.tensor(stats['train_oracle_rot_err']).mean() * 180 / 3.1415)
        print('pos err percent',
              torch.tensor(stats['train_oracle_pos_err']).mean() * 100 / 0.1)
        # print(oracle_tensor.std() / np.sqrt(len(stats['train_oracle_loss'])))
        pdb.set_trace()

    if ukf_test:
        stiffness = 2500
        POP = 16
        V200 = False
        CUBE_XML = 'assets/cube_mujoco.xml'

        study_name = f'mujoco_cube_{stiffness}_ukf_test'
        os.system(f'rm -r results/{study_name}')

        optimizer_config = OptimizerConfig(lr=1e-4, wd=0., patience=0)

        learnable_config = DeepLearnableSystemConfig()
        data_config = DataConfig(
            n_pop=POP,
            n_train=POP - 2,
            n_valid=1,
            n_test=1,
            t_history=16,
            storage=study_name
            # T_skip = TSKIP,
            # static_noise = torch.zeros(12)#,
            # dynamic_noise = torch.zeros(12)
            # noiser_type = GaussianWhiteNoiser
        )

        experiment_config = MuJoCoExperimentConfig(
            xml=CUBE_XML,
            learnable_config=learnable_config,
            optimizer_config=optimizer_config,
            data_config=data_config,
            stiffness=stiffness,
            v200=V200)

        experiment = MuJoCoExperiment(experiment_config)
        _, best_valid_loss, learned_system, train_traj, valid_traj, test_traj = experiment.train(
        )
        dataset = experiment.data_manager.slice(train_traj)
        base_system = experiment.get_base_system()
        oracle_system = experiment.get_oracle_system()
        oracle_loss = []
        base_loss = []
        from time import time

        t0 = time()
        N = 0
        M = 0
        BL_MIN = 1e-3
        for (x, y) in dataset:
            M += 1
            bl = experiment.evaluation_loss(x.clone().unsqueeze(0),
                                            y.unsqueeze(0), base_system)
            if bl > BL_MIN:
                N += 1
                base_loss.append(bl)
                oracle_loss.append(
                    experiment.evaluation_loss(x.clone().unsqueeze(0),
                                               y.unsqueeze(0), oracle_system))
        dur = time() - t0
        print(dur, N, dur / N, M)
        itemize: Callable = lambda l: [i.item() for i in l]
        base_loss = itemize(base_loss)
        oracle_loss = itemize(oracle_loss)
        print(sum(base_loss) / N, sum(oracle_loss) / N)
        import matplotlib.pyplot as plt

        # pdb.set_trace()
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(base_loss, oracle_loss)
        ax.set_yscale('log')
        ax.set_xscale('log')
        min_loss = min(base_loss + oracle_loss)
        max_loss = max(base_loss + oracle_loss)
        bounds = [min_loss, max_loss]
        ax.plot(bounds, bounds)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        plt.show()
