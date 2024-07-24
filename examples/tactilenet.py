"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import time
from typing import cast, List

import sys
import pdb

import click
import numpy as np
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict
import pickle
import git

from dair_pll import file_utils
from dair_pll.dataset_generation import DataGenerationConfig, \
    ExperimentDatasetGenerator
from dair_pll.dataset_management import DataConfig, TrajectorySliceConfig
from dair_pll.deep_learnable_model import MLP
from dair_pll.deep_learnable_system import DeepLearnableSystemConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeMultibodyLearnableTactileExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeDeepLearnableExperiment, DrakeMultibodyLearnableTactileExperimentConfig
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig, \
    SupervisedLearningExperimentConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_terms import InertiaLearn
from dair_pll.state_space import ConstantSampler, UniformSampler, GaussianWhiteNoiser, \
    FloatingBaseSpace, FixedBaseSpace, ProductSpace
from dair_pll.system import System


# Possible systems on which to run PLL
CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
ASYMMETRIC_SYSTEM = 'asymmetric'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM, ASYMMETRIC_SYSTEM]

# Possible dataset types
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]

# Possible simulation data augmentations.
VORTEX_AUGMENTATION = 'vortex'
VISCOUS_AUGMENTATION = 'viscous'
GRAVITY_AUGMENTATION = 'gravity'
AUGMENTED_FORCE_TYPES = [VORTEX_AUGMENTATION, VISCOUS_AUGMENTATION,
                         GRAVITY_AUGMENTATION]

# Possible inertial parameterizations to learn for the elbow system.
# The options are:
# 0 - none (0 parameters)
# 1 - masses (n_bodies - 1 parameters)
# 2 - CoMs (3*n_bodies parameters)
# 3 - CoMs and masses (4*n_bodies - 1 parameters)
# 4 - all (10*n_bodies - 1 parameters)
INERTIA_PARAM_CHOICES = [str(i) for i in range(5)]
INERTIA_PARAM_DESCRIPTIONS = [
    'learn no inertial parameters (0 * n_bodies)',
    'learn only masses and not the first mass (n_bodies - 1)',
    'learn only centers of mass (3 * n_bodies)',
    'learn masses (except first) and centers of mass (4 * n_bodies - 1)',
    'learn all parameters (except first mass) (10 * n_bodies - 1)']
INERTIA_PARAM_OPTIONS = ['none', 'masses', 'CoMs', 'CoMs and masses', 'all']


# File management.
CUBE_DATA_ASSET = 'contactnets_cube'
ELBOW_DATA_ASSET = 'contactnets_elbow'
CUBE_BOX_URDF_ASSET = 'contactnets_cube.urdf'
CUBE_MESH_URDF_ASSET = 'contactnets_cube_mesh.urdf'
ELBOW_BOX_URDF_ASSET = 'contactnets_elbow.urdf'
ELBOW_MESH_URDF_ASSET = 'contactnets_elbow_mesh.urdf'
ASYMMETRIC_URDF_ASSET = 'contactnets_asymmetric.urdf'

REAL_DATA_ASSETS = {CUBE_SYSTEM: CUBE_DATA_ASSET, ELBOW_SYSTEM: ELBOW_DATA_ASSET}

MESH_TYPE = 'mesh'
BOX_TYPE = 'box'
POLYGON_TYPE = 'polygon'
GEOMETRY_TYPES = [BOX_TYPE, MESH_TYPE, POLYGON_TYPE]

CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET,
              BOX_TYPE: CUBE_BOX_URDF_ASSET,
              POLYGON_TYPE: CUBE_MESH_URDF_ASSET}
ELBOW_URDFS = {MESH_TYPE: ELBOW_MESH_URDF_ASSET,
               BOX_TYPE: ELBOW_BOX_URDF_ASSET,
               POLYGON_TYPE: ELBOW_MESH_URDF_ASSET}
ASYMMETRIC_URDFS = {MESH_TYPE: ASYMMETRIC_URDF_ASSET,
                    POLYGON_TYPE: ASYMMETRIC_URDF_ASSET}
TRUE_URDFS = {CUBE_SYSTEM: CUBE_URDFS, ELBOW_SYSTEM: ELBOW_URDFS,
              ASYMMETRIC_SYSTEM: ASYMMETRIC_URDFS}


CUBE_BOX_URDF_ASSET_BAD = 'contactnets_cube_bad_init.urdf'
CUBE_BOX_URDF_ASSET_SMALL = 'contactnets_cube_small_init.urdf'
CUBE_MESH_URDF_ASSET_SMALL = 'contactnets_cube_mesh_small_init.urdf'
ELBOW_BOX_URDF_ASSET_BAD = 'contactnets_elbow_bad_init.urdf'
ELBOW_BOX_URDF_ASSET_SMALL = 'contactnets_elbow_small_init.urdf'
ELBOW_MESH_URDF_ASSET_SMALL = 'contactnets_elbow_mesh_small_init.urdf'
CUBE_BOX_WRONG_URDFS = {'bad': CUBE_BOX_URDF_ASSET_BAD,
                        'small': CUBE_BOX_URDF_ASSET_SMALL}
CUBE_MESH_WRONG_URDFS = {'small': CUBE_MESH_URDF_ASSET_SMALL}
ELBOW_BOX_WRONG_URDFS = {'bad': ELBOW_BOX_URDF_ASSET_BAD,
                         'small': ELBOW_BOX_URDF_ASSET_SMALL}
ELBOW_MESH_WRONG_URDFS = {'small': ELBOW_MESH_URDF_ASSET_SMALL}
WRONG_BOX_URDFS = {CUBE_SYSTEM: CUBE_BOX_WRONG_URDFS,
                   ELBOW_SYSTEM: ELBOW_BOX_WRONG_URDFS}
WRONG_MESH_URDFS = {CUBE_SYSTEM: CUBE_MESH_WRONG_URDFS,
                    ELBOW_SYSTEM: ELBOW_MESH_WRONG_URDFS}
WRONG_URDFS_BY_GEOM_THEN_SYSTEM = {MESH_TYPE: WRONG_MESH_URDFS,
                                   POLYGON_TYPE: WRONG_MESH_URDFS,
                                   BOX_TYPE: WRONG_BOX_URDFS}

REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))

# Data configuration.
DT = 0.0068

# Generation configuration.
CUBE_X_0 = torch.tensor(
    [0., 0.0524 + 0.02, 0.,# Cube Q in 2D (x, z, theta)
#     1., 0., 0., 0., 0., 0., 0.5, # Robot Floating Base Q
     0.5, #0., 0.0524, # Robot finger_0 Q 1D (x)
     -0.5, #0., 0.0524, # Robot finger_1 Q 1D (x)
     0., 0., 0., # Cube V in 2D (dx, dz, dtheta)
#     0., 0., 0., 0., 0., -.075, # Robot Floating Base V
     0., #0., 0., # Robot V finger_0 1D (dx)
     0., #0., 0., # Robot V finger_1 1D (dx)
     ])
ROBOT_DESIRED = np.array(
    [0., 0., # 0., 0.0195 + 0.0524, # Desired Robot Q (1d)
     0., 0., # 0., 0., # Desired Robot V (1d)
    ])
ELBOW_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, np.pi, 0., 0., 0., 0., 0., -.075, 0.])
ASYMMETRIC_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
X_0S = {CUBE_SYSTEM: CUBE_X_0, ELBOW_SYSTEM: ELBOW_X_0,
        ASYMMETRIC_SYSTEM: ASYMMETRIC_X_0}
CUBE_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, 6., 6., 6., 1.5, 1.5, .075
])
ELBOW_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, np.pi, 6., 6., 6., 1.5,
    1.5, .075, 6.
])
ASYMMETRIC_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, 6., 6., 6., 1.5, 1.5, .075
])
SAMPLER_RANGES = {
    CUBE_SYSTEM: CUBE_SAMPLER_RANGE,
    ELBOW_SYSTEM: ELBOW_SAMPLER_RANGE,
    ASYMMETRIC_SYSTEM: ASYMMETRIC_SAMPLER_RANGE
}
TRAJECTORY_LENGTHS = {CUBE_SYSTEM: 300, ELBOW_SYSTEM: 120, ASYMMETRIC_SYSTEM: 80}

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
ELBOW_LR = 1e-3
ASYMMETRIC_LR = 1e-3
LRS = {CUBE_SYSTEM: CUBE_LR, ELBOW_SYSTEM: ELBOW_LR,
       ASYMMETRIC_SYSTEM: ASYMMETRIC_LR}
CUBE_WD = 0.0
ELBOW_WD = 0.0  #1e-4
ASYMMETRIC_WD = 0.0
WDS = {CUBE_SYSTEM: CUBE_WD, ELBOW_SYSTEM: ELBOW_WD,
       ASYMMETRIC_SYSTEM: ASYMMETRIC_WD}
DEFAULT_WEIGHT_RANGE = (1e-2, 1e2)
EPOCHS = 200            # change this (originally 500)
PATIENCE = EPOCHS       # change this (originally EPOCHS)

WANDB_DEFAULT_PROJECT = 'dair_pll-examples'


def main(storage_folder_name: str = "",
         run_name: str = "",
         system: str = CUBE_SYSTEM,
         source: str = SIM_SOURCE,
         structured: bool = True,
         contactnets: bool = True,
         geometry: str = BOX_TYPE,
         regenerate: bool = False,
         dataset_size: int = 1,
         inertia_params: int = 1,
         true_sys: bool = True,
         wandb_project: str = WANDB_DEFAULT_PROJECT,
         w_pred: float = 1e0,
         w_comp: float = 1e0,
         w_diss: float = 1e0,
         w_pen: float = 2e1,
         w_res: float = 1e0,
         w_res_w: float = 1e0,
         w_dev: float = 2e1,
         do_residual: bool = False,
         g_frac: float = 1.0):
    """Execute ContactNets basic example on a system.

    Args:
        storage_folder_name: name of outer storage directory.
        run_name: name of experiment run.
        system: Which system to learn.
        source: Where to get data from.
        contactnets: Whether to use ContactNets or prediction loss.
        geometry: How to represent geometry (box, mesh, or polygon).
        regenerate: Whether save updated URDF's each epoch.
        dataset_size: Number of trajectories for train/val/test.
        inertia_params: What inertial parameters to learn.
        true_sys: Whether to start with the "true" URDF or poor initialization.
        wandb_project: What W&B project to store results under.
        w_pred: Weight of prediction term in ContactNets loss.
        w_comp: Weight of complementarity term in ContactNets loss.
        w_diss: Weight of dissipation term in ContactNets loss.
        w_pen: Weight of penetration term in ContactNets loss.
        w_res: Weight of residual regularization term in loss.
        w_dev: Weight of deviation from measured contact forces term in loss.
        do_residual: Whether to add residual physics block.
        g_frac: Fraction of gravity to use with initial model.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    torch.set_default_device('cuda')

    # Unpack inertia bitmask
    inertia_mode = InertiaLearn(
        mass = bool((int(inertia_params) // 1) % 2),
        com = bool((int(inertia_params) // 2) % 2),
        inertia = bool((int(inertia_params) // 4) % 2),
    )

    print(f'Starting test under \'{storage_folder_name}\' ' \
         + f'with name \'{run_name}\':' \
         + f'\n\tPerforming on system: {system} \n\twith source: {source}' \
         + f'\n\twith structured parameterization: {structured}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\twith geometry represented as: {geometry}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\tinertia learning mode: {inertia_params} == {inertia_mode}' \
         + f'\n\tloss weights (pred, comp, diss, pen, res, res_w, dev): ' \
         + f'({w_pred}, {w_comp}, {w_diss}, {w_pen}, {w_res}, {w_res_w}, {w_dev})' \
         + f'\n\twith residual: {do_residual}' \
         + f'\n\tand starting with provided true_sys={true_sys}' \
         + f'\n\twith gravity fraction (if gravity): {g_frac}')

    simulation = source == SIM_SOURCE
    dynamic = source == DYNAMIC_SOURCE

    storage_name = os.path.join(REPO_DIR, 'results', storage_folder_name)

    # If this script is used in conjuction with pll_manager.py, then the file
    # management is taken care of there.

    print(f'\nStoring data at    {file_utils.data_dir(storage_name)}')
    print(f'Storing results at {file_utils.run_dir(storage_name, run_name)}')

    # Next, build the configuration of the learning experiment.

    # If starting with true system, no need to train, since we probably just
    # want to generate statistics.
    num_epochs = 0 if true_sys else EPOCHS

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig(lr=Float(LRS[system]),
                                       wd=Float(WDS[system]),
                                       patience=PATIENCE,
                                       epochs=num_epochs,
                                       batch_size=Int(int(TRAJECTORY_LENGTHS[system])))

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    # first, select urdfs
    urdf_asset = TRUE_URDFS[system][geometry]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf, 'robot': file_utils.get_asset("spherebot.urdf")}
    bad_init_urdfs = {system: file_utils.get_asset(CUBE_BOX_URDF_ASSET_BAD), 'robot': file_utils.get_asset("spherebot.urdf")}

    additional_system_builders = (["dair_pll.drake_utils.pid_controller_builder"], [{"desired_state": ROBOT_DESIRED, "kp": 2.0, "kd": 100.0}])

    base_config = DrakeSystemConfig(urdfs=urdfs, 
        additional_system_builders=additional_system_builders[0], 
        additional_system_kwargs=additional_system_builders[1],
        use_meshcat=False
    )

    # how to slice trajectories into training datapoints
    # Give robot_state from previous and next time steps
    # Give actuation and contact forces simulated from previous time step
    # NOTE: Simulation goes (calc net actuation/forces -> calc next state), so
    # next state's net_actuation / contact_forces are from the previous time step.
    # TODO: HACK "time" is needed to index into predicted trajectory
    slice_config = TrajectorySliceConfig(
        his_state_keys = ["robot_state", "net_actuation", "contact_forces", "time"],
        pred_state_keys = ["robot_state", "time"],
        shuffle = False,
    )


    # Describes configuration of the data
    data_config = DataConfig(dt=DT,
                             train_fraction=1.0,
                             valid_fraction=0.0,
                             test_fraction=0.0,
                             slice_config=slice_config,
                             update_dynamically=False)

    if structured:
        loss = MultibodyLosses.TACTILENET_LOSS if contactnets else \
               MultibodyLosses.PREDICTION_LOSS

        learnable_config = MultibodyLearnableSystemConfig(
            urdfs=bad_init_urdfs, loss=loss, inertia_mode=inertia_mode,
            constant_bodies = ["finger_0", "finger_1"],
            w_pred=w_pred,
            w_comp = Float(w_comp, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            w_diss = Float(w_diss, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            w_pen = Float(w_pen, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            w_res = Float(w_res, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            w_res_w = Float(w_res_w, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            w_dev = Float(w_dev, log=True, distribution=DEFAULT_WEIGHT_RANGE),
            do_residual=do_residual, represent_geometry_as=geometry,
            # TODO: Re-add
            # randomize_initialization = not true_sys, 
            randomize_initialization = False, 
            g_frac=g_frac,
            # Use the same additional system builders as the base system
            additional_system_builders=additional_system_builders[0], 
            additional_system_kwargs=additional_system_builders[1],
            use_meshcat=True,
        )

    else:
        learnable_config = DeepLearnableSystemConfig(
            layers=4, hidden_size=256,
            nonlinearity=torch.nn.Tanh, model_constructor=MLP)

    # Combines everything into config for entire experiment.
    experiment_config = DrakeMultibodyLearnableTactileExperimentConfig(
        trajectory_model_name = system,
        data_config=data_config,
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        storage=storage_name,
        run_name=run_name,
        run_wandb=True,
        wandb_project=wandb_project,
        full_evaluation_period=EPOCHS if dynamic else 1,
        update_geometry_in_videos=True  # ignored for deep learnable experiments
    )

    # Make experiment.
    experiment = DrakeMultibodyLearnableTactileExperiment(experiment_config)

    # Prepare data.
    x_0 = X_0S[system]

    # Simulate one trajectory
    #experiment.get_base_system().simulate(x_0.reshape(1, -1), experiment.get_base_system().carry_callback(), 3000)

    # For simulation, specify the following:
    data_generation_config = DataGenerationConfig(
        dt=DT,
        # timestep
        n_pop=dataset_size,
        # How many trajectories to simulate
        trajectory_length=TRAJECTORY_LENGTHS[system],
        # trajectory length
        x_0=x_0,
        # A nominal initial state
        sampler_type=ConstantSampler,
        # use uniform distribution to sample ``x_0``
        sampler_kwargs={"x_0": x_0},
        # Other arguments for the smapler
        noiser_type=None,
        # Distribution of noise in trajectory data (No Noise).
        storage=storage_name
        # where to store trajectories
    )

    data_generation_system = experiment.get_base_system()

    from pydrake.multibody.plant import MultibodyPlant
    def carry_callback(keys: List[str], plant: MultibodyPlant) -> Tensor:
        carry = TensorDict({}, [1])
        for key in keys:
            subkeys = tuple(key.split("."))
            size = 1
            if subkeys[0] == "contact_forces":
                size = 3
            else:
                size = plant.GetOutputPort(subkeys[0]).size()
            carry.set(tuple(key.split(".")), torch.zeros(1, size))
        return carry

    from functools import partial
    data_generation_system.set_carry_sampler(partial(carry_callback, keys=["net_actuation", "robot_state", "contact_forces.finger_0", "contact_forces.finger_1"], plant=data_generation_system.plant_diagram.plant))

    generator = ExperimentDatasetGenerator(
        data_generation_system, data_generation_config)
    print(f'Generating (or getting existing) simulation trajectories.\n')
    generator.generate()

    # Test Data Loading
    #from dair_pll.dataset_management import ExperimentDataManager
    #edm = ExperimentDataManager(
    #            storage_name, data_config)
    #train, val, test = edm.get_updated_trajectory_sets()

    # Trains system and saves final results.
    print(f'\nTraining the model.')
    learned_system, stats = experiment.generate_results(
        regenerate_callback if regenerate else default_epoch_callback)

    input(f'Done!')




@click.command()
@click.argument('storage_folder_name')
@click.argument('run_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--structured/--end-to-end',
              default=True,
              help="whether to train structured parameters or deep network.")
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--geometry',
              type=click.Choice(GEOMETRY_TYPES, case_sensitive=True),
              default=BOX_TYPE,
              help="how to represent geometry.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--dataset-size',
              default=1,
              help="dataset size")
@click.option('--inertia-params',
              type=click.IntRange(0, 7),
              default=1,
              help="Bitmap of what inertia params to learn: inertia-com-mass (e.g. 0 == none, 1 == mass only, 7 == all)")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--wandb-project',
              type = str,
              default=WANDB_DEFAULT_PROJECT,
              help="what W&B project to save results under.")
@click.option('--w-pred',
              type=float,
              default=2e1,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=2e1,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=2e1,
              help="weight of penetration term in ContactNets loss")
@click.option('--w-res',
              type=float,
              default=1e0,
              help="weight of residual norm regularization term in loss")
@click.option('--w-res-w',
              type=float,
              default=1e0,
              help="weight of residual weight regularization term in loss")
@click.option('--w-dev',
              type=float,
              default=2e4,
              help="weight of deviation from measured contact forces in ContactNets loss")
@click.option('--residual/--no-residual',
              default=False,
              help="whether to include residual physics or not.")
@click.option('--g-frac',
              type=float,
              default=1e0,
              help="fraction of gravity constant to use.")
def main_command(storage_folder_name: str, run_name: str, system: str,
                 source: str, structured: bool, contactnets: bool,
                 geometry: str, regenerate: bool, dataset_size: int,
                 inertia_params: str, true_sys: bool,
                 wandb_project: str, w_pred: float, w_comp: float,
                 w_diss: float, w_pen: float, w_res: float, w_res_w: float,
                 w_dev: float,
                 residual: bool, g_frac: float):
    """Executes main function with argument interface."""
    assert storage_folder_name is not None
    assert run_name is not None

    main(storage_folder_name, run_name, system, source, structured, contactnets,
         geometry, regenerate, dataset_size, inertia_params,
         true_sys, wandb_project, w_pred, w_comp, w_diss, w_pen, w_res, w_res_w, w_dev,
         residual, g_frac)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
