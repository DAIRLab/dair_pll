"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import time
from typing import cast

import sys
import pdb

import click
import numpy as np
import torch
from torch import Tensor
import pickle
import git

from dair_pll import file_utils
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperiment
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import SupervisedLearningExperimentConfig
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.system import System



REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))



def main(storage_folder_name: str = "", run_name: str = ""):
    """Restart a ContactNets experiment run.

    Args:
        storage_folder_name: name of outer storage directory.
        run_name: name of experiment run.
    """
    storage_name = os.path.join(REPO_DIR, 'results', storage_folder_name)

    # Combines everything into config for entire experiment.
    experiment_config = file_utils.load_configuration(storage_name, run_name)
    print(f'Loaded original experiment configuration.')

    # Makes experiment.
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    def regenerate_callback(epoch: int, learned_system: System,
                            train_loss: Tensor,
                            best_valid_loss: Tensor) -> None:
        default_epoch_callback(epoch, learned_system, train_loss,
                               best_valid_loss)
        cast(MultibodyLearnableSystem, learned_system).generate_updated_urdfs()

    # Trains system and saves final results.
    _, _, learned_system = experiment.train(regenerate_callback)

    learned_system = cast(MultibodyLearnableSystem, learned_system)

    # Save the final urdf.
    print(f'\nSaving the final learned parameters.')
    learned_system.generate_updated_urdfs()

    # Save final statistics.
    print(f'\nSaving the final performance statistics.')
    stats = experiment.get_results(default_epoch_callback)
    print(f'Done!')



@click.command()
@click.argument('storage_folder_name')
@click.argument('run_name')
def main_command(storage_folder_name: str, run_name: str):
    """Executes main function with argument interface."""
    assert storage_folder_name is not None
    assert run_name is not None

    main(storage_folder_name, run_name)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
