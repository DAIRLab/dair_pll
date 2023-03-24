"""Utility functions for managing saved files for training models.

File system is organized around a "storage" directory associated with data and
training runs. The functions herein can be used to return absolute paths of and
summary information about the content of this directory.
"""
import glob
import os
from os import path
from typing import List, Callable

TRAJ_EXTENSION = '.pt'  # trajectory file
HYPERPARAMETERS_EXTENSION = '.json'  # hyperparameter set file
STATS_EXTENSION = '.pkl'  # experiment statistics
DATA_SUBFOLDER_NAME = 'data'
RUNS_SUBFOLDER_NAME = 'runs'
STUDIES_SUBFOLDER_NAME = 'studies'
URDFS_SUBFOLDER_NAME = 'urdfs'
WANDB_SUBFOLDER_NAME = 'wandb'
TRAJECTORY_GIF_DEFAULT_NAME = 'trajectory.gif'
FINAL_EVALUATION_NAME = f'statistics{STATS_EXTENSION}'
HYPERPARAMETERS_FILENAME = f'optimal_hyperparameters{HYPERPARAMETERS_EXTENSION}'
"""str: extensions for saved files"""


def assure_created(directory: str) -> str:
    """Wrapper to put around directory paths which ensure their existence.

    Args:
        directory: Path of directory that may not exist.

    Returns:
        ``directory``, Which is ensured to exist by recursive mkdir.
    """
    directory = path.abspath(directory)
    if not path.exists(directory):
        assure_created(path.dirname(directory))
        os.mkdir(directory)
    return directory


MAIN_DIR = path.dirname(path.dirname(__file__))
ASSETS_DIR = assure_created(os.path.join(MAIN_DIR, 'assets'))
# str: locations of key static directories


def get_asset(asset_file_basename: str) -> str:
    """Gets

    Args:
        asset_file_basename: Basename of asset file located in ``ASSET_DIR``

    Returns:
        Asset's absolute path.
    """
    return os.path.join(ASSETS_DIR, asset_file_basename)


def assure_storage_tree_created(storage_name: str) -> None:
    """Assure that all subdirectories of specified storage are created.

    Args:
        storage_name: name of storage directory.
    """
    storage_directories = [data_dir,
                           all_runs_dir,
                           all_studies_dir]  # type: List[Callable[[str],str]]

    for directory in storage_directories:
        assure_created(directory(storage_name))


def import_data_to_storage(storage_name: str, import_data_dir: str) -> None:
    """Import data in external folder into data directory.

    Args:
        storage_name: Name of storage for data import.
        import_data_dir: Directory to import data from.
    """
    # check if data is synchronized already
    storage_data_dir = data_dir(storage_name)
    storage_traj_count = get_numeric_file_count(storage_data_dir,
                                                TRAJ_EXTENSION)
    data_traj_count = get_numeric_file_count(import_data_dir, TRAJ_EXTENSION)

    # overwrite in case of any discrepancies
    if storage_traj_count != data_traj_count:
        os.system(f'rm -r {storage_data_dir}')
        os.system(f'cp -r {import_data_dir} {storage_data_dir}')


def storage_dir(storage_name: str) -> str:
    """Absolute path of storage directory"""
    # return assure_created(os.path.join(RESULTS_DIR, storage_name))
    return assure_created(storage_name)


def data_dir(storage_name: str) -> str:
    """Absolute path of training/validation/test trajectories directory"""
    return assure_created(path.join(storage_dir(storage_name),
                                    DATA_SUBFOLDER_NAME))


def all_runs_dir(storage_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(path.join(storage_dir(storage_name),
                                    RUNS_SUBFOLDER_NAME))


def all_studies_dir(storage_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(path.join(storage_dir(storage_name),
                                    STUDIES_SUBFOLDER_NAME))


def delete(file_name: str) -> None:
    """Removes file at path specified by ``file_name``"""
    if path.exists(file_name):
        os.remove(file_name)


def get_numeric_file_count(directory: str,
                           extension: str = TRAJ_EXTENSION) -> int:
    """Count number of whole-number-named files.

    If folder ``/fldr`` has contents (7.pt, 11.pt, 4.pt), then::

        get_numeric_file_count("/fldr", ".pt") == 3

    Args:
        directory: Directory to tally file count in
        extension: Extension of files to be counted

    Returns:
        Number of files in specified ``directory`` with specified
        ``extension`` that have an integer basename.
    """
    return len(glob.glob(path.join(directory, './[0-9]*' + extension)))


def get_trajectory_count(storage_name: str):
    """Count number of trajectories on disc in storage directory"""
    return get_numeric_file_count(data_dir(storage_name), TRAJ_EXTENSION)


def trajectory_file(storage_name: str, num_trajectory: int) -> str:
    """Absolute path of specific trajectory in storage"""
    return path.join(data_dir(storage_name),
                     f'{num_trajectory}{TRAJ_EXTENSION}')


def run_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of run-specific storage folder."""
    return assure_created(path.join(all_runs_dir(storage_name),
                                    run_name))


def get_trajectory_video_filename(storage_name: str, run_name: str) -> str:
    """Return the filepath of the temporary rollout video gif."""
    return path.join(run_dir(storage_name, run_name),
                     TRAJECTORY_GIF_DEFAULT_NAME)


def get_final_evaluation_filename(storage_name: str, run_name: str) -> str:
    """Return the filepath of the temporary rollout video gif."""
    return path.join(run_dir(storage_name, run_name),
                     FINAL_EVALUATION_NAME)


def get_learned_urdf_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of learned model URDF storage directory."""
    return assure_created(path.join(run_dir(storage_name, run_name),
                                    URDFS_SUBFOLDER_NAME))


def wandb_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(path.join(run_dir(storage_name, run_name),
                                    WANDB_SUBFOLDER_NAME))


def get_evaluation_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment run statistics file."""
    return path.join(run_dir(storage_name, run_name),
                     FINAL_EVALUATION_NAME)


def study_dir(storage_name: str, study_name: str) -> str:
    """Absolute path of study-specific storage folder."""
    return assure_created(path.join(all_studies_dir(storage_name),
                                    study_name))


def hyperparameter_opt_run_name(study_name: str, trial_number: int) -> str:
    """Experiment run name for hyperparameter optimization trial."""
    return f'{study_name}_hyperparameter_opt_{trial_number}'


def sweep_run_name(study_name: str, sweep_run: int, n_train: int) -> str:
    """Experiment run name for dataset size sweep study."""
    return f'{study_name}_sweep_{sweep_run}_n_train_{n_train}'


def get_hyperparameter_filename(storage_name: str, study_name: str) -> str:
    """Absolute path of optimized hyperparameters for a study"""
    return path.join(study_dir(storage_name, study_name),
                     HYPERPARAMETERS_FILENAME)
