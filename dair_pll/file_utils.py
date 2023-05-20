"""Utility functions for managing saved files for training models.

File system is organized around a "storage" directory associated with data and
training runs. The functions herein can be used to return absolute paths of and
summary information about the content of this directory.
"""
import glob
import json
import os
import random
import pickle
from os import path
from typing import List, Callable, BinaryIO, Any, TextIO, Optional

from dair_pll.experiment_config import SupervisedLearningExperimentConfig

TRAJ_EXTENSION = '.pt'  # trajectory file
HYPERPARAMETERS_EXTENSION = '.json'  # hyperparameter set file
STATS_EXTENSION = '.pkl'  # experiment statistics
CONFIG_EXTENSION = '.pkl'
CHECKPOINT_EXTENSION = '.pt'
DATA_SUBFOLDER_NAME = 'data'
LEARNING_DATA_SUBFOLDER_NAME = 'learning'
GROUND_TRUTH_DATA_SUBFOLDER_NAME = 'ground_truth'
RUNS_SUBFOLDER_NAME = 'runs'
STUDIES_SUBFOLDER_NAME = 'studies'
URDFS_SUBFOLDER_NAME = 'urdfs'
WANDB_SUBFOLDER_NAME = 'wandb'
TRAJECTORY_GIF_DEFAULT_NAME = 'trajectory.gif'
FINAL_EVALUATION_NAME = f'statistics{STATS_EXTENSION}'
HYPERPARAMETERS_FILENAME = f'optimal_hyperparameters{HYPERPARAMETERS_EXTENSION}'
CONFIG_FILENAME = f'config{CONFIG_EXTENSION}'
CHECKPOINT_FILENAME = f'checkpoint{CHECKPOINT_EXTENSION}'
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
LOG_DIR = assure_created(os.path.join(MAIN_DIR, 'logs'))
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
    storage_directories = [data_dir, all_runs_dir,
                           all_studies_dir]  # type: List[Callable[[str],str]]

    for directory in storage_directories:
        assure_created(directory(storage_name))


def list_file_nums(path: str) -> List[int]:
    file_nums = []
    for file_name in os.listdir(path):
        file_nums.append(int(file_name.split('.')[0]))
    return file_nums


def import_data_to_storage(storage_name: str, import_data_dir: str,
                           num: int = None) -> List[int]:
    """Import data in external folder into data directory.

    Args:
        storage_name: Name of storage for data import.
        import_data_dir: Directory to import data from.
        num: Number of trajectories to import.  If `None`, imports all in the
          directory.
    """
    output_directories = [
        ground_truth_data_dir(storage_name),
        learning_data_dir(storage_name)
    ]
    data_traj_count = get_numeric_file_count(import_data_dir, TRAJ_EXTENSION)
    run_indices = [i for i in range(data_traj_count)]

    target_traj_number = data_traj_count if num is None else \
                         min(num, data_traj_count)

    # Check if data is synchronized already.
    for output_directory in output_directories:
        storage_traj_count = get_numeric_file_count(output_directory,
                                                    TRAJ_EXTENSION)

        # Overwrite in case of any discrepancies.
        if storage_traj_count != target_traj_number:

            # Copy entire directory if all trajectories are desired.
            if target_traj_number == data_traj_count:
                for output_dir in output_directories:
                    os.system(f'rm -r {output_dir}')
                    os.system(f'cp -r {import_data_dir} {output_dir}')

            # Copy a random subset of trajectories if want a smaller number.
            else:
                random.shuffle(run_indices)
                run_indices = run_indices[:target_traj_number]
                for output_dir in output_directories:
                    os.system(f'rm -r {output_dir}')
                    os.system(f'mkdir {output_dir}')

                    # Copy over a random selection of trajectories, numbering
                    # from 0.
                    for i, run in zip(range(target_traj_number), run_indices):
                        os.system(f'cp {import_data_dir}/{run}.pt ' + \
                                  f'{output_dir}/{i}.pt')

            # Can terminate outer loop over output directories since all output
            # directories are written to at once.
            return


def storage_dir(storage_name: str) -> str:
    """Absolute path of storage directory"""
    # return assure_created(os.path.join(RESULTS_DIR, storage_name))
    return assure_created(storage_name)


def data_dir(storage_name: str) -> str:
    """Absolute path of data folder."""
    return assure_created(
        path.join(storage_dir(storage_name), DATA_SUBFOLDER_NAME))


def learning_data_dir(storage_name: str) -> str:
    """Absolute path of folder for data preprocessed for
    training/validation."""
    return assure_created(
        path.join(data_dir(storage_name), LEARNING_DATA_SUBFOLDER_NAME))


def ground_truth_data_dir(storage_name: str) -> str:
    """Absolute path of folder for raw unprocessed trajectories."""
    return assure_created(
        path.join(data_dir(storage_name), GROUND_TRUTH_DATA_SUBFOLDER_NAME))


def all_runs_dir(storage_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(
        path.join(storage_dir(storage_name), RUNS_SUBFOLDER_NAME))


def all_studies_dir(storage_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(
        path.join(storage_dir(storage_name), STUDIES_SUBFOLDER_NAME))


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


def get_trajectory_count(trajectory_dir: str):
    """Count number of trajectories on disk in given directory."""
    return get_numeric_file_count(trajectory_dir, TRAJ_EXTENSION)


def trajectory_file(trajectory_dir: str, num_trajectory: int) -> str:
    """Absolute path of specific trajectory in storage"""
    return path.join(trajectory_dir,
                     f'{num_trajectory}{TRAJ_EXTENSION}')


def run_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of run-specific storage folder."""
    return assure_created(path.join(all_runs_dir(storage_name), run_name))


def get_trajectory_video_filename(storage_name: str, run_name: str) -> str:
    """Return the filepath of the temporary rollout video gif."""
    return path.join(run_dir(storage_name, run_name),
                     TRAJECTORY_GIF_DEFAULT_NAME)


def get_learned_urdf_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of learned model URDF storage directory."""
    return assure_created(
        path.join(run_dir(storage_name, run_name), URDFS_SUBFOLDER_NAME))


def wandb_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(
        path.join(run_dir(storage_name, run_name), WANDB_SUBFOLDER_NAME))


def get_evaluation_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment run statistics file."""
    return path.join(run_dir(storage_name, run_name), FINAL_EVALUATION_NAME)


def get_configuration_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment configuration."""
    return path.join(run_dir(storage_name, run_name), CONFIG_FILENAME)


def get_model_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment configuration."""
    return path.join(run_dir(storage_name, run_name), CHECKPOINT_FILENAME)


def get_geometrically_accurate_urdf(urdf_name: str) -> str:
    """Replaces urdf_name with the name of a urdf corresponding to the same
    system with accurate geometry.

    Args:
        urdf_name: Name of a URDF file located in ``ASSET_DIR``

    Returns:
        The name of a URDF file located in ``ASSET_DIR`` that contains the true
        geometry of the system.
    """
    URDF_MAP = {'contactnets_cube_bad_init.urdf': 'contactnets_cube.urdf',
                'contactnets_cube_small_init.urdf': 'contactnets_cube.urdf',
                'contactnets_cube.urdf': 'contactnets_cube.urdf',
                'contactnets_cube_mesh.urdf': 'contactnets_cube_mesh.urdf',
                'contactnets_elbow_bad_init.urdf': 'contactnets_elbow.urdf',
                'contactnets_elbow_small_init.urdf': 'contactnets_elbow.urdf',
                'contactnets_elbow.urdf': 'contactnets_elbow.urdf',
                'contactnets_elbow_mesh.urdf': 'contactnets_elbow_mesh.urdf',
                'contactnets_asymmetric.urdf': 'contactnets_asymmetric.urdf'}
    base_name = urdf_name.split('/')[-1]

    if base_name in URDF_MAP.keys():
        return get_asset(URDF_MAP[base_name])
    else:
        print(f'Could not find geometrically accurate version of {base_name};' +
              f' defaulting to using it directly.')
        return get_asset(base_name)


def study_dir(storage_name: str, study_name: str) -> str:
    """Absolute path of study-specific storage folder."""
    return assure_created(path.join(all_studies_dir(storage_name), study_name))


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


def load_binary(filename: str, load_callback: Callable[[BinaryIO], Any]) -> Any:
    """Load binary file"""
    with open(filename, 'rb') as file:
        value = load_callback(file)
    return value


def load_string(filename: str,
                load_callback: Optional[Callable[[TextIO], Any]] = None) -> Any:
    """Load text file"""
    with open(filename, 'r', encoding='utf8') as file:
        if load_callback:
            value = load_callback(file)
        else:
            value = file.read()
    return value


def save_binary(filename: str, value: Any,
                save_callback: Callable[[Any, BinaryIO], None]) -> None:
    """Save binary file."""
    with open(filename, 'wb') as file:
        save_callback(value, file)


def save_string(
        filename: str,
        value: Any,
        save_callback: Optional[Callable[[Any, TextIO], None]] = None) -> None:
    """Save text file."""
    with open(filename, 'w', encoding='utf8') as file:
        if save_callback:
            save_callback(value, file)
        else:
            assert isinstance(value, str)
            file.write(value)


def load_configuration(storage_name: str, run_name: str) -> \
        SupervisedLearningExperimentConfig:
    """Load configuration file."""
    configuration_filename = get_configuration_filename(storage_name, run_name)
    configuration = load_binary(configuration_filename, pickle.load)
    assert isinstance(configuration, SupervisedLearningExperimentConfig)
    return configuration


def save_configuration(storage_name: str, run_name: str,
                       config: SupervisedLearningExperimentConfig) -> None:
    """Save configuration file."""
    configuration_filename = get_configuration_filename(storage_name, run_name)
    save_binary(configuration_filename, config, pickle.dump)


def load_evaluation(storage_name: str, run_name: str) -> Any:
    """Load evaluation file."""
    evaluation_filename = get_evaluation_filename(storage_name, run_name)
    return load_binary(evaluation_filename, pickle.load)


def save_evaluation(storage_name: str, run_name: str, evaluation: Any) -> None:
    """Save evaluation file."""
    evaluation_filename = get_evaluation_filename(storage_name, run_name)
    save_binary(evaluation_filename, evaluation, pickle.dump)


def load_hyperparameters(storage_name: str, study_name: str) -> Any:
    """Load hyperparameter file."""
    hyperparameter_filename = get_hyperparameter_filename(
        storage_name, study_name)
    return load_string(hyperparameter_filename, json.load)


def save_hyperparameters(storage_name: str, study_name: str,
                         hyperparameters: Any) -> None:
    """Save hyperparameter file."""
    hyperparameter_filename = get_hyperparameter_filename(
        storage_name, study_name)
    save_string(hyperparameter_filename, hyperparameters, json.dump)
