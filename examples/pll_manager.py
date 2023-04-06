"""Manager for starting GRASP cluster PLL jobs."""
import os
import os.path as op
import git
import click
import subprocess
import time
import pdb
import fnmatch
from typing import List, Optional

from dair_pll import file_utils


# Possible categories for automatic run name generation.
TEST = 'test'
DEV = 'dev'
CATEGORIES = [TEST, DEV]

# Possible run name regex patterns.
CUBE_TEST_PATTERN = 'tc??'
ELBOW_TEST_PATTERN = 'te??'
CUBE_DEV_PATTERN =  'dc??'
ELBOW_DEV_PATTERN = 'de??'

# Possible systems on which to run PLL
CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM]

# Possible dataset types
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]

# Possible results management options
OVERWRITE_DATA_AND_RUNS = 'data_and_runs'
OVERWRITE_SINGLE_RUN_KEEP_DATA = 'run'
OVERWRITE_NOTHING = 'nothing'
OVERWRITE_RESULTS = [OVERWRITE_DATA_AND_RUNS,
                     OVERWRITE_SINGLE_RUN_KEEP_DATA,
                     OVERWRITE_NOTHING]

# Possible inertial parameterizations to learn for the elbow system.
# The options are:
# 0 - learn no inertial parameters (0 for elbow)
# 1 - learn the mass of second and beyond links (1 for elbow)
# 2 - learn the locations of all links' centers of mass (6 for elbow)
# 3 - learn second and beyond masses and all centers of mass (7 for elbow)
# 4 - learn all parameters except mass of first link (19 for elbow)
INERTIA_PARAM_CHOICES = ['0', '1', '2', '3', '4']
INERTIA_PARAM_DESCRIPTIONS = [
    'learn no inertial parameters (0 * n_bodies)',
    'learn only masses and not the first mass (n_bodies - 1)',
    'learn only locations of centers of mass (3 * n_bodies)',
    'learn masses (except first) and centers of mass (4 * n_bodies - 1)',
    'learn all parameters (except first mass) (10 * n_bodies - 1)']
INERTIA_PARAM_OPTIONS = ['none', 'masses', 'CoMs', 'CoMs and masses', 'all']


def create_instance(storage_folder_name: str, run_name: str, system: str,
                    source: str, contactnets: bool, box: bool, regenerate: bool,
                    dataset_size: int, local: bool, inertia_params: str,
                    true_sys: bool):
    print(f'Generating experiment {storage_folder_name}/{run_name}')

    base_file = 'startup'
    out_file = f'{base_file}_{storage_folder_name}_{run_name}.bash'

    # use local template if running locally
    base_file += '_local.bash' if local else '.bash'

    base_file = op.join(op.dirname(__file__), base_file)
    out_file = op.join(op.dirname(__file__), out_file)


    script = open(base_file, 'r').read()

    script = script.replace('{storage_folder_name}', storage_folder_name)
    script = script.replace('{run_name}', run_name)
    
    train_options = f' --system={system} --source={source}' + \
                    f' --dataset-size={dataset_size}' + \
                    f' --inertia-params={inertia_params}'
    train_options += ' --contactnets' if contactnets else ' --prediction'
    train_options += ' --box' if box else ' --mesh'
    train_options += ' --regenerate' if regenerate else ' --no-regenerate'
    train_options += ' --true-sys' if true_sys else ' --wrong-sys'

    script = script.replace('{train_args}', train_options)

    repo = git.Repo(search_parent_directories=True)
    git_folder = repo.git.rev_parse("--show-toplevel")
    git_folder = op.normpath(git_folder)
    script = script.replace('{pll_dir}', git_folder)

    commit_hash = repo.head.object.hexsha
    script = script.replace('{hash}', commit_hash)

    with open(out_file, "w") as of:
        of.write(script)

    train_cmd = ['bash', out_file] if local else ['sbatch', out_file]
    print(f'Creating and queuing {out_file}')
    ec = subprocess.run(train_cmd)
    print(f'Queued file.')


def get_slurm_from_instances(instances: List[str], prefix='pll'):
    jobids = []
    for instance in instances:
        cmd = ['squeue', f'--user={os.getlogin()}', '--format', '%.18i',
               '--noheader', '--name', f'{prefix}_{instance}']
        ps = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        ps.wait()
        (out, err) = ps.communicate()
        out = out.decode('unicode-escape')
        out = ''.join(i for i in out if i.isdigit())
        if len(out) > 0:
            jobids.append(out)
    return jobids


def attach_tb(name: str, local: bool = False):
    print(f'Working to attach tensorboard...')
    repo = git.Repo(search_parent_directories=True)
    git_folder = repo.git.rev_parse("--show-toplevel")
    git_folder = op.normpath(git_folder)

    if not local:
        tb_script = op.join(op.dirname(__file__), 'tensorboard.bash')
        tb_logfile = op.join(git_folder, 'logs', 'tensorboard_' + name + '.txt')
        os.system(f'rm {tb_logfile}')
        tboard_cmd = ['sbatch', f'--output={tb_logfile}', \
            f'--job-name=tb_{name}'.format(tb_logfile), tb_script, \
            op.join(git_folder, 'results', name, 'tensorboard'), name]
        ec = subprocess.run(tboard_cmd)

        # wait for and report tensorboard url
        print('Waiting on TensorBoard startup ...')
        lines = []
        while not op.exists(tb_logfile):
            time.sleep(0.1)
        while len(lines) < 1:
            with open(tb_logfile) as f:
                lines = f.readlines()
            time.sleep(1.0)
        print(f'\nTensorBoard running on {lines[0]}\n')

    else:
        tb_script = op.join(op.dirname(__file__), 'tensorboard_local.bash')
        tboard_cmd = ['bash', tb_script,
                      op.join(git_folder, 'results', name, 'tensorboard'), name]
        print(f'Starting local TensorBoard command:  {tboard_cmd}')
        ec = subprocess.run(tboard_cmd)

    print(f'Ran tensorboard command.')


def take_care_of_file_management(overwrite: str, storage_name: str,
                                 run_name: str) -> None:
    """Take care of file management.

    Todo:
        This isn't fool-proof.  Even if overwrite is set to nothing or to keep
        data, the data gets overwritten if the dataset size is different.
    """
    if overwrite == OVERWRITE_DATA_AND_RUNS:
        os.system(f'rm -r {file_utils.storage_dir(storage_name)}')

    elif overwrite == OVERWRITE_SINGLE_RUN_KEEP_DATA:
        os.system(f'rm -r {file_utils.run_dir(storage_name, run_name)}')

    elif overwrite == OVERWRITE_NOTHING:
        # Do nothing.  If the experiment and run did not already exist, it will
        # make it.  Otherwise it will continue the experiment run.
        pass

    else:
        raise NotImplementedError('Choose 1 of 3 result overwriting options')


def experiment_class_command(category: str, run_name: str, system: str,
    contactnets: bool, box: bool, regenerate: bool, local: bool,
    inertia_params: str, true_sys: bool, overwrite: str):
    """Executes main function with argument interface."""

    assert category in CATEGORIES

    def get_run_name_pattern(category, system):
        run_name_pattern = 't' if category == TEST else 'd'
        run_name_pattern += 'c' if system == CUBE_SYSTEM else 'e'
        run_name_pattern += '??'
        return run_name_pattern

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)

    commits_ahead = sum(1 for _ in repo.iter_commits('origin/main..main'))
    if commits_ahead > 0:
        if not click.confirm(f'You are {commits_ahead} commits ahead of' \
                             + f' main branch, continue?'):
            raise RuntimeError('Make sure you have pushed commits!')

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if len(changed_files) > 0:
        print('Uncommitted changes to:')
        print(changed_files)
        if not click.confirm('Continue?'):
            raise RuntimeError('Make sure you have committed changes!')

    # First, take care of data management and how to keep track of results.
    storage_folder_name = f'{category}_{system}'
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', storage_folder_name)
    if run_name is None:
        last_run_num = int(os.listdir(storage_name)[-1].split('t')[-1])
        run_name = category[0]
        run_name += 'c' if system==CUBE_SYSTEM else 'e'
        run_name += str(last_run_num+1).zfill(2)
    run_name_pattern = get_run_name_pattern(category, system)
    assert fnmatch.fnmatch(run_name, run_name_pattern)

    print(f'\nOverwrite set to {overwrite}.')

    if op.isdir(op.join(storage_name, 'runs', run_name)):
        if not click.confirm(f'\nPause!  Experiment \'' \
                             + f'{storage_folder_name}/{run_name}\'' \
                             + f' already taken, continue?'):
            raise RuntimeError('Choose a new run name next time.')

    take_care_of_file_management(overwrite, storage_name, run_name)
    dataset_size = 4 if category == TEST else 64
    source = SIM_SOURCE

    # Continue creating PLL instance.
    create_instance(storage_folder_name, run_name, system, source, contactnets,
                    box, regenerate, dataset_size, local, inertia_params,
                    true_sys)


@click.group()
def cli():
    pass


@cli.command('create')
@click.argument('storage_folder_name')
@click.argument('run_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--dataset-size',
              default=512,
              help="dataset size")
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
def create_command(storage_folder_name: str, run_name: str, system: str,
                   source: str, contactnets: bool, box: bool, regenerate: bool,
                   dataset_size: int, local: bool, inertia_params: str,
                   true_sys: bool, overwrite: str):
    """Executes main function with argument interface."""

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)

    commits_ahead = sum(1 for _ in repo.iter_commits('origin/main..main'))
    if commits_ahead > 0:
        if not click.confirm(f'You are {commits_ahead} commits ahead of' \
                             + f' main branch, continue?'):
            raise RuntimeError('Make sure you have pushed commits!')

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if len(changed_files) > 0:
        print('Uncommitted changes to:')
        print(changed_files)
        if not click.confirm('Continue?'):
            raise RuntimeError('Make sure you have committed changes!')

    # First, take care of data management and how to keep track of results.
    assert storage_folder_name is not None
    assert run_name is not None
    assert '-' not in run_name
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', storage_folder_name)

    print(f'\nOverwrite set to {overwrite}.')

    if op.isdir(op.join(storage_name, 'runs', run_name)):
        if not click.confirm(f'\nPause!  Experiment \'' \
                             + f'{storage_folder_name}/{run_name}\'' \
                             + f' already taken, continue?'):
            raise RuntimeError('Choose a new run name next time.')
    elif op.isdir(storage_name):
        dataset_size_in_folder = file_utils.get_numeric_file_count(
            file_utils.learning_data_dir(storage_name))
        if not click.confirm(f'\nPause!  Experiment storage \'' \
                             + f'{storage_folder_name}\'' \
                             + f' already taken with {dataset_size_in_folder}' \
                             + f' dataset size, continue?'):
            raise RuntimeError('Choose a new storage name next time.')

    take_care_of_file_management(overwrite, storage_name, run_name)

    # Check if experiment name was given and if it already exists.  We also
    # don't want hyphens in the name since that's how the sweep instances
    # are created.
    assert storage_folder_name is not None
    assert '-' not in run_name
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', storage_folder_name, run_name)
    if op.isdir(storage_name):
        if not click.confirm(f'\nPause!  Experiment name \'' \
                             + f'{storage_folder_name}/{run_name}\'' \
                             + f' already taken, continue (overwrite)?'):
            raise RuntimeError('Choose a new name next time.')

    # clear the results directory, per user input
    os.system(f'rm -r {file_utils.run_dir(storage_folder_name, run_name)}')

    # Continue creating PLL instance.
    create_instance(storage_folder_name, run_name, system, source, contactnets,
                    box, regenerate, dataset_size, local, inertia_params,
                    true_sys)


@cli.command('test')
@click.argument('run_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
def test_command(run_name: str, system: str, contactnets: bool, box: bool,
                 regenerate: bool, local: bool, inertia_params: str,
                 true_sys: bool, overwrite: str):
    """Executes main function with argument interface."""
    experiment_class_command('test', run_name, system, contactnets, box,
                             regenerate, local, inertia_params, true_sys,
                             overwrite)

@cli.command('dev')
@click.argument('run_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
def dev_command(run_name: str, system: str, contactnets: bool, box: bool,
                regenerate: bool, local: bool, inertia_params: str,
                true_sys: bool, overwrite: str):
    """Executes main function with argument interface."""
    experiment_class_command('dev', run_name, system, contactnets, box,
                             regenerate, local, inertia_params, true_sys,
                             overwrite)


@cli.command('sweep')
@click.argument('sweep_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
def sweep_command(sweep_name: str, system: str, source: str,
                  contactnets: bool, box: bool, regenerate: bool,
                  local: bool, inertia_params: str, true_sys: bool):
    """Starts a series of instances, sweeping over dataset size."""

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)

    commits_ahead = sum(1 for _ in repo.iter_commits('origin/main..main'))
    if commits_ahead > 0:
        if not click.confirm(f'You are {commits_ahead} commits ahead of' \
                             + f' main branch, continue?'):
            raise RuntimeError('Make sure you have pushed commits!')

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if len(changed_files) > 0:
        print('Uncommitted changes to:')
        print(changed_files)
        if not click.confirm('Continue?'):
            raise RuntimeError('Make sure you have committed changes!')

    # Check if experiment name was given and if it already exists.
    assert sweep_name is not None
    assert '-' not in sweep_name
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', f'{sweep_name}-2')
    if op.isdir(storage_name):
        raise RuntimeError(f'It appears the sweep experiment name ' \
                           + f'\'{sweep_name}\' is already taken.  Choose' \
                           + f' a new name next time.')

    # No need to clear the results directory because it should be assured
    # to be empty.

    # Create a PLL instance for every dataset size from 4 to 512 (2^2 to
    # 2^9).
    for dataset_exponent in range(2, 10):
        dataset_size = 2**dataset_exponent
        exp_name = f'{sweep_name}-{dataset_exponent}'
        create_instance(exp_name, system, source, contactnets, box,
                        regenerate, dataset_size, local, inertia_params,
                        true_sys)



@cli.command('detach')
@click.argument('instance')
def detach(instance: str):
    """Deletes Tensorboard task associated with experiment name."""
    jobid = get_slurm_from_instances([instance], prefix='tb')[0]
    os.system(f'scancel {jobid}')


@cli.command('attach')
@click.argument('instance')
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
def attach(instance: str, local: bool):
    """Attaches Tensorboard task to experiment name."""
    attach_tb(instance, local)



if __name__ == '__main__':
    cli()
