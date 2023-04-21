"""Manager for starting GRASP cluster PLL jobs."""
import os
import os.path as op
import git
import click
import subprocess
import time
import pdb
import fnmatch
import wandb
from typing import List, Optional

from dair_pll import file_utils

from dair_pll.multibody_learnable_system import LOSS_INERTIA_AGNOSTIC, \
    LOSS_PLL_ORIGINAL, LOSS_BALANCED, LOSS_POWER, LOSS_CONTACT_VELOCITY, \
    LOSS_VARIATIONS, LOSS_VARIATION_NUMBERS


# Possible categories for automatic run name generation.
TEST = 'test'
DEV = 'dev'
SWEEP = 'sweep'
HYPERPARAMETER = 'hyperparam'
CATEGORIES = [TEST, DEV, SWEEP, HYPERPARAMETER]

# Possible systems on which to run PLL
CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM]

# Possible run name regex patterns.
CUBE_TEST_PATTERN = 'tc??'
ELBOW_TEST_PATTERN = 'te??'
CUBE_DEV_PATTERN =  'dc??'
ELBOW_DEV_PATTERN = 'de??'
CUBE_SWEEP_PATTERN = 'sc??'
ELBOW_SWEEP_PATTERN = 'se??'
RUN_PREFIX_TO_FOLDER_NAME = {'tc': f'{TEST}_{CUBE_SYSTEM}',
                             'te': f'{TEST}_{ELBOW_SYSTEM}',
                             'dc': f'{DEV}_{CUBE_SYSTEM}',
                             'de': f'{DEV}_{ELBOW_SYSTEM}',
                             'sc': f'{SWEEP}_{CUBE_SYSTEM}',
                             'se': f'{SWEEP}_{ELBOW_SYSTEM}',
                             'hc': f'{HYPERPARAMETER}_{CUBE_SYSTEM}',
                             'he': f'{HYPERPARAMETER}_{ELBOW_SYSTEM}'}

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

# Possible W&B project names.
WANDB_PROJECT_CLUSTER = 'dair_pll-cluster'
WANDB_PROJECT_LOCAL = 'dair_pll-dev'
WANDB_PROJECTS = {True: WANDB_PROJECT_LOCAL,
                  False: WANDB_PROJECT_CLUSTER}

# Possible inertial parameterizations to learn for the elbow system.
# The options are:
# 0 - learn no inertial parameters (0 for elbow)
# 1 - learn the mass of second and beyond links (1 for elbow)
# 2 - learn the locations of all links' centers of mass (6 for elbow)
# 3 - learn second and beyond masses and all centers of mass (7 for elbow)
# 4 - learn all parameters except mass of first link (19 for elbow)
INERTIA_PARAM_CHOICES = [str(i) for i in range(5)]
INERTIA_PARAM_DESCRIPTIONS = [
    'learn no inertial parameters (0 * n_bodies)',
    'learn only masses and not the first mass (n_bodies - 1)',
    'learn only locations of centers of mass (3 * n_bodies)',
    'learn masses (except first) and centers of mass (4 * n_bodies - 1)',
    'learn all parameters (except first mass) (10 * n_bodies - 1)']
INERTIA_PARAM_OPTIONS = ['none', 'masses', 'CoMs', 'CoMs and masses', 'all']

WANDB_NO_GROUP_MESSAGE = \
    'echo "Not exporting WANDB_RUN_GROUP since restarting."'

# Weights to try in hyperparameter search
HYPERPARAMETER_WEIGHTS = [1e-2, 1e-1, 1e0, 1e1, 1e2]


def create_instance(storage_folder_name: str, run_name: str,
                    system: str = CUBE_SYSTEM,
                    source: str = SIM_SOURCE,
                    contactnets: bool = True,
                    box: bool = True,
                    regenerate: bool = True,
                    dataset_size: int = 0,
                    local: bool = False,
                    inertia_params: str = '4',
                    loss_variation: str = '0',
                    true_sys: bool = True,
                    restart: bool = False,
                    wandb_group_id: str = None,
                    w_pred: float = 1e0,
                    w_comp: float = 1e0,
                    w_diss: float = 1e0,
                    w_pen: float = 1e0):
    print(f'Generating experiment {storage_folder_name}/{run_name}')

    if wandb_group_id is None:
        wandb_group_id = '' if restart else \
                         f'{run_name}_{wandb.util.generate_id()}'

    base_file = 'startup'
    out_file = f'{base_file}_{storage_folder_name}_{run_name}.bash'

    # use local template if running locally
    base_file += '_local.bash' if local else '.bash'

    base_file = op.join(op.dirname(__file__), base_file)
    out_file = op.join(op.dirname(__file__), out_file)


    script = open(base_file, 'r').read()

    script = script.replace('{storage_folder_name}', storage_folder_name)
    script = script.replace('{run_name}', run_name)
    script = script.replace('{restart}', 'true' if restart else 'false')
    script = script.replace('{wandb_group_id}', wandb_group_id)

    train_options = ''
    
    if not restart:
        train_options = f' --system={system} --source={source}' + \
                        f' --dataset-size={dataset_size}' + \
                        f' --inertia-params={inertia_params}' + \
                        f' --loss-variation={loss_variation}'
        train_options += ' --contactnets' if contactnets else ' --prediction'
        train_options += ' --box' if box else ' --mesh'
        train_options += ' --regenerate' if regenerate else ' --no-regenerate'
        train_options += ' --true-sys' if true_sys else ' --wrong-sys'
        train_options += f' --wandb-project={WANDB_PROJECTS[local]}'

        if contactnets:
            train_options += f' --w-pred={w_pred}'
            train_options += f' --w-comp={w_comp}'
            train_options += f' --w-diss={w_diss}'
            train_options += f' --w-pen={w_pen}'

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


def check_for_git_updates(repo):
    """Check for git updates."""
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


def experiment_class_command(category: str, run_name: str, system: str,
    contactnets: bool, box: bool, regenerate: bool, local: bool,
    inertia_params: str, loss_variation: str, true_sys: bool, overwrite: str,
    w_pred: float, w_comp: float, w_diss: float, w_pen: float,
    dataset_exponent: int = None, last_run_num: int = None, number: int = 1):
    """Executes main function with argument interface."""

    assert category in CATEGORIES
    if dataset_exponent is not None:
        assert dataset_exponent in range(2, 10)

    def get_run_name_pattern(category, system):
        run_name_pattern = category[0]  # t/d/s/h for test/dev/sweep/hyperparam
        run_name_pattern += system[0]   # c for cube or e for elbow
        run_name_pattern += '????' if category==HYPERPARAMETER else '??'
        run_name_pattern += '-?' if category==SWEEP else ''
        return run_name_pattern

    # First, take care of data management and how to keep track of results.
    storage_folder_name = f'{category}_{system}'
    storage_folder_name += f'-{dataset_exponent}' if category==SWEEP else ''

    repo = git.Repo(search_parent_directories=True)
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', storage_folder_name)
    
    if run_name is None:
        nums_to_display = 4 if category == HYPERPARAMETER else 2
        if last_run_num is None:
            runs_dir = file_utils.all_runs_dir(storage_name)
            runs_list = sorted(os.listdir(runs_dir))
            if len(runs_list) > 0:
                last_run_name = runs_list[-1]
                last_run_num = int(last_run_name.split('-')[0][2:])
            else:
                last_run_num = -1
        run_name = category[0]
        run_name += 'c' if system==CUBE_SYSTEM else 'e'
        run_name += str(last_run_num+1).zfill(nums_to_display)
        run_name += f'-{dataset_exponent}' if category==SWEEP else ''

    run_name_pattern = get_run_name_pattern(category, system)
    assert fnmatch.fnmatch(run_name, run_name_pattern)

    print(f'\nOverwrite set to {overwrite}.')

    if op.isdir(op.join(storage_name, 'runs', run_name)):
        if not click.confirm(f'\nPause!  Experiment \'' \
                             + f'{storage_folder_name}/{run_name}\'' \
                             + f' already taken, continue?'):
            raise RuntimeError('Choose a new run name next time.')
    elif number > 1 and op.isdir(op.join(storage_name, 'runs', f'{run_name}-0')):
        raise RuntimeError(f'Found experiment run {storage_folder_name}/' + \
                           f'{run_name}-0.  Choose a new base name next time.')

    dataset_size = 4 if category == TEST else \
                  64 if category == DEV else \
                  512 if category == HYPERPARAMETER else \
                  2**dataset_exponent
    source = REAL_SOURCE if category == SWEEP else SIM_SOURCE

    names = [run_name] if number == 1 else \
            [f'{run_name}-{i}' for i in range(number)]
    wandb_group_id = None if number == 1 else \
                     f'{run_name}_{wandb.util.generate_id()}'

    for run_name_i in names:
        take_care_of_file_management(overwrite, storage_name, run_name_i)

        # Continue creating PLL instance.
        create_instance(storage_folder_name, run_name_i, system=system,
                        source=source, contactnets=contactnets, box=box,
                        regenerate=regenerate, dataset_size=dataset_size,
                        local=local, inertia_params=inertia_params,
                        loss_variation=loss_variation, true_sys=true_sys,
                        restart=False, wandb_group_id=wandb_group_id,
                        w_pred=w_pred, w_comp=w_comp, w_diss=w_diss,
                        w_pen=w_pen)


@click.group()
def cli():
    pass


@cli.command('create')
@click.argument('storage_folder_name')
@click.argument('run_name')
@click.option('--number',
              default=1,
              help="number of grouped identical experiments to run")
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
@click.option('--loss-variation',
              type=click.Choice(LOSS_VARIATION_NUMBERS),
              default='0',
              help="ContactNets loss variation")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
@click.option('--w-pred',
              type=float,
              default=1e0,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=1e0,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=1e0,
              help="weight of penetration term in ContactNets loss")
def create_command(storage_folder_name: str, run_name: str, number: int,
                   system: str, source: str, contactnets: bool, box: bool,
                   regenerate: bool, dataset_size: int, local: bool,
                   inertia_params: str, loss_variation: str, true_sys: bool,
                   overwrite: str, w_pred: float, w_comp: float,
                   w_diss: float, w_pen: float):
    """Executes main function with argument interface."""

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    # First, take care of data management and how to keep track of results.
    # Check if experiment name was given and if it already exists.  We also
    # don't want hyphens in the name since that's how the sweep instances
    # are created.
    assert loss_variation in LOSS_VARIATION_NUMBERS
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
    elif op.isdir(op.join(storage_name, 'runs', f'{run_name}-0')):
        raise RuntimeError(f'Found experiment run {storage_folder_name}/' + \
                           f'{run_name}-0.  Choose a new base name next time.')
    elif op.isdir(storage_name):
        dataset_size_in_folder = file_utils.get_numeric_file_count(
            file_utils.learning_data_dir(storage_name))
        if not click.confirm(f'\nPause!  Experiment storage \'' \
                             + f'{storage_folder_name}\'' \
                             + f' already taken with {dataset_size_in_folder}' \
                             + f' dataset size, continue?'):
            raise RuntimeError('Choose a new storage name next time.')

    names = [run_name] if number == 1 else \
            [f'{run_name}-{i}' for i in range(number)]
    wandb_group_id = None if number == 1 else \
                     f'{run_name}_{wandb.util.generate_id()}'

    for run_name_i in names:
        take_care_of_file_management(overwrite, storage_name, run_name_i)

        # Continue creating PLL instance.
        create_instance(storage_folder_name, run_name_i, system, source,
                        contactnets, box, regenerate, dataset_size, local,
                        inertia_params, loss_variation, true_sys, False,
                        wandb_group_id=wandb_group_id, w_pred=w_pred,
                        w_comp=w_comp, w_diss=w_diss, w_pen=w_pen)


@cli.command('test')
@click.option('--run_name',
              type=str,
              default=None)
@click.option('--number',
              default=1,
              help="number of grouped identical experiments to run")
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
@click.option('--loss-variation',
              type=click.Choice(LOSS_VARIATION_NUMBERS),
              default='0',
              help="ContactNets loss variation")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
@click.option('--w-pred',
              type=float,
              default=1e0,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=1e0,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=1e0,
              help="weight of penetration term in ContactNets loss")
def test_command(run_name: str, number: int, system: str, contactnets: bool,
                 box: bool, regenerate: bool, local: bool, inertia_params: str,
                 loss_variation: str, true_sys: bool, overwrite: str,
                 w_pred: float, w_comp: float, w_diss: float, w_pen: float):
    """Executes main function with argument interface."""
    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    experiment_class_command('test', run_name, system=system,
                             contactnets=contactnets, box=box,
                             regenerate=regenerate, local=local, 
                             inertia_params=inertia_params,
                             loss_variation=loss_variation, true_sys=true_sys,
                             overwrite=overwrite, number=number, w_pred=w_pred,
                             w_comp=w_comp, w_diss=w_diss, w_pen=w_pen)

@cli.command('dev')
@click.option('--run_name',
              type=str,
              default=None)
@click.option('--number',
              default=1,
              help="number of grouped identical experiments to run")
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
@click.option('--loss-variation',
              type=click.Choice(LOSS_VARIATION_NUMBERS),
              default='0',
              help="ContactNets loss variation")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--overwrite',
              type=click.Choice(OVERWRITE_RESULTS, case_sensitive=True),
              default=OVERWRITE_NOTHING)
@click.option('--w-pred',
              type=float,
              default=1e0,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=1e0,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=1e0,
              help="weight of penetration term in ContactNets loss")
def dev_command(run_name: str, number: int, system: str, contactnets: bool,
                box: bool, regenerate: bool, local: bool, inertia_params: str,
                loss_variation: str, true_sys: bool, overwrite: str,
                w_pred: float, w_comp: float, w_diss: float, w_pen: float):
    """Executes main function with argument interface."""
    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    experiment_class_command('dev', run_name, system=system,
                             contactnets=contactnets, box=box,
                             regenerate=regenerate, local=local, 
                             inertia_params=inertia_params, 
                             loss_variation=loss_variation, true_sys=true_sys,
                             overwrite=overwrite, number=number, w_pred=w_pred,
                             w_comp=w_comp, w_diss=w_diss, w_pen=w_pen)


@cli.command('restart')
@click.argument('run_name')
@click.option('--storage-folder-name',
              type=str,
              default='')
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
def restart_command(run_name: str, storage_folder_name: str, local: bool):
    """Restarts a previously started run."""

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    # Figure out the storage folder name if not provided
    if storage_folder_name == '':
        assert len(run_name) == 4
        assert run_name[0] in ['t', 'd']
        assert run_name[1] in ['c', 'e']
        assert int(run_name[2:]) + 1

        storage_folder_name = RUN_PREFIX_TO_FOLDER_NAME[run_name[:2]]

    # Check that both the storage folder name and run name exist.
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', storage_folder_name)

    if not op.isdir(op.join(storage_name, 'runs', run_name)):
        raise RuntimeError(f'Error!  Could not find run under ' + \
                           f'{storage_folder_name} with run {run_name}.')

    print(f'Found experiment run \'{run_name}\' in \'{storage_folder_name}\'')

    create_instance(storage_folder_name, run_name, local=local, restart=True)




@cli.command('sweep')
@click.option('--sweep_name',
              type=str,
              default=None)
@click.option('--number',
              default=1,
              help="number of grouped identical experiments to run")
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
@click.option('--loss-variation',
              type=click.Choice(LOSS_VARIATION_NUMBERS),
              default='0',
              help="ContactNets loss variation")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--w-pred',
              type=float,
              default=1e0,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=1e0,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=1e0,
              help="weight of penetration term in ContactNets loss")
def sweep_command(sweep_name: str, number: int, system: str, contactnets: bool,
                  box: bool, regenerate: bool, local: bool, inertia_params: str,
                  loss_variation: str, true_sys: bool, w_pred: float,
                  w_comp: float, w_diss: float, w_pen: float):
    """Starts a series of instances, sweeping over dataset size."""
    assert sweep_name is None or '-' not in sweep_name

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    # First determine what run number to use so they are consistent for each
    # dataset size.
    last_run_num = -1
    repo = git.Repo(search_parent_directories=True)
    repo_dir = repo.git.rev_parse("--show-toplevel")
    partial_storage_name = op.join(repo_dir, 'results', f'sweep_{system}')
    
    for dataset_exponent in range(2, 10):
        storage_name = f'{partial_storage_name}-{dataset_exponent}'
        if op.isdir(storage_name):
            runs_dir = file_utils.all_runs_dir(storage_name)
            runs_list = sorted(os.listdir(runs_dir))
            if len(runs_list) > 0:
                last_run_num = max(last_run_num, int(runs_list[-1][2:4]))

    # Create a pll instance for every dataset size from 4 to 512
    for dataset_exponent in range(2, 10):
        experiment_class_command('sweep', sweep_name, system=system,
                                 contactnets=contactnets, box=box,
                                 regenerate=regenerate, local=local, 
                                 inertia_params=inertia_params,
                                 loss_variation=loss_variation,
                                 true_sys=true_sys,
                                 dataset_exponent=dataset_exponent,
                                 last_run_num=last_run_num,
                                 overwrite=OVERWRITE_NOTHING,
                                 number=number, w_pred=w_pred, w_comp=w_comp,
                                 w_diss=w_diss, w_pen=w_pen)



@cli.command('hyperparam')
@click.option('--hp_name',
              type=str,
              default=None)
@click.option('--number',
              default=1,
              help="number of grouped identical experiments to run")
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
def hyperparameter_command(hp_name: str, number: int, system: str,
                           contactnets: bool, box: bool, regenerate: bool,
                           local: bool, inertia_params: str, true_sys: bool):
    """Starts a series of instances, sweeping over dataset size."""
    assert hp_name is None or '-' not in hp_name

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)
    check_for_git_updates(repo)

    # First determine what run number to use so they are consistent for each
    # dataset size.
    last_run_num = -1
    repo = git.Repo(search_parent_directories=True)
    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', f'hyperparam_{system}')

    if op.isdir(storage_name):
        runs_dir = file_utils.all_runs_dir(storage_name)
        runs_list = sorted(os.listdir(runs_dir))
        if len(runs_list) > 0:
            last_run_num = max(last_run_num, int(runs_list[-1][2:6]))

    # Search over weights for 3 of the loss components for loss variations 1, 2,
    # and 3 (leave out 0 since it's a scaled version of 1).
    w_pred = 1e0
    for w_comp in HYPERPARAMETER_WEIGHTS:
        for w_diss in HYPERPARAMETER_WEIGHTS:
            for w_pen in HYPERPARAMETER_WEIGHTS:
                if w_comp == w_diss == w_pen == 1e0:
                    # Already ran many tests with (1, 1, 1, 1) weights, so can
                    # skip repeating this hyperparameter set.
                    continue

                for loss_variation in [1, 2, 3]:
                    experiment_class_command(
                        'hyperparam', hp_name, system=system,
                        contactnets=contactnets, box=box,
                        regenerate=regenerate, local=local,
                        inertia_params=inertia_params,
                        loss_variation=loss_variation, true_sys=true_sys,
                        last_run_num=last_run_num, overwrite=OVERWRITE_NOTHING,
                        number=number, w_pred=w_pred, w_comp=w_comp,
                        w_diss=w_diss, w_pen=w_pen)
                    last_run_num += 1



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
