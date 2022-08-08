"""Manager for starting GRASP cluster PLL jobs."""
import os
import os.path as op
import git
import click
import subprocess



CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM]
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]


def create_instance(name: str, system: str, source: str, contactnets: bool,
                 	box: bool, regenerate: bool, dataset_size: int, local: bool,
                 	videos: bool):
	print(f'Generating experiment {name}')

	base_file = 'startup'

	out_file = base_file + '_' + name + '.bash'

	# use local template if running locally
	base_file += '_local.bash' if local else '.bash'

	base_file = op.join(op.dirname(__file__), base_file)
	out_file = op.join(op.dirname(__file__), out_file)


	script = open(base_file, 'r').read()

	script = script.replace('{name}', name)
	script = script.replace('{gen_videos}', 'true') if videos else script.replace('{gen_videos}', 'false')

	train_options = f' --system={system} --source={source} --dataset-size={dataset_size}'
	train_options += ' --box' if box else ' --mesh'
	train_options += ' --contactnets' if contactnets else ' --prediction'
	train_options += ' --regenerate' if regenerate else ' --no-regenerate'
	train_options += ' --videos' if videos else ' --no-videos'
	train_options += ' --local' if local else ' --cluster'

	script = script.replace('{train_args}', train_options)

	if not local:
		firefox_dir = '/mnt/beegfs/scratch/bibit/firefox'
		script = script.replace('{firefox_dir}', firefox_dir)

	repo = git.Repo(search_parent_directories=True)
	git_folder = repo.git.rev_parse("--show-toplevel")
	git_folder = op.normpath(git_folder)
	script = script.replace('{pll_dir}', git_folder)

	commit_hash = repo.head.object.hexsha
	script = script.replace('{hash}', commit_hash)

	with open(out_file, "w") as of:
		of.write(script)

	train_cmd = ['bash', out_file] if local else ['sbatch', out_file]
	ec = subprocess.run(train_cmd)



@click.command()
@click.argument('name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train and test with ContactNets/prediction loss.")
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
@click.option('--videos/--no-videos',
			  default=False,
			  help="whether to generate videos or not.")
def main_command(name: str, system: str, source: str, contactnets: bool,
                 box: bool, regenerate: bool, dataset_size: int, local: bool,
                 videos: bool):
    """Executes main function with argument interface."""

    # Check if git repository has uncommitted changes.
    repo = git.Repo(search_parent_directories=True)

    commits_ahead = sum(1 for _ in repo.iter_commits('origin/main..main'))
    if commits_ahead > 0:
        if not click.confirm(f'You are {commits_ahead} commits ahead of main branch, continue?'):
            raise RuntimeError('Make sure you have pushed commits!')

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if len(changed_files) > 0:
        print('Uncommitted changes to:')
        print(changed_files)
        if not click.confirm('Continue?'):
            raise RuntimeError('Make sure you have committed changes!')

    # Check if experiment name already exists.
    assert name is not None

    repo_dir = repo.git.rev_parse("--show-toplevel")
    storage_name = op.join(repo_dir, 'results', name)
    if op.isdir(storage_name):
        if not click.confirm(f'\nPause!  Experiment name \'{name}\' already taken, continue (overwrite)?'):
            raise RuntimeError('Choose a new name next time.')

	# clear the results directory, per user input
	storage_name = os.path.join(repo_dir, 'results', name)
	os.system(f'rm -r {file_utils.storage_dir(storage_name)}')

    # Continue creating PLL instance.
    create_instance(name, system, source, contactnets, box, regenerate, dataset_size, local, videos)


if __name__ == '__main__':
    main_command()
