import os
import os.path as op
import git
import click
import time
import multiprocessing


@click.command()
@click.argument('name')
@click.option('--local/--cluster',
			  default=False,
			  help="whether running script locally or on cluster.")
def main_command(name: str, local: bool):
	# get the git repository folder
	repo = git.Repo(search_parent_directories=True)
	git_folder = repo.git.rev_parse("--show-toplevel")
	git_folder = op.normpath(git_folder)

	# get the tensorboard bash script
	tb_script = op.join(git_folder, 'examples', 'tensorboard.bash')

	# make a tensorboard log file
	tb_logfile = op.join(git_folder, 'logs', 'tensorboard_' + name + '.txt')
	os.system(f'rm {tb_logfile}')
	tb_folder = op.join(git_folder, 'results', name, 'tensorboard')

	# make and start tensorboard command
	if local:
		tboard_cmd = f'bash {tb_script} {tb_folder} {name} >> {tb_logfile}'
	else:
		tboard_cmd = f'sbatch --output={tb_logfile} --job-name=tb_{name} {tb_script} {tb_folder} {name}'

	print(f'\ntboard_cmd:\n{tboard_cmd}\n')

	thread = multiprocessing.Process(target=os.system, args=(tboard_cmd,))
	thread.start()

	# wait for tensorboard url
	print('Waiting on TensorBoard startup ...')
	lines = []
	while not op.exists(tb_logfile):
		time.sleep(0.1)
	while len(lines) < 1:
		with open(tb_logfile) as f:
			lines = f.readlines()
		time.sleep(1.0)
	print('')
	print(f'TensorBoard running on {lines[0]}')
	print('')
	print('Running training setup')



if __name__ == '__main__':
	main_command()
