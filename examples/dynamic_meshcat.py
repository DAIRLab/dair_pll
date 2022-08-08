"""Script that finds the latest meshcat server URL for the current experiment."""


import os
import git
import pdb

from dair_pll import file_utils


EXP_KEY = 'PLL_EXPERIMENT'


def main_command():
	# First, get the latest meshcat server URL for the current experiment.
	pll_name = os.getenv(EXP_KEY)

	mc_log_name = file_utils.LOG_DIR + f'/meshcat_{pll_name}.txt'
	meshcat_log = open(mc_log_name, 'r').read()

	new_url = meshcat_log.split('web_url=')[-1].split('\n')[0]

	# Second, write a new static html file using the updated URL.
	base_html = file_utils.get_asset('static.html')
	script = open(base_html, 'r').read()

	script = script.replace('http://127.0.0.1:7000/static/', new_url)

	repo = git.Repo(search_parent_directories=True)
	repo_dir = repo.git.rev_parse("--show-toplevel")
	storage_name = os.path.join(repo_dir, 'results', pll_name)
	storage_loc = file_utils.storage_dir(storage_name)

	out_file = storage_loc + f'/static.html'
	with open(out_file, "w") as of:
		of.write(script)


if __name__ == '__main__':
    main_command()