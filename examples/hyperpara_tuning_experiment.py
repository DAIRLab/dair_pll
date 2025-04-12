import numpy as np
import gin
from typing import Dict
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from rss_experiment import Experiment
import git
from typing import List, Any 

from chamfer_distance import get_chamfer_distance
from action_library import sample_action, ActionLibrary
#DEBUG
from ipdb import set_trace

import csv 
import time

from sim_with_lcm import Simulation


num_steps = 3

# Initial hyperparameter values 
base_hyperparams = {
    "w_pred": 5e0,
    "w_q_pred": 1e1,
    "w_comp" : 2e1,
    "w_fdiss" : 1e0,
    "w_ndiss" : 1e0,
    "w_pen" : 1e2,
    "w_dev" : 1e5,
    "w_norm" : 1e0,
    "epoch" : 50,
}

hyperparam_ranges = {
    'w_pred': np.logspace(1, 4, num_steps),
    'w_q_pred': np.logspace(1, 3, num_steps),
    'w_comp': np.logspace(1, 3, num_steps),
    'w_fdiss': np.logspace(1, 3, num_steps),
    'w_ndiss': np.logspace(1, 3, num_steps),
    'w_pen': np.logspace(1, 3, num_steps),
    'w_dev': np.logspace(2, 4, num_steps),
    'w_norm': np.logspace(1, 3, num_steps),
    'epoch': np.linspace(50, 500, num_steps),
}


def main(num_steps: int = 3,
         workspace_radius: float = 0.15,
         sphere_radius: float = 0.01575,
         ):
    """
    Main function to run the hyperparameter tuning experiment
    """

    actions_enum = [7]

    cfd_cache = []
    max_iter = 20
    epislon = 1e-3

    num_of_params = len(base_hyperparams) 

    csv_file = CSV_File()
    
    # Initialize simulation
    sim = Simulation()
    learned_model = Experiment()
    
    while True:

        for param_name, current_param in base_hyperparams.items():        

            # concatenate the candidate parameters with the current parameter
            param_range_ = np.concatenate([hyperparam_ranges[param_name], [current_param]])
            chamfer_distances = np.zeros((num_steps + 1,))

            # each trial
            for idx, candi_param in enumerate(param_range_):
                # insert candidate parameter into hyperparameter dict
                base_hyperparams[param_name] = candi_param
                print(f"Testing with these hyperparameters: {base_hyperparams}")

                # execute actions
                assert len(actions_enum) > 0
                assert np.all(np.array(actions_enum) > 0 and np.array(actions_enum) < 8)
        
                for act_idx in actions_enum:
                    action = sample_action(workspace_radius=workspace_radius,
                                        sphere_radius=sphere_radius,
                                        library=act_idx)
                
                    learned_model.data_collection(action)

                learned_model.data_train(base_hyperparams)
                chamfer_distances[idx] = learned_model.chamfer_distance()
                print(f"Chamfer distance: {chamfer_distances[idx]}")
                
                # reset sim and experiment
                sim.reset()
                learned_model.reset()
            
            optml_idx = np.argmin(chamfer_distances)

            if optml_idx == 0:
                print("Warning: Optimal parameter is the first in the testing range for hyparam: ", param_name)
            elif optml_idx == num_steps - 2:
                print("Warning: Optimal parameter is the last in the testing range for hyparam: ", param_name)
            elif optml_idx == num_steps - 1:
                print("Warning: Could not find a better parameter than the current one for hyparam: ", param_name)

            # save optimal parameter into dict
            base_hyperparams[param_name] = param_range_[optml_idx]
            cfd_cache.append((chamfer_distances[optml_idx]))

            csv_file.write_row(
                params=base_hyperparams,
                chamfer_distance=chamfer_distances[optml_idx]
            )

        

        # # once the each hyperparam is tuned, check convergence
        # delta_cfd = cfd_cache[-1:-num_of_params]
        set_trace()


class CSV_File():
    def __init__(self):
        filename = f"hyperparam_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        first_row = [name for name in base_hyperparams.keys()]
        first_row.append("cfd")

        with open(filename, 'w', newline='') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',')
            file_writer.writerow(first_row)

        self.filename = filename

    def write_row(self, 
                  params: Dict[str, Any],
                  chamfer_distance: float,
                  ):
        
        values = [val for val in params.values()]
        values.append(chamfer_distance)

        # Open file in append mode for each write
        with open(self.filename, 'a', newline='') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',')
            file_writer.writerow(values)


if __name__ == "__main__":
    REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )
    DEFAULT_CONFIG = "rss_experiment.gin"

    config_file = DEFAULT_CONFIG

    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))


    main()







# w_pred: float,
# w_q_pred: float,
# w_comp: float,
# w_fdiss: float,
# w_ndiss: float,
# w_pen: float,
# w_dev: float,
# w_norm: float,
# ):


