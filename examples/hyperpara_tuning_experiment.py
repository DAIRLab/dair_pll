import os
import subprocess

import numpy as np
import gin
from typing import Dict

from rss_experiment import Experiment
import git
from typing import List, Any 

from action_library import sample_action, ActionLibrary

import csv 
import time

#DEBUG
from ipdb import set_trace

class CSV_File():
    def __init__(self, hyperparams):
        filename = f"hyperparam_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        first_row = [name for name in hyperparams.keys()]
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

class hyperparam_tuning:
    hyperparams = {
        "w_pred": 5e0,
        "w_q_pred": 1e1,
        "w_comp" : 2e1,
        "w_fdiss" : 1.27,
        "w_ndiss" : 1e0,
        "w_pen" : 1e2,
        "w_dev" : 1e5,
        "w_norm" : 1e0,
        "epoch" : 50,
    }
        
    def __init__(self):
        self.csv_file = CSV_File(self.hyperparams)

        self.cfd_prev = 0
        self.cfd = -np.inf

        self.iter = 0
        self.workspace_radius = 0.15
        self.sphere_radius = 0.01575

        self.cfd = self.trial()

        self.csv_file.write_row(
            params=self.hyperparams,
            chamfer_distance=self.cfd
            )
        
    def terminate(self):
        process = self.process
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def trial(self,
              actions_enum: List[int] = [7],
              ) -> float:
        hyperparams = self.hyperparams
        workspace_radius = self.workspace_radius
        sphere_radius = self.sphere_radius

        self.process = subprocess.Popen(["python", "examples/sim_with_lcm.py"])
        time.sleep(5)
        learned_model = Experiment()

        # execute actions
        assert len(actions_enum) > 0
        assert np.all(np.array(actions_enum) > 0 and np.array(actions_enum) < 8)

        for act_idx in actions_enum:
            action = sample_action(workspace_radius=workspace_radius,
                                sphere_radius=sphere_radius,
                                library=act_idx)
        
            learned_model.data_collection(action)

        learned_model.data_train(hyperparams)
        chamfer_distance = learned_model.chamfer_distance().cpu().detach()
        
        # reset sim and experiment
        learned_model.reset()
        self.process.terminate()

        return chamfer_distance

    def coordinate_descent(self, 
                           step_size: float = 0.2,
                           max_iter: int = np.inf,
                           epsilon: float = 1e-7,
                           actions_enum: List[int] = [7],
                           ):
        """
        Run hyperparameter tuning experiment using coordinate descent algorithm
        https://en.wikipedia.org/wiki/Coordinate_descent
        """
        chamfer_distances = np.zeros((3,))

        hyperparams = self.hyperparams
        #while self.iter <= max_iter and (self.cfd - self.cfd_prev) > epsilon:
        while self.iter <= max_iter:
            self.cfd_prev = self.cfd

            for param_name, current_param in hyperparams.items():
                rand_num = np.random.rand() + 1
                candi_params = np.array([
                    current_param - step_size*current_param*rand_num, 
                    current_param + step_size*current_param*rand_num,
                    ])
                # epoch must be an int
                candi_params = candi_params.astype(int) if param_name == 'epoch' else candi_params
                
                chamfer_distances[2] = self.cfd 
                
                for idx, candi_param in enumerate(candi_params):
                    hyperparams[param_name] = candi_param
                    print(f"Testing with these hyperpacandi_paramrameters: {hyperparams}")

                    chamfer_distances[idx] = self.trial(actions_enum)

                opt_idx = np.argmin(chamfer_distances)
                # save optimal parameter into dict
                hyperparams[param_name] = candi_params[opt_idx] if opt_idx != 2 else current_param
                print(f"Candidate chamfer distances: {chamfer_distances}")
                self.cfd = chamfer_distances[opt_idx]

                self.csv_file.write_row(
                    params=hyperparams,
                    chamfer_distance=chamfer_distances[opt_idx]
                )

                self.iter += 1
                self.hyperparams = hyperparams

if __name__ == "__main__":
    REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )
    DEFAULT_CONFIG = "rss_experiment.gin"

    config_file = DEFAULT_CONFIG
    
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))

    exp = hyperparam_tuning()
    try:
        exp.coordinate_descent()
    finally:
        exp.terminate()