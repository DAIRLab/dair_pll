import os
import subprocess
import sys
import signal
import numpy as np
import gin
import git
from typing import Dict, List, Any 
import csv 
import time
from scipy.stats import loguniform, uniform

from action_library import sample_action, ActionLibrary
from examples.experiment import Experiment
from trifinger_lcm_service import TrifingerLCMService

#DEBUG
from ipdb import set_trace
import logging

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
    # hyperparams = {
    #     "w_pred": 5e0,
    #     "w_q_pred": 1e1,
    #     "w_comp" : 2e1,
    #     "w_fdiss" : 1.27,
    #     "w_ndiss" : 1e0,
    #     "w_pen" : 1e2,
    #     "w_dev" : 1e5,
    #     "w_norm" : 1e0,
    #     "epoch" : 50,
    # }
    hyperparams = {
        "w_pred": 3361.0808430682923,
        "w_q_pred": 2.1600355729130523,
        "w_comp": 1042.262542257485,
        "w_fdiss": 278.42832684261674,
        "w_ndiss": 8.539420054746449,
        "w_pen": 226.5058660208208,
        "w_dev": 1542.401683179897,
        "w_norm": 362.39421199892433,
        "epoch": 500,  # Integer value for epochs
    }
        
    def __init__(self):
        self.csv_file = CSV_File(self.hyperparams)
        self.learned_model = Experiment()

        self.cfd_prev = 0
        self.cfd = -np.inf

        self.iter = 0
        self.workspace_radius = 0.15
        self.sphere_radius = 0.01575

        #self.trifinger_lcm_ = TrifingerLCMService()

        # get cfd for the user's intial guess
        self.cfd = self.trial()
        self.csv_file.write_row(
            params=self.hyperparams,
            chamfer_distance=self.cfd
            )

        # find a better initial guess compared to the user's
        self.hyperparams = self.get_best_guess()

        
    def get_best_guess(self,
                       iters: int = 100):

        best_cfd = self.cfd
        best_hyperparams = dict()

        for _ in range(iters):
            hyperparams = {
            "w_pred": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_q_pred": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_comp": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_fdiss": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_ndiss": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_pen": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "w_dev": float(loguniform.rvs(a=1e2, b=1e5, size=1)),
            "w_norm": float(loguniform.rvs(a=1e0, b=1e4, size=1)),
            "epoch": float(uniform.rvs(loc=50, scale=450, size=1)), 
            }

            self.hyperparams = hyperparams
            cfd_candi = self.trial()

            self.csv_file.write_row(
                params=hyperparams,
                chamfer_distance=cfd_candi,
                )

            if cfd_candi < best_cfd:
                best_cfd = cfd_candi
                best_hyperparams = hyperparams

        return best_hyperparams
        

    def trial(self,
              actions_enum: List[Any] = [ActionLibrary.XPINCH, 
                                         ActionLibrary.YPINCH, 
                                         ActionLibrary.ZPINCH, 
                                         ActionLibrary.CORNERSINGLE,
                                         ],
              ) -> float:
        hyperparams = self.hyperparams
        workspace_radius = self.workspace_radius
        sphere_radius = self.sphere_radius
        learned_model = self.learned_model

        self.process = subprocess.Popen(["python", "examples/sim_with_lcm.py"])
        time.sleep(5)
        self.learned_model.start_sim()

        # execute actions
        for act_idx in actions_enum:
            action = sample_action(workspace_radius=workspace_radius,
                                   sphere_radius=sphere_radius,
                                   library=act_idx)
        
            learned_model.data_collection(action)

        learned_model.data_train(hyperparams)
        chamfer_distance = learned_model.chamfer_distance()
        
        # reset sim and experiment
        self.process.terminate()

        return float(chamfer_distance.cpu().detach())

    def coordinate_descent(self, 
                           step_size: float = 0.4,
                           max_iter: int = 500,
                           epsilon: float = 1e-7,
                           actions_enum: List[Any] = [ActionLibrary.CORNERSINGLE,],
                           ):
        """
        Run hyperparameter tuning experiment using coordinate descent algorithm
        Link:
            https://en.wikipedia.org/wiki/Coordinate_descent
        """
        chamfer_distances = np.zeros((3,))

        hyperparams = self.hyperparams
        #while self.iter <= max_iter and (self.cfd - self.cfd_prev) > epsilon:
        while self.iter <= max_iter:
            self.cfd_prev = self.cfd

            for param_name, current_param in hyperparams.items():
                logging.info(f"Minimizing hyperparameter: {param_name}")
                while True:
                    rand_num = np.random.rand()*0.5 + 0.5
                    candi_params = np.array([
                        current_param - step_size*current_param*rand_num, 
                        current_param + step_size*current_param*rand_num,
                        ])
                    # epoch must be an int
                    candi_params = np.clip(candi_params.astype(int), 50, 500) if param_name == 'epoch' else candi_params
                    
                    chamfer_distances[2] = self.cfd 
                    
                    for idx, candi_param in enumerate(candi_params):
                        hyperparams[param_name] = candi_param
                        logging.info(f"Testing with these hyperpacandi_paramrameters: {hyperparams}")

                        chamfer_distances[idx] = self.trial()

                    opt_idx = np.argmin(chamfer_distances)
                    self.cfd = chamfer_distances[opt_idx]

                    self.csv_file.write_row(
                    params=hyperparams,
                    chamfer_distance=chamfer_distances[opt_idx],
                    )

                    self.iter += 1
    
                    logging.info(f"Candidate chamfer distances(-/+/.): {chamfer_distances}")
                    # save optimal parameter into dict
                    if (chamfer_distances[1] > chamfer_distances[2]) and (chamfer_distances[0] > chamfer_distances[2]):
                        logging.info("Local minima found!") 
                        self.hyperparams = hyperparams
                        break

                    else:
                        logging.info("Continuing descent") 
                        # the optimal paramter becomes the new current parameter
                        current_param = candi_params[opt_idx] 
                        hyperparams[param_name] = current_param
                        self.hyperparams = hyperparams
    def terminate(self):
        self.process.terminate()  # Send SIGTERM
        try:
            self.process.wait(timeout=5)  # Give it time to shut down
        except subprocess.TimeoutExpired:
            print("Child didn’t exit, force killing.")
            self.process.kill()  # Send SIGKILL

    def signal_handler(self,
                       sig, 
                       frame,
                       ):
        """ Handle SIGINT"""
        self.terminate()
        sys.exit(0)

def main():
    """Entry point"""
    if '-debug' in sys.argv:
        logging.basicConfig(level=logging.DEBUG)
    elif '-info' in sys.argv:
        logging.basicConfig(level=logging.INFO)

    REPO_DIR = os.path.normpath(git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))
    DEFAULT_CONFIG = "rss_experiment.gin"
    gin.parse_config_file(os.path.join(REPO_DIR, "config", DEFAULT_CONFIG))

    experiment = hyperparam_tuning()
    signal.signal(signal.SIGINT, experiment.signal_handler)
    experiment.coordinate_descent()


if __name__ == "__main__":
    main()