import numpy as np
import gin
from typing import Dict
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from rss_experiment import sim_experiment
import git

from chamfer_distance import get_chamfer_distance
#DEBUG
import ipdb

#ipdb.set_trace()

@gin.configurable
def main(
         weights: Dict,
         ):
    """
    Main function to run the hyperparameter tuning experiment
    """
    ipdb.set_trace()
    learned_model = sim_experiment()
    learned_model.set_weights()


    kernel_ = RBF(length_scale=1.0)
    surrogate_model = GaussianProcessRegressor(
                             kernel = kernel_,
                             ).fit(X,Y)    
    
    ipdb.set_trace()


    model_accuracy = get_chamfer_distance()




if __name__ == "__main__":
    REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )
    DEFAULT_CONFIG = "hyperpara_tuning_experiment.gin"

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


