# DAIRLab Physics-based Learning Library

## API Documentation
https://dairlab.github.io/dair_pll


## Running the Code

We recommend working within a virtual environment.

1. Install with `pip install -e .`

2. From the main folder, run a test using `python examples/contactnets_simple.py <storage name> <run name>`. By default, 200 epochs are run (with a patience of 10, i.e. early stopping of validation loss does not drop for 10 epochs). These can be modified directly in `contactnets_simple.py` (TODO: make these command line flags)

_Notes: by default, all weights are set to 1.0, and simulation training operates on 512 trajectories (which can take a while per epoch). These can be modified with command line args documented in `contactnets_simple.py`_

3. In practice, it is recommended to set a higher weight on the penetration loss, e.g., `python examples/contactnets_simple.py <storage name> <run name> --w-pen 20.0`


## Attribution notes
* The GitHub Action documentation build scripts are based on [Anne Gentle](https://github.com/annegentle)'s great example here: https://github.com/annegentle/create-demo
* Some functions (such as [`rotation_matrix_from_one_vector`](https://dairlab.github.io/dair_pll/dair_pll.tensor_utils.html#dair_pll.tensor_utils.rotation_matrix_from_one_vector)) are Pytorch reimplementations of [drake](https://github.com/RobotLocomotion/drake) functionality, and are attributed accordingly in their documentation.
* This code contains a repackaged version of the Manifold Unscented Kalman Filter developed by [Martin Brossard et al.](https://github.com/CAOR-MINES-ParisTech/ukfm)

  
