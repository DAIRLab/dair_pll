# DAIRLab Physics-based Learning Library (Active Tactile Exploration Branch)

This codebase was used to run the experiments presented in our [ICRA 2026 Paper](https://dairlab.github.io/activetactile/).

A clean version, which will also be migrated to JAX, is currently being developed at https://github.com/DAIRLab/dair_exploration

## API Documentation
https://dairlab.github.io/dair_pll


## Running the Code

We recommend working within a virtual environment.

1. Install with `pip install -e .`

2. To run a simulation (necessary without a Trifinger), execute `./examples/sim_with_lcm.py`. Configuration options are in `config/sim_with_lcm.gin`. Visualize the sim with Meshcat at `http://localhost:7000` or whichever port is reported by the code.

3. To run an active tactile exploration , execute `./examples/active_exploration.py`. Configuration options are in `config/active_exploration.gin`. Visualize the results with Meshcat at `http://localhost:7001` or whichever port is reported by the code.


## Attribution notes
* The GitHub Action documentation build scripts are based on [Anne Gentle](https://github.com/annegentle)'s great example here: https://github.com/annegentle/create-demo
* Some functions (such as [`rotation_matrix_from_one_vector`](https://dairlab.github.io/dair_pll/dair_pll.tensor_utils.html#dair_pll.tensor_utils.rotation_matrix_from_one_vector)) are Pytorch reimplementations of [drake](https://github.com/RobotLocomotion/drake) functionality, and are attributed accordingly in their documentation.
* This code contains a repackaged version of the Manifold Unscented Kalman Filter developed by [Martin Brossard et al.](https://github.com/CAOR-MINES-ParisTech/ukfm)

## Citation

```
@inproceedings{gordon2026active,
  title={Active Tactile Exploration for Rigid Body Pose and Shape Estimation}, 
  author={Ethan K. Gordon and Bruke Baraki and Hien Bui and Michael Posa},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2026},
  url={https://arxiv.org/abs/2510.13595}
}
```

  
