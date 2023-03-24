"""Interface for logging training progress to Weights and Biases."""
from typing import Dict, Tuple, Optional, Any

import numpy as np
import wandb

from dair_pll.hyperparameter import hyperparameter_values
from dair_pll.system import MeshSummary


def _write_scalars(epoch: int, scalars: Dict[str, float]) -> None:
    """Logs scalars."""
    wandb.log(scalars, step=epoch)


def _write_videos(epoch: int, videos: Dict[str, Tuple[np.ndarray,
                                                      int]]) -> None:
    """Logs videos."""
    wandb_videos = {
        video_name: wandb.Video(video_array[0], fps=fps)
        for video_name, (video_array, fps) in videos.items()
    }
    wandb.log(wandb_videos, step=epoch)


def _write_meshes(epoch: int, meshes: Dict[str, MeshSummary]) -> None:
    """Logs meshes."""
    wandb_meshes = {
        mesh_name: wandb.Object3D(mesh_summary.vertices.detach().numpy())
        for mesh_name, mesh_summary in meshes.items()
    }
    wandb.log(wandb_meshes, step=epoch)


class WeightsAndBiasesManager:
    """Manages logging of the training process.

    Given a set of scalars, videos, and meshes, writes to Weights and Biases
    at https://wandb.ai .
    """
    run_name: str
    """Unique name for Weights and Biases experiment run."""
    directory: str
    """Absolute path to store metadata."""
    project_name: Optional[str]
    """Unique name for W&B project, analogous to a ``dair_pll`` experiment."""

    def __init__(self, run_name: str, directory: str,
                 project_name: Optional[str]):
        self.run_name = run_name
        self.directory = directory
        self.project_name = project_name

    def launch(self) -> None:
        """Launches experiment run on Weights & Biases."""
        wandb.init(project=self.project_name,
                   dir=self.directory,
                   name=self.run_name,
                   id=self.run_name)

    @staticmethod
    def log_config(config: Any):
        """Log experiment hyperparameter values."""
        wandb.log(hyperparameter_values(config))
        wandb.log({"config": str(config)})

    @staticmethod
    def update(epoch: int, scalars: Dict[str, float],
               videos: Dict[str, Tuple[np.ndarray, int]],
               meshes: Dict[str, MeshSummary]) -> None:
        """Write new epoch summary to Weights and Biases.

        Args:
            epoch: Current epoch in training process
            scalars: Scalars to log.
            videos: Videos to log.
            meshes: Meshes to log.
        """

        _write_scalars(epoch, scalars)
        _write_videos(epoch, videos)
        _write_meshes(epoch, meshes)
