"""Interface for logging training progress to Weights and Biases."""
from typing import Dict, Tuple

import numpy as np
import wandb

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
    experiment_name: str
    """Unique identifier for Weights and Biases experiment."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    def launch(self) -> None:
        """Launches experiment run on Weights & Biases."""
        wandb.init(project=self.experiment_name)

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
