"""Interface for logging training progress to Weights and Biases."""
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import wandb

from dair_pll.hyperparameter import hyperparameter_values
from dair_pll.system import MeshSummary

WANDB_ALLOW = "allow"
WANDB_NEVER = "never"


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


@dataclass
class WeightsAndBiasesManager:
    """Manages logging of the training process.

    Given a set of scalars, videos, and meshes, writes to Weights and Biases
    at https://wandb.ai .
    """
    run_name: str
    """Display name for Weights and Biases experiment run."""
    directory: str
    """Absolute path to store metadata."""
    project_name: str
    """Unique name for W&B project, analogous to a ``dair_pll`` experiment."""
    resume_from_id: Optional[str] = None
    """Allow W&B to resume a unique run ID if provided."""

    def _setup_wandb_run_id(self) -> str:
        """Generates unique run ID for Weights and Biases by concatenating
        the run name and a timestamp. If resumption is allowed, returns the
        saved run ID."""
        if self.resume_from_id is not None:
            return self.resume_from_id

        timestamp = str(time.time_ns() // 1000)

        return f"{self.run_name}_{timestamp}"

    def launch(self) -> str:
        r"""Launches experiment run on Weights & Biases.

        Returns:
            The run ID of the launched run.
        """
        resuming = self.resume_from_id is not None
        wandb_run_id = self._setup_wandb_run_id()

        wandb.init(project=self.project_name,
                   dir=self.directory,
                   name=self.run_name,
                   id=wandb_run_id,
                   config={},
                   resume=WANDB_ALLOW if resuming else WANDB_NEVER,
                   allow_val_change=True) # TODO: revert temprary hack

        return wandb_run_id

    def finish(self):
        """Finishes the Weights and Biases run, making it possible to start a
        new run in the same process."""
        print('WANDB FINISH CALLED')
        wandb.finish()

    @staticmethod
    def log_config(config: Any):
        """Log experiment hyperparameter values."""
        wandb.config.update(hyperparameter_values(config),
                            allow_val_change=True) # TODO: revert temprary hack
        wandb.config.update({"ExperimentConfig": str(config)},
                            allow_val_change=True) # TODO: revert temprary hack

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
        print(epoch)
        print(scalars)
        _write_scalars(epoch, scalars)
        _write_videos(epoch, videos)
        _write_meshes(epoch, meshes)
