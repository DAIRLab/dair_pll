"""Interface for logging training progress to Tensorboard."""
import contextlib
import multiprocessing
import os
import socket
from typing import Dict, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter  # type: ignore
from torch import Tensor

from dair_pll.system import MeshSummary


class TensorboardManager:
    """Manages logging of the training process.

    Given a set of scalars, videos, and meshes, writes to tensorboard.

    Additionally, manages the spinning up and shutting down of a separate
    process to run the tensorboard server.
    """
    writer: SummaryWriter
    """TensorboardX writer for logging to the tensorboard files."""
    thread: multiprocessing.Process
    r"""Thread for :py:func:`os.system` call to ``tensorboard``\ ."""

    def __init__(self, folder: str):
        self.folder = folder

    def create_writer(self) -> None:
        """Creates :py:class:`tensorboardX.SummaryWriter` for interfacing
        with Tensorboard."""
        folder = self.folder
        self.writer = SummaryWriter(folder)

    def launch(self) -> None:
        """Launches tensorboard thread"""
        folder = self.folder

        # pylint: disable=E1103
        port = 6006 + torch.randint(900, (1,)).item()
        # Use threading so tensorboard is automatically closed on process end
        command = 'tensorboard --samples_per_plugin images=0 --bind_all ' \
                  f'--port {port} --logdir {folder} > /dev/null '\
                  f'--reload_interval=1 --window_title {socket.gethostname()} '\
                  f'2>&1'
        #self.thread = threading.Thread(target=os.system, args=(command,))
        self.thread = multiprocessing.Process(target=os.system, args=(command,))
        self.thread.start()

        print(f'Launching tensorboard on http://localhost:{port}')
        self.create_writer()

    def stop(self):
        """Stops tensorboard thread."""
        self.thread.terminate()

    def update(self, epoch: int, scalars: Dict[str, float],
               videos: Dict[str, Tuple[np.ndarray, int]],
               meshes: Dict[str, MeshSummary]) -> None:
        """Write new epoch summary to Tensorboard.

        Args:
            epoch: Current epoch in training process
            scalars: Scalars to log.
            videos: Videos to log.
            meshes: Meshes to log.
        """

        self.__write_scalars(epoch, scalars)
        self.__write_videos(epoch, videos)
        self.__write_meshes(epoch, meshes)

        # Should just have to flush--for some reason, need to close??
        self.writer.close()

    def __write_scalars(self, epoch: int, scalars: Dict[str, float]) -> None:
        """Logs scalars."""
        for field in scalars.keys():
            # pylint: disable=E1103
            self.writer.add_scalar(field, torch.tensor(scalars[field]), epoch)

    def __write_videos(self, epoch: int, videos: Dict[str, Tuple[np.ndarray,
                                                                 int]]) -> None:
        """Logs videos."""
        with open(os.devnull, "w",
                  encoding='utf-8') as f, contextlib.redirect_stdout(f):
            for video_name in videos.keys():
                video, fps = videos[video_name]
                self.writer.add_video(
                    video_name,
                    video,  # type: ignore
                    epoch,
                    fps=fps // 4)

    def __write_meshes(self, epoch: int, meshes: Dict[str,
                                                      MeshSummary]) -> None:
        """Logs meshes."""
        for mesh_name, mesh in meshes.items():
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = vertices - vertices.mean(dim=0, keepdim=True)
            vertices = vertices / vertices.std(dim=0).max()

            color = Tensor([194, 0, 77]).to(torch.int)
            color = color.reshape((1, 1, 3)).repeat((1, vertices.shape[0], 1))
            colors = color.cpu().numpy()
            vertices = vertices.unsqueeze(0).cpu().detach().numpy()
            faces = faces.unsqueeze(0).numpy()
            self.writer.add_mesh(f"{mesh_name}_convex",
                                 vertices,
                                 colors,
                                 faces,
                                 global_step=epoch)
