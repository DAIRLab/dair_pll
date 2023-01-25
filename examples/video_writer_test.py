# Test script for doing forced triggering for Drake's VideoWriter.

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.plant import CoulombFriction
from pydrake.multibody.tree import world_model_instance
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.geometry import HalfSpace
from pydrake.visualization import VideoWriter

from dair_pll import file_utils

from PIL import Image, ImageSequence

from copy import deepcopy

import numpy as np
import pdb


DUMMY_DT = 0.001  # "DUMMY" because goal to force trigger, not run a simulation.

URDFS = {"cube": file_utils.get_asset("contactnets_cube.urdf")}

# In order of [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz].
STATE_TRAJS = {
	"ground": np.array([[], [], [], []]),
	"cube": np.array([[1, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0],
					  [1, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
					  [1, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
					  [1, 0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0]])}


# Going through PLL code:
# Starts in meshcat_utils.generate_visualization_system()
# plant_diagram = MultibodyPlantDiagram(urdfs, dt, enable_visualizer)
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, DUMMY_DT)
parser = Parser(plant)

# Build [model instance index] list, starting with world model, which is always
# added by default.
model_ids = [world_model_instance()]
model_ids.extend([parser.AddModelFromFile(urdf, name) \
				  for name, urdf in URDFS.items()])

# Add video writer to diagram.  From drake_utils.MultibodyPlantDiagram
sensor_pose = RigidTransform(RollPitchYaw([-np.pi/2, 0, np.pi/2]), [2., 0., 0.])
video_writer = VideoWriter.AddToBuilder(filename="output.gif",
										builder=builder,
                                        sensor_pose=sensor_pose)

# Add ground plane at z=0.
halfspace_transform = RigidTransform()
friction = CoulombFriction(1.0, 1.0)
plant.RegisterCollisionGeometry(plant.world_body(), halfspace_transform,
                                HalfSpace(), "ground", friction)

# Builds and initialize simulator from diagram
plant.Finalize()
diagram = builder.Build()
diagram.CreateDefaultContext()
sim = Simulator(diagram)
sim.Initialize()
sim.set_publish_every_time_step(False)


def set_plant_state(traj_idx):
    # Set state initial condition in internal Drake ``Simulator`` context.
    global plant, sim, model_ids

    sim_context = sim.get_mutable_context()
    sim_context.SetTime(DUMMY_DT/4)
    plant_context = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())

    # Iterate over every object in the plant.
    for model_id, obj_key in zip(model_ids, STATE_TRAJS.keys()):
	    q_drake = STATE_TRAJS[obj_key][traj_idx, :7]
	    v_drake = STATE_TRAJS[obj_key][traj_idx, 7:]

	    plant.SetPositions(plant_context, model_id, q_drake)
	    plant.SetVelocities(plant_context, model_id, v_drake)

    sim.Initialize()



for i in range(STATE_TRAJS["ground"].shape[0]):
	# Set the plant state.
	set_plant_state(i)

	# Force trigger a video writer publish event.
	sim_context = sim.get_mutable_context()
	video_context = video_writer.GetMyContextFromRoot(sim_context)
	video_writer._publish(video_context)

print(f'Published {i+1} frames via video_writer._publish()')

# Save the gif.
video_writer.Save()

# Since Drake's VideoWriter defaults to not looping gifs, re-load and re-save
# the gif to ensure it loops.
im = Image.open(video_writer._filename)
im.save(f'_{video_writer._filename}', save_all=True, loop=0)
im.close()

# Check how many frames there are.
im = Image.open(f'_{video_writer._filename}')
print(f'Frames in resulting gif: {len(list(ImageSequence.Iterator(im)))}')
print(f'Saved original gif at: {video_writer._filename}')
print(f'Saved looping gif at: _{video_writer._filename}')

# pdb.set_trace()

