import numpy as np
import random
import math
import genesis as gs
import torch
import argparse
from collections import deque

parser = argparse.ArgumentParser(description="Simulate Pupper")
parser.add_argument("--vis", action="store_true", help="Enable visualization")
parser.add_argument("--use_terrain", action="store_true", help="Use terrain")
parser.add_argument("--n_envs", type=int, default=1000, help="Number of environments")
parser.add_argument("--go2", action="store_true", help="Use Go2")
parser.add_argument("--mesh", action="store_true", help="Use mesh")
args = parser.parse_args()

VIS = args.vis
USE_TERRAIN = args.use_terrain
N_ENVS = args.n_envs
DT = 0.02
SUBSTEPS = 5
RESET_STEPS = 50


def small_quaternions(batch_size=8, max_angle_deg=30, max_yaw_deg=180):
    """
    Returns a batch of quaternions (as a torch.Tensor of shape (batch_size, 4)).
    Each quaternion has random pitch and roll in [-max_angle_deg, max_angle_deg]
    (in degrees) and yaw = 0, then converted to radians and into quaternion form.

    Args:
        batch_size (int): Number of quaternions to generate.
        max_angle_deg (float): Maximum magnitude of pitch/roll in degrees.

    Returns:
        torch.Tensor: Shape (batch_size, 4). Each row is (w, x, y, z).
    """

    # Random pitch and roll in [-max_angle_deg, max_angle_deg]
    pitch_deg = (torch.rand(batch_size) * 2 - 1) * max_angle_deg
    roll_deg = (torch.rand(batch_size) * 2 - 1) * max_angle_deg
    yaw_deg = (torch.rand(batch_size) * 2 - 1) * max_yaw_deg

    # Convert degrees to radians
    pitch_rad = pitch_deg * math.pi / 180.0
    roll_rad = roll_deg * math.pi / 180.0
    yaw_rad = yaw_deg * math.pi / 180.0

    # Half angles
    half_pitch = pitch_rad / 2
    half_roll = roll_rad / 2
    half_yaw = yaw_rad / 2

    # cosines and sines of half angles
    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    # Convert Euler angles -> Quaternion
    # Roll (X), Pitch (Y), Yaw (Z) in intrinsic rotation order
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Stack to form (batch_size, 4) quaternions
    q = torch.stack([w, x, y, z], dim=-1)

    # Normalize quaternions (though they should already be nearly normalized)
    q = q / q.norm(dim=-1, keepdim=True)

    return q


########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        n_rendered_envs=min(N_ENVS, 100),
    ),
    sim_options=gs.options.SimOptions(
        dt=DT,
        substeps=SUBSTEPS,
    ),
    rigid_options=gs.options.RigidOptions(
        integrator=gs.integrator.implicitfast,
        dt=DT,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_self_collision=True,
        enable_joint_limit=True,
        # use_contact_island=True,
    ),
    show_viewer=VIS,
)

########################## entities ##########################

if args.go2:
    pupper = scene.add_entity(
        gs.morphs.URDF(
            file="/home/nathankau/pupperv3/rl/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf",
            merge_fixed_links=True,
            pos=np.array([0, 0, 0.8]),
            quat=np.array([1.0, 0, 0, 0]),
        ),
        vis_mode="collision",
    )
else:
    pupper = scene.add_entity(
        gs.morphs.MJCF(
            file="/home/nathankau/pupperv3/ros2_ws/src/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.position.full_collision.xml",
        ),
        vis_mode="collision",
    )
terrain_cfg = {
    "vertical_scale": 0.10,
    "horizontal_scale": 0.1,
    "n_subterrains": (3, 3),
    "subterrain_size": (8, 8),
    "subterrain_types": [
        # ["flat_terrain", "random_uniform_terrain", "fractal_terrain"],
        # ["pyramid_sloped_terrain", "stepping_stones_terrain", "sloped_terrain"],
        # ["wave_terrain", "stairs_terrain", "pyramid_stairs_terrain"],
        # [
        #     "flat_terrain",
        #     "flat_terrain",
        #     "flat_terrain",
        # ],
        # [
        #     "flat_terrain",
        #     "flat_terrain",
        #     "flat_terrain",
        # ],
        # [
        #     "flat_terrain",
        #     "flat_terrain",
        #     "flat_terrain",
        # ],
        ["flat_terrain", "random_uniform_terrain", "random_uniform_terrain"],
        ["pyramid_sloped_terrain", "stepping_stones_terrain", "pyramid_stairs_terrain"],
        ["wave_terrain", "stairs_terrain", "pyramid_stairs_terrain"],
    ],
}

if USE_TERRAIN:
    terrain_morph = gs.morphs.Terrain(
        pos=(-12.0, -12.0, 0.0),
        n_subterrains=terrain_cfg["n_subterrains"],
        horizontal_scale=terrain_cfg["horizontal_scale"],
        vertical_scale=terrain_cfg["vertical_scale"],
        subterrain_size=terrain_cfg["subterrain_size"],
        subterrain_types=terrain_cfg["subterrain_types"],
    )

    terrain = scene.add_entity(
        terrain_morph,
        vis_mode="collision",
    )
elif args.mesh:
    mesh_terrain = gs.morphs.Mesh(
        file="/home/nathankau/Downloads/mesh.obj",
        pos=(0, 0, -1),
        scale=(1, 1, 1),
        convexify=False,
        decompose_nonconvex=True,
        fixed=True,
        coacd_options=gs.options.CoacdOptions(
            threshold=0.01, preprocess_resolution=100
        ),
    )
    scene.add_entity(
        mesh_terrain,
        vis_mode="collision",
    )
else:
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
# ball = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(-0.4, 7.44, 0.42)))
# ball2 = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(-0.4, 7.44, 1.5)))


# import ipdb

# ipdb.set_trace()

########################## build ##########################
scene.build(n_envs=N_ENVS)
if not args.go2:
    jnt_names = [
        "leg_front_r_1",
        "leg_front_r_2",
        "leg_front_r_3",
        "leg_front_l_1",
        "leg_front_l_2",
        "leg_front_l_3",
        "leg_back_r_1",
        "leg_back_r_2",
        "leg_back_r_3",
        "leg_back_l_1",
        "leg_back_l_2",
        "leg_back_l_3",
    ]
else:
    jnt_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

dofs_idx = [pupper.get_joint(name).dof_idx_local for name in jnt_names]

############ Optional: set control gains ############
# set positional gains
pupper.set_dofs_kp(
    # kp=np.ones((N_ENVS, 12)) * 5.0,
    kp=np.ones(12) * 0.25,
    dofs_idx_local=dofs_idx,
)
# set velocity gains
pupper.set_dofs_kv(
    # kv=np.ones((N_ENVS, 12)) * 0.25,
    kv=np.ones(12) * 0.25,
    dofs_idx_local=dofs_idx,
)
# set force range for safety
pupper.set_dofs_force_range(
    # lower=np.ones((N_ENVS, 12)) * -1.0,
    # upper=np.ones((N_ENVS, 12)) * 1.0,
    lower=np.ones(12) * -1.0,
    upper=np.ones(12) * 1.0,
    dofs_idx_local=dofs_idx,
)


def reset_envs(env_idxs):
    pos = (torch.rand((len(env_idxs), 3)) * 2 - 1) * 10.0
    pos[:, 2] = 2.0
    pupper.set_pos(
        pos,
        zero_velocity=True,
        envs_idx=env_idxs,
    )

    # set random quaternion orientation
    # generate random quatenrion
    q = small_quaternions(batch_size=len(env_idxs), max_angle_deg=90)
    pupper.set_quat(
        q,
        zero_velocity=True,
        envs_idx=env_idxs,
    )

    pupper.set_dofs_position(
        position=torch.zeros(len(env_idxs), 12),
        dofs_idx_local=dofs_idx,
        zero_velocity=True,
        envs_idx=env_idxs,
    )


# PD control
i = 0
pos_history = deque(maxlen=50)
vel_history = deque(maxlen=50)
while True:
    print("step:", i, "total steps: ", i * N_ENVS)  # , pupper.get_qpos())
    if i % RESET_STEPS == 0:
        reset_envs(torch.arange(N_ENVS))

    pupper.control_dofs_position((torch.rand((N_ENVS, 12)) * 2 - 1) * 0.2, dofs_idx)

    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    # print("control force:", pupper.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    # print("internal force:", pupper.get_dofs_force(dofs_idx))

    high_vel = pupper.get_links_vel().norm(dim=-1) > 100  # shape (N_ENVS, 13)
    if high_vel.any():
        high_vel_idx = high_vel.any(dim=-1).nonzero().flatten()
        gs.logger.warning(
            f"velocity too high, resetting {high_vel_idx.detach().cpu().numpy()}"
        )
        reset_envs(high_vel_idx)

    pos_history.append(pupper.get_pos().clone().detach())
    vel_history.append(pupper.get_links_vel().clone().detach())
    scene.step()

    isnans = torch.isnan(pupper.get_pos())
    if isnans.any():
        import ipdb

        pos_tensor = torch.stack(pos_history)
        print(pos_tensor[:, isnans.nonzero()[0, 0], :])

        vel_tensor = torch.stack(vel_history)
        print(vel_tensor[:, isnans.nonzero()[0, 0], 0])

        ipdb.set_trace()
    i += 1
    # import ipdb

    # ipdb.set_trace()
