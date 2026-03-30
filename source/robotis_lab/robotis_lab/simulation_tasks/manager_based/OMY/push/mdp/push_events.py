# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sample_two_object_poses(
    min_separation: float = 0.15,
    max_separation: float = 0.5,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """
    Sample positions for two objects with separation constraints.

    Args:
        min_separation: Minimum distance between objects.
        max_separation: Maximum distance between objects.
        pose_range: Dictionary with 'x', 'y', 'z', 'roll', 'pitch', 'yaw' ranges.
        max_sample_tries: Maximum number of sampling attempts.

    Returns:
        List of two poses [x, y, z, roll, pitch, yaw].
    """
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]

    # Sample first object
    pose_1 = [random.uniform(range[0], range[1]) for range in range_list]

    # Sample second object with separation constraints
    for j in range(max_sample_tries):
        pose_2 = [random.uniform(range[0], range[1]) for range in range_list]

        # Check distance between objects
        distance = math.dist(pose_1[:3], pose_2[:3])

        # Accept if within separation constraints or max tries reached
        if (min_separation <= distance <= max_separation) or (j == max_sample_tries - 1):
            break

    return [pose_1, pose_2]


def randomize_two_objects_on_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.15,
    max_separation: float = 0.5,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """
    Randomize positions of two objects on table with separation constraints.

    Args:
        env: Environment instance.
        env_ids: Environment IDs to reset.
        asset_cfgs: List of two SceneEntityCfg for the objects.
        min_separation: Minimum distance between objects (meters).
        max_separation: Maximum distance between objects (meters).
        pose_range: Dictionary specifying x, y, z, roll, pitch, yaw ranges.
        max_sample_tries: Maximum sampling attempts per environment.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_two_object_poses(
            min_separation=min_separation,
            max_separation=max_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )

            # Reset object velocities
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default joint pose for the robot.

    This function should be called during startup mode to set the default
    pose that the robot will reset to. It only sets the default values,
    and does NOT write to simulation. The actual reset to this pose happens
    during reset events.

    Args:
        env: The environment instance.
        env_ids: The environment IDs (unused, for compatibility).
        default_pose: List of joint positions for the default pose.
        asset_cfg: The scene entity configuration for the robot asset.
    """
    # Set the default pose for robots in all envs
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def reset_robot_to_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot to its default joint pose on environment reset.

    This function reads from asset.data.default_joint_pos and writes
    the joint state to simulation. Should be called during reset mode.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        asset_cfg: The scene entity configuration for the robot asset.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get default pose and reset velocity
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Set targets and write to simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    print(f"Resetting robot to default pose: {joint_pos[0].cpu().numpy()}")  # Debug print for first env
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]] = None,
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation within the given ranges."""
    if pose_range is None:
        pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }

    asset: Camera = env.scene[asset_cfg.name]

    # Store initial positions and quaternions once
    if not hasattr(asset, "_initial_pos_w"):
        asset._initial_pos_w = asset.data.pos_w.clone()
        asset._initial_quat_w_ros = asset.data.quat_w_ros.clone()
        asset._initial_quat_w_opengl = asset.data.quat_w_opengl.clone()
        asset._initial_quat_w_world = asset.data.quat_w_world.clone()

    ori_pos_w = asset._initial_pos_w
    if convention == "ros":
        ori_quat_w = asset._initial_quat_w_ros
    elif convention == "opengl":
        ori_quat_w = asset._initial_quat_w_opengl
    elif convention == "world":
        ori_quat_w = asset._initial_quat_w_world

    # Get pose ranges
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    # Sample random offsets for each environment independently
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    # Apply per-env randomization
    for i, env_id in enumerate(env_ids.tolist()):
        pos = ori_pos_w[env_id, 0:3] + rand_samples[i, 0:3]
        ori_delta = math_utils.quat_from_euler_xyz(
            rand_samples[i, 3], rand_samples[i, 4], rand_samples[i, 5]
        )
        ori = math_utils.quat_mul(ori_quat_w[env_id], ori_delta)
        asset.set_world_poses(
            pos.unsqueeze(0), ori.unsqueeze(0), env_ids=torch.tensor([env_id], device=asset.device), convention=convention
        )
