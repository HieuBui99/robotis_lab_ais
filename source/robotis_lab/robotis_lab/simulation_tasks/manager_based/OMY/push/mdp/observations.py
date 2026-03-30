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

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_rel_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint positions for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel_rel_name(env: ManagerBasedEnv, joint_names: list[str], asset_name: str = "robot") -> torch.Tensor:
    """
    Returns the relative joint velocities for the specified joint names.

    Args:
        env: ManagerBasedEnv instance.
        joint_names: List of joint names to extract.
        asset_name: Name of the asset (default: "robot").

    Returns:
        torch.Tensor of shape [1, len(joint_names)]
    """
    asset: Articulation = env.scene[asset_name]

    joint_ids = [asset.joint_names.index(name) for name in joint_names]

    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def pusher_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("pusher_object")
) -> torch.Tensor:
    """Pusher object position in robot root frame."""
    pusher: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]

    pusher_pos_w = pusher.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform to robot frame
    pusher_pos_b = quat_rotate_inverse(robot_quat_w, pusher_pos_w - robot_pos_w)

    return pusher_pos_b


def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object")
) -> torch.Tensor:
    """Target object position in robot root frame."""
    target: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]

    target_pos_w = target.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Transform to robot frame
    target_pos_b = quat_rotate_inverse(robot_quat_w, target_pos_w - robot_pos_w)

    return target_pos_b


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position in world frame relative to environment origin."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gripper joint positions."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def contact_pusher(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Binary signal: True when robot is in contact with pusher object."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    obj_pos_w = obj.data.root_pos_w

    distance = torch.norm(ee_pos_w - obj_pos_w, dim=-1)
    contact = (distance < threshold).float()

    return contact.unsqueeze(-1)


def objects_close(
    env: ManagerBasedRLEnv,
    pusher_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Binary signal: True when objects are within threshold distance."""
    pusher: RigidObject = env.scene[pusher_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pusher_pos_w = pusher.data.root_pos_w
    target_pos_w = target.data.root_pos_w

    distance = torch.norm(pusher_pos_w - target_pos_w, dim=-1)
    close = (distance < threshold).float()

    return close.unsqueeze(-1)


def objects_close_success(
    env: ManagerBasedRLEnv,
    pusher_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Success termination: True when objects are within threshold distance.

    This is used for automatic success detection during demonstration recording.
    Returns a boolean tensor (not float like objects_close observation).
    """
    pusher: RigidObject = env.scene[pusher_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pusher_pos_w = pusher.data.root_pos_w
    target_pos_w = target.data.root_pos_w

    distance = torch.norm(pusher_pos_w - target_pos_w, dim=-1)
    success = distance < threshold

    return success
