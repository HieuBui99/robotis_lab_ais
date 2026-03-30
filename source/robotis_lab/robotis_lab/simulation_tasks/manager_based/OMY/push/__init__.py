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

import gymnasium as gym
import os

from . import (
    agents,
    ik_rel_env_cfg,
    joint_pos_env_cfg,
)

from .mimic_env import OMYPushMimicEnv
from . import mimic_env_cfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="RobotisLab-Push-OMY-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.OMYPushEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-Push-OMY-IK-Rel-Mimic-v0",
    entry_point="robotis_lab.simulation_tasks.manager_based.OMY.push:OMYPushMimicEnv",
    kwargs={
        "env_cfg_entry_point": mimic_env_cfg.OMYPushMimicEnvCfg,
    },
    disable_env_checker=True,
)


## Joint Position Control

gym.register(
    id="RobotisLab-Push-OMY-Joint-Pos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.OMYPushJointPosEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)