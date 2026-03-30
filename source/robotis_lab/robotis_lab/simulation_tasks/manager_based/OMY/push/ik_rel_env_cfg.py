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

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils

from robotis_lab.simulation_tasks.manager_based.OMY.push.mdp import push_events
from robotis_lab.simulation_tasks.manager_based.OMY.push.push_env_cfg import PushEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.OMY import OMY_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_omy_arm_pose = EventTerm(
        func=push_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [-0.13842978172359469, -0.1842454734844208, 2.587226374961472, -0.8321845764770508, 1.5707004512008669, 2.3968449783325197e-05, 0.49900540540540544, 0.0, 0.0, 0.0],
        },
    )

    reset_robot_position = EventTerm(
        func=push_events.reset_robot_to_default_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Disabled: randomize_omy_joint_state - uncomment if you want randomization
    # randomize_omy_joint_state = EventTerm(
    #     func=push_events.randomize_joint_by_gaussian_offset,
    #     mode="reset",
    #     params={
    #         "mean": 0.0,
    #         "std": 0.02,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    randomize_object_positions = EventTerm(
        func=push_events.randomize_two_objects_on_table,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.5), "y": (-0.20, 0.20), "z": (0.015, 0.015), "yaw": (0, 0)},
            "min_separation": 0.15,
            "max_separation": 0.5,
            "asset_cfgs": [SceneEntityCfg("pusher_object"), SceneEntityCfg("target_object")],
        },
    )

    randomize_wrist_camera = EventTerm(
        func=push_events.randomize_camera_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cam_wrist"),
            "pose_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.005, 0.005),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "convention": "ros",
        },
    )

    randomize_top_camera = EventTerm(
        func=push_events.randomize_camera_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cam_top"),
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
            "convention": "ros",
        },
    )


@configclass
class OMYPushEnvCfg(PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set OMY as robot with custom initial pose
        self.scene.robot = OMY_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=OMY_CFG.init_state.replace(
                joint_pos={
                    "joint1": -0.13842978172359469,
                    "joint2": -0.1842454734844208,
                    "joint3": 2.587226374961472,
                    "joint4": -0.8321845764770508,
                    "joint5": 1.5707004512008669,
                    "joint6": 2.3968449783325197e-05,
                    "rh_r1_joint": 0.49900540540540544,
                }
            ),
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (OMY) with IK control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.15,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rh_r1_joint"],
            open_command_expr={"rh_r1_joint": 0.0},
            close_command_expr={"rh_r1_joint": 0.7},
        )

        # Rigid body properties for each object
        object_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set pusher object (blue block) - 120x30x30mm
        self.scene.pusher_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PusherObject",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, -0.15, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(2.96, 0.74, 0.74),  # 120mm x 30mm x 30mm
                rigid_props=object_properties,
                semantic_tags=[("class", "pusher_object")],
            ),
        )

        # Set target object (red block) - 120x30x30mm
        self.scene.target_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetObject",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.15, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(2.96, 0.74, 0.74),  # 120mm x 30mm x 30mm
                rigid_props=object_properties,
                semantic_tags=[("class", "target_object")],
            ),
        )

        # Wrist Camera (mounted on link6)
        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/link6/cam_wrist",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=11.8,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.08, 0.07),
                rot=(0.5, -0.5, -0.5, -0.5),
                convention="isaac",
            )
        )

        # Top Camera (mounted in world frame)
        self.scene.cam_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_top",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 1.0),  # 1m above ground, centered over table
                rot=(0.7071068, 0.0, 0.0, -0.7071068),  # 90° pitch down (looking at table)
                convention="isaac",
            )
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.248, 0.0],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_r2",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_l2",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )
