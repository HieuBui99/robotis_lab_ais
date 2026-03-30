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

"""Leader device controller for robot teleoperation via DDS."""

import numpy as np
import os
import threading
import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from pynput.keyboard import Listener
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager


@dataclass
class LeaderDeviceCfg(DeviceCfg):
    """Configuration for leader device teleoperation."""

    topic_name: str = "/leader/action"
    """DDS topic name to subscribe to for joint trajectory commands."""

    gripper_threshold: float = 0.35
    """Threshold for gripper position. Values above this are considered closed."""

    domain_id: int | None = None
    """DDS domain ID. If None, uses ROS_DOMAIN_ID environment variable or defaults to 30."""


class LeaderDevice(DeviceBase):
    """A leader device controller for robot teleoperation via DDS.

    This class reads joint angles from a leader robot publishing JointTrajectory messages
    via DDS and outputs them as absolute joint positions for the follower robot to mirror.

    The follower robot's joints will directly sync up with the leader device's joints.

    Args:
        cfg: Configuration object for the leader device.

    Key Features:
        - Asynchronous DDS message reading in background thread
        - Direct absolute position output (no delta computation)
        - Thread-safe state management
        - Gripper command mapping from continuous to binary

    DDS Topic Format:
        - Topic: /leader/action (or configured topic_name)
        - Type: JointTrajectory
        - Expected joints: joint1-joint6, rh_r1_joint (OMY robot)
    """

    def __init__(self, cfg: LeaderDeviceCfg, env=None):
        """Initialize the leader device interface.

        Args:
            cfg: Configuration object for leader device settings.
            env: The environment instance (optional, not used in current implementation).
        """
        super().__init__()

        # Store configuration
        self._cfg = cfg
        self._sim_device = cfg.sim_device

        # DDS configuration
        if cfg.domain_id is None:
            self._domain_id = int(os.getenv("ROS_DOMAIN_ID", 30))
        else:
            self._domain_id = cfg.domain_id

        # State variables
        self._lock = threading.Lock()
        self._running = True
        self._latest_joint_positions = None
        self._additional_callbacks = {}
        self._started = False  # Whether to send actions to simulation
        self._reset_state = False  # Whether to reset environment
        self.listener = None  # Keyboard listener

        # OMY joint configuration
        self._joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"]

        # Initialize DDS subscriber
        self._topic_manager = TopicManager(domain_id=self._domain_id)
        self._reader = self._topic_manager.topic_reader(
            topic_name=cfg.topic_name,
            topic_type=JointTrajectory_
        )

        # Start background thread for DDS reading
        self._thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self._thread.start()

        # Start keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        print(f"LeaderDevice initialized: listening on topic '{cfg.topic_name}' (domain {self._domain_id})")
        print("[LeaderDevice] Recording paused. Press 'B' to start recording.")
        print("[LeaderDevice] Keyboard controls active:")
        print("  B - Start/resume recording")
        print("  R - Reset/discard episode")
        print("  N - Save successful episode")

    def __del__(self):
        """Clean up resources when device is destroyed."""
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self, 'listener') and self.listener is not None:
            self.listener.stop()
        if hasattr(self, '_reader'):
            try:
                self._reader.Close()
            except Exception:
                pass

    def __str__(self) -> str:
        """Returns: A string containing information about the leader device."""
        msg = f"Leader Device Controller: {self.__class__.__name__}\n"
        msg += f"\tTopic: {self._cfg.topic_name}\n"
        msg += f"\tDomain ID: {self._domain_id}\n"
        msg += f"\tGripper Threshold: {self._cfg.gripper_threshold}"
        return msg

    def reset(self):
        """Reset the internal state."""
        # Nothing to reset - we don't track previous states
        pass

    def add_callback(self, key: Any, func: Callable):
        """Add callback function for keyboard events.

        Args:
            key: The callback identifier.
            func: The function to call. The callback function should not take any arguments.
        """
        self._additional_callbacks[key] = func

    def _call_callback(self, key: str):
        """Trigger registered callback for a key.

        Args:
            key: The callback identifier to trigger.
        """
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def _on_press(self, key):
        """Keyboard event handler.

        Args:
            key: The key that was pressed.
        """
        try:
            if key.char == 'b':
                # Start recording
                self._started = True
                self._reset_state = False
                self._call_callback("START")
                print("[LeaderDevice] Recording STARTED")
            elif key.char == 'r':
                # Reset/discard episode
                self._started = False
                self._reset_state = True
                self._call_callback("R")
                print("[LeaderDevice] Resetting episode (discarded)")
            elif key.char == 'n':
                # Save successful episode and STOP recording
                self._started = False  # Stop recording
                self._reset_state = True
                self._call_callback("N")
                print("[LeaderDevice] Saving successful episode")
        except AttributeError:
            # Handle special keys (non-character keys)
            pass

    def advance(self) -> torch.Tensor | None:
        """Get current joint positions from leader device.

        This method reads the latest joint positions from the leader device and
        outputs them as absolute positions for the follower robot to mirror.

        Returns:
            torch.Tensor | None: A 7-element tensor containing absolute joint positions:
                [j1_pos, j2_pos, j3_pos, j4_pos, j5_pos, j6_pos, gripper_cmd]
                Returns None if reset is requested.
        """
        # Check if we should reset
        if self._reset_state:
            # Return None to signal reset needed
            self._reset_state = False  # Clear the flag
            return None

        # Check if recording is active
        if not self._started:
            # Return zero action (no movement)
            return torch.zeros(7, dtype=torch.float32, device=self._sim_device)

        # Get latest joint positions (thread-safe)
        with self._lock:
            current_joints = self._latest_joint_positions

        # If no data received yet, return zero command
        if current_joints is None:
            return torch.zeros(7, dtype=torch.float32, device=self._sim_device)

        # Map gripper position to binary command
        gripper_cmd = self._map_gripper(current_joints[6])

        # Output absolute positions (6 arm joints + gripper)
        command = np.concatenate([current_joints[:6], [gripper_cmd]])

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _subscriber_loop(self):
        """Background thread loop that continuously reads DDS messages."""
        try:
            while self._running:
                try:
                    for msg in self._reader.take_iter():
                        if msg and msg.points:
                            # Extract the latest trajectory point
                            latest_point = msg.points[-1]

                            # Map joint names to positions
                            joint_dict = dict(zip(msg.joint_names, latest_point.positions))

                            # Extract positions in correct order
                            joint_positions = np.array([
                                joint_dict.get(name, 0.0) for name in self._joint_names
                            ], dtype=np.float64)

                            # Store latest positions (thread-safe)
                            with self._lock:
                                self._latest_joint_positions = joint_positions
                except Exception as e:
                    print(f"LeaderDevice subscriber error: {e}")
        except Exception as e:
            print(f"LeaderDevice subscriber thread error: {e}")
        finally:
            if self._running:
                print("LeaderDevice subscriber thread terminated unexpectedly")

    def _map_gripper(self, gripper_pos: float) -> float:
        """Map continuous gripper position to binary command.

        Args:
            gripper_pos: Continuous gripper joint position.

        Returns:
            float: Binary gripper command (+1.0 for open, -1.0 for close).
        """
        # If gripper position is above threshold, it's closed
        if gripper_pos > self._cfg.gripper_threshold:
            return -1.0  # Close gripper
        else:
            return 1.0  # Open gripper
