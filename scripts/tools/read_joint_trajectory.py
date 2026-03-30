#!/usr/bin/env python3
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

"""
Script to read and print joint states from the leader/joint_trajectory topic.

This script subscribes to the 'leader/joint_trajectory' topic and prints out
the joint names and positions in a human-readable format.

Usage:
    python read_joint_trajectory.py
"""

import os
import time
import threading
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager


def format_joint_states(msg):
    """Format joint trajectory message into readable string."""
    if not msg or not msg.points:
        return None

    # Get the last point in the trajectory (most recent command)
    point = msg.points[-1]

    # Build formatted output
    output = []
    output.append("=" * 60)
    output.append(f"Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
    output.append(f"Number of joints: {len(msg.joint_names)}")
    output.append("-" * 60)
    output.append("Joint States:")

    # Print each joint with its position
    for name, pos in zip(msg.joint_names, point.positions):
        output.append(f"  {name:20s}: {pos:8.4f} rad ({pos * 57.2958:8.2f} deg)")

    # Add velocities if available
    if point.velocities:
        output.append("-" * 60)
        output.append("Velocities:")
        for name, vel in zip(msg.joint_names, point.velocities):
            output.append(f"  {name:20s}: {vel:8.4f} rad/s")

    # Add accelerations if available
    if point.accelerations:
        output.append("-" * 60)
        output.append("Accelerations:")
        for name, acc in zip(msg.joint_names, point.accelerations):
            output.append(f"  {name:20s}: {acc:8.4f} rad/s²")

    output.append("=" * 60)

    return "\n".join(output)


class JointTrajectoryReader:
    """Class to handle joint trajectory reading in a separate thread."""

    def __init__(self, domain_id=0):
        self.domain_id = domain_id
        self.running = False
        self.message_count = 0
        self.thread = None
        self.lock = threading.Lock()
        self.current_message = None

        # Initialize topic manager and reader
        print(f"Initializing topic reader on domain {self.domain_id}...")
        topic_manager = TopicManager(domain_id=self.domain_id)
        self.reader = topic_manager.topic_reader(
            topic_name="leader/action",
            topic_type=JointTrajectory_
        )
        print(f"Reader created for DDS topic: rt/leader/action")

    def _subscriber_callback(self):
        """Callback function that runs in a separate thread."""
        print("Waiting for joint trajectory messages...")
        print("Subscriber thread running. Polling for messages...")

        iteration_count = 0
        try:
            while self.running:
                iteration_count += 1

                # Debug: Print every 100 iterations to show we're alive
                if iteration_count % 100 == 0:
                    print(f"[Debug] Still polling... (iteration {iteration_count})")

                # Try using take() which blocks until data is available
                messages = self.reader.take(N=10)  # Take up to 10 messages at once

                if messages:
                    print(f"[Debug] Received {len(messages)} message(s)")
                    for msg in messages:
                        if msg:
                            with self.lock:
                                self.message_count += 1
                                current_count = self.message_count
                                self.current_message = msg
                            formatted_output = format_joint_states(msg)
                            if formatted_output:
                                print(formatted_output)
                                print(f"Total messages received: {current_count}\n")
                else:
                    # Small sleep to prevent busy-waiting if no messages
                    time.sleep(0.01)  # 10ms delay

        except Exception as e:
            print(f"Error in subscriber thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Subscriber thread stopped.")

    def start(self):
        """Start the subscriber thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._subscriber_callback, daemon=True)
            self.thread.start()
            print("Subscriber thread started.")

    def stop(self):
        """Stop the subscriber thread."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            try:
                self.reader.Close()
                print("Reader closed successfully.")
            except Exception as e:
                print(f"Error closing reader: {e}")


def main():
    """Main function to subscribe and print joint trajectory data."""
    # Get ROS domain ID from environment
    domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
    print(f"Starting joint trajectory reader on domain {domain_id}...")
    print("Subscribing to topic: leader/joint_trajectory")
    print("Press Ctrl+C to stop.\n")

    # Create and start the reader
    trajectory_reader = JointTrajectoryReader(domain_id=domain_id)
    trajectory_reader.start()

    try:
        # Keep main thread alive
        last_count = 0
        while True:
            with trajectory_reader.lock:
                print(trajectory_reader.current_message)  # Debug: Print the current message object
            time.sleep(1.0)  # Check every second
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        trajectory_reader.stop()


if __name__ == "__main__":
    main()
