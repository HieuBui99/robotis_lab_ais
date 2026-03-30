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
Script to test DDS connection and list discovered entities.

This is a simplified diagnostic tool that attempts to connect to DDS
and verify basic connectivity.

Usage:
    Run inside Docker container:
    python list_dds_topics.py [domain_id]
"""

import os
import sys
import time


def test_dds_connection(domain_id):
    """Test basic DDS connection."""
    print(f"Testing DDS connection on domain {domain_id}...")
    print("Please wait for discovery...\n")

    try:
        from cyclonedds.domain import DomainParticipant
        participant = DomainParticipant(domain_id)
        print(f"✓ Successfully created DomainParticipant on domain {domain_id}")

        # Wait for discovery
        time.sleep(3)
        print("✓ Discovery period completed")

        # Try to import the builtin module to check what's available
        print("\nChecking cyclonedds.builtin module...")
        try:
            from cyclonedds import builtin
            available_attrs = [attr for attr in dir(builtin) if not attr.startswith('_')]
            print(f"Available in builtin module: {available_attrs}")
        except Exception as e:
            print(f"✗ Error checking builtin: {e}")

        print("\n" + "="*80)
        print("DDS Connection Test Complete")
        print("="*80)
        print("\nIf you're trying to debug topic subscription:")
        print("1. Make sure the publisher is running")
        print("2. Verify ROS_DOMAIN_ID matches on both sides")
        print("3. Check network connectivity")
        print("4. Try running: read_joint_trajectory.py")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    # Get domain ID from command line or environment
    if len(sys.argv) > 1:
        domain_id = int(sys.argv[1])
    else:
        domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))

    print(f"DDS Connection Test Tool")
    print(f"Domain ID: {domain_id}")
    print("="*80)

    try:
        test_dds_connection(domain_id)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


if __name__ == "__main__":
    main()

