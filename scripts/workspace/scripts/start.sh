#!/bin/bash

# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# Docker Workspace Container Startup Script
################################################################################
#
# DESCRIPTION:
#   This script is the primary initialization script executed when the ACALSim
#   Docker workspace container starts. It handles critical user and group ID
#   synchronization between the host system and the container, ensuring proper
#   file permissions for shared volumes. The script configures the container's
#   user environment to match the host user's UID/GID, preventing permission
#   issues when accessing mounted directories.
#
# PURPOSE:
#   - Synchronize container user UID/GID with host system user
#   - Ensure correct ownership and permissions for mounted volumes
#   - Configure container environment for seamless file sharing with host
#   - Keep container running indefinitely after initialization
#   - Provide a signal file to indicate successful startup completion
#
# WORKFLOW:
#   Step 1: Change Ownership of Home Directory
#     - Checks if USERID and GROUPID environment variables are set
#     - Compares them with current container user's UID/GID
#     - If different, recursively changes ownership of all files in home directory
#     - Updates ownership of hidden files (dotfiles) and regular files
#     - Changes ownership of the home directory itself
#
#   Step 2: Configure Group ID (GID)
#     - Checks if GROUPID is set and differs from current GID
#     - If target GID doesn't exist: modifies current group to use new GID
#     - If target GID exists: adds user to existing group and sets as primary
#     - Ensures user has proper group membership for file access
#
#   Step 3: Configure User ID (UID)
#     - Checks if USERID is set and differs from current UID
#     - Modifies the container user's UID to match the host user
#     - Preserves user name and home directory location
#     - Critical for proper file permission handling on mounted volumes
#
#   Step 4: Signal Completion
#     - Creates /docker/start.sh.done file as completion marker
#     - Other scripts can check for this file to ensure startup finished
#     - Useful for orchestration and initialization dependencies
#
#   Step 5: Keep Container Running
#     - Executes 'tail -f /dev/null' to prevent container from exiting
#     - Allows container to remain active for interactive use
#     - Container stays alive until explicitly stopped
#
# USAGE:
#   This script is automatically executed when the container starts. It should
#   not be called manually by users under normal circumstances.
#
#   Example 1: Starting container with docker-compose
#     docker-compose up -d
#     # start.sh is executed automatically as container's ENTRYPOINT or CMD
#
#   Example 2: Starting container with docker run
#     docker run -d \
#       -e USERID=$(id -u) \
#       -e GROUPID=$(id -g) \
#       -v /host/path/projects:/home/user/projects \
#       acalsim-workspace
#     # start.sh runs automatically, synchronizing UID/GID
#
#   Example 3: Manual execution (for debugging)
#     docker exec acalsim-workspace /path/to/start.sh
#     # Useful for troubleshooting startup issues
#
# EXPECTED OUTPUT:
#   The script produces minimal output under normal operation. On success:
#
#     (No output - silent execution)
#     Container continues running in background
#     File /docker/start.sh.done is created
#
#   If UID/GID changes are made, you might see sudo prompts internally:
#
#     [sudo] password for user: (auto-handled if properly configured)
#
#   To verify successful execution:
#     docker exec acalsim-workspace ls -la /docker/start.sh.done
#     docker exec acalsim-workspace id
#     # Should show USERID and GROUPID matching host values
#
# ENVIRONMENT VARIABLES:
#   USERID (required):
#     The user ID from the host system that should be used in the container.
#     Typically passed as: -e USERID=$(id -u)
#     Example: USERID=1000
#     Default: If not set, no UID synchronization occurs
#
#   GROUPID (required):
#     The group ID from the host system that should be used in the container.
#     Typically passed as: -e GROUPID=$(id -g)
#     Example: GROUPID=1000
#     Default: If not set, no GID synchronization occurs
#
# PREREQUISITES:
#   - Docker container with sudo configured for the container user
#   - Container user must have passwordless sudo access (or sudo configured)
#   - Base user and group must exist in container before script runs
#   - /docker directory must exist with write permissions
#   - sudo, usermod, groupmod, chown commands must be available
#   - Home directory must exist at /home/<username>
#
# DOCKER INTEGRATION:
#   This script should be configured as the container's startup command:
#
#   In Dockerfile:
#     ENTRYPOINT ["/path/to/start.sh"]
#     # or
#     CMD ["/path/to/start.sh"]
#
#   In docker-compose.yml:
#     services:
#       workspace:
#         image: acalsim-workspace
#         command: /path/to/start.sh
#         environment:
#           - USERID=${USERID}
#           - GROUPID=${GROUPID}
#         volumes:
#           - ./projects:/home/user/projects
#
# FILE PERMISSIONS AND OWNERSHIP:
#   Understanding the UID/GID synchronization:
#
#   Problem: Container user UID (e.g., 1001) differs from host user UID (e.g., 1000)
#   Impact: Files created in container appear owned by wrong user on host
#   Solution: This script changes container user UID to match host user UID
#
#   Example scenario:
#     Host user: alice (UID=1000, GID=1000)
#     Container user: developer (UID=1001, GID=1001)
#     After script: developer (UID=1000, GID=1000)
#     Result: Files created by 'developer' in container appear owned by 'alice' on host
#
# EXIT CODES:
#   The script uses 'tail -f /dev/null' to keep running, so it normally doesn't exit.
#   If the script encounters errors before the tail command:
#
#   0   - Success (rare, as script should keep running)
#   1   - General error (file operations, permission issues)
#   126 - Permission denied (sudo not configured properly)
#   127 - Command not found (usermod, groupmod, or chown missing)
#
# SIGNAL FILE:
#   /docker/start.sh.done
#     Created after successful initialization
#     Can be checked by other scripts to ensure startup is complete
#     Example: while [ ! -f /docker/start.sh.done ]; do sleep 1; done
#
# SECURITY CONSIDERATIONS:
#   - Script requires sudo access for user modification commands
#   - Ensure sudo is configured securely in the container
#   - USERID and GROUPID should come from trusted sources
#   - Be cautious when running containers with host user privileges
#   - Changing UIDs affects all files owned by the original user
#
# TROUBLESHOOTING:
#   Problem: Permission denied errors when accessing mounted volumes
#   Solution: Verify USERID and GROUPID are correctly passed to container
#             Check: docker exec container_name id
#             Should match: id on host system
#
#   Problem: Container exits immediately after starting
#   Solution: Check for errors in script execution
#             Run: docker logs container_name
#             Look for: usermod, groupmod, or chown errors
#
#   Problem: Home directory files owned by wrong user
#   Solution: Ensure script runs before user creates files
#             Check: ls -la /home/$(id -un)
#             Re-run ownership change if needed
#
#   Problem: "Group already exists" errors
#   Solution: Script handles this by adding user to existing group
#             If issues persist, check /etc/group in container
#
#   Problem: Sudo prompts for password
#   Solution: Configure passwordless sudo for container user
#             Add to /etc/sudoers: user ALL=(ALL) NOPASSWD:ALL
#
# NOTES:
#   - The script uses [[ ]] for conditional tests (bash-specific syntax)
#   - Sudo commands assume passwordless sudo is configured
#   - The tail -f /dev/null trick is a common Docker pattern for keeping containers alive
#   - Hidden files (dotfiles) are handled separately with .[!.]* pattern
#   - Script is idempotent: safe to run multiple times with same values
#
# SEE ALSO:
#   - login.sh: Script executed when user logs into the container
#   - startup.sh: Displays welcome message and workspace information
#   - Docker ENTRYPOINT documentation: https://docs.docker.com/engine/reference/builder/#entrypoint
#   - Linux user management: man usermod, man groupmod, man chown
#
################################################################################

# This script will be executed while the container is starting.

# change the ownership of the home directory
if [[ (${USERID} != "" && ${GROUPID} != "") && (${USERID} != "$(id -u)" || ${GROUPID} != "$(id -g)") ]]; then
	for file in "/home/$(id -un)"/.[!.]* "/home/$(id -un)"/*; do
		sudo chown "${USERID}":"${GROUPID}" "${file}"
	done
	sudo chown "${USERID}":"${GROUPID}" "/home/$(id -un)"
fi

# configure the GID of the user in container to match the host system
if [[ ${GROUPID} != "" && ${GROUPID} != "$(id -g)" ]]; then
	if [[ $(getent group "${GROUPID}") == "" ]]; then
		sudo groupmod -g "${GROUPID}" "$(id -gn)"
	else
		sudo usermod -aG "$(getent group "${GROUPID}" | cut -d: -f1)" "$(id -un)"
		sudo usermod -g "$(getent group "${GROUPID}" | cut -d: -f1)" "$(id -un)"
	fi
fi

# configure the UID of the user in container to match the host system
if [[ ${USERID} != "" && ${USERID} != "$(id -u)" ]]; then
	sudo usermod -u "${USERID}" "$(id -un)"
fi

# create a file to indicate this script has finished
touch /docker/start.sh.done

# keep the container running
tail -f /dev/null
