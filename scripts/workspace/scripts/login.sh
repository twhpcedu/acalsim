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
# Docker Workspace Login Script
################################################################################
#
# DESCRIPTION:
#   This script serves as the entry point for interactive login sessions into
#   the ACALSim Docker workspace container. It is typically executed when a
#   user attaches to a running container via 'docker exec' or similar commands.
#   The script displays a welcome message and environment information to help
#   users understand the workspace layout and available resources.
#
# PURPOSE:
#   - Provide a user-friendly greeting when entering the Docker container
#   - Display workspace configuration and mounted directory mappings
#   - Show available commands and utilities within the container
#   - Help users navigate the containerized development environment
#
# WORKFLOW:
#   1. Script is invoked when user logs into the container
#   2. Executes the startup script located at /usr/local/bin/startup
#   3. Startup script displays:
#      - Welcome banner with version information
#      - List of mounted directories (container path : host path)
#      - Available container commands (startup, ssh-keygen, exit)
#      - Development guidelines and best practices
#   4. User is presented with an interactive shell in the container
#
# USAGE:
#   This script is typically not called directly by users. Instead, it is
#   automatically executed when entering the container. Common scenarios:
#
#   Example 1: Login via docker exec
#     docker exec -it acalsim-workspace bash
#     # login.sh is executed automatically, displaying welcome message
#
#   Example 2: Attach to running container
#     docker attach acalsim-workspace
#     # Container shell appears with startup information
#
#   Example 3: Manual invocation (if needed)
#     ./login.sh
#     # Displays workspace information and environment details
#
# EXPECTED OUTPUT:
#   When executed, you will see:
#
#     Welcome to use the Docker Workspace for ACALSim projects (version X.X.X).
#
#     Develop all your projects in '~/projects/' to keep them after the container closed.
#     In addition, '~/.ssh' and '~/.vscode-server' are also mounted to your host machine.
#     You can easily share files between the workspace and your host machine via these mounted folders.
#
#     The following are some useful commands for this environment.
#
#     [ Container ]
#         startup     : Show this message.
#         ssh-keygen  : Generate SSH private & public key.
#         exit        : Leave the container.
#
#     [ Mounted Directories (WORKSPACE : HOST) ]
#         ~/projects/             : /path/to/host/projects/
#         ~/.config/              : /path/to/host/temp/config/
#         ~/.ssh/                 : /path/to/host/temp/.ssh/
#         ~/.cache/pre-commit/    : /path/to/host/temp/pre-commit/
#         ~/.vscode-server/       : /path/to/host/temp/.vscode-server/
#
# ENVIRONMENT VARIABLES:
#   MOUNT_DIR (via startup script):
#     The base directory on the host machine where workspace volumes are mounted.
#     This variable is used to display the host-side paths of mounted directories.
#     Example: MOUNT_DIR=/Users/username/acalsim-workspace
#
# DEPENDENCIES:
#   - /usr/local/bin/startup: The startup script that displays environment info
#   - bash: Required shell interpreter
#   - Docker container must be properly configured with environment variables
#
# PREREQUISITES:
#   - Container must be running (started via start.sh)
#   - User permissions must be properly configured
#   - Mounted volumes should be in place for proper directory mapping display
#   - startup script must exist at /usr/local/bin/startup
#
# INTEGRATION WITH OTHER SCRIPTS:
#   - start.sh: Starts the container (must run before login)
#   - startup.sh: Displays the welcome message (called by this script)
#   - This script is typically configured as the container's login shell or
#     executed automatically via .bashrc or .profile
#
# EXIT CODES:
#   0  - Success: Startup script executed successfully
#   127 - Command not found: /usr/local/bin/startup does not exist
#   126 - Permission denied: Startup script is not executable
#   Other - Errors from the startup script execution
#
# NOTES:
#   - This script should be kept simple to avoid login failures
#   - Any errors in startup script will be displayed but won't prevent login
#   - The script uses bash explicitly to ensure consistent behavior
#   - Can be re-run manually by typing 'startup' in the container shell
#
# TROUBLESHOOTING:
#   Problem: Welcome message doesn't appear
#   Solution: Check if /usr/local/bin/startup exists and is executable
#             Run: ls -l /usr/local/bin/startup
#
#   Problem: Mounted directories show incorrect paths
#   Solution: Verify MOUNT_DIR environment variable is set correctly
#             Run: echo $MOUNT_DIR
#
#   Problem: Permission denied errors
#   Solution: Ensure startup script has execute permissions
#             Run: sudo chmod +x /usr/local/bin/startup
#
# SEE ALSO:
#   - start.sh: Container startup and initialization script
#   - startup.sh: Welcome message and information display script
#   - Docker documentation: https://docs.docker.com/engine/reference/commandline/exec/
#
################################################################################

bash "/usr/local/bin/startup"
