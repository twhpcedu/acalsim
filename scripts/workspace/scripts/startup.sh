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
# Docker Workspace Welcome Message and Information Display Script
################################################################################
#
# DESCRIPTION:
#   This script displays a comprehensive welcome message and workspace
#   information when users log into the ACALSim Docker workspace container.
#   It provides essential information about the containerized development
#   environment, including mounted directories, available commands, and
#   best practices for working within the workspace.
#
# PURPOSE:
#   - Welcome users to the ACALSim Docker workspace environment
#   - Display version information about the workspace
#   - Show which directories are mounted and their host mappings
#   - List available commands for container management
#   - Provide guidance on where to store persistent data
#   - Help users understand the workspace layout and capabilities
#
# WORKFLOW:
#   Step 1: Display Welcome Banner
#     - Shows ACALSim workspace welcome message
#     - Displays version information (placeholder 'VERSION' to be replaced)
#     - Provides visual separation for easy reading
#
#   Step 2: Explain Persistent Storage
#     - Instructs users to work in ~/projects/ for data persistence
#     - Explains which directories survive container restarts
#     - Highlights mounted directories (.ssh, .vscode-server)
#
#   Step 3: List Available Commands
#     - Shows container-specific commands (startup, ssh-keygen, exit)
#     - Provides brief description of each command's purpose
#     - Helps users navigate the container environment
#
#   Step 4: Display Mounted Directory Mappings
#     - Lists all mounted directories in workspace
#     - Shows corresponding host machine paths
#     - Uses MOUNT_DIR environment variable for host path display
#     - Format: CONTAINER_PATH : HOST_PATH
#
#   Step 5: Output via cat and HEREDOC
#     - Uses cat <<EOF for clean, readable output
#     - Allows embedded variables (MOUNT_DIR) to be expanded
#     - Produces formatted text without complex echo statements
#
# USAGE:
#   This script is typically called automatically via login.sh, but can also
#   be executed manually at any time.
#
#   Example 1: Automatic execution on login
#     docker exec -it acalsim-workspace bash
#     # startup.sh is called automatically by login.sh
#     # Welcome message appears immediately
#
#   Example 2: Manual execution to redisplay information
#     startup
#     # Displays the welcome message again
#     # Useful if terminal was cleared or information is needed
#
#   Example 3: Direct script invocation
#     /usr/local/bin/startup
#     # or
#     bash /path/to/startup.sh
#     # Shows workspace information
#
#   Example 4: Pipe to file for documentation
#     startup > workspace-info.txt
#     # Save workspace configuration details to a file
#
# EXPECTED OUTPUT:
#   When executed, the script produces the following output:
#
#     Welcome to use the Docker Workspace for ACALSim projects (version 1.0.0).
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
#         ~/projects/             : /Users/username/acalsim-workspace/projects/
#         ~/.config/              : /Users/username/acalsim-workspace/temp/config/
#         ~/.ssh/                 : /Users/username/acalsim-workspace/temp/.ssh/
#         ~/.cache/pre-commit/    : /Users/username/acalsim-workspace/temp/pre-commit/
#         ~/.vscode-server/       : /Users/username/acalsim-workspace/temp/.vscode-server/
#
# ENVIRONMENT VARIABLES:
#   MOUNT_DIR (required):
#     The base directory on the host machine where workspace volumes are mounted.
#     This variable is displayed in the mounted directories section to show users
#     where their container files are located on the host system.
#
#     Example values:
#       MOUNT_DIR=/Users/username/acalsim-workspace
#       MOUNT_DIR=/home/username/docker-workspaces/acalsim
#
#     Must be set when container is started:
#       docker run -e MOUNT_DIR=/path/to/workspace acalsim-workspace
#
#     If not set, output will show:
#       ~/projects/             : /projects/
#       (Missing leading path component)
#
# MOUNTED DIRECTORIES EXPLAINED:
#   The script displays several mounted directories. Here's what each is for:
#
#   ~/projects/ -> ${MOUNT_DIR}/projects/
#     Primary workspace directory for all development projects
#     All code, data, and project files should be stored here
#     Persists across container restarts and rebuilds
#     Shared between container and host for easy file access
#
#   ~/.config/ -> ${MOUNT_DIR}/temp/config/
#     Application configuration files and settings
#     Preserves tool configurations (git config, editor settings, etc.)
#     Allows consistent environment across container sessions
#
#   ~/.ssh/ -> ${MOUNT_DIR}/temp/.ssh/
#     SSH keys and configuration for Git operations
#     Enables SSH authentication without recreating keys
#     Preserves known_hosts and SSH config files
#
#   ~/.cache/pre-commit/ -> ${MOUNT_DIR}/temp/pre-commit/
#     Pre-commit hook cache for faster execution
#     Stores downloaded hook repositories
#     Reduces setup time when using pre-commit hooks
#
#   ~/.vscode-server/ -> ${MOUNT_DIR}/temp/.vscode-server/
#     VS Code Remote Development server files
#     Preserves VS Code extensions and settings
#     Enables seamless VS Code remote development experience
#
# AVAILABLE COMMANDS:
#   The script lists these commands for container interaction:
#
#   startup:
#     Re-displays this welcome message and information
#     Usage: startup
#     Useful when information is needed after terminal is cleared
#
#   ssh-keygen:
#     Standard SSH key generation utility
#     Usage: ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
#     Creates SSH keys for Git authentication
#
#   exit:
#     Exits the current container shell session
#     Usage: exit
#     Disconnects from container but leaves it running
#
# PREREQUISITES:
#   - Bash shell environment
#   - MOUNT_DIR environment variable should be set (optional but recommended)
#   - Script should be installed at /usr/local/bin/startup or similar location
#   - Execute permissions on the script file
#
# INSTALLATION:
#   To make this script available as 'startup' command:
#
#   1. Copy to system binary directory:
#      sudo cp startup.sh /usr/local/bin/startup
#
#   2. Set execute permissions:
#      sudo chmod +x /usr/local/bin/startup
#
#   3. Verify installation:
#      which startup
#      # Should output: /usr/local/bin/startup
#
#   4. Test execution:
#      startup
#      # Should display welcome message
#
# VERSION PLACEHOLDER:
#   The script contains a VERSION placeholder that should be replaced during
#   container build or at runtime:
#
#   During Dockerfile build:
#     RUN sed -i 's/VERSION/1.0.0/g' /usr/local/bin/startup
#
#   At container startup:
#     export WORKSPACE_VERSION="1.0.0"
#     sed -i "s/VERSION/${WORKSPACE_VERSION}/g" /usr/local/bin/startup
#
#   Or use envsubst for more robust replacement:
#     export VERSION="1.0.0"
#     envsubst < startup.sh > /usr/local/bin/startup
#
# EXIT CODES:
#   0 - Success: Message displayed successfully
#   1 - Error: Failed to execute cat or heredoc
#
#   The script typically always succeeds unless there are severe system issues.
#   Even if MOUNT_DIR is not set, the script will still run and display output.
#
# INTEGRATION WITH OTHER SCRIPTS:
#   login.sh:
#     Calls this script when user logs into container
#     Provides automatic welcome message on container entry
#
#   start.sh:
#     Container initialization script (runs before this)
#     Sets up environment that this script displays
#
#   Typical call chain:
#     1. Container starts -> start.sh executes
#     2. User logs in -> login.sh executes
#     3. login.sh calls -> startup.sh (this script)
#     4. Welcome message displayed
#
# CUSTOMIZATION:
#   To customize the welcome message:
#
#   1. Edit the text between cat <<EOF and EOF
#   2. Add new sections following the existing format
#   3. Use ${VARIABLE} syntax for environment variable expansion
#   4. Maintain consistent indentation for readability
#
#   Example: Add a new mounted directory
#     ~/.custom-config/       : ${MOUNT_DIR}/temp/.custom-config/
#
# NOTES:
#   - Uses HEREDOC (<<EOF) for multi-line output
#   - Environment variables in HEREDOC are expanded by default
#   - Clean, readable format makes it easy to modify message content
#   - Script has no side effects - only displays information
#   - Safe to run multiple times without any negative impact
#   - Output is sent to stdout and can be redirected or piped
#
# TROUBLESHOOTING:
#   Problem: MOUNT_DIR shows as empty in output
#   Solution: Ensure MOUNT_DIR is set when starting the container
#             docker run -e MOUNT_DIR=/path/to/workspace ...
#
#   Problem: Command not found when typing 'startup'
#   Solution: Check if script is in PATH and executable
#             Run: ls -l /usr/local/bin/startup
#             Run: chmod +x /usr/local/bin/startup
#
#   Problem: VERSION placeholder not replaced
#   Solution: Add version replacement to Dockerfile or entrypoint
#             RUN sed -i 's/VERSION/1.0.0/' /usr/local/bin/startup
#
#   Problem: Mounted directories don't match actual mounts
#   Solution: Update the HEREDOC content to reflect actual volume mounts
#             Check actual mounts: docker inspect container_name
#
# SEE ALSO:
#   - login.sh: Container login script that calls this script
#   - start.sh: Container initialization and startup script
#   - Bash HEREDOC: https://tldp.org/LDP/abs/html/here-docs.html
#   - Docker volumes: https://docs.docker.com/storage/volumes/
#
################################################################################

cat <<EOF

    Welcome to use the Docker Workspace for ACALSim projects (version VERSION).

    Develop all your projects in '~/projects/' to keep them after the container closed.
    In addition, '~/.ssh' and '~/.vscode-server' are also mounted to your host machine.
    You can easily share files between the workspace and your host machine via these mounted folders.

    The following are some useful commands for this environment.

    [ Container ]
        startup     : Show this message.
        ssh-keygen  : Generate SSH private & public key.
        exit        : Leave the container.

    [ Mounted Directories (WORKSPACE : HOST) ]
        ~/projects/             : ${MOUNT_DIR}/projects/
        ~/.config/              : ${MOUNT_DIR}/temp/config/
        ~/.ssh/                 : ${MOUNT_DIR}/temp/.ssh/
        ~/.cache/pre-commit/    : ${MOUNT_DIR}/temp/pre-commit/
        ~/.vscode-server/       : ${MOUNT_DIR}/temp/.vscode-server/

EOF
