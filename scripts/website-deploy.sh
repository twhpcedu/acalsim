#!/bin/bash

# ==============================================================================
# Beta Documentation Server Deployment Script
# ==============================================================================
#
# Description:
#   Automated deployment script for updating the ACALSim documentation on the
#   beta documentation server. Connects via SSH and pulls the latest changes
#   from the Git repository.
#
# Usage:
#   ./website-deploy.sh
#
# Prerequisites:
#   1. Environment variables must be set:
#      - BETA_DOCSERVER_USERNAME: SSH username for the doc server
#      - BETA_DOCSERVER_IP: IP address or hostname of the doc server
#   2. Password file must exist:
#      - .secure_files/beta_docserver_password.txt
#   3. sshpass must be installed for password-based authentication
#
# Required Tools:
#   - sshpass: For non-interactive SSH password authentication
#   - ssh: OpenSSH client for remote connection
#   - git: Version control (on remote server)
#
# Workflow:
#   1. Read environment variables for server connection details
#   2. Load SSH password from secure file
#   3. Connect to beta doc server via SSH with password authentication
#   4. Navigate to repository directory on remote server
#   5. Execute 'git pull' to update to latest version
#   6. Disconnect from server
#
# Security Notes:
#   - Password stored in .secure_files/ (should be in .gitignore)
#   - StrictHostKeyChecking disabled (suitable for known beta servers)
#   - Consider using SSH keys instead of password authentication for production
#
# Exit Codes:
#   0: Success - Repository updated successfully on beta server
#   1: Failure - Connection error, authentication failure, or git pull failed
#
# Example Output:
#   Connecting to user@192.168.1.100...
#   Already up to date. (or commit messages if updates were pulled)
#
# Author: ACAL Playlab
# Copyright: 2023-2025 Playlab/ACAL
# ==============================================================================

# Construct SSH connection string from environment variables
SERVER="${BETA_DOCSERVER_USERNAME}@${BETA_DOCSERVER_IP}"

# Path to file containing SSH password
PASSWORD_FILE="./.secure_files/beta_docserver_password.txt"

# Repository directory on the remote server
REPO_DIRECTORY="/home/${BETA_DOCSERVER_USERNAME}/acalsim-workspace/projects/acalsim/"

# Read password from secure file
PASSWORD=$(<"${PASSWORD_FILE}")

# Connect to server and update repository
# -p: Provide password
# -o StrictHostKeyChecking=no: Skip host key verification (beta environment)
# -ttv: Force pseudo-terminal allocation and verbose output
sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no -ttv "${SERVER}" <<EOF
cd $REPO_DIRECTORY
git pull
exit
EOF
