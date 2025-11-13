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

# SST Environment Setup Script
# Source this script to configure SST environment in Docker container

# Detect SST installation location
SST_LOCATIONS=(
	"/home/user/projects/acalsim/sst-core/sst-core-install"
	"/home/user/projects/acalsim/build/sst-core/sst-core-install"
	"/usr/local"
)

SST_FOUND=false

# Check if SST is already in PATH
if command -v sst &>/dev/null; then
	echo "SST found in system PATH"
	echo "  SST version: $(sst --version 2>&1 | head -1)"
	SST_FOUND=true
else
	# Try known locations
	for location in "${SST_LOCATIONS[@]}"; do
		if [ -d "$location" ] && [ -f "$location/bin/sst" ]; then
			export SST_CORE_HOME=$location
			export PATH=$SST_CORE_HOME/bin:$PATH
			export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

			echo "SST environment configured:"
			echo "  SST_CORE_HOME: $SST_CORE_HOME"
			echo "  SST version: $(sst --version 2>&1 | head -1)"
			SST_FOUND=true
			break
		fi
	done
fi

if [ "$SST_FOUND" = false ]; then
	echo "ERROR: SST not found in any known location"
	echo ""
	echo "Checked locations:"
	for location in "${SST_LOCATIONS[@]}"; do
		echo "  - $location"
	done
	echo ""
	echo "Please install SST-Core first. See:"
	echo "  - /home/user/projects/acalsim/SST-INTEGRATION.md"
	echo "  - src/qemu-sst/DOCKER.md"
	echo ""
	echo "Quick install:"
	echo "  cd /home/user/projects/acalsim"
	echo "  # Follow Step 0 in SST-INTEGRATION.md"
	return 1
fi

# Verify SST is now accessible
if ! command -v sst &>/dev/null; then
	echo "ERROR: SST still not found after setup"
	return 1
fi

if ! command -v sst-config &>/dev/null; then
	echo "ERROR: sst-config not found after setup"
	return 1
fi

echo "✓ SST environment ready"
