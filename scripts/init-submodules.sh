#!/bin/bash
# Copyright 2023-2026 Playlab/ACAL
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

set -e

echo "Initializing repository and submodules..."
rm -rf third-party/*/

echo "Attempting to initialize git submodules..."
git submodule init || true
git submodule update --init --recursive --depth 1 || echo "Git submodule update failed, trying manual clone..."

if [ ! -d "third-party/googletest/.git" ]; then
	echo "Manually cloning missing submodules..."
	cd third-party
	[ ! -d "googletest/.git" ] && git clone --depth 1 https://github.com/google/googletest.git || true
	[ ! -d "json/.git" ] && git clone --depth 1 https://github.com/nlohmann/json.git || true
	[ ! -d "CLI11/.git" ] && git clone --depth 1 https://github.com/CLIUtils/CLI11.git || true
	[ ! -d "cpp-channel/.git" ] && git clone --depth 1 https://github.com/andreiavrammsd/cpp-channel.git || true
	[ ! -d "systemc/.git" ] && git clone --depth 1 https://github.com/accellera-official/systemc.git || true
	cd ..
fi

echo "Verifying submodules..."
find third-party -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | xargs -I {} echo "Found {} third-party directories"

pre-commit install || echo "pre-commit not installed, skipping hook installation"

echo "Repository initialization complete"
