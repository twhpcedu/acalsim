#!/usr/bin/env python3
"""Automated License Header Management Tool for ACAL Project.

This script automates the process of checking and updating license headers across
all source files in the ACAL project. It uses the 'license-header-checker' tool
to ensure that all source files (C++, Python, CMake, Markdown, etc.) contain the
proper Apache 2.0 license header.

The script processes different file types with appropriate comment styles:
    - C++ files (.h, .hpp, .cc, .cpp): C-style block comments
    - Python files (.py), CMake (.cmake), and text (.txt): Hash-style comments
    - Markdown files (.md): HTML-style comments

Workflow:
    1. Changes working directory to the project root
    2. Processes each file pattern group (C++, Python, Markdown)
    3. Runs license-header-checker with appropriate parameters
    4. Parses output to count modified files
    5. Exits with code 1 if any files were modified (for CI/CD validation)

Exit Codes:
    0: All files have correct license headers
    1: Some files were modified (headers added or replaced)

Usage:
    Basic usage (from any directory):
        $ python3 scripts/license-update.py

    Typical use cases:
        # Check and update all license headers
        $ ./scripts/license-update.py

        # In CI/CD pipeline (fails if headers are missing/incorrect)
        $ python3 scripts/license-update.py && echo "All headers OK" || echo "Headers updated"

Example Output:
    When all headers are correct:
        $ python3 scripts/license-update.py
        Processing C++ files...
        100 licenses ok, 0 licenses replaced, 0 licenses added
        Processing Python files...
        50 licenses ok, 0 licenses replaced, 0 licenses added
        Processing Markdown files...
        20 licenses ok, 0 licenses replaced, 0 licenses added
        $ echo $?
        0

    When headers need updating:
        $ python3 scripts/license-update.py
        Processing C++ files...
        98 licenses ok, 2 licenses replaced, 1 licenses added
        $ echo $?
        1

Dependencies:
    - license-header-checker: External command-line tool for managing license headers
      (Must be installed and available in PATH)

Configuration:
    License templates are stored in: scripts/license-templates/
    - cpp.txt: Template for C++ files
    - py.txt: Template for Python/CMake files
    - md.txt: Template for Markdown files

Notes:
    - The script automatically excludes build directories, third-party code, and
      external dependencies
    - Existing license headers matching the regex patterns are replaced
    - The script preserves shebang lines in executable scripts
"""

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

from typing import Any, Dict, List
import os
import re
import subprocess

# Directory containing license header templates for different file types
LICENSE_TEXT_FILE_DIR: str = "scripts/license-templates"

# Regex pattern to remove ANSI color codes from command output
# Matches sequences like \033[0m, \033[1;32m, etc.
COLOR_REGEX: re.Pattern = re.compile(r"\033\[[0-9;]+m")

# Regex pattern to extract statistics from license-header-checker output
# Captures three groups: (ok_count, replaced_count, added_count)
CHECK_RESULT_RE: re.Pattern = re.compile(
    r"(\d+) licenses ok, (\d+) licenses replaced, (\d+) licenses added"
)

# Configuration for different file type groups
# Each entry specifies:
#   - path: Root directory to search
#   - ignore-paths: Directories to exclude from processing
#   - contents: Name of license template file (in LICENSE_TEXT_FILE_DIR)
#   - re: Regex pattern to match existing license headers (None = auto-detect)
#   - extensions: List of file extensions to process
PATTERN_LIST: List[Dict[str, Any]] = [
    {
        # C++ source and header files
        "path": ".",
        "ignore-paths": ["build", "include/external", "libs/external", "third-party"],
        "contents": "cpp.txt",
        "re": None,  # Auto-detect C++ comment blocks
        "extensions": ["h", "hh", "inl", "hpp", "cc", "cpp"]
    },
    {
        # Python, CMake, and text files (hash-style comments)
        "path": ".",
        "ignore-paths": [
            "build", LICENSE_TEXT_FILE_DIR, "scripts/workspace/dependency", "third-party"
        ],
        "contents": "py.txt",
        "re": "(?:#.*\n?)+",  # Match consecutive lines starting with #
        "extensions": ["cmake", "py", "txt"]
    },
    {
        # Markdown files (HTML-style comments)
        "path": ".",
        "ignore-paths": ["build", "third-party", "docs/license.md"],
        "contents": "md.txt",
        "re": "<!--[\s\S]*?-->",  # Match HTML comment blocks
        "extensions": ["md"]
    }
]


def move_to_root_dir() -> None:
	"""Change the current working directory to the project root.

	This function navigates from the script's location (scripts/) to the project
	root directory. This is necessary because all paths in PATTERN_LIST are
	relative to the project root.

	The function works by:
		1. Getting the absolute path of this script file
		2. Getting the directory containing the script (scripts/)
		3. Moving up one level to the project root

	Args:
		None

	Returns:
		None

	Raises:
		OSError: If the parent directory does not exist or is not accessible.

	Example:
		If script is at /home/user/acalsim/scripts/license-update.py,
		this function changes directory to /home/user/acalsim/
	"""
	os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def gen_command(info: Dict[str, Any]) -> List[str]:
	"""Generate a license-header-checker command from configuration dictionary.

	Constructs a command-line invocation for the license-header-checker tool
	based on the provided configuration. The command includes options for
	adding/replacing headers, ignoring paths, matching existing headers,
	and specifying target file extensions.

	Args:
		info (Dict[str, Any]): Configuration dictionary containing:
			- path (str): Root directory to search for files
			- ignore-paths (List[str]): Directories to exclude
			- contents (str): Filename of license template in LICENSE_TEXT_FILE_DIR
			- re (Optional[str]): Regex pattern to match existing license headers.
				If None, auto-detection is used
			- extensions (List[str]): File extensions to process (without dots)

	Returns:
		List[str]: Command and arguments ready for subprocess.run()

	Example:
		>>> config = {
		...     "path": ".",
		...     "ignore-paths": ["build", "third-party"],
		...     "contents": "cpp.txt",
		...     "re": None,
		...     "extensions": ["h", "cpp"]
		... }
		>>> cmd = gen_command(config)
		>>> print(cmd)
		['license-header-checker', '-a', '-r', '-i', 'build,third-party',
		 'scripts/license-templates/cpp.txt', '.', 'h', 'cpp']

	Command-line flags used:
		-a: Add license headers to files that don't have them
		-r: Replace existing license headers that don't match
		-i: Comma-separated list of directories to ignore
		-e: Regular expression to match existing license headers
	"""
	cmd: List[str] = ["license-header-checker", "-a", "-r"]

	# The list of ignored directories
	if len(info["ignore-paths"]) > 0:
		cmd.extend(["-i", ",".join(info["ignore-paths"])])

	# The regular expression of existing license headers
	if info["re"] is not None:
		cmd.extend(["-e", info["re"]])

	# The file contains the contexts of license header
	cmd.append(f"{LICENSE_TEXT_FILE_DIR}/{info['contents']}")

	# The path of source files
	cmd.append(info["path"])

	# The file extensions to be processed
	cmd.extend(info["extensions"])

	return cmd


def main() -> None:
	"""Execute license header checking and updating for all file types.

	Iterates through each file pattern configuration in PATTERN_LIST and runs
	the license-header-checker tool. For each pattern group, it:
		1. Generates the appropriate command
		2. Executes the command and captures output
		3. Strips ANSI color codes from output
		4. Parses the output to extract statistics
		5. Tracks whether any files were modified

	The function exits with code 1 if ANY files were added or replaced across
	all pattern groups, indicating that the repository had files with missing
	or incorrect license headers. This behavior is useful for CI/CD pipelines
	to ensure all committed code has proper license headers.

	Args:
		None

	Returns:
		None (exits with code 1 if modifications were made)

	Exit Codes:
		0 (implicit): All files have correct license headers
		1 (explicit): At least one file was modified (header added/replaced)

	Raises:
		subprocess.SubprocessError: If license-header-checker command fails
		AttributeError: If output format doesn't match expected pattern

	Example Execution Flow:
		Processing C++ files:
			Command: ['license-header-checker', '-a', '-r', '-i', 'build,...',
			          'scripts/license-templates/cpp.txt', '.', 'h', 'cpp']
			Output: "100 licenses ok, 0 licenses replaced, 0 licenses added"
			Result: successful=True (no changes)

		Processing Python files:
			Command: ['license-header-checker', '-a', '-r', '-i', 'build,...',
			          '-e', '(?:#.*\n?)+', 'scripts/license-templates/py.txt',
			          '.', 'py', 'cmake', 'txt']
			Output: "48 licenses ok, 2 licenses replaced, 0 licenses added"
			Result: successful=False (2 files updated)

		Final: exit(1) due to modifications
	"""
	successful: bool = True

	for pattern in PATTERN_LIST:
		# Generate command for this file type group
		command: List[str] = gen_command(info=pattern)

		# Execute license-header-checker and capture output
		result: subprocess.CompletedProcess = subprocess.run(
		    command, capture_output=True, text=True
		)

		# Remove ANSI color codes for easier parsing
		stdout: str = COLOR_REGEX.sub("", result.stdout)

		# Extract statistics from output using regex
		match: re.Match[str] = re.search(pattern=CHECK_RESULT_RE, string=stdout)
		replaced_cnt: int = int(match.group(2))  # Number of headers replaced
		added_cnt: int = int(match.group(3))  # Number of headers added

		# Track whether this pattern group had any modifications
		# Both replaced and added counts must be 0 for success
		successful = (replaced_cnt == 0 and added_cnt == 0) & successful

	# Exit with error code if any files were modified
	if not successful:
		exit(1)


if __name__ == '__main__':
	# Change to project root directory before processing files
	move_to_root_dir()

	# Execute the main license checking workflow
	main()
