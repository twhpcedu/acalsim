#!/usr/bin/env python3
"""Automated regression testing framework for ACAL simulator projects.

This script provides a comprehensive regression testing system that compiles and
executes multiple simulator projects with different build configurations (Debug,
Release, GTest). It validates simulation outputs, captures logs, and reports
results in a formatted manner.

The regression framework:
    1. Configures and builds projects using CMake
    2. Executes compiled binaries with specified arguments
    3. Validates simulation completion ticks against expected values/ranges
    4. Captures all outputs to organized log files
    5. Reports pass/fail status with color-coded terminal output

Typical Usage:
    Run all regression tests with default 30-second timeout:
        $ python scripts/regression.py

    Run tests with custom timeout (60 seconds):
        $ python scripts/regression.py --timeout 60

    Run tests with no timeout (useful for debugging):
        $ python scripts/regression.py --timeout 0

Expected Output:
    [  Debug  ] Simple CPU Test .................... Passed
    [ Release ] Cache Hierarchy Test ............... Passed
    [  Debug  ] Multi-Core Simulation .............. Execute Timeout
    [ Release ] Memory System Test ................. Compile Failed

Log Structure:
    build/regression/
        debug/
            example1/
                build.log       # CMake configuration and compilation output
                test1-exec.log  # Execution logs with pre/post steps
        release/
            example2/
                build.log
                test2-exec.log

Project Configuration:
    Projects are defined in the PROJ_ARR configuration imported from acalsim module.
    Each project specifies:
        - name: Display name for reporting
        - log-name: Log file prefix
        - src-subdir: Source directory name
        - exec-args: Command-line arguments for execution
        - compile-mode: Build type (Debug/Release/GTest)
        - pre-steps: Commands to run before simulation
        - post-steps: Commands to run after simulation
        - total-tick: Expected simulation completion tick (exact value)
        - total-tick-range: Expected tick range [min, max] (alternative to total-tick)

Validation:
    The script validates simulation correctness by:
        1. Checking that simulation completes (prints completion message)
        2. Verifying completion tick matches expected value (if total-tick specified)
        3. Verifying completion tick falls within range (if total-tick-range specified)

    Validation failures result in "Incorrect Result" status.

Exit Codes:
    0: All tests passed
    1: One or more tests failed or keyboard interrupt

Dependencies:
    - click: Command-line interface framework
    - CMake: Build system
    - Configured acalsim projects (PROJ_ARR configuration)

Note:
    This script must be run from the project root directory or will automatically
    change to it using move_to_root_dir().
"""

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

from typing import Dict, List, Tuple, Optional
import sys
import os
import re
import subprocess

import click

from acalsim import PROJ_ARR

# Directory and file path constants
BUILD_DIR: str = "build"  # Root build directory for compiled binaries
REGRESSION_LOG_DIR: str = "build/regression"  # Directory for all regression test logs
BUILD_LOG_FILENAME: str = "build.log"  # Filename for CMake/compilation logs
EXEC_LOG_FILENAME: str = "exec.log"  # Filename suffix for execution logs

# Build configuration constants
COMPILE_TYPE_LIST: List[str] = ["Debug", "Release", "GTest", "SST"]  # Supported build types
IGNORED_COMPILE_TYPE: List[str] = [
    COMPILE_TYPE_LIST[2], COMPILE_TYPE_LIST[3]
]  # Build types that skip CMAKE_BUILD_TYPE flag (GTest, SST)

# Regular expressions for log parsing
COLOR_REGEX: re.Pattern = re.compile(r"\033\[[0-9;]+m")  # Matches ANSI color escape codes
# Matches both native ACALSim and SST completion messages
TIMETICK_REGEX: re.Pattern = re.compile(
    r"(?:Tick=(\d+) Info: \[.+\] Simulation complete\.|All simulators done at tick (\d+))"
)  # Extracts simulation completion tick

# Global formatting variables (set dynamically by task_analysis())
MAX_COMPILEMODE_LENGTH: int = 0  # Maximum compile mode string length for aligned output
MAX_LINE_WIDTH: int = 0  # Maximum line width for result formatting


class ValidationError(Exception):
	"""Custom exception raised when simulation output validation fails.

	This exception is raised when:
	    - Simulation does not print completion message
	    - Completion tick does not match expected value
	    - Completion tick falls outside expected range

	Args:
	    message (str): Detailed error message describing the validation failure.

	Example:
	    >>> raise ValidationError("Expected tick 1000, got 999")
	"""

	def __init__(self, message):
		super().__init__(message)


def move_to_root_dir() -> None:
	"""Change working directory to the project root.

	Navigates from the script's location (scripts/) up one level to the project
	root directory. This ensures all relative paths used in the script (like
	'build/', 'src/', etc.) resolve correctly regardless of where the script
	is invoked from.

	Args:
	    None

	Returns:
	    None

	Example:
	    If script is at /path/to/project/scripts/regression.py,
	    sets cwd to /path/to/project/
	"""
	os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def remove_existing_logs() -> None:
	"""Remove existing regression log directory to start fresh.

	Deletes the entire regression log directory tree if it exists. This ensures
	each regression run starts with a clean slate and prevents confusion from
	stale log files from previous runs.

	Args:
	    None

	Returns:
	    None

	Raises:
	    subprocess.CalledProcessError: If the rm command fails.

	Note:
	    This function is safe to call even if REGRESSION_LOG_DIR doesn't exist,
	    as it checks for directory existence before attempting removal.
	"""
	if os.path.isdir(REGRESSION_LOG_DIR):
		subprocess.run(["rm", "-r", REGRESSION_LOG_DIR], check=True)


def task_analysis() -> None:
	"""Analyze project configurations and set global formatting parameters.

	Scans all projects in PROJ_ARR to determine the maximum string lengths needed
	for aligned terminal output. Sets global variables MAX_COMPILEMODE_LENGTH and
	MAX_LINE_WIDTH to ensure consistent formatting across all test results.

	This function must be called before test execution to properly format the
	output display.

	Args:
	    None

	Returns:
	    None

	Side Effects:
	    Modifies global variables:
	        - MAX_COMPILEMODE_LENGTH: Set to longest compile-mode + 2 for padding
	        - MAX_LINE_WIDTH: Set to longest name/log-name + 24 for formatting

	Example:
	    If PROJ_ARR contains projects with compile modes ["Debug", "Release"],
	    MAX_COMPILEMODE_LENGTH will be set to 9 (len("Release") + 2).
	"""
	global MAX_COMPILEMODE_LENGTH
	global MAX_LINE_WIDTH
	MAX_COMPILEMODE_LENGTH = max(len(x["compile-mode"]) for x in PROJ_ARR) + 2
	MAX_LINE_WIDTH = max(max(len(x["name"]), len(x["log-name"])) for x in PROJ_ARR) + 24


def exec_cmd(
    cmd: List[str],
    log: str,
    file_mode: str = "w",
    timeout: Optional[float] = None,
    verify_sim_tick_value: Optional[int] = None,
    verify_sim_tick_range: Optional[Tuple[int, int]] = None
) -> None:
	"""Execute a command and log its output with optional simulation validation.

	Runs a subprocess command, captures stdout/stderr, strips ANSI color codes,
	and writes cleaned output to a log file. Optionally validates simulation
	completion tick against expected value or range.

	The function parses command output for simulation completion messages matching
	the pattern "Tick=<number> Info: [...] Simulation complete." and extracts
	the completion tick for validation.

	Args:
	    cmd (List[str]): Command and arguments to execute (e.g., ["cmake", "-B", "build"]).
	    log (str): Path to log file for writing command output.
	    file_mode (str, optional): File open mode ('w' for write, 'a' for append).
	        Defaults to "w".
	    timeout (Optional[float], optional): Maximum execution time in seconds.
	        None means no timeout. Defaults to None.
	    verify_sim_tick_value (Optional[int], optional): Expected exact simulation
	        completion tick. If provided, raises ValidationError if tick doesn't match.
	        Defaults to None.
	    verify_sim_tick_range (Optional[Tuple[int, int]], optional): Expected tick
	        range as (min, max). If provided, raises ValidationError if tick falls
	        outside range. Defaults to None.

	Returns:
	    None

	Raises:
	    subprocess.CalledProcessError: If the command exits with non-zero status.
	        Output is captured to log file.
	    subprocess.TimeoutExpired: If command execution exceeds timeout.
	        No output is captured to prevent excessive logs.
	    ValidationError: If simulation validation fails (missing completion message,
	        tick mismatch, or tick out of range).

	Example:
	    >>> exec_cmd(
	    ...     cmd=["./simulator", "--config", "test.yaml"],
	    ...     log="sim.log",
	    ...     timeout=60,
	    ...     verify_sim_tick_value=1000
	    ... )
	    # Runs simulator and validates it completes at tick 1000

	    >>> exec_cmd(
	    ...     cmd=["cmake", "--build", "build"],
	    ...     log="build.log",
	    ...     file_mode="a"
	    ... )
	    # Appends build output to existing build.log

	Note:
	    - ANSI color codes are stripped from all output before logging
	    - Both stdout and stderr are combined into single stream
	    - Validation only occurs if verification parameters are provided
	"""
	try:
		# Execute command with or without timeout
		if timeout != None:
			proc = subprocess.run(
			    cmd,
			    timeout=timeout,
			    stdout=subprocess.PIPE,
			    stderr=subprocess.STDOUT,
			    text=True,
			    check=True
			)
		else:
			proc = subprocess.run(
			    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
			)

		# Track simulation completion tick (extracted from output)
		sim_complete_tick: Optional[int] = None

		# Write output to log file, stripping color codes and extracting tick
		with open(log, mode=file_mode) as f_log:
			for line in proc.stdout.splitlines():
				line = COLOR_REGEX.sub("", line)  # Remove ANSI color codes
				m = re.search(TIMETICK_REGEX, line)
				# Extract tick from whichever group matched (group 1 for native, group 2 for SST)
				if m:
					sim_complete_tick = int(m.group(1) if m.group(1) else m.group(2))
				f_log.write(f"{line}\n")

		# Validation phase: Check if simulation completed when validation is requested
		if (verify_sim_tick_value or verify_sim_tick_range) and not sim_complete_tick:
			raise ValidationError("The simulation did not print the complete time.")

		# Validate exact tick value match
		if verify_sim_tick_value and not (verify_sim_tick_value == sim_complete_tick):
			raise ValidationError(
			    f"The simulation did not complete at the expected tick. Received tick: {sim_complete_tick}"
			)

		# Validate tick falls within expected range
		if verify_sim_tick_range and not (
		    verify_sim_tick_range[0] <= sim_complete_tick <= verify_sim_tick_range[1]
		):
			raise ValidationError(
			    f"The simulation did not complete at the expected tick range. Received tick: {sim_complete_tick}"
			)

	except subprocess.CalledProcessError as e:
		# Command failed: capture error output to log
		with open(log, mode=file_mode) as f_log:
			for line in e.stdout.splitlines() if e.stdout else []:
				line = COLOR_REGEX.sub("", line)
				f_log.write(f"{line}\n")
			for line in e.stderr.splitlines() if e.stderr else []:
				line = COLOR_REGEX.sub("", line)
				f_log.write(f"{line}\n")
			f_log.write(f"The command (or program) exited with return code {e.returncode}.")
		raise e

	except subprocess.TimeoutExpired as e:
		# Command timed out: write notice instead of potentially huge output
		with open(log, mode=file_mode) as f_log:
			f_log.write(
			    "The outputs are not captured because a program that times out might generate an excessive amount of logs.\n"
			)
		raise e


def print_result(
    title: str,
    mode: str,
    result: bool,
    stage: str = "",
    exception: Optional[Exception] = None
) -> None:
	"""Print formatted test result to terminal with color-coding.

	Displays a single test result line with aligned formatting, showing the
	compile mode, test name, and result status. Results are color-coded:
	green for pass, red for failures/errors.

	The output format is:
	    [<mode>] <title> ............... <result>

	Args:
	    title (str): Test name to display (e.g., "Simple CPU Test").
	    mode (str): Compile mode (e.g., "Debug", "Release").
	    result (bool): True for pass, False for failure.
	    stage (str, optional): Stage where failure occurred ("Compile" or "Execute").
	        Only used when result is False. Defaults to "".
	    exception (Optional[Exception], optional): Exception that caused failure.
	        Used to determine specific error type. Defaults to None.

	Returns:
	    None

	Result Status Messages:
	    - "Passed": Test completed successfully (result=True)
	    - "Compile Failed": Compilation failed (CalledProcessError during compile)
	    - "Execute Failed": Execution failed (CalledProcessError during execution)
	    - "Compile Timeout": Compilation timed out (TimeoutExpired during compile)
	    - "Execute Timeout": Execution timed out (TimeoutExpired during execution)
	    - "Incorrect Result": Validation failed (ValidationError)
	    - "Regression Error": Unexpected exception type

	Example Output:
	    [  Debug  ] Cache Coherence Test ................ Passed
	    [ Release ] Memory Ordering Test ................ Execute Failed
	    [  GTest  ] Unit Test Suite .................... Compile Timeout

	Note:
	    Requires MAX_COMPILEMODE_LENGTH and MAX_LINE_WIDTH to be set by
	    task_analysis() for proper alignment.
	"""
	COLOR_CODE: Dict[str, str] = {"RED": '\033[0;31m', "GREEN": '\033[0;32m', "NC": '\033[0m'}
	INTERNAL_ERROR_MSG: str = "Regression Error"

	# Determine result message based on success/failure type
	result_str: str = ""

	if result:
		result_str = "Passed"
	elif type(exception) == subprocess.CalledProcessError:
		result_str = stage + " " + "Failed" if len(stage) > 0 else "Failed"
	elif type(exception) == subprocess.TimeoutExpired:
		result_str = stage + " " + "Timeout" if len(stage) > 0 else "Timeout"
	elif type(exception) == ValidationError:
		result_str = "Incorrect Result"
	else:
		result_str = INTERNAL_ERROR_MSG

	# Apply color coding: green for success, red for failure
	colored_result_str: str = (
	    COLOR_CODE["GREEN"] + result_str if result else COLOR_CODE["RED"] + result_str
	) + COLOR_CODE["NC"]

	# Print formatted result line with alignment
	print(
	    f"[{mode.center(MAX_COMPILEMODE_LENGTH)}] {title} {'.'*(MAX_LINE_WIDTH - len(title) - len(result_str))} {colored_result_str}"
	)

	# For internal errors, print additional debug information
	if result_str == INTERNAL_ERROR_MSG and exception:
		print(
		    " " * (MAX_COMPILEMODE_LENGTH + 3) + f"{exception.__class__.__name__}: {str(exception)}"
		)


def exec_proj(
    name: str,
    log_name: str,
    src_dirname: str,
    exec_args: List[str],
    compile_mode: str,
    pre_steps: List[str],
    post_steps: List[str],
    timeout: Optional[int],
    expected_tick_value: Optional[int] = None,
    expected_tick_range: Optional[Tuple[int, int]] = None
) -> bool:
	"""Execute complete test workflow for a single project: configure, compile, run.

	Orchestrates the full testing lifecycle for one project configuration:
	    1. Creates log directory structure
	    2. Configures CMake with appropriate build type
	    3. Compiles the target binary
	    4. Runs pre-execution commands
	    5. Executes the simulator with validation
	    6. Runs post-execution commands
	    7. Reports results via print_result()

	Args:
	    name (str): Human-readable project name for display.
	    log_name (str): Prefix for execution log filename.
	    src_dirname (str): Source subdirectory name (becomes CMake target name).
	        For GTest builds, 'g' is prepended automatically.
	    exec_args (List[str]): Command-line arguments to pass to the executable.
	    compile_mode (str): Build type - must be one of COMPILE_TYPE_LIST
	        ("Debug", "Release", "GTest").
	    pre_steps (List[str]): Shell commands to run before simulation.
	        Each command is a list of arguments.
	    post_steps (List[str]): Shell commands to run after successful simulation.
	        Each command is a list of arguments.
	    timeout (Optional[int]): Maximum execution time in seconds for simulation.
	        None means no timeout.
	    expected_tick_value (Optional[int], optional): Exact expected completion tick.
	        Validation fails if simulation completes at different tick. Defaults to None.
	    expected_tick_range (Optional[Tuple[int, int]], optional): Expected tick range
	        as (min, max). Validation fails if tick falls outside. Defaults to None.

	Returns:
	    bool: True if all stages passed, False otherwise.

	Raises:
	    RuntimeError: If compile_mode is not in COMPILE_TYPE_LIST.

	Side Effects:
	    - Creates directory structure under REGRESSION_LOG_DIR
	    - Writes build.log and <log_name>-exec.log files
	    - Prints test result to stdout via print_result()

	Log File Structure:
	    build/regression/<compile_mode>/<src_dirname>/
	        build.log           # CMake configuration and compilation output
	        <log_name>-exec.log # Pre-steps, simulation, and post-steps output

	Example:
	    >>> exec_proj(
	    ...     name="Simple Cache Test",
	    ...     log_name="cache_test",
	    ...     src_dirname="cache_sim",
	    ...     exec_args=["--config", "simple.yaml"],
	    ...     compile_mode="Debug",
	    ...     pre_steps=[["mkdir", "-p", "output"]],
	    ...     post_steps=[["python", "verify.py"]],
	    ...     timeout=60,
	    ...     expected_tick_value=1000
	    ... )
	    True  # If all stages succeed

	Workflow:
	    COMPILE STAGE:
	        cmake -B build/debug -DCMAKE_BUILD_TYPE=Debug -DNO_LOGS=OFF
	        cmake --build build/debug -j<cpus> --target cache_sim

	    EXECUTE STAGE:
	        mkdir -p output             # pre_steps
	        ./build/debug/cache_sim --config simple.yaml  # main execution
	        python verify.py            # post_steps

	Note:
	    - Compilation failure stops execution (no simulation attempted)
	    - Execution failure skips post_steps
	    - All exceptions are caught and reported via print_result()
	"""
	# Setup paths for build directory and log files
	build_dir: str = os.path.join(BUILD_DIR, compile_mode.lower())
	log_dir: str = os.path.join(REGRESSION_LOG_DIR, compile_mode.lower(), src_dirname)
	build_log: str = os.path.join(log_dir, BUILD_LOG_FILENAME)
	exec_log: str = os.path.join(log_dir, f"{log_name}-{EXEC_LOG_FILENAME}")

	# Validate compile mode
	if compile_mode not in COMPILE_TYPE_LIST:
		raise RuntimeError(f"The compile mode '{compile_mode}' is unsupported.")

	# GTest targets have 'g' prefix convention
	if compile_mode == "GTest":
		src_dirname = "g" + src_dirname

	try:
		# COMPILE STAGE

		# Create log directory structure
		subprocess.run(["mkdir", "-p", log_dir], check=True)

		if compile_mode == "SST":
			# SST mode: Build and install SST element library
			# SST elements are built using their own Makefile in src/<src_dirname>
			sst_src_dir = os.path.join("src", src_dirname)

			# Detect if running inside Docker container
			in_docker = os.path.exists("/.dockerenv")

			if in_docker:
				# Running inside Docker - execute SST commands with bash to set environment
				# SST environment variables
				sst_env = (
				    "export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install && "
				    "export PATH=$SST_CORE_HOME/bin:$PATH && "
				    "export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH"
				)

				# Clean previous build
				exec_cmd(
				    cmd=["bash", "-c", f"cd {sst_src_dir} && {sst_env} && make clean"],
				    log=build_log,
				    file_mode="w"
				)

				# Build SST element library
				exec_cmd(
				    cmd=["bash", "-c", f"cd {sst_src_dir} && {sst_env} && make -j$(nproc)"],
				    log=build_log,
				    file_mode="a"
				)

				# Install SST element library
				exec_cmd(
				    cmd=["bash", "-c", f"cd {sst_src_dir} && {sst_env} && make install"],
				    log=build_log,
				    file_mode="a"
				)
			else:
				# Running on host - use docker exec to run commands in container
				docker_cmd = "/usr/local/bin/docker"
				docker_container = "acalsim-workspace"
				docker_project_dir = "/home/user/projects/acalsim"
				docker_sst_dir = f"{docker_project_dir}/{sst_src_dir}"

				# SST environment variables
				sst_env = (
				    "export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install && "
				    "export PATH=$SST_CORE_HOME/bin:$PATH && "
				    "export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH"
				)

				# Clean previous build
				exec_cmd(
				    cmd=[
				        docker_cmd, "exec", docker_container, "bash", "-c",
				        f"cd {docker_sst_dir} && {sst_env} && make clean"
				    ],
				    log=build_log,
				    file_mode="w"
				)

				# Build SST element library
				exec_cmd(
				    cmd=[
				        docker_cmd, "exec", docker_container, "bash", "-c",
				        f"cd {docker_sst_dir} && {sst_env} && make -j$(nproc)"
				    ],
				    log=build_log,
				    file_mode="a"
				)

				# Install SST element library
				exec_cmd(
				    cmd=[
				        docker_cmd, "exec", docker_container, "bash", "-c",
				        f"cd {docker_sst_dir} && {sst_env} && make install"
				    ],
				    log=build_log,
				    file_mode="a"
				)
		else:
			# Standard CMake-based build for Debug/Release/GTest modes
			# Configure CMake with build type
			# Note: GTest mode skips CMAKE_BUILD_TYPE flag as it's in IGNORED_COMPILE_TYPE
			exec_cmd(
			    cmd=[
			        "cmake", "-B", build_dir, f"-DCMAKE_BUILD_TYPE={compile_mode}"
			        if compile_mode not in IGNORED_COMPILE_TYPE else "", "-DNO_LOGS=OFF"
			    ],
			    log=build_log,
			    file_mode="w"
			)

			# Compile the target using all available CPU cores
			exec_cmd(
			    cmd=["cmake", "--build", build_dir, f"-j{os.cpu_count()}", "--target", src_dirname],
			    log=build_log,
			    file_mode="a"
			)

	except Exception as e:
		# Compilation failed - report and abort (don't attempt execution)
		print_result(title=name, mode=compile_mode, result=False, stage="Compile", exception=e)
		return False

	try:
		# EXECUTE STAGE

		# Run pre-execution setup commands
		with open(exec_log, mode="w", encoding="utf-8") as file:
			file.write("======= Pre-steps =======\n")
		for cmd in pre_steps:
			exec_cmd(cmd=cmd, log=exec_log, file_mode="a")

		# Run the main simulation with validation
		with open(exec_log, mode="a", encoding="utf-8") as file:
			file.write("\n======= Simulation =======\n")

		if compile_mode == "SST":
			# SST mode: Run sst command with Python config file
			# exec_args[0] should be the path to the Python config file
			sst_src_dir = os.path.join("src", src_dirname)
			sst_config = exec_args[0] if exec_args else "examples/riscv_single_core.py"

			# Detect if running inside Docker container
			in_docker = os.path.exists("/.dockerenv")

			if in_docker:
				# Running inside Docker - execute SST with bash to set environment
				# SST environment variables
				sst_env = (
				    "export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install && "
				    "export PATH=$SST_CORE_HOME/bin:$PATH && "
				    "export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH"
				)

				exec_cmd(
				    cmd=["bash", "-c", f"cd {sst_src_dir} && {sst_env} && sst {sst_config}"],
				    log=exec_log,
				    file_mode="a",
				    timeout=timeout,
				    verify_sim_tick_value=expected_tick_value,
				    verify_sim_tick_range=expected_tick_range
				)
			else:
				# Running on host - use docker exec to run sst in container
				docker_cmd = "/usr/local/bin/docker"
				docker_container = "acalsim-workspace"
				docker_project_dir = "/home/user/projects/acalsim"
				docker_sst_dir = f"{docker_project_dir}/{sst_src_dir}"

				# SST environment variables
				sst_env = (
				    "export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install && "
				    "export PATH=$SST_CORE_HOME/bin:$PATH && "
				    "export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH"
				)

				exec_cmd(
				    cmd=[
				        docker_cmd, "exec", docker_container, "bash", "-c",
				        f"cd {docker_sst_dir} && {sst_env} && sst {sst_config}"
				    ],
				    log=exec_log,
				    file_mode="a",
				    timeout=timeout,
				    verify_sim_tick_value=expected_tick_value,
				    verify_sim_tick_range=expected_tick_range
				)
		else:
			# Standard execution for Debug/Release/GTest modes
			exec_cmd(
			    cmd=[os.path.join(build_dir, src_dirname)] + exec_args,
			    log=exec_log,
			    file_mode="a",
			    timeout=timeout,
			    verify_sim_tick_value=expected_tick_value,
			    verify_sim_tick_range=expected_tick_range
			)

		# Run post-execution cleanup/verification commands
		with open(exec_log, mode="a", encoding="utf-8") as file:
			file.write("\n======= Post-steps =======\n")
		for cmd in post_steps:
			exec_cmd(cmd=cmd, log=exec_log, file_mode="a")

		# All stages passed successfully
		print_result(title=name, mode=compile_mode, result=True)

	except Exception as e:
		# Execution/validation failed - report failure
		print_result(title=name, mode=compile_mode, result=False, stage="Execute", exception=e)
		return False

	return True


def test_projects(timeout: Optional[int]) -> None:
	"""Execute regression tests for all configured projects.

	Iterates through all projects defined in PROJ_ARR and executes each one
	using exec_proj(). Tracks overall success/failure status across all tests
	and exits with error code if any test fails.

	Args:
	    timeout (Optional[int]): Maximum execution time in seconds for each
	        simulation. Passed to exec_proj() for all projects. None means
	        no timeout.

	Returns:
	    None

	Side Effects:
	    - Calls exec_proj() for each project in PROJ_ARR
	    - Prints test results to stdout via exec_proj()
	    - Exits with code 1 if any test fails

	Exit Behavior:
	    - All tests pass: Returns normally (exit code 0)
	    - One or more tests fail: Calls exit(1)

	Example:
	    If PROJ_ARR contains:
	        [
	            {"name": "Test A", "compile-mode": "Debug", ...},
	            {"name": "Test B", "compile-mode": "Release", ...}
	        ]

	    Output:
	        [  Debug  ] Test A ............................ Passed
	        [ Release ] Test B ............................ Passed

	Note:
	    - Tests continue running even if some fail (fail-late behavior)
	    - Uses bitwise AND (&) to accumulate failures across all tests
	    - Extracts tick validation parameters from project config if present
	"""
	success: bool = True

	# Execute all configured projects
	for proj in PROJ_ARR:
		success = exec_proj(
		    name=proj["name"],
		    log_name=proj["log-name"],
		    src_dirname=proj["src-subdir"],
		    exec_args=proj["exec-args"],
		    compile_mode=proj["compile-mode"],
		    pre_steps=proj["pre-steps"],
		    post_steps=proj["post-steps"],
		    timeout=timeout,
		    expected_tick_value=proj["total-tick"] if "total-tick" in proj else None,
		    expected_tick_range=tuple(proj["total-tick-range"])
		    if "total-tick-range" in proj else None
		) & success  # Bitwise AND to track cumulative success

	# Exit with error if any test failed
	if not success:
		exit(1)


@click.command()
@click.option(
    '--timeout',
    help='Time limitation in seconds for executing each test example.',
    default=30,
    required=False,
    type=int
)
def main(timeout: Optional[int]) -> None:
	"""Main entry point for regression testing framework.

	Orchestrates the complete regression testing workflow:
	    1. Changes to project root directory
	    2. Removes old log files
	    3. Analyzes project configurations for output formatting
	    4. Executes all regression tests

	Handles keyboard interrupts gracefully by printing termination message
	and exiting cleanly.

	Args:
	    timeout (Optional[int]): Maximum execution time in seconds for each
	        simulation. Default is 30 seconds. Set to 0 or None for no timeout.

	Returns:
	    None

	Exit Codes:
	    0: All tests passed successfully
	    1: One or more tests failed OR keyboard interrupt received

	Command-Line Usage:
	    Run with default 30-second timeout:
	        $ python scripts/regression.py

	    Run with 60-second timeout:
	        $ python scripts/regression.py --timeout 60

	    Run with no timeout (wait indefinitely):
	        $ python scripts/regression.py --timeout 0

	    Show help:
	        $ python scripts/regression.py --help

	Example Session:
	    $ python scripts/regression.py --timeout 45
	    [  Debug  ] Simple Cache Test .................. Passed
	    [ Release ] Multi-Core Simulation ............... Passed
	    [  Debug  ] Memory Ordering Test ................ Execute Failed
	    # (exits with code 1 due to failure)

	Note:
	    - Script automatically navigates to project root
	    - Old regression logs are deleted at start of each run
	    - Ctrl+C during execution triggers clean shutdown
	"""
	try:
		move_to_root_dir()  # Navigate to project root
		remove_existing_logs()  # Clean up old test logs
		task_analysis()  # Calculate output formatting parameters
		test_projects(timeout=timeout)  # Run all regression tests
	except KeyboardInterrupt as e:
		# User interrupted with Ctrl+C - exit gracefully
		print("\n" + "Regression test is terminated.", file=sys.stderr)
		exit(1)


if __name__ == '__main__':
	main()
