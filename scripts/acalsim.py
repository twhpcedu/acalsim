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
"""ACAL Simulator Project Configuration Module.

This module defines the project configurations for the ACAL (Accelerator Architecture Lab)
simulator regression testing framework. It contains a comprehensive list of test projects,
each configured with specific build parameters, execution arguments, and expected simulation
outcomes.

The configurations support testing various ThreadManager implementations (both production-validated
and experimental), multiple hardware simulation scenarios, and different compilation modes for
thorough regression testing of the ACAL simulation framework.

Project Structure:
    Each project configuration is a dictionary with the following schema:

    - name (str): Human-readable project identifier used for display
    - log-name (str): Filename prefix for logs (build.log, exec.log)
    - src-subdir (str): Source directory path relative to src/
    - exec-args (List[str]): Command-line arguments passed to the executable
    - compile-mode (str): Compilation mode - "Debug", "Release", or "GTest"
    - pre-steps (List[str]): Shell commands to execute before compilation
    - post-steps (List[str]): Shell commands to execute after execution
    - total-tick (int, optional): Expected simulation tick count for validation
    - total-tick-range (List[int], optional): Expected [min, max] tick range

ThreadManager Implementations:
    Production (validated in ICCAD 2025):
        - PriorityQueue: Priority-based task scheduling
        - Barrier: Barrier-synchronized parallel execution
        - PrebuiltTaskList: Pre-compiled static task scheduling
        - LocalTaskQueue: Thread-local task queue implementation

    Experimental (under development):
        - V4, V5, V7, V8: Next-generation ThreadManager variants

Test Categories:
    - Basic functionality: test, testConfig, testCommunication
    - Hardware components: testChannel, testSimPort, testPETile, testCrossBar
    - Accelerator systems: testAccelerator
    - SystemC integration: testSTSim, testSTSystemC, testSimChannel
    - RISC-V processor: riscvSimTemplate, riscv load_store_simple, riscv branch_simple
    - Resource management: testResourceRecycling
    - Unit testing: UnitTest with GTest framework

Usage:
    This module is imported by regression.py for automated testing:

        from acalsim import PROJ_ARR

        # Iterate through all project configurations
        for project in PROJ_ARR:
            build_project(project)
            run_simulation(project)
            validate_results(project)

Example Project Configuration:
    {
        "name": "testAccelerator : ThreadManager PriorityQueue",
        "log-name": "testAccelerator_priorityqueue",
        "src-subdir": "testAccelerator",
        "exec-args": ["--threadmanager", "PriorityQueue"],
        "compile-mode": "Debug",
        "pre-steps": [],
        "post-steps": [],
        "total-tick": 521
    }

Expected Workflow:
    1. regression.py reads PROJ_ARR from this module
    2. For each project configuration:
       a. Execute pre-steps (if any)
       b. Compile project in specified compile-mode
       c. Run executable with exec-args
       d. Validate simulation completes at expected total-tick
       e. Execute post-steps (if any)
       f. Save logs to build/regression/{log-name}/
    3. Generate regression test report

Output Example:
    When used with regression.py, produces logs like:

        build/regression/testAccelerator_priorityqueue/
            build.log    - CMake and compilation output
            exec.log     - Simulation execution output with tick count

    Simulation completion message in exec.log:
        Tick=521 Info: [ThreadManager] Simulation complete.

Notes:
    - All paths in src-subdir are relative to the project's src/ directory
    - Config files in exec-args use paths relative to project root
    - Multiple projects can share the same src-subdir with different exec-args
    - GTest mode projects require --googletest flag in exec-args
    - Production ThreadManagers have consistent expected tick counts across test cases

See Also:
    regression.py: Main regression testing script that uses these configurations
    build/regression/: Directory where test logs are stored

Author:
    Playlab/ACAL Team

Version:
    Updated for ICCAD 2025 validation
"""

from typing import Any, Dict, List

# Constants
"""
Project Template Schema:

PROJ_TEMPLATE = {
    "name": "",                 # Full descriptive name for display
    "log-name": "",            # Log file prefix (no spaces)
    "src-subdir": "",          # Source directory under src/
    "exec-args": [],           # Command-line arguments (List[str])
    "compile-mode": "Debug",   # Options: "Debug", "Release", "GTest"
    "pre-steps": [],           # Pre-compilation shell commands (List[str])
    "post-steps": [],          # Post-execution shell commands (List[str])
    "total-tick": 0,           # (Optional) Expected final simulation tick (int)
    "total-tick-range": [0, 10] # (Optional) Expected tick range [min, max] (List[int])
}

Field Descriptions:
    name: Human-readable identifier displayed in test output
    log-name: Filesystem-safe name for log files (build.log, exec.log)
    src-subdir: Relative path from project src/ to project source
    exec-args: Arguments passed to compiled executable
    compile-mode: CMake build type determining optimization and debug symbols
    pre-steps: Commands executed before cmake/make (e.g., code generation)
    post-steps: Commands executed after successful simulation (e.g., result validation)
    total-tick: Exact expected simulation tick for validation (fails if mismatch)
    total-tick-range: Acceptable tick range for non-deterministic simulations

Validation Behavior:
    - If total-tick is set: Simulation must complete at exactly this tick
    - If total-tick-range is set: Simulation must complete within [min, max] ticks
    - If neither is set: No tick validation performed (e.g., ProjectTemplate)
    - If both are set: total-tick takes precedence

Example Usage Scenarios:
    1. Simple test without validation:
       {"name": "test", "log-name": "test", "src-subdir": "test",
        "exec-args": [], "compile-mode": "Debug", "pre-steps": [],
        "post-steps": [], "total-tick": 30}

    2. Accelerator with specific ThreadManager:
       {"name": "testAccelerator : ThreadManager Barrier",
        "log-name": "testAccelerator_barrier",
        "src-subdir": "testAccelerator",
        "exec-args": ["--threadmanager", "Barrier"],
        "compile-mode": "Debug", "pre-steps": [], "post-steps": [],
        "total-tick": 521}

    3. GTest unit tests:
       {"name": "testPETile", "log-name": "testPETile",
        "src-subdir": "testPETile",
        "exec-args": ["--config", "src/testPETile/configs.json", "--googletest"],
        "compile-mode": "GTest", "pre-steps": [], "post-steps": [],
        "total-tick": 164}

    4. RISC-V assembly test:
       {"name": "riscv load_store_simple test",
        "log-name": "riscv_load_store_simple",
        "src-subdir": "riscv",
        "exec-args": ["--asm_file_path", "src/riscv/asm/load_store_simple.txt"],
        "compile-mode": "Debug", "pre-steps": [], "post-steps": [],
        "total-tick": 13}
"""

# ==============================================================================
# PROJECT CONFIGURATION ARRAY
# ==============================================================================
# This array contains all project configurations for regression testing.
# Projects are organized by category and ThreadManager implementation type.
# The regression.py script iterates through this array to build, execute,
# and validate each project configuration.
# ==============================================================================

PROJ_ARR: List[Dict[str, Any]] = []

# ------------------------------------------------------------------------------
# Template and Basic Test Projects
# ------------------------------------------------------------------------------
# These projects provide templates and basic functionality testing without
# specific ThreadManager requirements.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "ProjectTemplate",
    "log-name": "ProjectTemplate",
    "src-subdir": "ProjectTemplate",
    "exec-args": [],  # No runtime arguments required
    "compile-mode": "Debug",
    "pre-steps": [],  # No pre-compilation steps
    "post-steps": [],  # No post-execution steps
    "total-tick": 0  # No tick validation (template only)
})

PROJ_ARR.append({
    "name": "test",
    "log-name": "test",
    "src-subdir": "test",
    "exec-args": [],
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 30  # Expected to complete at exactly tick 30
})

# ------------------------------------------------------------------------------
# testAccelerator - Production ThreadManagers (ICCAD 2025 validated)
# ------------------------------------------------------------------------------
# Tests the accelerator simulation framework with various production-ready
# ThreadManager implementations. All variants should complete at tick 521.
# These ThreadManagers have been validated and published in ICCAD 2025.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager PriorityQueue",
    "log-name": "testAccelerator_priorityqueue",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "PriorityQueue"],  # Priority-based scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521  # All testAccelerator variants complete at tick 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager Barrier",
    "log-name": "testAccelerator_barrier",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "Barrier"],  # Barrier-synchronized execution
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager PrebuiltTaskList",
    "log-name": "testAccelerator_prebuilttasklist",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "PrebuiltTaskList"],  # Pre-compiled task scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager LocalTaskQueue",
    "log-name": "testAccelerator_localtaskqueue",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "LocalTaskQueue"],  # Thread-local task queues
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

# ------------------------------------------------------------------------------
# testAccelerator - Experimental ThreadManagers
# ------------------------------------------------------------------------------
# Experimental next-generation ThreadManager implementations under development.
# Not yet validated in published research. Expected tick: 521.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager V4",
    "log-name": "testAccelerator_v4",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "V4"],  # Experimental variant 4
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager V5",
    "log-name": "testAccelerator_v5",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "V5"],  # Experimental variant 5
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager V7",
    "log-name": "testAccelerator_v7",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "V7"],  # Experimental variant 7
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

PROJ_ARR.append({
    "name": "testAccelerator : ThreadManager V8",
    "log-name": "testAccelerator_v8",
    "src-subdir": "testAccelerator",
    "exec-args": ["--threadmanager", "V8"],  # Experimental variant 8
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 521
})

# ------------------------------------------------------------------------------
# testChannel - Production ThreadManagers (ICCAD 2025 validated)
# ------------------------------------------------------------------------------
# Tests channel communication mechanisms with production ThreadManagers.
# All variants should complete at tick 26.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testChannel : ThreadManager PriorityQueue",
    "log-name": "testChannel_priorityqueue",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "PriorityQueue"],  # Priority-based scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26  # All testChannel variants complete at tick 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager Barrier",
    "log-name": "testChannel_barrier",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "Barrier"],  # Barrier-synchronized execution
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager PrebuiltTaskList",
    "log-name": "testChannel_prebuilttasklist",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "PrebuiltTaskList"],  # Pre-compiled task scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager LocalTaskQueue",
    "log-name": "testChannel_localtaskqueue",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "LocalTaskQueue"],  # Thread-local task queues
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

# ------------------------------------------------------------------------------
# testChannel - Experimental ThreadManagers
# ------------------------------------------------------------------------------
# Experimental ThreadManager variants for channel communication testing.
# Expected tick: 26.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testChannel : ThreadManager V4",
    "log-name": "testChannel_v4",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "V4"],  # Experimental variant 4
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager V5",
    "log-name": "testChannel_v5",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "V5"],  # Experimental variant 5
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager V7",
    "log-name": "testChannel_v7",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "V7"],  # Experimental variant 7
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

PROJ_ARR.append({
    "name": "testChannel : ThreadManager V8",
    "log-name": "testChannel_v8",
    "src-subdir": "testChannel",
    "exec-args": ["--threadmanager", "V8"],  # Experimental variant 8
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 26
})

# ------------------------------------------------------------------------------
# Standalone Functional Tests
# ------------------------------------------------------------------------------
# Tests for specific functionality without ThreadManager variation.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testCommunication",
    "log-name": "testCommunication",
    "src-subdir": "testCommunication",
    "exec-args": [],  # Basic communication test without arguments
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 8  # Simple communication test completes quickly
})

PROJ_ARR.append({
    "name": "testConfig",
    "log-name": "testConfig",
    "src-subdir": "testConfig",
    "exec-args": [
        "--config",
        "src/testConfig/config.json",  # Primary config file
        "--config",
        "src/testConfig/config2.json",  # Secondary config file
        "--config",
        "src/testConfig/acalsim_config.json",  # Framework config
        "--test_int",
        "10",  # Test integer parameter
        "--test_float",
        "2.0",  # Test float parameter
        "--test_int_enum",
        "1"  # Test enumeration parameter
    ],
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 0  # Config validation test, no simulation execution
})

PROJ_ARR.append({
    "name": "testPETile",
    "log-name": "testPETile",
    "src-subdir": "testPETile",
    "exec-args": [
        "--config",
        "src/testPETile/configs.json",  # PE Tile configuration
        "--googletest"  # Run as Google Test suite
    ],
    "compile-mode": "GTest",  # Uses Google Test framework
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 164  # Processing Element Tile test simulation
})

PROJ_ARR.append({
    "name": "testResourceRecycling",
    "log-name": "testResourceRecycling",
    "src-subdir": "testResourceRecycling",
    "exec-args": [],  # Resource management stress test
    "compile-mode": "Release",  # Release mode for performance testing
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 80000  # Long-running test for memory/resource validation
})

# ------------------------------------------------------------------------------
# testSimPort - Production ThreadManagers (ICCAD 2025 validated)
# ------------------------------------------------------------------------------
# Tests simulation port functionality with production ThreadManagers.
# All variants should complete at tick 312.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager PriorityQueue",
    "log-name": "testSimPort_priorityqueue",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "PriorityQueue"],  # Priority-based scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312  # All testSimPort variants complete at tick 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager Barrier",
    "log-name": "testSimPort_barrier",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "Barrier"],  # Barrier-synchronized execution
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager PrebuiltTaskList",
    "log-name": "testSimPort_prebuilttasklist",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "PrebuiltTaskList"],  # Pre-compiled task scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager LocalTaskQueue",
    "log-name": "testSimPort_localtaskqueue",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "LocalTaskQueue"],  # Thread-local task queues
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

# ------------------------------------------------------------------------------
# testSimPort - Experimental ThreadManagers
# ------------------------------------------------------------------------------
# Experimental ThreadManager variants for simulation port testing.
# Expected tick: 312.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager V4",
    "log-name": "testSimPort_v4",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "V4"],  # Experimental variant 4
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager V5",
    "log-name": "testSimPort_v5",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "V5"],  # Experimental variant 5
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager V7",
    "log-name": "testSimPort_v7",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "V7"],  # Experimental variant 7
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

PROJ_ARR.append({
    "name": "testSimPort : ThreadManager V8",
    "log-name": "testSimPort_v8",
    "src-subdir": "testSimPort",
    "exec-args": ["--threadmanager", "V8"],  # Experimental variant 8
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 312
})

# ------------------------------------------------------------------------------
# SystemC Integration Tests
# ------------------------------------------------------------------------------
# Tests for SystemC integration and simulation channel functionality.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testSTSim",
    "log-name": "testSTSim",
    "src-subdir": "testSTSim",
    "exec-args": [],  # Minimal SystemC template test
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 1  # Minimal test completes at tick 1
})

# ------------------------------------------------------------------------------
# testSTSystemC - Production ThreadManagers (ICCAD 2025 validated)
# ------------------------------------------------------------------------------
# SystemC integration testing with production ThreadManagers.
# All variants should complete at tick 793.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testSTSystemC : ThreadManager PriorityQueue",
    "log-name": "testSTSystemC_priorityqueue",
    "src-subdir": "testSTSystemC",
    "exec-args": ["--threadmanager", "PriorityQueue"],  # Priority-based scheduling
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 793  # SystemC tests complete at tick 793
})

PROJ_ARR.append({
    "name": "testSTSystemC : ThreadManager Barrier",
    "log-name": "testSTSystemC_barrier",
    "src-subdir": "testSTSystemC",
    "exec-args": ["--threadmanager", "Barrier"],  # Barrier-synchronized execution
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 793
})

# ------------------------------------------------------------------------------
# testSimChannel - Simulation Channel Tests
# ------------------------------------------------------------------------------
# Tests for simulation channel communication in both GTest and normal modes.
# Note: Same project with different configurations (GTest vs Debug).
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testSimChannel",
    "log-name": "testSimChannel",
    "src-subdir": "testSimChannel",
    "exec-args": [
        "--googletest",  # Run as Google Test suite
        "--gtest_filter=*",  # Run all test cases
    ],
    "compile-mode": "GTest",  # Google Test framework mode
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 4  # GTest mode completes at tick 4
})

PROJ_ARR.append({
    "name": "testSimChannel",
    "log-name": "testSimChannel",
    "src-subdir": "testSimChannel",
    "exec-args": [],  # Normal simulation mode
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 27  # Debug mode completes at tick 27
})

# ------------------------------------------------------------------------------
# RISC-V Processor Tests
# ------------------------------------------------------------------------------
# RISC-V instruction set architecture simulation tests with various assembly
# programs to validate processor implementation.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "riscvSimTemplate",
    "log-name": "riscvSimTemplate",
    "src-subdir": "riscvSimTemplate",
    "exec-args": [],  # Template RISC-V simulation
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 9  # Simple template test
})

PROJ_ARR.append({
    "name": "riscv load_store_simple test",
    "log-name": "riscv_load_store_simple",
    "src-subdir": "riscv",
    "exec-args": [
        "--asm_file_path",
        "src/riscv/asm/load_store_simple.txt"  # Load/store assembly
    ],
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 13  # Load/store instructions test
})

PROJ_ARR.append({
    "name": "riscv branch_simple test",
    "log-name": "riscv_branch_simple",
    "src-subdir": "riscv",
    "exec-args": [
        "--asm_file_path",
        "src/riscv/asm/branch_simple.txt"  # Branch assembly
    ],
    "compile-mode": "Debug",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 28  # Branch instructions test
})

# ------------------------------------------------------------------------------
# testCrossBar - Production ThreadManagers (ICCAD 2025 validated)
# ------------------------------------------------------------------------------
# CrossBar interconnect tests with production ThreadManagers.
# Tests a large interconnect: 132 masters, 5 slaves, 10 requests each.
# All variants should complete at tick 268.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testCrossBar : ThreadManager PriorityQueue",
    "log-name": "testCrossBar_priorityqueue",
    "src-subdir": "testCrossBar",
    "exec-args": [
        "--n_master=132",  # Number of master devices
        "--n_slave=5",  # Number of slave devices
        "--n_requests=10",  # Requests per master
        "--threadmanager=PriorityQueue"  # Priority-based scheduling
    ],
    "compile-mode": "GTest",  # Uses Google Test framework
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 268  # All testCrossBar variants complete at tick 268
})

PROJ_ARR.append({
    "name": "testCrossBar : ThreadManager Barrier",
    "log-name": "testCrossBar_barrier",
    "src-subdir": "testCrossBar",
    "exec-args": [
        "--n_master=132",
        "--n_slave=5",
        "--n_requests=10",
        "--threadmanager=Barrier"  # Barrier-synchronized execution
    ],
    "compile-mode": "GTest",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 268
})

# ------------------------------------------------------------------------------
# testCrossBar - Experimental ThreadManagers
# ------------------------------------------------------------------------------
# Experimental ThreadManager variants for CrossBar interconnect testing.
# Same configuration: 132 masters, 5 slaves, 10 requests. Expected tick: 268.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "testCrossBar : ThreadManager V4",
    "log-name": "testCrossBar_v4",
    "src-subdir": "testCrossBar",
    "exec-args": [
        "--n_master=132",
        "--n_slave=5",
        "--n_requests=10",
        "--threadmanager=V4"  # Experimental variant 4
    ],
    "compile-mode": "GTest",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 268
})

PROJ_ARR.append({
    "name": "testCrossBar : ThreadManager V7",
    "log-name": "testCrossBar_v7",
    "src-subdir": "testCrossBar",
    "exec-args": [
        "--n_master=132",
        "--n_slave=5",
        "--n_requests=10",
        "--threadmanager=V7"  # Experimental variant 7
    ],
    "compile-mode": "GTest",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 268
})

PROJ_ARR.append({
    "name": "testCrossBar : ThreadManager V8",
    "log-name": "testCrossBar_v8",
    "src-subdir": "testCrossBar",
    "exec-args": [
        "--n_master=132",
        "--n_slave=5",
        "--n_requests=10",
        "--threadmanager=V8"  # Experimental variant 8
    ],
    "compile-mode": "GTest",
    "pre-steps": [],
    "post-steps": [],
    "total-tick": 268
})

# ------------------------------------------------------------------------------
# Unit Testing Suite
# ------------------------------------------------------------------------------
# Comprehensive unit test suite using Google Test framework.
# Contains tests for core framework components, utilities, and data structures.
# No specific tick validation as this runs multiple independent test cases.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "UnitTest",
    "log-name": "UnitTest",
    "src-subdir": "UnitTest",
    "exec-args": [],  # Runs all unit tests without filters
    "compile-mode": "GTest",  # Google Test framework
    "pre-steps": [],
    "post-steps": [],
    # Note: total-tick omitted - UnitTest contains multiple test cases
    # with varying tick counts, so no single tick validation is performed
})

# ------------------------------------------------------------------------------
# SST Integration Tests - RISC-V
# ------------------------------------------------------------------------------
# Tests for SST (Structural Simulation Toolkit) integration with RISC-V processor.
# These tests use SST's Python configuration files and run via the 'sst' command.
# The SST element library must be built and installed before running these tests.
# ------------------------------------------------------------------------------

PROJ_ARR.append({
    "name": "SST RISC-V single core (branch_simple)",
    "log-name": "sst_riscv_single_core_branch",
    "src-subdir": "sst-riscv",  # SST integration directory
    "exec-args": ["examples/riscv_single_core.py"],  # Python config file for SST
    "compile-mode": "SST",  # Special mode for SST integration tests
    "pre-steps": [],  # Build/install happens in SST compile mode
    "post-steps": [],
    "total-tick": 7  # Expected simulation tick for branch_simple.txt test
})

# ==============================================================================
# END OF PROJECT CONFIGURATION ARRAY
# ==============================================================================
# Total projects configured: 51
# - Production ThreadManager variants: 4 implementations across multiple tests
# - Experimental ThreadManager variants: 4 implementations under development
# - Standalone tests: Basic functionality, config, communication, resources
# - Hardware tests: Accelerator, Channel, SimPort, CrossBar, PETile
# - SystemC tests: STSim, STSystemC, SimChannel
# - RISC-V tests: Template, load/store, branch instructions
# - SST Integration: RISC-V processor in SST framework
# - Unit tests: Comprehensive framework validation
# ==============================================================================
