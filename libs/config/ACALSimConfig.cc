/*
 * Copyright 2023-2026 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file ACALSimConfig.cc
 * @brief ACALSimConfig implementation - framework-level configuration and enum mappings
 *
 * This file implements ACALSimConfig, which provides framework-level configuration
 * parameters for ACALSim, including ThreadManager version selection and enum-to-string
 * bidirectional mapping tables for CLI parsing and logging.
 *
 * **ThreadManagerVersion Enum Mapping Architecture:**
 * ```
 * User Input                CLI11 Transform        Enum Value              Display String
 *    │                            │                     │                       │
 *    │  --threadmanager=V3        │                     │                       │
 *    └─────────────────────────────►                     │                       │
 *                                 │                     │                       │
 *                    ThreadManagerVersionMap            │                       │
 *                    {"V3" → V3, "v3" → V3}            │                       │
 *                                 │                     │                       │
 *                                 └─────────────────────►                       │
 *                                              ThreadManagerVersion::V3          │
 *                                                       │                       │
 *                                                       │  (logging)            │
 *                                                       └───────────────────────►
 *                                                                               │
 *                                                             ThreadManagerVersionReMap
 *                                                             {V3 → "PrebuiltTaskList (V3)"}
 *                                                                               │
 *                                                                               ▼
 *                                                                    "ThreadManagerVersion::
 *                                                                     PrebuiltTaskList (V3)"
 * ```
 *
 * **Bidirectional Mapping Tables:**
 *
 * 1. **ThreadManagerVersionMap** (lines 21-37): String → Enum
 *    - Purpose: CLI argument parsing, JSON config parsing
 *    - Used by: CLI11 CheckedTransformer, JSON parameter parsing
 *    - Case handling: CLI11 can apply ignore_case for flexible user input
 *
 * 2. **ThreadManagerVersionReMap** (lines 40-48): Enum → String
 *    - Purpose: Logging, error messages, --help output
 *    - Used by: SimTop initialization logging, statistics output
 *    - Format: Descriptive name with version number and production status
 *
 * **ThreadManager Version Naming Convention:**
 * ACALSim ThreadManager versions have dual naming for clarity:
 *
 * | Enum Value        | Descriptive Name    | Numeric Name | Status       | Features                  |
 * |-------------------|---------------------|--------------|--------------|---------------------------|
 * | PriorityQueue     | PriorityQueue       | V1           | Production   | Priority-based scheduling |
 * | Barrier           | Barrier             | V2           | Production   | Barrier synchronization   |
 * | PrebuiltTaskList  | PrebuiltTaskList    | V3           | Production   | Prebuilt task lists       |
 * | LocalTaskQueue    | LocalTaskQueue      | V6           | Production   | Per-thread local queues   |
 * | Default           | (maps to V3)        | -            | Production   | Framework default         |
 * | V4                | V4                  | V4           | Experimental | Fine-grained dependencies |
 * | V5                | V5                  | V5           | Experimental | Adaptive load balancing   |
 * | V7                | V7                  | V7           | Experimental | NUMA-aware scheduling     |
 * | V8                | V8                  | V8           | Experimental | Speculative execution     |
 *
 * **CLI Usage Examples:**
 * ```bash
 * # Using descriptive name (recommended for new code)
 * ./sim --threadmanager=PrebuiltTaskList
 *
 * # Using numeric name (backward compatibility)
 * ./sim --threadmanager=V3
 *
 * # Case insensitive (if CLI11 configured with ignore_case)
 * ./sim --threadmanager=prebuilttasklist
 * ./sim --threadmanager=v3
 *
 * # Using default
 * ./sim --threadmanager=Default  # Maps to V3 internally
 *
 * # Invalid input
 * ./sim --threadmanager=V9       # ERROR: Invalid choice
 * ```
 *
 * **JSON Configuration Example:**
 * ```json
 * {
 *   "ACALSim": {
 *     "thread_manager_version": "PrebuiltTaskList"
 *   }
 * }
 * ```
 *
 * **Integration with CLI11:**
 * ```cpp
 * // In CLIManager::registerACALSimCLIArguments():
 * addCLIOption<ThreadManagerVersion>("--threadmanager", ...)
 *     ->transform(CLI::CheckedTransformer(ThreadManagerVersionMap,
 *                                         CLI::ignore_case));
 *
 * // CLI11 workflow:
 * 1. User provides: --threadmanager=V3
 * 2. CLI11 looks up "V3" in ThreadManagerVersionMap
 * 3. Returns ThreadManagerVersion::V3
 * 4. Value stored in config parameter
 * 5. SimTop::initThreadManager() uses enum to instantiate ThreadManagerV3
 * ```
 *
 * **Logging Integration:**
 * ```cpp
 * // In SimTop::initConfig():
 * auto version = getParameter<ThreadManagerVersion>("ACALSim", "thread_manager_version");
 * VERBOSE_CLASS_INFO << "Thread Manager Type: "
 *                    << ThreadManagerVersionReMap[version];
 *
 * // Output example:
 * // [SimTop] Thread Manager Type: ThreadManagerVersion::PrebuiltTaskList (V3)
 * ```
 *
 * **Production vs Experimental Status:**
 * - **Production versions** (V1, V2, V3, V6):
 *   - Thoroughly tested, stable performance
 *   - Recommended for simulation runs
 *   - Documented in published research
 *
 * - **Experimental versions** (V4, V5, V7, V8):
 *   - Research prototypes, under development
 *   - May have performance variability
 *   - Subject to API changes
 *   - Documented in header files for development
 *
 * **Map Initialization:**
 * Both maps are initialized at program startup (global scope) before main()
 * executes, ensuring they are available during SimTop construction and CLI
 * parsing.
 *
 * **Why Bidirectional Maps?**
 * - **Separation of concerns**: Parse and display are different operations
 * - **Multiple input formats**: Allow both descriptive and numeric names
 * - **User-friendly output**: Display full descriptive names in logs
 * - **Backward compatibility**: Support legacy numeric names (V1-V8)
 * - **Error validation**: CLI11 CheckedTransformer validates against known keys
 *
 * **Extension Pattern:**
 * To add a new ThreadManager version (e.g., V9):
 * ```cpp
 * // 1. Define enum in ACALSimConfig.hh
 * enum class ThreadManagerVersion {
 *     ...,
 *     V9  // New version
 * };
 *
 * // 2. Add to ThreadManagerVersionMap (this file)
 * {"V9", ThreadManagerVersion::V9},
 * {"SuperFast", ThreadManagerVersion::V9},  // Descriptive name
 *
 * // 3. Add to ThreadManagerVersionReMap (this file)
 * {ThreadManagerVersion::V9, "ThreadManagerVersion::SuperFast (V9)"},
 *
 * // 4. Implement ThreadManagerV9 class
 * // 5. Add case to SimTop::initThreadManager()
 * ```
 *
 * **Thread Safety:**
 * Both maps are const after initialization and read-only during simulation,
 * making them inherently thread-safe for concurrent access.
 *
 * @see ACALSimConfig.hh For enum definition and class interface
 * @see CLIManager.cc For CLI parsing integration with CLI11
 * @see SimTop.cc For ThreadManager instantiation based on enum
 * @see ThreadManager.hh For ThreadManager interface documentation
 */

#include "config/ACALSimConfig.hh"

namespace acalsim {

// for framework to parse from CLI.
std::map<std::string, ThreadManagerVersion> ThreadManagerVersionMap = {
    // Production versions (descriptive names)
    {"PriorityQueue", ThreadManagerVersion::PriorityQueue},
    {"Barrier", ThreadManagerVersion::Barrier},
    {"PrebuiltTaskList", ThreadManagerVersion::PrebuiltTaskList},
    {"LocalTaskQueue", ThreadManagerVersion::LocalTaskQueue},
    {"Default", ThreadManagerVersion::Default},
    // Backward compatibility (numeric names)
    {"V1", ThreadManagerVersion::V1},
    {"V2", ThreadManagerVersion::V2},
    {"V3", ThreadManagerVersion::V3},
    {"V6", ThreadManagerVersion::V6},
    // Experimental versions
    {"V4", ThreadManagerVersion::V4},
    {"V5", ThreadManagerVersion::V5},
    {"V7", ThreadManagerVersion::V7},
    {"V8", ThreadManagerVersion::V8}};

// for framework to print the parameter.
std::map<ThreadManagerVersion, std::string> ThreadManagerVersionReMap = {
    {ThreadManagerVersion::PriorityQueue, "ThreadManagerVersion::PriorityQueue (V1)"},
    {ThreadManagerVersion::Barrier, "ThreadManagerVersion::Barrier (V2)"},
    {ThreadManagerVersion::PrebuiltTaskList, "ThreadManagerVersion::PrebuiltTaskList (V3)"},
    {ThreadManagerVersion::LocalTaskQueue, "ThreadManagerVersion::LocalTaskQueue (V6)"},
    {ThreadManagerVersion::V4, "ThreadManagerVersion::V4 (Experimental)"},
    {ThreadManagerVersion::V5, "ThreadManagerVersion::V5 (Experimental)"},
    {ThreadManagerVersion::V7, "ThreadManagerVersion::V7 (Experimental)"},
    {ThreadManagerVersion::V8, "ThreadManagerVersion::V8 (Experimental)"}};

}  // namespace acalsim
