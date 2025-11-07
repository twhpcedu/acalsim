/*
 * Copyright 2023-2025 Playlab/ACAL
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
 * @file CLIManager.cc
 * @brief CLIManager implementation - command-line interface integration with SimConfig
 *
 * This file implements CLIManager's core functionality for bridging command-line
 * arguments with SimConfig parameters using the CLI11 library, enabling runtime
 * configuration overrides via command line.
 *
 * **Deferred Parameter Application Architecture:**
 * ```
 * Command Line:         CLI11 Parsing:         Deferred Storage:      Final Application:
 * ./sim --cores=8   →   Parse with CLI11   →   cliParameters[]    →   setCLIParametersToSimConfig()
 *                                                   │                           │
 *                                                   │ CLIParameter {            │ For each CLIParameter:
 *                                                   │   configName: "cpu"       │   updateFunc()
 *                                                   │   paramName: "cores"      │     └─ setParameter("cores", 8)
 *                                                   │   updateFunc: λ           │
 *                                                   │     └─ Captures value=8   │
 *                                                   │ }                         │
 *                                                   │                           │
 *                                                   └───────────────────────────┘
 *
 * Why deferred?
 * - Allows CLI to override config file values
 * - Config files parsed first (lower priority)
 * - CLI values applied last (highest priority)
 * ```
 *
 * **Configuration Priority Enforcement:**
 * ```
 * SimTop Initialization Sequence:
 *   │
 *   ├─ 1. registerConfigs()
 *   │     └─ Set default values in code
 *   │        Example: config->addParameter("cores", 4)
 *   │
 *   ├─ 2. registerCLIArguments()
 *   │     └─ Register CLI options (creates CLIParameter entries)
 *   │        Example: addCLIOption<int>("--cores", "CPU cores", "cpu", "cores")
 *   │
 *   ├─ 3. parseCLIArguments(argc, argv)
 *   │     └─ CLI11 parses command line, populates CLI option values
 *   │        Example: User runs: ./sim --cores=8
 *   │        Result: CLI11 stores 8 in option variable
 *   │
 *   ├─ 4. parseConfigFiles(configFilePaths)
 *   │     └─ Load JSON configs, override defaults
 *   │        Example: config.json contains "cores": 6
 *   │        Result: Parameter updated from 4 → 6
 *   │
 *   └─ 5. setCLIParametersToSimConfig()  ◄── THIS FILE
 *        └─ Apply CLI overrides, final priority
 *           Example: Execute updateFunc() → setParameter("cores", 8)
 *           Result: Parameter updated from 6 → 8 (CLI wins)
 *
 * Final result: cores = 8 (CLI) > 6 (JSON) > 4 (default)
 * ```
 *
 * **CLIParameter Deferred Update Mechanism:**
 * ```cpp
 * // When user calls addCLIOption<int>("--cores", ..., "cpu", "cores"):
 *
 * 1. CLI11 creates option and links to storage variable:
 *    int cores_value;
 *    app.add_option("--cores", cores_value, "CPU cores");
 *
 * 2. addCLIOption creates lambda capturing the storage variable:
 *    auto updateFunc = [this, cores_value]() {
 *        this->getConfig("cpu")->setParameter<int>("cores", cores_value);
 *    };
 *
 * 3. addCLIParameter stores the lambda:
 *    cliParameters.push_back({
 *        configName: "cpu",
 *        paramName: "cores",
 *        updateFunc: updateFunc  // Lambda captured cores_value
 *    });
 *
 * 4. Later, setCLIParametersToSimConfig() executes all lambdas:
 *    for (auto& cli_param : cliParameters) {
 *        cli_param.updateFunc();  // Applies CLI value to SimConfig
 *    }
 * ```
 *
 * **Framework CLI Arguments (registerACALSimCLIArguments):**
 * This method registers standard ACALSim framework command-line options:
 *
 * | Option                | Type            | Description                        | Default    |
 * |-----------------------|-----------------|-------------------------------------|-----------|
 * | `-g, --googletest`    | Flag (bool)     | Enable Google Test mode            | false     |
 * | `-c, --config`        | Multi-string    | Config file paths (0+ files)       | []        |
 * | `-t, --trace`         | String          | Trace output filename              | "trace"   |
 * | `--threadmanager`     | Enum            | ThreadManager version (V1-V8)      | (config)  |
 * | `--threads`           | Int             | Number of worker threads (0=auto)  | 0         |
 *
 * **CLI11 Transform Feature (ThreadManager Selection):**
 * ```cpp
 * // ThreadManagerVersion is an enum, need string→enum conversion
 * addCLIOption<ThreadManagerVersion>("--threadmanager", ...)
 *     ->transform(CLI::CheckedTransformer(ThreadManagerVersionMap, CLI::ignore_case));
 *
 * // ThreadManagerVersionMap:
 * {
 *     {"V1", ThreadManagerVersion::V1},
 *     {"V2", ThreadManagerVersion::V2},
 *     {"v3", ThreadManagerVersion::V3},  // Case insensitive
 *     ...
 * }
 *
 * // Usage:
 * ./sim --threadmanager=V3   → Valid
 * ./sim --threadmanager=v3   → Valid (ignore_case)
 * ./sim --threadmanager=V9   → Error: invalid choice
 * ```
 *
 * **Multi-Config File Support:**
 * ```bash
 * # Single config file
 * ./sim --config=base.json
 *
 * # Multiple config files (applied in order)
 * ./sim --config=base.json --config=platform.json --config=override.json
 *
 * # CLI11 expected(0, -1) allows 0 or more occurrences
 * # Each --config adds to configFilePathsFromCLI vector
 * # SimConfigManager::parseConfigFiles() processes them sequentially
 * ```
 *
 * **Google Test Integration:**
 * ```cpp
 * // When running under Google Test framework:
 * ./test_simulator --googletest
 *
 * // CLIManager::isGTestMode() returns true
 * // Simulators can use this to:
 * if (isGTestMode()) {
 *     // Disable tracing (reduce test output)
 *     // Use deterministic seeds
 *     // Enable assertions
 *     // Reduce simulation duration
 * }
 * ```
 *
 * **Implementation Functions:**
 *
 * 1. **setCLIParametersToSimConfig()**: (lines 23-25)
 *    - Executes all deferred CLI parameter updates
 *    - Iterates through cliParameters vector
 *    - Calls each updateFunc() lambda to apply CLI value
 *    - Must be called AFTER parseConfigFiles() for correct priority
 *
 * 2. **addCLIParameter()**: (lines 27-31)
 *    - Internal method to register deferred parameter update
 *    - Stores CLIParameter with configName, paramName, and lambda
 *    - Called by addCLIOption<T>() template (in .inl file)
 *    - Enables deferred execution pattern
 *
 * 3. **registerACALSimCLIArguments()**: (lines 33-50)
 *    - Registers framework-level CLI options
 *    - Uses CLI11 API (add_flag, add_option)
 *    - Sets up enum transforms for ThreadManagerVersion
 *    - Called automatically during SimTop initialization
 *
 * **CLI11 API Methods Used:**
 * - `add_flag()`: Boolean flags (--googletest sets gTestMode=true)
 * - `add_option()`: Typed options (--threads=8)
 * - `default_val()`: Set default for flags
 * - `default_str()`: Set default display string
 * - `expected(0, -1)`: Allow 0 or more occurrences (multi-config)
 * - `transform()`: String→enum conversion with validation
 *
 * **Example Complete CLI Session:**
 * ```bash
 * # Command line
 * ./sim --config=base.json --cores=8 --frequency=3.5 --threadmanager=V3 --threads=4
 *
 * # Execution flow:
 * 1. registerACALSimCLIArguments() registers --threadmanager, --threads
 * 2. registerCLIArguments() registers --cores, --frequency (user-defined)
 * 3. parseCLIArguments() parses command line:
 *    - configFilePathsFromCLI = ["base.json"]
 *    - cores CLI value = 8
 *    - frequency CLI value = 3.5
 *    - thread_manager_version = V3
 *    - nCustomThreads = 4
 * 4. parseConfigFiles(["base.json"]) loads JSON (cores might be 4)
 * 5. setCLIParametersToSimConfig() applies CLI overrides:
 *    - cores: 4 → 8 (CLI wins)
 *    - frequency: 2.5 → 3.5 (CLI wins)
 *
 * # Result: CLI values have highest priority
 * ```
 *
 * @see CLIManager.hh For interface documentation and addCLIOption<T> template
 * @see SimConfig.cc For parameter storage and type system
 * @see SimConfigManager.cc For configuration orchestration
 * @see CLI11 documentation (https://github.com/CLIUtils/CLI11)
 */

#include "config/CLIManager.hh"

#include "config/ACALSimConfig.hh"

namespace acalsim {

void CLIManager::setCLIParametersToSimConfig() {
	for (auto cli_param : this->cliParameters) { cli_param.updateFunc(); }
}

void CLIManager::addCLIParameter(const std::string& _configName, const std::string& _paramName,
                                 std::function<void()> _updateFunc) {
	auto cli_param = CLIParameter{_configName, _paramName, _updateFunc};
	this->cliParameters.push_back(cli_param);
}

void CLIManager::registerACALSimCLIArguments() {
	this->getCLIApp()
	    ->add_flag("-g,--googletest", this->gTestMode, "Enable or disable Google Test Framework")
	    ->default_val(this->gTestMode);
	this->getCLIApp()
	    ->add_option("-c,--config", this->configFilePathsFromCLI, "Specifies the path(s) to configuration file(s).")
	    ->expected(0, -1);
	this->getCLIApp()
	    ->add_option("-t,--trace", this->tracingJsonFileName, "Specify the name of the trace file for logging.")
	    ->default_str(this->tracingJsonFileName);
	this->addCLIOption<ThreadManagerVersion>("--threadmanager", "Specify the version of the ThreadManager.", "ACALSim",
	                                         "thread_manager_version")
	    ->transform(CLI::CheckedTransformer(ThreadManagerVersionMap, CLI::ignore_case));
	this->getCLIApp()
	    ->add_option("--threads", this->nCustomThreads,
	                 "Set the number of threads for the simulation. Use 0 for the default configuration.")
	    ->default_str(std::to_string(this->nCustomThreads));
}

}  // namespace acalsim
