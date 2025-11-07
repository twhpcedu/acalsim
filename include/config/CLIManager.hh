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
 * @file CLIManager.hh
 * @brief Command-line interface management with configuration integration
 *
 * CLIManager bridges command-line arguments with SimConfig parameter system,
 * enabling runtime configuration overrides via CLI using the CLI11 library.
 *
 * **CLI Integration Architecture:**
 * ```
 * Command Line                CLIManager              SimConfigManager
 *     │                           │                         │
 *     │  --cores=8                │                         │
 *     │  --frequency=4.2          │                         │
 *     │  --config=cpu.json        │                         │
 *     └──────────────────────────►│                         │
 *                                 │  parseCLIArguments()    │
 *                                 │         │               │
 *                                 │         ▼               │
 *                                 │  CLIParameter list      │
 *                                 │   - cores → 8           │
 *                                 │   - frequency → 4.2     │
 *                                 │         │               │
 *                                 │  setCLIParametersTo...()│
 *                                 │         │               │
 *                                 └─────────┼──────────────►│
 *                                           │               │
 *                                           ▼               ▼
 *                                     SimConfig updates parameters
 * ```
 *
 * **Configuration Priority (highest to lowest):**
 * ```
 * 1. Command-line arguments (--option=value)
 *    ↓ overrides
 * 2. JSON configuration files (--config=file.json)
 *    ↓ overrides
 * 3. Default values in code (addParameter(..., defaultValue))
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * class MySimulator : public SimBase, public CLIManager {
 * public:
 *     MySimulator(int argc, char** argv)
 *         : CLIManager("MySimulator", {"config/base.json"}) {
 *         registerConfigs();
 *         registerCLIArguments();
 *         parseCLIArguments(argc, argv);
 *         parseConfigFiles(configFilePaths);
 *         setCLIParametersToSimConfig();  // CLI overrides config files
 *     }
 *
 * protected:
 *     void registerConfigs() override {
 *         auto* config = new SimConfig("cpu");
 *         config->addParameter("cores", 4);
 *         config->addParameter("frequency", 3.5);
 *         addConfig("cpu", config);
 *     }
 *
 *     void registerCLIArguments() override {
 *         // Add CLI option that maps to config parameter
 *         addCLIOption<int>("--cores,-c", "Number of CPU cores",
 *                          "cpu", "cores");
 *         addCLIOption<double>("--frequency,-f", "CPU frequency (GHz)",
 *                             "cpu", "frequency");
 *     }
 * };
 *
 * // Command line:
 * // ./sim --cores=8 --frequency=4.2 --config=override.json
 * ```
 *
 * **Common CLI Patterns:**
 * ```cpp
 * // Boolean flags
 * addCLIOption<bool>("--enable-tracing", "Enable tracing",
 *                    "debug", "tracing");
 *
 * // Numeric options
 * addCLIOption<int>("--threads", "Number of threads",
 *                   "runtime", "num_threads");
 *
 * // String options
 * addCLIOption<std::string>("--output", "Output directory",
 *                           "io", "output_dir");
 *
 * // Struct members
 * addCLIOption<size_t, CacheConfig>("--cache-size", "L1 cache size",
 *                                   "cache", "l1", "size");
 * ```
 *
 * @see SimConfigManager For parameter management
 * @see SimConfig For configuration object
 * @see CLI11 library (https://github.com/CLIUtils/CLI11)
 */

#pragma once

#include <string>

// ACALSim Library
#include "config/SimConfig.hh"
#include "config/SimConfigManager.hh"

// Third-Party Library
#include <CLI/CLI.hpp>

namespace acalsim {

/**
 * @class CLIManager
 * @brief Command-line interface manager with SimConfig integration
 *
 * CLIManager extends SimConfigManager to provide CLI argument parsing using CLI11
 * library, enabling runtime configuration overrides from command line.
 *
 * **Design Pattern:**
 * - Inherits from SimConfigManager for config parameter management
 * - Uses CLI11 library for robust command-line parsing
 * - Deferred parameter application (CLI overrides config files)
 * - Support for simple and struct member parameters
 *
 * **Initialization Flow:**
 * ```
 * 1. Construct CLIManager with default config files
 * 2. registerConfigs() - Define parameters with defaults
 * 3. registerCLIArguments() - Map CLI options to parameters
 * 4. parseCLIArguments() - Parse command line
 * 5. parseConfigFiles() - Load JSON configs
 * 6. setCLIParametersToSimConfig() - Apply CLI overrides
 * ```
 *
 * @note Most simulators inherit from both SimBase and CLIManager
 * @see SimConfigManager, CLI11
 */
class CLIManager : public SimConfigManager {
	/**
	 * @struct CLIParameter
	 * @brief Internal structure tracking CLI-to-config parameter mappings
	 *
	 * Stores deferred parameter updates from CLI arguments. The updateFunc
	 * lambda captures the CLI-provided value and applies it to SimConfig
	 * when setCLIParametersToSimConfig() is called.
	 *
	 * **Purpose:**
	 * Allows CLI arguments to override configuration file values by deferring
	 * parameter application until after config files are parsed.
	 */
	struct CLIParameter {
		std::string           configName;  ///< SimConfig name (e.g., "cpu")
		std::string           paramName;   ///< Parameter name (e.g., "cores")
		std::function<void()> updateFunc;  ///< Lambda to apply CLI value to config
	};

public:
	/**
	 * @brief Construct CLI manager with default config files
	 *
	 * Initializes CLIManager with default configuration file paths and tracing
	 * settings. Sets up CLI11 application for argument parsing.
	 *
	 * @param name Manager identifier (typically simulator name)
	 * @param _configFilePaths Default config file paths (loaded before CLI overrides)
	 * @param _tracingJsonFileName Tracing output file name (default: "trace")
	 *
	 * **Usage:**
	 * ```cpp
	 * // With default config files
	 * CLIManager cli("MySim", {"config/base.json", "config/platform.json"});
	 *
	 * // Minimal (no default configs)
	 * CLIManager cli("MySim");
	 *
	 * // With custom tracing file
	 * CLIManager cli("MySim", {"config/sim.json"}, "mytrace");
	 * ```
	 *
	 * @note After construction, call registerCLIArguments() → parseCLIArguments()
	 */
	CLIManager(const std::string& name, const std::vector<std::string>& _configFilePaths = {},
	           const std::string& _tracingJsonFileName = "trace")
	    : SimConfigManager("SimConfigManager"),
	      configFilePaths(_configFilePaths),
	      gTestMode(false),
	      tracingJsonFileName(_tracingJsonFileName) {}

	/**
	 * @brief Virtual destructor
	 *
	 * Default destructor. Base class SimConfigManager handles cleanup.
	 */
	virtual ~CLIManager() = default;

	/**
	 * @brief Check if running in Google Test mode
	 *
	 * Returns true if gTest mode flag is set, indicating simulator is running
	 * under unit testing framework.
	 *
	 * @return bool True if in gTest mode, false otherwise
	 *
	 * **Usage:**
	 * ```cpp
	 * if (isGTestMode()) {
	 *     // Disable tracing, reduce verbosity for tests
	 *     config.setParameter("tracing", false);
	 * }
	 * ```
	 *
	 * @note Set via --gtest CLI flag or programmatically
	 */
	bool isGTestMode() const { return this->gTestMode; }

protected:
	/**
	 * @brief Register ACALSim framework-level CLI arguments
	 *
	 * Registers standard framework arguments like --config, --threads, --gtest.
	 * Called automatically by framework initialization.
	 *
	 * @note Internal use - applications don't typically call this directly
	 */
	void registerACALSimCLIArguments();

	/**
	 * @brief Register application-specific CLI arguments (override in derived classes)
	 *
	 * Virtual hook for subclasses to register custom command-line options that
	 * map to their SimConfig parameters.
	 *
	 * **Override Pattern:**
	 * ```cpp
	 * void registerCLIArguments() override {
	 *     addCLIOption<int>("--cores", "CPU cores", "cpu", "cores");
	 *     addCLIOption<double>("--freq", "CPU frequency", "cpu", "frequency");
	 *     addCLIOption<bool>("--trace", "Enable tracing", "debug", "tracing");
	 * }
	 * ```
	 *
	 * @note Called after registerACALSimCLIArguments() and before parseCLIArguments()
	 * @see addCLIOption()
	 */
	virtual void registerCLIArguments() {}

	/**
	 * @brief Parse command-line arguments
	 *
	 * Parses argc/argv using CLI11 library. Exits on parse errors with help message.
	 *
	 * @param argc Argument count from main()
	 * @param argv Argument array from main()
	 *
	 * **Usage:**
	 * ```cpp
	 * int main(int argc, char** argv) {
	 *     MySimulator sim(argc, argv);
	 *     sim.parseCLIArguments(argc, argv);  // Usually in constructor
	 * }
	 * ```
	 *
	 * @throws CLI::ParseError Exits with error code if parsing fails
	 * @note Ensures UTF-8 encoding for argv
	 */
	void parseCLIArguments(int argc, char** argv) {
		argv = this->app.ensure_utf8(argv);
		try {
			this->app.parse(argc, argv);
		} catch (const CLI::ParseError& e) { exit(this->app.exit(e)); }
	}

	/**
	 * @brief Add CLI option mapped to SimConfig parameter
	 *
	 * Registers a command-line option that maps to a SimConfig parameter. When the
	 * option is provided, a lambda is stored to update the parameter after config
	 * files are parsed (enabling CLI to override configs).
	 *
	 * @tparam T Parameter type (int, double, bool, std::string, custom struct, etc.)
	 * @tparam TStruct Struct type (void for simple parameters, struct type for members)
	 * @param _optionName CLI option name (e.g., "--cores,-c" for long and short)
	 * @param _optionDescription Help text shown in --help
	 * @param _configName SimConfig name containing the parameter
	 * @param _paramName Parameter name within SimConfig
	 * @param _memberName Struct member name (empty "" for simple parameters)
	 * @param _defaultValue CLI11 default value flag
	 * @return CLI::Option* Pointer to CLI11 option (for further configuration)
	 *
	 * **Usage:**
	 * ```cpp
	 * // Simple parameter
	 * addCLIOption<int>("--cores,-c", "CPU cores", "cpu", "cores");
	 *
	 * // Struct member
	 * addCLIOption<size_t, CacheConfig>("--cache-size", "Cache size",
	 *                                   "cache", "l1", "size");
	 * ```
	 *
	 * @note Implementation in .inl file (template)
	 * @see setCLIParametersToSimConfig()
	 */
	template <typename T, typename TStruct = void>
	inline CLI::Option* addCLIOption(const std::string& _optionName, const std::string& _optionDescription,
	                                 const std::string& _configName, const std::string& _paramName,
	                                 const std::string& _memberName = "", const bool& _defaultValue = true);

	/**
	 * @brief Apply CLI parameter overrides to SimConfig
	 *
	 * Executes all deferred CLI parameter updates, applying command-line values
	 * to SimConfig objects. Must be called AFTER parseConfigFiles() to ensure
	 * CLI arguments override config file values.
	 *
	 * **Usage:**
	 * ```cpp
	 * registerConfigs();
	 * registerCLIArguments();
	 * parseCLIArguments(argc, argv);
	 * parseConfigFiles(configFilePaths);  // Load JSON
	 * setCLIParametersToSimConfig();      // Apply CLI overrides
	 * ```
	 *
	 * @note Call order is critical for correct priority: CLI > JSON > defaults
	 */
	void setCLIParametersToSimConfig();

	/**
	 * @brief Get CLI11 application instance
	 *
	 * Returns pointer to internal CLI::App for advanced CLI11 operations.
	 *
	 * @return CLI::App* Pointer to CLI11 application
	 * @note Rarely needed - use addCLIOption() for most cases
	 */
	CLI::App* getCLIApp() const { return const_cast<CLI::App*>(&app); }

	/// @brief Tracing JSON output filename (without extension)
	std::string tracingJsonFileName;

	/// @brief Number of custom threads (if overridden via CLI)
	int nCustomThreads = 0;

	/// @brief Default config file paths (from constructor)
	std::vector<std::string> configFilePaths = {};

	/// @brief Additional config files specified via --config CLI argument
	std::vector<std::string> configFilePathsFromCLI = {};

	/// @brief Google Test mode flag
	bool gTestMode;

private:
	/**
	 * @brief Register CLI parameter for deferred update
	 *
	 * Adds a CLIParameter entry to internal list. The updateFunc lambda will
	 * be executed when setCLIParametersToSimConfig() is called.
	 *
	 * @param _configName SimConfig name
	 * @param _paramName Parameter name
	 * @param _updateFunc Lambda capturing CLI value to apply to config
	 *
	 * @note Internal method - use addCLIOption() instead
	 */
	void addCLIParameter(const std::string& _configName, const std::string& _paramName,
	                     std::function<void()> _updateFunc);

	/// @brief List of pending CLI parameter updates
	std::vector<CLIParameter> cliParameters;

	/// @brief CLI11 application instance for argument parsing
	CLI::App app{"ACALSim Description"};
};

}  // end of namespace acalsim

#include "config/CLIManager.inl"
