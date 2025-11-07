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
 * @file SimConfigManager.cc
 * @brief SimConfigManager implementation - multi-config orchestration and JSON parsing
 *
 * This file implements SimConfigManager, which orchestrates multiple SimConfig objects
 * and provides JSON configuration file parsing with merge capabilities for hierarchical
 * configuration management in ACALSim simulations.
 *
 * **Multi-Config Architecture:**
 * ```
 * SimConfigManager (SimTop inherits)
 *   │
 *   ├─ configs map:
 *   │    │
 *   │    ├─ "ACALSim" → ACALSimConfig
 *   │    │                ├─ thread_manager_version
 *   │    │                └─ num_threads
 *   │    │
 *   │    ├─ "cpu" → CPUConfig
 *   │    │           ├─ cores
 *   │    │           ├─ frequency
 *   │    │           └─ arch
 *   │    │
 *   │    ├─ "memory" → MemoryConfig
 *   │    │              ├─ size
 *   │    │              └─ latency
 *   │    │
 *   │    └─ "cache" → CacheConfig
 *   │                  ├─ l1_size
 *   │                  └─ l2_size
 *   │
 *   └─ parseConfigFiles(paths)
 *        └─ Routes JSON sections to corresponding SimConfig objects
 * ```
 *
 * **JSON File Structure Mapping:**
 * ```json
 * // simulation_config.json
 * {
 *   "ACALSim": {           // → ACALSimConfig::parseParameters()
 *     "thread_manager_version": "V3",
 *     "num_threads": 8
 *   },
 *   "cpu": {               // → CPUConfig::parseParameters()
 *     "cores": 4,
 *     "frequency": 3.5
 *   },
 *   "memory": {            // → MemoryConfig::parseParameters()
 *     "size": 8192,
 *     "latency": 100
 *   }
 * }
 * ```
 *
 * **Configuration Registration Flow:**
 * ```
 * SimTop Initialization:
 *   │
 *   ├─ 1. registerConfigs() (user override)
 *   │     ├─ auto cpu_config = new CPUConfig("CPU");
 *   │     ├─ cpu_config->addParameter("cores", 4);
 *   │     ├─ cpu_config->addParameter("frequency", 2.5);
 *   │     │
 *   │     └─ addConfig("cpu", cpu_config)  ◄── THIS FILE (lines 29-34)
 *   │          ├─ Verify no duplicate names
 *   │          ├─ Log registration (VERBOSE mode)
 *   │          └─ Store in configs["cpu"] = cpu_config
 *   │
 *   └─ 2. parseConfigFiles(paths)  ◄── THIS FILE (lines 63-96)
 *        └─ Routes JSON sections to registered configs
 * ```
 *
 * **Multi-File JSON Merging (deprecated mergeJSONConfigs):**
 * Note: mergeJSONConfigs() (lines 36-61) exists but is deprecated. Current
 * implementation uses sequential parsing with duplicate key detection instead.
 *
 * ```
 * OLD Approach (mergeJSONConfigs):
 *   base.json + override.json → merged JSON → parse once
 *
 * CURRENT Approach (parseConfigFiles):
 *   base.json → parse → override.json → parse (with duplicate detection)
 *
 * Why changed?
 * - Better error reporting (know which file has duplicates)
 * - Explicit ordering guarantees
 * - Clearer semantics for overrides
 * ```
 *
 * **JSON Parsing Flow (parseConfigFiles):**
 * ```
 * parseConfigFiles(["base.json", "override.json"])
 *   │
 *   ├─ For each config file path:
 *   │   │
 *   │   ├─ 1. Validate file exists
 *   │   │     └─ ASSERT if file not found
 *   │   │
 *   │   ├─ 2. Open and parse JSON
 *   │   │     ├─ Try nlohmann::json::parse()
 *   │   │     └─ Catch parse_error, log and skip file
 *   │   │
 *   │   ├─ 3. For each top-level key in JSON:
 *   │   │     │
 *   │   │     ├─ Check for duplicates across files
 *   │   │     │   └─ ERROR if key already processed
 *   │   │     │
 *   │   │     ├─ Lookup SimConfig by key name
 *   │   │     │   ├─ Found: config->parseParameters(params)
 *   │   │     │   │          └─ Delegate to SimConfig::parseParameters()
 *   │   │     │   │
 *   │   │     │   └─ Not found: WARNING (unregistered config)
 *   │   │     │
 *   │   │     └─ Mark key as processed
 *   │   │
 *   │   └─ Return
 *   │
 *   └─ All files processed
 * ```
 *
 * **Duplicate Key Detection:**
 * The processed_keys set tracks which config sections have been seen across
 * all files to detect conflicts:
 *
 * ```
 * base.json:
 *   { "cpu": {...}, "memory": {...} }
 *
 * override.json:
 *   { "cache": {...}, "cpu": {...} }  // ERROR: "cpu" already in base.json
 *
 * Result:
 *   processed_keys: {"cpu", "memory"}
 *   When parsing override.json:
 *     - "cache" → OK, add to processed_keys
 *     - "cpu" → ERROR: Duplicate key found
 * ```
 *
 * **Error Handling Strategy:**
 * | Error Type                  | Severity | Action                        | Line  |
 * |----------------------------|----------|--------------------------------|-------|
 * | File does not exist        | FATAL    | ASSERT, halt execution        | 67    |
 * | File cannot be opened      | FATAL    | ASSERT, halt execution        | 70    |
 * | JSON parsing error         | ERROR    | Log error, skip file          | 76    |
 * | Duplicate config key       | ERROR    | Log error, continue parsing   | 82    |
 * | Unregistered config key    | WARNING  | Log warning, skip section     | 89-92 |
 *
 * **Configuration Lookup (getConfig):**
 * ```
 * getConfig("cpu")
 *   │
 *   ├─ Search configs map for "cpu"
 *   │   ├─ Found: return CPUConfig*
 *   │   └─ Not found: ASSERT (critical error)
 *   │
 *   └─ Return SimConfig*
 *
 * Usage:
 *   auto cpu_config = manager->getConfig("cpu");
 *   int cores = cpu_config->getParameter<int>("cores");
 * ```
 *
 * **Integration with CLI Override System:**
 * ```
 * Initialization Order (from SimTop::initConfig):
 *   1. registerConfigs()              // Set defaults in code
 *   2. registerCLIArguments()         // Register CLI options
 *   3. parseCLIArguments()            // Parse command line
 *   4. parseConfigFiles(paths)        // Load JSON (THIS FILE)
 *      └─ Overrides defaults with JSON values
 *   5. setCLIParametersToSimConfig()  // CLI overrides JSON
 *      └─ CLI has highest priority
 *
 * Priority: CLI > JSON > Defaults
 * ```
 *
 * **Example Multi-File Configuration:**
 * ```cpp
 * // SimTop initialization
 * class MySimTop : public SimTop {
 * protected:
 *     void registerConfigs() override {
 *         addConfig("cpu", new CPUConfig());
 *         addConfig("memory", new MemoryConfig());
 *         addConfig("cache", new CacheConfig());
 *     }
 * };
 *
 * // File: base_platform.json
 * {
 *     "cpu": {
 *         "cores": 4,
 *         "frequency": 2.5
 *     },
 *     "memory": {
 *         "size": 4096
 *     }
 * }
 *
 * // File: workload_specific.json
 * {
 *     "cache": {
 *         "l1_size": 32768,
 *         "l2_size": 262144
 *     }
 * }
 *
 * // Usage:
 * std::vector<std::string> paths = {"base_platform.json", "workload_specific.json"};
 * simTop->parseConfigFiles(paths);
 * // Result: "cpu", "memory", "cache" all configured
 * ```
 *
 * **Advantages of Multi-Config Design:**
 * - **Separation of Concerns**: Each component has its own config object
 * - **Reusability**: CPUConfig can be reused across different simulators
 * - **Scalability**: Easy to add new config sections without modifying SimTop
 * - **Type Safety**: Each config enforces its own parameter types
 * - **Hierarchical Organization**: Large simulations with 10+ config sections
 *
 * **JSON Library Integration:**
 * Uses nlohmann::json for parsing:
 * - `json::parse()`: Parse JSON from stream with exception handling
 * - `items()`: Iterate over top-level keys
 * - `parse_error`: Exception type for invalid JSON
 * - Automatic file I/O with std::fstream integration
 *
 * **Logging Levels:**
 * - VERBOSE_CLASS_INFO: Config registration (line 32)
 * - LABELED_WARNING: Unregistered config keys (line 89)
 * - LABELED_ERROR: Duplicate keys, JSON parse errors (lines 76, 82)
 * - LABELED_ASSERT_MSG: File not found, file open failure (lines 67, 70)
 *
 * @see SimConfigManager.hh For interface documentation
 * @see SimConfig.cc For individual config parameter parsing
 * @see CLIManager.cc For CLI override integration
 * @see nlohmann::json documentation (https://github.com/nlohmann/json)
 */

#include "config/SimConfigManager.hh"

#include <fstream>
#include <unordered_set>

#include "config/SimConfig.hh"

// Third-Party Library
#include <nlohmann/json.hpp>

namespace acalsim {

void SimConfigManager::addConfig(const std::string& name, SimConfig* config) {
	CLASS_ASSERT_MSG(!this->configs.contains(name),
	                 "SimConfig `" + name + "` is already registered in SimConfigManager.");
	VERBOSE_CLASS_INFO << "Adding SimConfig: " << name;
	this->configs.emplace(name, config);
}

nlohmann::json mergeJSONConfigs(const std::vector<std::string>& configFilePaths) {
	nlohmann::json mergedConfig;

	for (const auto& path : configFilePaths) {
		if (path == "") continue;

		if (!std::filesystem::exists(path)) {
			std::cerr << "Warning: File " << path << " does not exist.\n";
			continue;
		}

		std::fstream f(path);
		if (!f.is_open()) {
			std::cerr << "Error opening file: " << path << "\n";
			continue;
		}

		nlohmann::json j;
		f >> j;  // Parse the current file

		// Merge the current file into the main config
		mergedConfig.merge_patch(j);  // Merge with priority to the new config
	}

	return mergedConfig;
}

void SimConfigManager::parseConfigFiles(const std::vector<std::string>& _configFilePaths) {
	std::unordered_set<std::string> processed_keys;

	for (const auto& path : _configFilePaths) {
		LABELED_ASSERT_MSG(std::filesystem::exists(path), this->name, "Warning: File " << path << " does not exist.");

		std::fstream f(path);
		LABELED_ASSERT_MSG(f.is_open(), this->name, "Error opening file: " << path);

		nlohmann::json j;
		try {
			j = nlohmann::json::parse(f);
		} catch (const nlohmann::json::parse_error& e) {
			LABELED_ERROR(this->name) << "JSON parsing error: " << e.what() << " in file " << path << "\n";
		}

		// Check for duplicate keys and merge the current file into the merged config
		for (const auto& [key, params] : j.items()) {
			if (processed_keys.find(key) != processed_keys.end()) {
				LABELED_ERROR(this->name) << "Duplicate key found: '" << key << "' in file " << path << ".\n";
			}
			processed_keys.insert(key);

			if (auto iter = this->configs.find(key); iter != this->configs.end()) {
				iter->second->parseParameters(params);
			} else {
				LABELED_WARNING(this->name)
				    << "Unrecognized configuration key: \'" << key << "` found in \'" << path
				    << "`. This key is not registered in the ACALSim Framework. "
				    << "Please use SimTop::addConfig(\"config_name\", config_ptr) to register this configuration.";
			}
		}
	}
}

SimConfig* SimConfigManager::getConfig(const std::string& _configName) const {
	auto iter = this->configs.find(_configName);
	CLASS_ASSERT_MSG(iter != this->configs.end(), "The config \'" + _configName + "\' does not exist.");
	return iter->second;
}

}  // namespace acalsim
