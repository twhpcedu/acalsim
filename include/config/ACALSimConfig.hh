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
 * @file ACALSimConfig.hh
 * @brief Framework-level configuration for ACALSim parallel simulation
 *
 * ACALSimConfig provides global configuration options for the ACALSim framework,
 * including thread manager version selection, JSON serialization, and string
 * conversion utilities.
 *
 * **Configuration Hierarchy:**
 * ```
 * ACALSimConfig (Framework-level)
 *   ↓ extends
 * SimConfig (Base configuration class)
 *   ↓ contains
 * Parameter<ThreadManagerVersion>
 * ```
 *
 * **ThreadManager Version Selection:**
 * ACALSim supports multiple ThreadManager implementations validated in published
 * research, each optimized for different workload characteristics:
 *
 * ```
 * ┌─────────────────────┬────────────────────────┬──────────────────────────┐
 * │ Version             │ Optimization           │ Best For                 │
 * ├─────────────────────┼────────────────────────┼──────────────────────────┤
 * │ PriorityQueue (V1)  │ Default, general use   │ Sparse activation (DGXSim)│
 * │ Barrier (V2)        │ C++20 barrier sync     │ Exploration/research     │
 * │ PrebuiltTaskList(V3)│ Memory-intensive       │ GPUSim workloads         │
 * │ LocalTaskQueue (V6) │ Lock-optimized V1      │ High contention          │
 * └─────────────────────┴────────────────────────┴──────────────────────────┘
 *
 * Experimental versions (V4, V5, V7, V8): Research prototypes, not validated
 * for production use.
 * ```
 *
 * **Configuration Methods:**
 * 1. **Programmatic (C++ code):**
 * ```cpp
 * ACALSimConfig config;
 * config.setParameter("thread_manager_version",
 *                     ThreadManagerVersion::PrebuiltTaskList);
 * ```
 *
 * 2. **JSON configuration file:**
 * ```json
 * {
 *   "ACALSim": {
 *     "thread_manager_version": {
 *       "type": "ThreadManagerVersion",
 *       "version": 3
 *     }
 *   }
 * }
 * ```
 *
 * 3. **Command-line arguments:**
 * ```bash
 * ./acalsim_app --thread-manager=3
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * #include "config/ACALSimConfig.hh"
 *
 * // In SimTop or main():
 * class SimTop {
 * public:
 *     void initFramework() {
 *         // Create framework config
 *         auto* acalConfig = new ACALSimConfig();
 *
 *         // Set thread manager version
 *         acalConfig->setParameter("thread_manager_version",
 *                                  ThreadManagerVersion::PrebuiltTaskList);
 *
 *         // Or load from JSON
 *         acalConfig->parseConfigFile("config/framework.json");
 *
 *         // Get configured version
 *         auto version = acalConfig->getParameter<ThreadManagerVersion>(
 *             "thread_manager_version");
 *
 *         // Use it to create ThreadManager
 *         threadManager = createThreadManager(version);
 *     }
 * };
 * ```
 *
 * **Version String Conversion:**
 * ```cpp
 * // String to enum
 * ThreadManagerVersion v1 = ThreadManagerVersionMap["PriorityQueue"];
 * ThreadManagerVersion v3 = ThreadManagerVersionMap["PrebuiltTaskList"];
 *
 * // Enum to string
 * std::string name1 = ThreadManagerVersionReMap[ThreadManagerVersion::V1];
 * std::string name3 = ThreadManagerVersionReMap[ThreadManagerVersion::V3];
 * ```
 *
 * @see SimConfig For base configuration class
 * @see SimConfigManager For per-simulator configuration
 * @see ThreadManager For thread pool implementation
 */

#pragma once

#include <map>

#include "config/SimConfig.hh"

// Third-Party Library
#include <nlohmann/json.hpp>

namespace acalsim {

using json = nlohmann::json;

/**
 * @enum ThreadManagerVersion
 * @brief Enumeration for the different versions of thread manager.
 *
 * Production versions (validated in published research):
 * - PriorityQueue (V1): Default, optimized for sparse activation patterns (DGXSim)
 * - Barrier (V2): C++20 barrier-based synchronization (explored)
 * - PrebuiltTaskList (V3): Optimized for memory-intensive workloads (GPUSim)
 * - LocalTaskQueue (V6): Lock-optimized version of V1
 *
 * Experimental versions (not validated for production):
 * - V4, V5, V7, V8: Research prototypes
 */
enum class ThreadManagerVersion {
	// Production versions
	PriorityQueue    = 1,  // V1: Default, sparse activation patterns
	Barrier          = 2,  // V2: C++20 barrier-based (explored)
	PrebuiltTaskList = 3,  // V3: Memory-intensive workloads
	LocalTaskQueue   = 6,  // V6: Lock-optimized V1

	// Backward compatibility aliases
	V1 = 1,
	V2 = 2,
	V3 = 3,
	V6 = 6,

	// Experimental (not for production use)
	V4 = 4,
	V5 = 5,
	V7 = 7,
	V8 = 8,

	// Default
	Default = PriorityQueue
};

/**
 * @brief JSON serialization for ThreadManagerVersion enum.
 */
NLOHMANN_JSON_SERIALIZE_ENUM(ThreadManagerVersion, {{ThreadManagerVersion::PriorityQueue, 1},
                                                    {ThreadManagerVersion::Barrier, 2},
                                                    {ThreadManagerVersion::PrebuiltTaskList, 3},
                                                    {ThreadManagerVersion::V4, 4},
                                                    {ThreadManagerVersion::V5, 5},
                                                    {ThreadManagerVersion::LocalTaskQueue, 6},
                                                    {ThreadManagerVersion::V7, 7},
                                                    {ThreadManagerVersion::V8, 8},
                                                    {ThreadManagerVersion::V1, 1},
                                                    {ThreadManagerVersion::V2, 2},
                                                    {ThreadManagerVersion::V3, 3},
                                                    {ThreadManagerVersion::V6, 6}})

/**
 * @brief String-to-enum mapping for ThreadManagerVersion
 *
 * Maps human-readable version names to ThreadManagerVersion enum values.
 * Useful for parsing configuration files and command-line arguments.
 *
 * **Available Mappings:**
 * - "PriorityQueue" → ThreadManagerVersion::PriorityQueue
 * - "Barrier" → ThreadManagerVersion::Barrier
 * - "PrebuiltTaskList" → ThreadManagerVersion::PrebuiltTaskList
 * - "LocalTaskQueue" → ThreadManagerVersion::LocalTaskQueue
 * - "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8" → Corresponding versions
 *
 * **Usage:**
 * ```cpp
 * std::string userInput = "PrebuiltTaskList";
 * ThreadManagerVersion version = ThreadManagerVersionMap[userInput];
 * ```
 */
extern std::map<std::string, ThreadManagerVersion> ThreadManagerVersionMap;

/**
 * @brief Enum-to-string mapping for ThreadManagerVersion
 *
 * Reverse mapping from ThreadManagerVersion enum values to string names.
 * Useful for logging, error messages, and configuration export.
 *
 * **Usage:**
 * ```cpp
 * ThreadManagerVersion version = ThreadManagerVersion::PrebuiltTaskList;
 * std::string name = ThreadManagerVersionReMap[version];  // "PrebuiltTaskList"
 * ```
 */
extern std::map<ThreadManagerVersion, std::string> ThreadManagerVersionReMap;

/**
 * @class ACALSimConfig
 * @brief Framework-level configuration for ACALSim parallel simulation
 *
 * ACALSimConfig extends SimConfig to provide specialized configuration management
 * for the ACALSim framework, particularly for ThreadManager version selection.
 *
 * **Design Pattern:**
 * - Inherits from SimConfig for parameter management infrastructure
 * - Registers ThreadManagerVersion as user-defined parameter
 * - Provides JSON parsing for ThreadManagerVersion enum
 * - Default configuration uses ThreadManagerVersion::Default (PriorityQueue)
 *
 * **Parameter Registry:**
 * ```
 * ACALSimConfig
 *   └─ Parameter<ThreadManagerVersion> "thread_manager_version"
 *      └─ Default: ThreadManagerVersion::Default
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * // Create and configure
 * ACALSimConfig config;
 * config.setParameter("thread_manager_version",
 *                     ThreadManagerVersion::PrebuiltTaskList);
 *
 * // Load from JSON file
 * config.parseConfigFile("config/acalsim.json");
 *
 * // Access configuration
 * auto version = config.getParameter<ThreadManagerVersion>(
 *     "thread_manager_version");
 * ```
 *
 * @note Single global instance typically managed by SimTop
 * @see SimConfig, ThreadManagerVersion, ThreadManager
 */
class ACALSimConfig : public SimConfig {
public:
	/**
	 * @brief Construct framework configuration
	 *
	 * Initializes ACALSimConfig with default ThreadManagerVersion parameter
	 * set to ThreadManagerVersion::Default (PriorityQueue).
	 *
	 * @param name Configuration name (defaults to "ACALSim")
	 *
	 * **Usage:**
	 * ```cpp
	 * // With default name
	 * ACALSimConfig config;
	 *
	 * // With custom name
	 * ACALSimConfig config("MyFramework");
	 * ```
	 *
	 * @note Automatically registers "thread_manager_version" parameter
	 */
	ACALSimConfig(const std::string& name = "ACALSim") : SimConfig(name) {
		this->addParameter<ThreadManagerVersion>("thread_manager_version", ThreadManagerVersion::Default,
		                                         ParamType::USER_DEFINED);
	}

	/**
	 * @brief Destructor
	 *
	 * Default destructor. Base class SimConfig handles cleanup.
	 */
	~ACALSimConfig() = default;

	/**
	 * @brief Parse user-defined parameter from JSON
	 *
	 * Overrides SimConfig::parseParametersUserDefined() to handle
	 * ThreadManagerVersion enum deserialization from JSON.
	 *
	 * @param _param_name Parameter name (e.g., "thread_manager_version")
	 * @param _param_value JSON object containing type and value
	 *
	 * **Expected JSON Format:**
	 * ```json
	 * {
	 *   "thread_manager_version": {
	 *     "type": "ThreadManagerVersion",
	 *     "version": 3
	 *   }
	 * }
	 * ```
	 *
	 * **Supported Types:**
	 * - "ThreadManagerVersion": Deserializes to ThreadManagerVersion enum
	 *
	 * @note Logs warning if unrecognized type encountered
	 * @see SimConfig::parseParametersUserDefined()
	 */
	void parseParametersUserDefined(const std::string& _param_name, const json& _param_value) override {
		std::string data_type;
		_param_value.at("type").get_to(data_type);

		if (data_type == "ThreadManagerVersion") {
			auto r = _param_value.at("version").get<ThreadManagerVersion>();
			this->setParameter<ThreadManagerVersion>(_param_name, r);
		} else {
			LABELED_INFO(this->getName()) << "Undefined ParamType in parseParameterUserDefine()!";
		}
	}
};

}  // namespace acalsim
