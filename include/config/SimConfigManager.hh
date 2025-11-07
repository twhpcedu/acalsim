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
 * @file SimConfigManager.hh
 * @brief Configuration management and parameter registry for simulation components
 *
 * SimConfigManager provides a centralized registry for SimConfig objects, enabling
 * structured parameter management with JSON file parsing and runtime updates.
 *
 * **Configuration Architecture:**
 * ```
 * SimConfigManager
 *   │
 *   ├─ SimConfig "cpu"
 *   │   ├─ Parameter<int> "cores"
 *   │   ├─ Parameter<double> "frequency"
 *   │   └─ Parameter<CacheConfig> "l1_cache"
 *   │       ├─ size
 *   │       ├─ associativity
 *   │       └─ latency
 *   │
 *   ├─ SimConfig "memory"
 *   │   ├─ Parameter<size_t> "dram_size"
 *   │   └─ Parameter<int> "channels"
 *   │
 *   └─ SimConfig "network"
 *       ├─ Parameter<int> "bandwidth"
 *       └─ Parameter<std::string> "topology"
 * ```
 *
 * **Configuration Workflow:**
 * ```
 * 1. Definition (in Simulator/Module):
 *    class CPUSimulator : public SimBase, public SimConfigManager {
 *    protected:
 *        void registerConfigs() override {
 *            auto* cpuConfig = new SimConfig("cpu");
 *            cpuConfig->addParameter("cores", 4);
 *            cpuConfig->addParameter("frequency", 3.5);
 *            addConfig("cpu", cpuConfig);
 *        }
 *    };
 *
 * 2. Loading (from JSON file):
 *    {
 *      "cpu": {
 *        "cores": 8,
 *        "frequency": 4.2
 *      }
 *    }
 *
 * 3. Parsing (in initialization):
 *    parseConfigFiles({"config/cpu.json"});
 *
 * 4. Access (at runtime):
 *    int cores = getParameter<int>("cpu", "cores");        // 8
 *    double freq = getParameter<double>("cpu", "frequency"); // 4.2
 *
 * 5. Runtime Update:
 *    updateParameter<int>("cpu", "cores", "", 16);
 * ```
 *
 * **Key Features:**
 * - **Hierarchical Organization**: Group related parameters in named SimConfig objects
 * - **Type-Safe Access**: Template-based getParameter<T>() with compile-time type checking
 * - **JSON Integration**: Parse configuration files to populate parameters
 * - **Struct Member Access**: Access nested struct fields directly
 * - **Runtime Updates**: Modify parameters during simulation
 * - **Virtual Inheritance**: Compatible with multiple inheritance hierarchies
 *
 * **Usage Example:**
 * ```cpp
 * // Define custom configuration struct
 * struct CacheConfig {
 *     size_t size;
 *     int associativity;
 *     int latency;
 * };
 *
 * class CacheSimulator : public SimBase, public SimConfigManager {
 * public:
 *     CacheSimulator() : SimConfigManager("cache_sim") {
 *         registerConfigs();
 *         parseConfigFiles({"config/cache.json"});
 *     }
 *
 * protected:
 *     void registerConfigs() override {
 *         auto* config = new SimConfig("cache");
 *
 *         // Simple parameters
 *         config->addParameter("enabled", true);
 *         config->addParameter("write_policy", std::string("writeback"));
 *
 *         // Struct parameter
 *         CacheConfig l1Config{32768, 8, 4};
 *         config->addParameter("l1", l1Config);
 *
 *         addConfig("cache", config);
 *     }
 *
 * public:
 *     void init() override {
 *         // Access simple parameters
 *         bool enabled = getParameter<bool>("cache", "enabled");
 *         std::string policy = getParameter<std::string>("cache", "write_policy");
 *
 *         // Access struct members
 *         size_t l1Size = getParameterMemberData<CacheConfig, size_t>(
 *             "cache", "l1", "size");
 *         int assoc = getParameterMemberData<CacheConfig, int>(
 *             "cache", "l1", "associativity");
 *
 *         initializeCache(l1Size, assoc, policy);
 *     }
 * };
 *
 * // JSON configuration file (cache.json):
 * {
 *   "cache": {
 *     "enabled": true,
 *     "write_policy": "writethrough",
 *     "l1": {
 *       "size": 65536,
 *       "associativity": 16,
 *       "latency": 3
 *     }
 *   }
 * }
 * ```
 *
 * @see SimConfig For individual configuration object implementation
 * @see Parameter For type-safe parameter storage
 * @see ACALSimConfig For framework-level global configuration
 */

#pragma once

#include <string>
#include <unordered_map>

// ACALSim Library
#include "config/SimConfig.hh"
#include "utils/HashableType.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**
 * @class SimConfigManager
 * @brief Configuration registry and parameter management for simulation components
 *
 * SimConfigManager aggregates multiple SimConfig objects, each representing a logical
 * grouping of related parameters. Provides type-safe parameter access, JSON file
 * parsing, and runtime parameter updates.
 *
 * **Design Pattern:**
 * - Uses virtual inheritance for compatibility with multiple inheritance hierarchies
 * - Friend relationship with SimConfig for internal access
 * - Template-based parameter access with compile-time type checking
 * - Supports nested struct member access via getParameterMemberData<>()
 *
 * **Typical Usage:**
 * ```
 * Simulator/Module
 *   ↓ inherits from
 * SimConfigManager
 *   ↓ contains
 * SimConfig objects ("cpu", "memory", "cache", etc.)
 *   ↓ each contains
 * Parameter<T> objects (typed parameters)
 * ```
 *
 * @note Most simulators and modules should inherit from both SimBase and SimConfigManager
 * @see SimConfig, Parameter, ACALSimConfig
 */
class SimConfigManager : virtual public HashableType {
	friend class SimConfig;

public:
	/**
	 * @brief Construct configuration manager with name
	 *
	 * @param _name Manager identifier (typically simulator/module name)
	 *
	 * **Usage:**
	 * ```cpp
	 * class CPUSimulator : public SimBase, public SimConfigManager {
	 * public:
	 *     CPUSimulator() : SimConfigManager("cpu_sim") { ... }
	 * };
	 * ```
	 */
	SimConfigManager(const std::string& _name) : name(_name) {}

	/**
	 * @brief Destructor - cleans up all registered SimConfig objects
	 *
	 * Iterates through all registered SimConfig objects and deletes them,
	 * preventing memory leaks.
	 *
	 * @note Logs deletion of each SimConfig in verbose mode
	 */
	~SimConfigManager() {
		for (auto& it : configs) {
			VERBOSE_CLASS_INFO << "Deleting SimConfig object : " << it.first;
			delete it.second;
		}
	}

	/**
	 * @brief Get parameter value with type safety
	 *
	 * Retrieves a parameter value from the specified SimConfig object. Uses
	 * template parameter T to ensure compile-time type checking.
	 *
	 * @tparam T Parameter type (int, double, std::string, custom struct, etc.)
	 * @param _configName Name of SimConfig object containing the parameter
	 * @param _paramName Name of parameter to retrieve
	 * @return T Parameter value of type T
	 *
	 * **Usage Example:**
	 * ```cpp
	 * // Simple types
	 * int cores = getParameter<int>("cpu", "cores");
	 * double freq = getParameter<double>("cpu", "frequency");
	 * std::string policy = getParameter<std::string>("cache", "write_policy");
	 *
	 * // Custom struct
	 * struct MemoryConfig { size_t size; int channels; };
	 * MemoryConfig memCfg = getParameter<MemoryConfig>("memory", "config");
	 * ```
	 *
	 * @throws std::runtime_error if config or parameter not found
	 * @note Type T must match parameter's stored type (enforced at runtime)
	 * @see getParameterMemberData() For accessing struct members
	 */
	template <typename T>
	T getParameter(const std::string& _configName, const std::string& _paramName) const {
		return this->getConfig(_configName)->getParameter<T>(_paramName);
	}

	/**
	 * @brief Get struct member value from parameter
	 *
	 * Accesses a specific member field within a struct-type parameter.
	 * Useful for complex configuration structs where only one field is needed.
	 *
	 * @tparam T_Struct Struct type containing the member
	 * @tparam T_Member Type of the member field to retrieve
	 * @param _configName Name of SimConfig object
	 * @param _paramName Name of parameter (must be struct type)
	 * @param _memberData Name of struct member field
	 * @return T_Member Value of the specified struct member
	 *
	 * **Usage Example:**
	 * ```cpp
	 * struct CacheConfig {
	 *     size_t size;
	 *     int associativity;
	 *     int latency;
	 * };
	 *
	 * // Access individual struct members without retrieving entire struct
	 * size_t cacheSize = getParameterMemberData<CacheConfig, size_t>(
	 *     "cache", "l1_config", "size");
	 * int assoc = getParameterMemberData<CacheConfig, int>(
	 *     "cache", "l1_config", "associativity");
	 * int latency = getParameterMemberData<CacheConfig, int>(
	 *     "cache", "l1_config", "latency");
	 * ```
	 *
	 * @throws std::runtime_error if config, parameter, or member not found
	 * @note Requires T_Struct to match parameter's stored type
	 * @see getParameter() For accessing entire struct
	 */
	template <typename T_Struct, typename T_Member>
	T_Member getParameterMemberData(const std::string& _configName, const std::string& _paramName,
	                                const std::string& _memberData) const {
		return this->getConfig(_configName)->getParameterMemberData<T_Struct, T_Member>(_paramName, _memberData);
	}

protected:
	/**
	 * @brief Register user-defined configurations (override in derived classes)
	 *
	 * Virtual hook for subclasses to register their SimConfig objects during
	 * initialization. Called automatically before parsing config files.
	 *
	 * **Override Pattern:**
	 * ```cpp
	 * class CPUSimulator : public SimBase, public SimConfigManager {
	 * protected:
	 *     void registerConfigs() override {
	 *         // Create and register CPU config
	 *         auto* cpuConfig = new SimConfig("cpu");
	 *         cpuConfig->addParameter("cores", 4);
	 *         cpuConfig->addParameter("frequency", 3.5);
	 *         addConfig("cpu", cpuConfig);
	 *
	 *         // Create and register cache config
	 *         auto* cacheConfig = new SimConfig("cache");
	 *         cacheConfig->addParameter("size", 32768);
	 *         cacheConfig->addParameter("associativity", 8);
	 *         addConfig("cache", cacheConfig);
	 *     }
	 * };
	 * ```
	 *
	 * @note Default implementation does nothing (no-op)
	 * @note Should be called once during initialization
	 * @see addConfig()
	 */
	virtual void registerConfigs() {}

	/**
	 * @brief Register SimConfig object in manager
	 *
	 * Adds a SimConfig object to the internal registry, making its parameters
	 * accessible via getParameter(). Takes ownership of the config pointer.
	 *
	 * @param _name Config identifier (e.g., "cpu", "memory", "cache")
	 * @param _config Pointer to SimConfig object (manager takes ownership)
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* config = new SimConfig("network");
	 * config->addParameter("bandwidth", 100);
	 * config->addParameter("latency", 5);
	 * addConfig("network", config);
	 * ```
	 *
	 * @note Called within registerConfigs() override
	 * @note Config will be deleted in destructor
	 * @see registerConfigs(), SimConfig
	 */
	void addConfig(const std::string& _name, SimConfig* _config);

	/**
	 * @brief Parse JSON configuration files and update parameters
	 *
	 * Reads one or more JSON configuration files and updates registered
	 * parameters with values from the files. Merges multiple files if provided.
	 *
	 * @param _configFilePaths Vector of JSON file paths to parse
	 *
	 * **JSON File Format:**
	 * ```json
	 * {
	 *   "cpu": {
	 *     "cores": 8,
	 *     "frequency": 4.2
	 *   },
	 *   "memory": {
	 *     "size": 16384,
	 *     "channels": 4
	 *   }
	 * }
	 * ```
	 *
	 * **Usage:**
	 * ```cpp
	 * // Single file
	 * parseConfigFiles({"config/cpu.json"});
	 *
	 * // Multiple files (merged)
	 * parseConfigFiles({"config/base.json", "config/override.json"});
	 * ```
	 *
	 * @note Call after registerConfigs() to override default values
	 * @note Later files override earlier files for conflicting parameters
	 * @throws std::runtime_error if file not found or invalid JSON
	 * @see registerConfigs(), addConfig()
	 */
	void parseConfigFiles(const std::vector<std::string>& _configFilePaths);

	/**
	 * @brief Update parameter value at runtime
	 *
	 * Modifies a parameter's value during simulation. Supports both simple
	 * parameters and struct member updates via compile-time if constexpr.
	 *
	 * @tparam T Value type to set
	 * @tparam TStruct Struct type (void for simple parameters)
	 * @param _configName Name of SimConfig containing the parameter
	 * @param _paramName Name of parameter to update
	 * @param _member_name Struct member name (empty string "" for simple parameters)
	 * @param _value New value to set
	 *
	 * **Usage:**
	 * ```cpp
	 * // Update simple parameter
	 * updateParameter<int>("cpu", "cores", "", 16);
	 * updateParameter<double>("cpu", "frequency", "", 4.5);
	 *
	 * // Update struct member
	 * struct CacheConfig { size_t size; int assoc; };
	 * updateParameter<size_t, CacheConfig>("cache", "l1", "size", 65536);
	 * updateParameter<int, CacheConfig>("cache", "l1", "assoc", 16);
	 * ```
	 *
	 * @note Logs update in verbose mode
	 * @throws std::runtime_error if config or parameter not found
	 * @see getParameter(), setParameter()
	 */
	template <typename T, typename TStruct = void>
	void updateParameter(const std::string& _configName, const std::string& _paramName, const std::string& _member_name,
	                     const T& _value) {
		if constexpr (std::is_same_v<TStruct, void>) {
			this->getConfig(_configName)->setParameter<T>(_paramName, _value);
			VERBOSE_CLASS_INFO << "Parameter \'" + _paramName + "\' is updated";
		} else {
			this->getConfig(_configName)->setParameterMemberData<TStruct, T>(_paramName, _member_name, _value);
			VERBOSE_CLASS_INFO << "Parameter \'" + _paramName + "::" + _member_name + "\' is updated";
		}
	}

	/**
	 * @brief Get SimConfig object by name (internal use)
	 *
	 * Retrieves pointer to SimConfig object from internal registry.
	 * Used internally by getParameter() and updateParameter().
	 *
	 * @param _configName Name of SimConfig to retrieve
	 * @return SimConfig* Pointer to config object
	 *
	 * @throws std::runtime_error if config not found
	 * @note Protected - use getParameter() for external access
	 */
	SimConfig* getConfig(const std::string& _configName) const;

private:
	/// @brief Registry mapping config names to SimConfig objects
	std::unordered_map<std::string, SimConfig*> configs;

	/// @brief Manager identifier (simulator/module name)
	const std::string name;
};

}  // end of namespace acalsim
