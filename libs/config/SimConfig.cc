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
 * @file SimConfig.cc
 * @brief SimConfig implementation - JSON parsing and parameter management
 *
 * This file implements the core functionality of SimConfig for parsing JSON
 * configuration files and managing typed parameters in ACALSim simulations.
 *
 * **JSON Parsing Flow:**
 * ```
 * Configuration File (config.json)
 *   │
 *   │  {
 *   │    "cpu": {
 *   │      "cores": 8,              // INT
 *   │      "frequency": 3.5,        // FLOAT
 *   │      "arch": "x86_64",        // STRING
 *   │      "boot_tick": 1000,       // TICK
 *   │      "cache": {               // USER_DEFINED (struct)
 *   │        "size": 32768,
 *   │        "associativity": 8
 *   │      }
 *   │    }
 *   │  }
 *   │
 *   ▼
 * SimConfigManager::parseConfigFiles(paths)
 *   │
 *   ├─ Read JSON file with nlohmann::json
 *   │
 *   ▼
 * SimConfig::parseParameters(json)
 *   │
 *   ├─ For each (param_name, param_value) in JSON:
 *   │   │
 *   │   ├─ 1. Check if parameter exists in parameters map
 *   │   │     └─ If not found: Log warning, skip parameter
 *   │   │
 *   │   ├─ 2. Dispatch by ParamType:
 *   │   │     │
 *   │   │     ├─ ParamType::INT
 *   │   │     │   └─ Extract int, call setParameter<int>(name, value)
 *   │   │     │
 *   │   │     ├─ ParamType::FLOAT
 *   │   │     │   └─ Extract float, call setParameter<float>(name, value)
 *   │   │     │
 *   │   │     ├─ ParamType::STRING
 *   │   │     │   └─ Extract string, call setParameter<string>(name, value)
 *   │   │     │
 *   │   │     ├─ ParamType::TICK
 *   │   │     │   └─ Extract int, cast to Tick, call setParameter<Tick>(name, value)
 *   │   │     │
 *   │   │     └─ ParamType::USER_DEFINED
 *   │   │         └─ Call parseParametersUserDefined(name, json_value)
 *   │   │             - User override point for custom struct parsing
 *   │   │             - Example: Parse CacheConfig struct from nested JSON
 *   │   │
 *   │   └─ 3. Log parameter update (VERBOSE mode)
 *   │
 *   └─ Return (all parameters updated from JSON)
 * ```
 *
 * **Type Dispatching Mechanism:**
 * The parseParameters() function uses a switch statement on ParamType to handle
 * different parameter types with type-safe extraction from JSON.
 *
 * | ParamType      | JSON Type    | C++ Type       | Example                  |
 * |----------------|--------------|----------------|--------------------------|
 * | INT            | number (int) | int            | "cores": 8               |
 * | FLOAT          | number       | float          | "frequency": 3.5         |
 * | STRING         | string       | std::string    | "arch": "x86_64"         |
 * | TICK           | number (int) | Tick (uint64_t)| "boot_tick": 1000        |
 * | USER_DEFINED   | object       | Custom struct  | "cache": {...}           |
 *
 * **User-Defined Type Extension:**
 * For custom struct types, users override parseParametersUserDefined():
 *
 * ```cpp
 * struct CacheConfig {
 *     size_t size;
 *     int associativity;
 * };
 *
 * class MyConfig : public SimConfig {
 * public:
 *     MyConfig() : SimConfig("MyConfig") {
 *         addParameter<CacheConfig>("cache", CacheConfig{32768, 8});
 *     }
 *
 * protected:
 *     void parseParametersUserDefined(const std::string& param_name,
 *                                     const json& param_value) override {
 *         if (param_name == "cache") {
 *             CacheConfig cfg;
 *             cfg.size = param_value["size"].get<size_t>();
 *             cfg.associativity = param_value["associativity"].get<int>();
 *             setParameter<CacheConfig>("cache", cfg);
 *             VERBOSE_LABELED_INFO(name) << "Parsed cache config: "
 *                                        << "size=" << cfg.size
 *                                        << " assoc=" << cfg.associativity;
 *         }
 *     }
 * };
 * ```
 *
 * **Parameter Replacement (updateParameter):**
 * The updateParameter() method replaces an existing parameter with a new one,
 * properly cleaning up the old Parameter<T> object.
 *
 * ```
 * updateParameter(name, new_param_ptr)
 *   │
 *   ├─ 1. Verify parameter exists (ASSERT if not found)
 *   ├─ 2. Delete old Parameter<T> object (free memory)
 *   ├─ 3. Store new Parameter<T> in parameters map
 *   └─ Return
 *
 * Use case: Runtime reconfiguration, parameter type changes
 * ```
 *
 * **Error Handling:**
 * - **Unknown parameters in JSON**: Logged as WARNING, skipped gracefully
 * - **Missing required parameters**: Not enforced here (user responsibility)
 * - **Type mismatches**: Will throw nlohmann::json exception during get<T>()
 * - **Non-existent parameter updates**: ASSERT failure with clear error message
 *
 * **JSON Library Integration:**
 * Uses nlohmann::json (https://github.com/nlohmann/json) for JSON parsing:
 * - Type-safe extraction with json.get<T>()
 * - Automatic type checking and conversions
 * - Iterator-based traversal with items()
 * - Exception handling for invalid JSON structures
 *
 * **Logging Verbosity:**
 * - VERBOSE_LABELED_INFO: Parameter updates (only with --verbose flag)
 * - LABELED_WARNING: Unknown parameters in JSON
 * - LABELED_ERROR: Invalid ParamType (should never happen)
 * - LABELED_ASSERT_MSG: Critical errors (parameter not found on update)
 *
 * **Example JSON Parsing Session:**
 * ```
 * Given SimConfig "cpu" with parameters:
 *   - cores: 4 (default)
 *   - frequency: 2.5 (default)
 *   - unknown_param: <not defined>
 *
 * Parsing JSON:
 *   {
 *     "cores": 8,
 *     "frequency": 3.5,
 *     "unknown_param": 42
 *   }
 *
 * Result:
 *   ✓ cores updated: 4 → 8
 *   ✓ frequency updated: 2.5 → 3.5
 *   ⚠ unknown_param: WARNING logged, skipped
 * ```
 *
 * @see SimConfig.hh For parameter management interface
 * @see SimConfigManager.cc For configuration orchestration
 * @see CLIManager.cc For command-line override integration
 */

#include "config/SimConfig.hh"

#include "sim/SimTop.hh"

namespace acalsim {

void SimConfig::parseParameters(const json& _params) {
	for (const auto& [param_name, param_value] : _params.items()) {
		/* set parameters with value in config file */

		if (!this->parameters.contains(param_name)) {
			LABELED_WARNING(this->name) << "The parameter \'" << param_name << "\' is not defined in \'" << this->name
			                            << "\'. It will be skipped during the config file parsing.";
			continue;
		}

		switch (this->parameters.at(param_name)->getType()) {
			case ParamType::INT: {
				auto i = param_value.get<int>();
				this->setParameter<int>(param_name, i);
				VERBOSE_LABELED_INFO(this->name) << param_name + " setted into " << i;
				break;
			}
			case ParamType::FLOAT: {
				auto f = param_value.get<float>();
				this->setParameter<float>(param_name, f);
				VERBOSE_LABELED_INFO(this->name) << param_name + " setted into " << f;
				break;
			}
			case ParamType::STRING: {
				auto s = param_value.get<std::string>();
				this->setParameter<std::string>(param_name, s);
				VERBOSE_LABELED_INFO(this->name) << param_name + " setted into " + s;
				break;
			}
			case ParamType::TICK: {
				auto t = (Tick)(param_value.get<int>());
				this->setParameter<Tick>(param_name, t);
				VERBOSE_LABELED_INFO(this->name) << param_name + " setted into " << t;
				break;
			}
			case ParamType::USER_DEFINED: {
				this->parseParametersUserDefined(param_name, param_value);
				break;
			}
			default: LABELED_ERROR(this->name) << "Undefined ParamType !"; break;
		}
	}
}

void SimConfig::updateParameter(const std::string& _name, ParameterBase* _param) {
	auto iter = this->parameters.find(_name);
	LABELED_ASSERT_MSG(iter != this->parameters.end(), this->name, "The parameter \'" + _name + "\' does not exist.");
	delete iter->second;
	this->parameters.at(_name) = _param;
}

}  // namespace acalsim
