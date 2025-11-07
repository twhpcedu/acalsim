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
 * @file main.cc
 * @brief Configuration System Demonstration Entry Point
 *
 * @details
 * This file serves as the entry point for demonstrating ACALSim's comprehensive configuration
 * management system. It showcases the integration of SimConfig, Parameter<T>, JSON configuration
 * parsing, and CLI11 command-line argument handling in a unified framework.
 *
 * # Configuration System Overview
 *
 * ACALSim provides a sophisticated multi-layered configuration system that allows simulator
 * parameters to be specified through multiple sources with a well-defined priority hierarchy:
 *
 * **Priority Order (highest to lowest):**
 * 1. Command-line arguments (CLI11)
 * 2. JSON configuration files (nlohmann::json)
 * 3. Programmatic defaults (in SimConfig constructors)
 *
 * # Architecture Components
 *
 * ## SimConfig Class Hierarchy
 * - **SimConfig**: Base class for configuration containers
 *   - Manages Parameter<T> instances for type-safe storage
 *   - Provides JSON parsing capabilities
 *   - Supports user-defined types through template specialization
 *
 * - **TestConfig**: Example configuration implementation
 *   - Demonstrates primitive types (int, float, string)
 *   - Shows enum class integration (TestIntEnum, TestStrEnum)
 *   - Illustrates complex struct handling (TestStruct)
 *
 * ## SimTop Integration
 * - **TestConfigTop**: Top-level simulator class
 *   - Registers configuration objects
 *   - Sets up CLI argument mappings
 *   - Retrieves and validates parameters
 *
 * # Parameter<T> Type System
 *
 * The Parameter<T> template provides type-safe, heterogeneous parameter storage:
 *
 * ```cpp
 * // Parameter registration in SimConfig constructor
 * this->addParameter<int>("test_int", -1, ParamType::INT);
 * this->addParameter<float>("test_float", -1.0, ParamType::FLOAT);
 * this->addParameter<std::string>("test_string", "default", ParamType::STRING);
 * this->addParameter<Tick>("test_tick", 1, ParamType::TICK);
 *
 * // Custom struct parameters
 * this->addParameter<TestStruct>("test_struct", TestStruct(), ParamType::USER_DEFINED);
 *
 * // Enum parameters
 * this->addParameter<TestIntEnum>("test_int_enum", TestIntEnum::INVALID, ParamType::USER_DEFINED);
 * ```
 *
 * # JSON Configuration Format
 *
 * Configuration files use JSON format with type annotations:
 *
 * ```json
 * {
 *   "TestConfig": {
 *     "test_int": 42,
 *     "test_float": 3.14,
 *     "test_string": "Hello from JSON",
 *     "test_tick": 1000,
 *     "test_int_enum": {
 *       "type": "TestIntEnum",
 *       "params": 2
 *     },
 *     "test_struct": {
 *       "type": "TestStruct",
 *       "params": {
 *         "test_struct_int": 100,
 *         "test_struct_float": 2.71,
 *         "test_struct_string": "Struct from JSON",
 *         "test_struct_tick": 500,
 *         "test_struct_int_enum": 1,
 *         "test_struct_str_enum": "2"
 *       }
 *     }
 *   }
 * }
 * ```
 *
 * # CLI11 Integration
 *
 * Command-line arguments are registered with descriptive options and type transformers:
 *
 * ```cpp
 * // Basic parameter mapping
 * this->addCLIOption<int>("--test_int", "Test Int Description",
 *                         "TestConfig", "test_int");
 *
 * // Enum with string-to-value transformer
 * this->addCLIOption<TestIntEnum>("--test_int_enum", "Test Int Enum Option",
 *                                 "TestConfig", "test_int_enum")
 *     ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 *
 * // Nested struct member access
 * this->addCLIOption<int, TestStruct>("--test_struct_int",
 *                                     "Test Struct / test_struct_int",
 *                                     "TestConfig", "test_struct", "test_struct_int");
 * ```
 *
 * # Configuration Priority Example
 *
 * Consider a parameter with multiple sources:
 *
 * 1. **Default** (in TestConfig constructor):
 *    ```cpp
 *    this->addParameter<int>("test_int", -1, ParamType::INT);  // Default: -1
 *    ```
 *
 * 2. **JSON Override** (in config.json):
 *    ```json
 *    { "TestConfig": { "test_int": 42 } }  // Override to: 42
 *    ```
 *
 * 3. **CLI Override** (command-line):
 *    ```bash
 *    ./testConfig --config config.json --test_int 100  // Final value: 100
 *    ```
 *
 * The final value will be 100 (CLI has highest priority).
 *
 * # Parameter Retrieval Patterns
 *
 * ## Direct Parameter Access
 * ```cpp
 * // Get entire parameter value
 * int value = this->getParameter<int>("TestConfig", "test_int");
 * TestStruct s = this->getParameter<TestStruct>("TestConfig", "test_struct");
 * ```
 *
 * ## Struct Member Access
 * ```cpp
 * // Access individual struct members without retrieving entire struct
 * int member_value = this->getParameterMemberData<TestStruct, int>(
 *     "TestConfig", "test_struct", "test_struct_int");
 * ```
 *
 * # User-Defined Type Integration
 *
 * ## 1. Define Custom Type
 * ```cpp
 * struct MyStruct {
 *     int field1;
 *     float field2;
 * };
 * ```
 *
 * ## 2. Add JSON Serialization
 * ```cpp
 * inline void from_json(const json& j, MyStruct& b) {
 *     j.at("field1").get_to(b.field1);
 *     j.at("field2").get_to(b.field2);
 * }
 * ```
 *
 * ## 3. Specialize Parameter Template
 * ```cpp
 * SPECIALIZE_PARAMETER(MyStruct, int, MAKE_MEMBER_PAIR(MyStruct, field1))
 * SPECIALIZE_PARAMETER(MyStruct, float, MAKE_MEMBER_PAIR(MyStruct, field2))
 * ```
 *
 * ## 4. Implement parseParametersUserDefined
 * ```cpp
 * void parseParametersUserDefined(const std::string& _param_name,
 *                                 const json& _param_value) override {
 *     std::string data_type;
 *     _param_value.at("type").get_to(data_type);
 *     if (data_type == "MyStruct") {
 *         auto r = _param_value.at("params").get<MyStruct>();
 *         this->setParameter<MyStruct>(_param_name, r);
 *     }
 * }
 * ```
 *
 * # Enum Integration Patterns
 *
 * ## Integer-Based Enums
 * ```cpp
 * enum class TestIntEnum { INVALID = 0, I_V1, I_V2, I_V3 };
 *
 * // JSON serialization mapping
 * NLOHMANN_JSON_SERIALIZE_ENUM(TestIntEnum, {
 *     {TestIntEnum::INVALID, nullptr},
 *     {TestIntEnum::I_V1, 1},
 *     {TestIntEnum::I_V2, 2},
 *     {TestIntEnum::I_V3, 3}
 * })
 *
 * // CLI transformation maps
 * std::map<std::string, TestIntEnum> TestIntEnumMap = {
 *     {"1", TestIntEnum::I_V1},
 *     {"2", TestIntEnum::I_V2},
 *     {"3", TestIntEnum::I_V3}
 * };
 * ```
 *
 * ## String-Based Enums
 * ```cpp
 * enum class TestStrEnum { INVALID = 0, S_V1, S_V2, S_V3 };
 *
 * NLOHMANN_JSON_SERIALIZE_ENUM(TestStrEnum, {
 *     {TestStrEnum::INVALID, nullptr},
 *     {TestStrEnum::S_V1, "1"},
 *     {TestStrEnum::S_V2, "2"},
 *     {TestStrEnum::S_V3, "3"}
 * })
 * ```
 *
 * # Configuration Validation
 *
 * The system provides several validation mechanisms:
 *
 * 1. **Type Safety**: Template-based Parameter<T> ensures compile-time type checking
 * 2. **CLI Transformers**: CheckedTransformer validates enum values from command-line
 * 3. **JSON Schema**: Type annotations in JSON prevent incorrect type assignments
 * 4. **Runtime Checks**: getParameter<T>() verifies parameter existence and type
 *
 * # Example Usage Workflow
 *
 * ## 1. Create Configuration File (config.json)
 * ```json
 * {
 *   "TestConfig": {
 *     "test_int": 42,
 *     "test_string": "JSON Config"
 *   }
 * }
 * ```
 *
 * ## 2. Run with Default Configuration
 * ```bash
 * ./testConfig
 * # Uses programmatic defaults
 * ```
 *
 * ## 3. Run with JSON Configuration
 * ```bash
 * ./testConfig --config config.json
 * # Defaults overridden by JSON values
 * ```
 *
 * ## 4. Run with CLI Overrides
 * ```bash
 * ./testConfig --config config.json --test_int 100 --test_string "CLI Override"
 * # JSON values overridden by CLI arguments
 * ```
 *
 * # Program Flow
 *
 * The main() function orchestrates the simulation lifecycle:
 *
 * 1. **Instantiation**: Create TestConfigTop instance
 *    - Sets up global 'top' pointer
 *    - Specifies simulator name
 *
 * 2. **Initialization**: Call top->init(argc, argv)
 *    - Registers configurations (registerConfigs)
 *    - Registers CLI arguments (registerCLIArguments)
 *    - Parses configuration file
 *    - Processes command-line arguments
 *    - Applies configuration priority rules
 *    - Registers simulators (registerSimulators)
 *
 * 3. **Execution**: Call top->run()
 *    - Executes simulation logic
 *
 * 4. **Cleanup**: Call top->finish()
 *    - Performs cleanup operations
 *    - Generates reports
 *
 * @see TestConfigTop::registerConfigs() for configuration registration
 * @see TestConfigTop::registerCLIArguments() for CLI argument setup
 * @see TestConfigTop::registerSimulators() for parameter retrieval examples
 * @see TestConfig for configuration implementation
 * @see SimConfig for base configuration class
 * @see Parameter for type-safe parameter storage
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 * @copyright Playlab/ACAL - Apache License 2.0
 */

#include "ACALSim.hh"
using namespace acalsim;

#include "TestConfigTop.hh"

/**
 * @brief Main entry point for configuration system demonstration
 *
 * @details
 * Demonstrates the complete configuration system workflow:
 * 1. Creates a TestConfigTop simulation instance
 * 2. Initializes the configuration system with CLI arguments
 * 3. Executes the simulation
 * 4. Performs cleanup
 *
 * The configuration system supports three-tier parameter specification:
 * - **Defaults**: Hardcoded in TestConfig constructor
 * - **JSON**: Loaded from configuration file
 * - **CLI**: Overridden by command-line arguments
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return int Exit status (0 for success)
 *
 * @par Example Invocations:
 * @code
 * // Use all defaults
 * ./testConfig
 *
 * // Load from JSON configuration
 * ./testConfig --config test_config.json
 *
 * // Override specific parameters via CLI
 * ./testConfig --config test_config.json --test_int 42 --test_string "Override"
 *
 * // Set enum values from command-line
 * ./testConfig --test_int_enum 2
 *
 * // Set nested struct members
 * ./testConfig --test_struct_int 100 --test_struct_float 3.14
 * @endcode
 *
 * @see TestConfigTop for the top-level simulator implementation
 * @see TestConfig for configuration parameter definitions
 * @see SimTop::init() for initialization details
 * @see SimTop::run() for execution details
 * @see SimTop::finish() for cleanup details
 */
int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	// Remember 1) to cast the top-level instance to the SimTop* type and set it to the global variable top
	// 2) Pass your own simulator class type to the STSim class template
	top = std::make_shared<TestConfigTop>("SimTop.TestConfig");
	top->init(argc, argv);
	top->run();
	top->finish();
	return 0;
}
