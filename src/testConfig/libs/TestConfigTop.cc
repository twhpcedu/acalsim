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
 * @file TestConfigTop.cc
 * @brief Top-Level Configuration Management Implementation
 *
 * @details
 * This file implements the TestConfigTop class, which serves as the top-level orchestrator
 * for the ACALSim configuration system demonstration. It shows how to integrate multiple
 * configuration objects (SimConfig derivatives), register CLI arguments using CLI11, and
 * retrieve configuration parameters using type-safe getParameter<T>() methods.
 *
 * # Class Responsibilities
 *
 * TestConfigTop inherits from SimTop and implements three critical lifecycle methods:
 *
 * 1. **registerConfigs()**: Register SimConfig instances into the configuration container
 * 2. **registerCLIArguments()**: Map CLI options to configuration parameters
 * 3. **registerSimulators()**: Retrieve and validate configuration values
 *
 * # Configuration Registration Pattern
 *
 * Configuration objects must be registered in the configContainer to enable:
 * - JSON parsing and parameter population
 * - CLI argument binding
 * - Parameter retrieval via getParameter<T>()
 *
 * ## Registration Workflow:
 * ```cpp
 * void registerConfigs() override {
 *     // 1. Instantiate configuration objects
 *     auto config = new TestConfig("Descriptive Name");
 *
 *     // 2. Register with short key for parameter access
 *     this->addConfig("TestConfig", config);
 * }
 * ```
 *
 * The registration key ("TestConfig") is used throughout the system:
 * - In JSON files: `{ "TestConfig": { "param": value } }`
 * - In CLI binding: `addCLIOption<T>("--opt", "desc", "TestConfig", "param")`
 * - In retrieval: `getParameter<T>("TestConfig", "param")`
 *
 * # CLI11 Integration Architecture
 *
 * CLI11 provides a robust command-line parsing framework with type safety and validation.
 * The addCLIOption<T>() template method creates bidirectional bindings between CLI arguments
 * and configuration parameters.
 *
 * ## Binding Types:
 *
 * ### 1. Basic Type Binding
 * Maps CLI argument directly to a parameter:
 * ```cpp
 * this->addCLIOption<int>("--test_int", "Integer parameter description",
 *                         "TestConfig", "test_int");
 * ```
 * - Argument: `--test_int 42`
 * - Sets: `TestConfig::test_int = 42`
 *
 * ### 2. Enum Binding with Transformer
 * Converts string input to enum values:
 * ```cpp
 * this->addCLIOption<TestIntEnum>("--test_int_enum", "Enum option description",
 *                                 "TestConfig", "test_int_enum")
 *     ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 * ```
 * - TestIntEnumMap: `{"1" -> TestIntEnum::I_V1, "2" -> TestIntEnum::I_V2, ...}`
 * - Argument: `--test_int_enum 2`
 * - Transformer: Converts "2" -> TestIntEnum::I_V2
 * - Validation: Rejects invalid values (e.g., "--test_int_enum 99")
 *
 * ### 3. Nested Struct Member Binding
 * Accesses individual fields within complex structs:
 * ```cpp
 * this->addCLIOption<int, TestStruct>("--test_struct_int",
 *                                     "Struct member description",
 *                                     "TestConfig", "test_struct", "test_struct_int");
 * ```
 * - Two-template pattern: `<MemberType, StructType>`
 * - Argument: `--test_struct_int 100`
 * - Sets: `TestConfig::test_struct.test_struct_int = 100`
 * - Does NOT require retrieving entire struct
 *
 * ## CLI11 Feature Usage:
 *
 * ### Type Transformers
 * Convert string inputs to strongly-typed values:
 * - **CheckedTransformer**: Validates against allowed values (enums)
 * - **ignore_case**: Case-insensitive string matching
 *
 * ### Automatic Type Validation
 * CLI11 validates types automatically:
 * ```bash
 * ./testConfig --test_int abc     # Error: not an integer
 * ./testConfig --test_float xyz   # Error: not a float
 * ./testConfig --test_int_enum 99 # Error: not in TestIntEnumMap
 * ```
 *
 * # Parameter Retrieval Patterns
 *
 * The registerSimulators() method demonstrates various parameter access patterns:
 *
 * ## 1. Direct Primitive Access
 * ```cpp
 * int value = this->getParameter<int>("TestConfig", "test_int");
 * float fval = this->getParameter<float>("TestConfig", "test_float");
 * std::string sval = this->getParameter<std::string>("TestConfig", "test_string");
 * ```
 *
 * ## 2. Enum Access with Reverse Mapping
 * ```cpp
 * TestIntEnum enum_val = this->getParameter<TestIntEnum>("TestConfig", "test_int_enum");
 * std::string enum_str = TestIntEnumReMap[enum_val];  // "TestIntEnum::I_V1"
 * ```
 *
 * ## 3. Complete Struct Retrieval
 * ```cpp
 * TestStruct s = this->getParameter<TestStruct>("TestConfig", "test_struct");
 * int member = s.test_struct_int;
 * float fmember = s.test_struct_float;
 * ```
 *
 * ## 4. Direct Struct Member Access
 * More efficient when only one field is needed:
 * ```cpp
 * int member = this->getParameterMemberData<TestStruct, int>(
 *     "TestConfig", "test_struct", "test_struct_int");
 * ```
 * - Template args: `<StructType, MemberType>`
 * - Avoids copying entire struct
 * - Type-safe member access
 *
 * # Configuration Priority Demonstration
 *
 * The system applies configuration in this order during SimTop::init():
 *
 * 1. **Default Values** (in TestConfig constructor)
 *    ```cpp
 *    this->addParameter<int>("test_int", -1, ParamType::INT);
 *    ```
 *
 * 2. **JSON Configuration** (loaded from --config file)
 *    ```json
 *    { "TestConfig": { "test_int": 42 } }
 *    ```
 *    Overrides default: `test_int = 42`
 *
 * 3. **CLI Arguments** (parsed by CLI11)
 *    ```bash
 *    --test_int 100
 *    ```
 *    Final value: `test_int = 100`
 *
 * # Type Safety Mechanisms
 *
 * ## Compile-Time Type Checking
 * Template-based parameter system:
 * ```cpp
 * // Correct: type matches parameter definition
 * int val = this->getParameter<int>("TestConfig", "test_int");
 *
 * // Compile error: type mismatch
 * float val = this->getParameter<float>("TestConfig", "test_int");
 * ```
 *
 * ## Runtime Validation
 * - Parameter existence checks
 * - Configuration key validation
 * - Member name verification (for struct access)
 *
 * # Multi-Configuration Support
 *
 * TestConfigTop demonstrates managing multiple independent configuration objects:
 *
 * ```cpp
 * auto config1 = new TestConfig("Config 1");    // Full-featured config
 * auto config2 = new TestConfig2("Config 2");   // Minimal config
 *
 * this->addConfig("TestConfig", config1);
 * this->addConfig("TestConfig2", config2);
 * ```
 *
 * Each configuration:
 * - Has independent parameter namespace
 * - Can have separate JSON sections
 * - Supports distinct CLI arguments
 * - Uses same retrieval interface
 *
 * Access pattern:
 * ```cpp
 * int val1 = this->getParameter<int>("TestConfig", "test_int");
 * int val2 = this->getParameter<int>("TestConfig2", "test_object");
 * ```
 *
 * # Initialization Sequence
 *
 * When SimTop::init(argc, argv) is called:
 *
 * 1. **registerConfigs()** is invoked
 *    - Creates configuration objects
 *    - Registers them in configContainer
 *
 * 2. **registerCLIArguments()** is invoked
 *    - Sets up CLI11 option parsers
 *    - Creates parameter bindings
 *
 * 3. **JSON Configuration Parsing**
 *    - Reads --config file if specified
 *    - Calls SimConfig::parseParameters() for each config
 *    - Updates registered parameters
 *
 * 4. **CLI Argument Processing**
 *    - CLI11 parses command-line arguments
 *    - Overrides JSON values via bindings
 *
 * 5. **registerSimulators()** is invoked
 *    - Retrieves finalized parameter values
 *    - Initializes simulator components
 *
 * # Example Configuration Scenarios
 *
 * ## Scenario 1: All Defaults
 * ```bash
 * ./testConfig
 * ```
 * Output: All values show defaults from TestConfig constructor
 *
 * ## Scenario 2: JSON Configuration
 * ```bash
 * ./testConfig --config example.json
 * ```
 * Output: Values from JSON override defaults
 *
 * ## Scenario 3: Selective CLI Overrides
 * ```bash
 * ./testConfig --config example.json --test_int 999 --test_string "Override"
 * ```
 * Output: CLI values override both JSON and defaults
 *
 * ## Scenario 4: Complex Struct Manipulation
 * ```bash
 * ./testConfig --test_struct_int 100 --test_struct_float 3.14 --test_struct_string "CLI"
 * ```
 * Output: Individual struct members set from CLI
 *
 * # Best Practices
 *
 * ## 1. Configuration Naming
 * - Use descriptive long names in constructor: `new TestConfig("Test configuration 1")`
 * - Use short, consistent keys for registration: `addConfig("TestConfig", config)`
 * - Keep keys consistent across JSON, CLI, and retrieval
 *
 * ## 2. CLI Argument Design
 * - Use clear, self-documenting option names: `--test_int` not `-ti`
 * - Provide meaningful descriptions for `--help` output
 * - Use transformers for enums to support human-readable values
 *
 * ## 3. Parameter Retrieval
 * - Use getParameter<T>() for primitive types and small structs
 * - Use getParameterMemberData<S,T>() for large structs when accessing single fields
 * - Validate critical parameters in registerSimulators()
 *
 * ## 4. Type Consistency
 * - Match template types exactly: int->int, float->float
 * - Use consistent enum types between CLI binding and retrieval
 * - Ensure struct member types match SPECIALIZE_PARAMETER declarations
 *
 * @see TestConfig for configuration parameter definitions
 * @see TestConfig2 for minimal configuration example
 * @see SimTop for base class functionality
 * @see SimConfig for parameter management base class
 * @see main.cc for program entry point and initialization
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 * @copyright Playlab/ACAL - Apache License 2.0
 */

#include "TestConfigTop.hh"

/**
 * @brief Register configuration objects into the configuration container
 *
 * @details
 * This method is called during SimTop::init() to register all SimConfig-derived objects
 * that will participate in the configuration system. Registered configurations can:
 * - Parse parameters from JSON files
 * - Receive values from CLI arguments
 * - Be queried via getParameter<T>()
 *
 * ## Registration Pattern:
 *
 * 1. **Instantiate** configuration objects with descriptive names:
 *    ```cpp
 *    auto config = new TestConfig("Test configuration 1");
 *    ```
 *    The name parameter is used for logging and identification.
 *
 * 2. **Register** configurations with short, consistent keys:
 *    ```cpp
 *    this->addConfig("TestConfig", config);
 *    ```
 *    The key ("TestConfig") becomes the namespace for all parameters in this config.
 *
 * ## Multi-Configuration Support:
 *
 * Multiple configuration objects enable modular parameter organization:
 * - **TestConfig**: Comprehensive configuration with various parameter types
 * - **TestConfig2**: Minimal configuration for demonstration
 *
 * Each configuration maintains independent parameter storage and can have separate
 * JSON sections and CLI arguments.
 *
 * ## Key Selection Guidelines:
 *
 * - Use short, memorable keys: "TestConfig" not "TestConfiguration1"
 * - Keep keys consistent across JSON structure, CLI bindings, and parameter access
 * - Use namespace-style naming for hierarchical configs: "Core", "Cache", "Memory"
 *
 * ## Memory Management:
 *
 * - Configuration objects are created with `new` and ownership transfers to SimTop
 * - SimTop's destructor handles cleanup via configContainer
 * - Do NOT delete configuration objects manually
 *
 * @par Example JSON Structure:
 * @code{.json}
 * {
 *   "TestConfig": {
 *     "test_int": 42,
 *     "test_float": 3.14
 *   },
 *   "TestConfig2": {
 *     "test_object": 100
 *   }
 * }
 * @endcode
 *
 * @note This method must be overridden in SimTop derivatives
 * @note Called automatically by SimTop::init() before JSON parsing
 * @note Configurations must be registered before registerCLIArguments() is called
 *
 * @see TestConfig::TestConfig() for parameter initialization
 * @see TestConfig2::TestConfig2() for minimal configuration example
 * @see SimTop::addConfig() for registration details
 * @see registerCLIArguments() for CLI binding to registered configs
 */
void TestConfigTop::registerConfigs() {
	/* 1. instantiate "TestConfig" in constructor of simulator. (Use long name to describe ConfigBase) */
	auto config  = new TestConfig("Test configuration 1");
	auto config2 = new TestConfig2("Test configuration 2");

	/* 2. add "TestConfig" into configContainer (Use short name to index ConfigBase) */
	this->addConfig("TestConfig", config);
	this->addConfig("TestConfig2", config2);
}

/**
 * @brief Register CLI11 command-line argument bindings to configuration parameters
 *
 * @details
 * This method establishes mappings between CLI11 command-line options and configuration
 * parameters registered in SimConfig objects. The addCLIOption<T>() method creates
 * bidirectional bindings that allow CLI arguments to override JSON and default values.
 *
 * ## CLI Binding Architecture:
 *
 * Each CLI option is bound to a specific parameter via template-based type matching:
 * ```cpp
 * this->addCLIOption<Type>("--option_name", "Description",
 *                          "ConfigKey", "param_name");
 * ```
 *
 * ## Binding Categories:
 *
 * ### 1. Enum Type with Transformer
 * Converts human-readable strings to enum values with validation:
 * ```cpp
 * this->addCLIOption<TestIntEnum>("--test_int_enum", "Description",
 *                                 "TestConfig", "test_int_enum")
 *     ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 * ```
 * - **CheckedTransformer**: Validates input against TestIntEnumMap keys
 * - **ignore_case**: Allows case-insensitive matching ("1", "1", etc.)
 * - **Rejection**: Invalid values (e.g., "99") cause CLI11 to report error
 *
 * Transform map example:
 * ```cpp
 * std::map<std::string, TestIntEnum> TestIntEnumMap = {
 *     {"1", TestIntEnum::I_V1},
 *     {"2", TestIntEnum::I_V2},
 *     {"3", TestIntEnum::I_V3}
 * };
 * ```
 *
 * ### 2. Primitive Type Bindings
 * Direct type-safe mappings for built-in types:
 *
 * #### Integer Parameter:
 * ```cpp
 * this->addCLIOption<int>("--test_int", "Test Int Description",
 *                         "TestConfig", "test_int");
 * ```
 * - CLI usage: `--test_int 42`
 * - Type validation: CLI11 ensures input is valid integer
 * - Sets: `TestConfig::parameters["test_int"] = 42`
 *
 * #### Float Parameter:
 * ```cpp
 * this->addCLIOption<float>("--test_float", "Test FLOAT Description",
 *                           "TestConfig", "test_float");
 * ```
 * - CLI usage: `--test_float 3.14`
 * - Type validation: CLI11 ensures input is valid float
 * - Sets: `TestConfig::parameters["test_float"] = 3.14`
 *
 * #### String Parameter:
 * ```cpp
 * this->addCLIOption<std::string>("--test_string", "Test String Description",
 *                                 "TestConfig", "test_string");
 * ```
 * - CLI usage: `--test_string "Hello World"`
 * - No type conversion needed (strings accepted as-is)
 * - Sets: `TestConfig::parameters["test_string"] = "Hello World"`
 *
 * ### 3. Struct Member Bindings
 * Provides direct access to nested struct fields without full struct retrieval:
 *
 * #### Struct Integer Member:
 * ```cpp
 * this->addCLIOption<int, TestStruct>("--test_struct_int",
 *                                     "Test Struct / test_struct_int Description",
 *                                     "TestConfig", "test_struct", "test_struct_int");
 * ```
 * - Template pattern: `<MemberType, StructType>`
 * - CLI usage: `--test_struct_int 100`
 * - Sets: `TestConfig::parameters["test_struct"].test_struct_int = 100`
 * - Only updates specified member, leaving other fields unchanged
 *
 * #### Struct Float Member:
 * ```cpp
 * this->addCLIOption<float, TestStruct>("--test_struct_float",
 *                                       "Test Struct / test_struct_float Description",
 *                                       "TestConfig", "test_struct", "test_struct_float");
 * ```
 * - CLI usage: `--test_struct_float 2.71`
 * - Independent of other struct members
 *
 * #### Struct String Member:
 * ```cpp
 * this->addCLIOption<std::string, TestStruct>("--test_struct_string",
 *                                             "Test Struct / test_struct_string Description",
 *                                             "TestConfig", "test_struct", "test_struct_string");
 * ```
 * - CLI usage: `--test_struct_string "CustomValue"`
 * - String escaping handled by shell/CLI11
 *
 * #### Struct Tick Member:
 * ```cpp
 * this->addCLIOption<Tick, TestStruct>("--test_struct_tick",
 *                                      "Test Struct / test_struct_tick Description",
 *                                      "TestConfig", "test_struct", "test_struct_tick");
 * ```
 * - Tick is a type alias (typically uint64_t)
 * - CLI usage: `--test_struct_tick 1000`
 *
 * ## Parameter Binding Requirements:
 *
 * For successful binding, parameters must:
 * 1. Be registered in SimConfig constructor via addParameter<T>()
 * 2. Have matching types between CLI binding and parameter registration
 * 3. For struct members, have SPECIALIZE_PARAMETER declarations
 *
 * Example registration in TestConfig:
 * ```cpp
 * TestConfig::TestConfig() {
 *     // Register parameter that CLI can bind to
 *     this->addParameter<int>("test_int", -1, ParamType::INT);
 *
 *     // Register struct parameter
 *     this->addParameter<TestStruct>("test_struct", TestStruct(), ParamType::USER_DEFINED);
 * }
 * ```
 *
 * ## CLI Priority and Override Behavior:
 *
 * CLI arguments have the highest priority in the configuration hierarchy:
 *
 * 1. **Default Value** (TestConfig constructor):
 *    ```cpp
 *    this->addParameter<int>("test_int", -1, ParamType::INT);  // Default: -1
 *    ```
 *
 * 2. **JSON Override**:
 *    ```json
 *    { "TestConfig": { "test_int": 42 } }  // Override to: 42
 *    ```
 *
 * 3. **CLI Override**:
 *    ```bash
 *    --test_int 100  // Final value: 100
 *    ```
 *
 * ## Help Text Generation:
 *
 * CLI11 automatically generates help text from registered options:
 * ```bash
 * ./testConfig --help
 * ```
 *
 * Output includes:
 * - All registered option names (--test_int, --test_float, etc.)
 * - Descriptions provided in second parameter
 * - Type information (inferred from templates)
 * - Required vs. optional status
 *
 * ## Type Safety and Validation:
 *
 * CLI11 provides multiple layers of validation:
 *
 * 1. **Type Checking**:
 *    - `--test_int abc` → Error: not an integer
 *    - `--test_float xyz` → Error: not a float
 *
 * 2. **Transformer Validation**:
 *    - `--test_int_enum 99` → Error: not in TestIntEnumMap
 *    - `--test_int_enum INVALID` → Error: not a valid key
 *
 * 3. **Compile-Time Checking**:
 *    - Template type mismatch causes compilation error
 *    - Ensures CLI binding matches parameter type exactly
 *
 * ## Usage Examples:
 *
 * ### Example 1: Basic Parameter Override
 * ```bash
 * ./testConfig --test_int 999 --test_string "Override"
 * ```
 * Sets test_int=999, test_string="Override", others use defaults/JSON
 *
 * ### Example 2: Enum Value Setting
 * ```bash
 * ./testConfig --test_int_enum 2
 * ```
 * Sets test_int_enum=TestIntEnum::I_V2 (via TestIntEnumMap)
 *
 * ### Example 3: Struct Member Manipulation
 * ```bash
 * ./testConfig --test_struct_int 100 --test_struct_float 3.14 --test_struct_string "Custom"
 * ```
 * Sets individual struct members independently
 *
 * ### Example 4: Combined Configuration
 * ```bash
 * ./testConfig --config config.json --test_int 500 --test_struct_tick 2000
 * ```
 * Loads JSON, then overrides specific parameters from CLI
 *
 * ## Best Practices:
 *
 * 1. **Option Naming**:
 *    - Use `--long-option` format (not single-dash `-o`)
 *    - Match parameter names closely: `test_int` → `--test_int`
 *    - Use descriptive names for struct members: `--test_struct_int` not `--tsi`
 *
 * 2. **Descriptions**:
 *    - Provide clear, concise descriptions for --help output
 *    - For struct members, indicate parent: "Test Struct / member_name"
 *    - Include value ranges or constraints if applicable
 *
 * 3. **Type Consistency**:
 *    - Match template type exactly to parameter registration
 *    - Use same enum types as in TestConfig
 *    - Ensure struct member types match SPECIALIZE_PARAMETER declarations
 *
 * 4. **Transformers**:
 *    - Always use CheckedTransformer for enums
 *    - Provide clear error messages via transformer maps
 *    - Consider case sensitivity (use ignore_case for flexibility)
 *
 * @note This method must be overridden in SimTop derivatives
 * @note Called automatically by SimTop::init() after registerConfigs()
 * @note CLI bindings are processed after JSON parsing, enabling overrides
 *
 * @see TestConfig for parameter registration
 * @see SimTop::addCLIOption() for binding implementation details
 * @see TestIntEnumMap for enum transformer map example
 * @see registerConfigs() for configuration registration
 * @see registerSimulators() for parameter retrieval after binding
 */
void TestConfigTop::registerCLIArguments() {
	this->addCLIOption<TestIntEnum>("--test_int_enum", "Test Int Enum Option Discription", "TestConfig",
	                                "test_int_enum")
	    ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));

	this->addCLIOption<int>("--test_int", "Test Int Discription", "TestConfig", "test_int");
	this->addCLIOption<float>("--test_float", "Test FLOAT Discription", "TestConfig", "test_float");
	this->addCLIOption<std::string>("--test_string", "Test String Discription", "TestConfig", "test_string");
	this->addCLIOption<int, TestStruct>("--test_struct_int", "Test Struct / test_struct_int Discription", "TestConfig",
	                                    "test_struct", "test_struct_int");
	this->addCLIOption<float, TestStruct>("--test_struct_float", "Test Struct / test_struct_float Discription",
	                                      "TestConfig", "test_struct", "test_struct_float");
	this->addCLIOption<std::string, TestStruct>("--test_struct_string", "Test Struct / test_struct_string Discription",
	                                            "TestConfig", "test_struct", "test_struct_string");
	this->addCLIOption<Tick, TestStruct>("--test_struct_tick", "Test Struct / test_struct_tick Discription",
	                                     "TestConfig", "test_struct", "test_struct_tick");
}

/**
 * @brief Retrieve configuration parameters and initialize simulator components
 *
 * @details
 * This method is called after configuration loading is complete (defaults + JSON + CLI)
 * and demonstrates comprehensive parameter retrieval patterns. It shows both direct
 * parameter access via getParameter<T>() and efficient struct member access via
 * getParameterMemberData<S,T>().
 *
 * ## Method Purpose:
 *
 * registerSimulators() serves multiple critical functions:
 * 1. **Parameter Retrieval**: Access finalized configuration values
 * 2. **Validation**: Verify parameters have valid values
 * 3. **Initialization**: Configure simulator components based on parameters
 * 4. **Demonstration**: Show various parameter access patterns
 *
 * ## Parameter Retrieval Patterns:
 *
 * ### 1. Primitive Type Access
 *
 * Direct retrieval of built-in types with template type specification:
 *
 * #### Integer Parameter:
 * ```cpp
 * int value = this->getParameter<int>("TestConfig", "test_int");
 * ```
 * - Template arg: Type of parameter (must match registration)
 * - Arg 1: Configuration key ("TestConfig")
 * - Arg 2: Parameter name ("test_int")
 * - Returns: Value after applying all overrides (default < JSON < CLI)
 *
 * #### Float Parameter:
 * ```cpp
 * float value = this->getParameter<float>("TestConfig", "test_float");
 * ```
 * - Type-safe: Template ensures correct type
 * - Runtime validation: Checks parameter exists
 *
 * #### String Parameter:
 * ```cpp
 * std::string value = this->getParameter<std::string>("TestConfig", "test_string");
 * ```
 * - Returns std::string copy
 * - Efficient for logging and display
 *
 * #### Tick Parameter (Time Values):
 * ```cpp
 * Tick value = this->getParameter<Tick>("TestConfig", "test_tick");
 * ```
 * - Tick is typically uint64_t (simulation time units)
 * - Used for timing, delays, and scheduling
 *
 * ### 2. Enum Parameter Access with Reverse Mapping
 *
 * Enums require forward and reverse mapping for usability:
 *
 * #### Integer-Based Enum:
 * ```cpp
 * TestIntEnum enum_val = this->getParameter<TestIntEnum>("TestConfig", "test_int_enum");
 * std::string enum_str = TestIntEnumReMap[enum_val];  // "TestIntEnum::I_V1"
 * ```
 * - **Forward map** (TestIntEnumMap): String → Enum (for CLI parsing)
 *   ```cpp
 *   std::map<std::string, TestIntEnum> TestIntEnumMap = {
 *       {"1", TestIntEnum::I_V1}, {"2", TestIntEnum::I_V2}, ...
 *   };
 *   ```
 * - **Reverse map** (TestIntEnumReMap): Enum → String (for display)
 *   ```cpp
 *   std::map<TestIntEnum, std::string> TestIntEnumReMap = {
 *       {TestIntEnum::I_V1, "TestIntEnum::I_V1"}, ...
 *   };
 *   ```
 *
 * #### String-Based Enum:
 * ```cpp
 * TestStrEnum enum_val = this->getParameter<TestStrEnum>("TestConfig", "test_str_enum");
 * std::string enum_str = TestStrEnumReMap[enum_val];  // "TestStrEnum::S_V1"
 * ```
 * - Same pattern as integer-based enums
 * - JSON serialization uses strings instead of integers
 *
 * ### 3. Complete Struct Retrieval
 *
 * Retrieve entire struct when multiple fields are needed:
 *
 * ```cpp
 * TestStruct s = this->getParameter<TestStruct>("TestConfig", "test_struct");
 * ```
 *
 * Access pattern after retrieval:
 * ```cpp
 * int int_val = s.test_struct_int;          // Direct member access
 * float float_val = s.test_struct_float;    // All fields available
 * std::string str_val = s.test_struct_string;
 * Tick tick_val = s.test_struct_tick;
 * TestIntEnum enum_val = s.test_struct_int_enum;
 * TestStrEnum str_enum = s.test_struct_str_enum;
 * ```
 *
 * **When to use:**
 * - Need multiple (3+) struct members
 * - Passing struct to another function
 * - Small structs (low copy cost)
 *
 * ### 4. Direct Struct Member Access (Efficient Pattern)
 *
 * Access individual struct members without copying entire struct:
 *
 * #### Integer Member:
 * ```cpp
 * int value = this->getParameterMemberData<TestStruct, int>(
 *     "TestConfig", "test_struct", "test_struct_int");
 * ```
 * - Template args: `<StructType, MemberType>`
 * - Arg 1: Configuration key
 * - Arg 2: Struct parameter name
 * - Arg 3: Member field name
 * - Returns: Copy of member value only (no struct copy)
 *
 * #### Float Member:
 * ```cpp
 * float value = this->getParameterMemberData<TestStruct, float>(
 *     "TestConfig", "test_struct", "test_struct_float");
 * ```
 *
 * #### String Member:
 * ```cpp
 * std::string value = this->getParameterMemberData<TestStruct, std::string>(
 *     "TestConfig", "test_struct", "test_struct_string");
 * ```
 *
 * #### Tick Member:
 * ```cpp
 * Tick value = this->getParameterMemberData<TestStruct, Tick>(
 *     "TestConfig", "test_struct", "test_struct_tick");
 * ```
 *
 * #### Enum Members:
 * ```cpp
 * TestIntEnum int_enum = this->getParameterMemberData<TestStruct, TestIntEnum>(
 *     "TestConfig", "test_struct", "test_struct_int_enum");
 * std::string enum_str = TestIntEnumReMap[int_enum];
 *
 * TestStrEnum str_enum = this->getParameterMemberData<TestStruct, TestStrEnum>(
 *     "TestConfig", "test_struct", "test_struct_str_enum");
 * std::string str = TestStrEnumReMap[str_enum];
 * ```
 *
 * **When to use:**
 * - Need only one or two struct members
 * - Large structs (avoid expensive copies)
 * - Frequent access to single fields
 *
 * **Performance comparison:**
 * ```cpp
 * // Less efficient: copies entire TestStruct
 * TestStruct s = this->getParameter<TestStruct>("TestConfig", "test_struct");
 * int value = s.test_struct_int;
 *
 * // More efficient: copies only int
 * int value = this->getParameterMemberData<TestStruct, int>(
 *     "TestConfig", "test_struct", "test_struct_int");
 * ```
 *
 * ### 5. Multi-Configuration Access
 *
 * Access parameters from different configuration objects:
 *
 * ```cpp
 * // From TestConfig
 * int val1 = this->getParameter<int>("TestConfig", "test_int");
 *
 * // From TestConfig2
 * int val2 = this->getParameter<int>("TestConfig2", "test_object");
 * ```
 *
 * Each configuration maintains independent namespace:
 * - No name conflicts between configurations
 * - Different parameter types allowed with same name
 * - Separate JSON sections for each
 *
 * ## Type Requirements:
 *
 * For getParameterMemberData<S,T>() to work, struct must have:
 *
 * 1. **SPECIALIZE_PARAMETER declarations** (in header):
 *    ```cpp
 *    SPECIALIZE_PARAMETER(TestStruct, int, MAKE_MEMBER_PAIR(TestStruct, test_struct_int))
 *    SPECIALIZE_PARAMETER(TestStruct, float, MAKE_MEMBER_PAIR(TestStruct, test_struct_float))
 *    ```
 *
 * 2. **from_json function** for JSON parsing:
 *    ```cpp
 *    inline void from_json(const json& j, TestStruct& b) {
 *        j.at("test_struct_int").get_to(b.test_struct_int);
 *        j.at("test_struct_float").get_to(b.test_struct_float);
 *        // ... other members
 *    }
 *    ```
 *
 * 3. **Parameter registration** in SimConfig constructor:
 *    ```cpp
 *    this->addParameter<TestStruct>("test_struct", TestStruct(), ParamType::USER_DEFINED);
 *    ```
 *
 * ## Logging and Display:
 *
 * The demonstration uses CLASS_INFO for formatted output:
 *
 * ```cpp
 * CLASS_INFO << "	- test_int : " << this->getParameter<int>("TestConfig", "test_int");
 * ```
 *
 * - **CLASS_INFO**: Logging macro with class context
 * - Shows finalized values after all overrides
 * - Demonstrates parameter access patterns
 * - Useful for debugging configuration issues
 *
 * For enum display, use reverse maps:
 * ```cpp
 * CLASS_INFO << "	- test_int_enum : " +
 *               TestIntEnumReMap[this->getParameter<TestIntEnum>("TestConfig", "test_int_enum")];
 * ```
 *
 * ## Error Handling:
 *
 * Parameter retrieval includes validation:
 * - **Non-existent parameter**: Runtime error with parameter name
 * - **Type mismatch**: Compile error (template type checking)
 * - **Invalid config key**: Runtime error with configuration name
 * - **Invalid member name**: Runtime error with member field name
 *
 * ## Configuration Value Flow:
 *
 * By the time registerSimulators() is called, values have been finalized:
 *
 * 1. **Defaults** (TestConfig constructor) → Base values
 * 2. **JSON** (if --config specified) → Override defaults
 * 3. **CLI** (command-line args) → Override JSON/defaults
 * 4. **getParameter<T>()** → Retrieve final value
 *
 * Example flow for test_int:
 * ```
 * Default: -1 (from addParameter<int>("test_int", -1, ...))
 *     ↓
 * JSON: 42 (from config.json: { "TestConfig": { "test_int": 42 } })
 *     ↓
 * CLI: 100 (from --test_int 100)
 *     ↓
 * getParameter<int>("TestConfig", "test_int") → Returns: 100
 * ```
 *
 * ## Best Practices:
 *
 * 1. **Type Safety**:
 *    - Always match template types exactly to parameter registration
 *    - Use auto when type is obvious: `auto value = this->getParameter<int>(...)`
 *
 * 2. **Efficiency**:
 *    - Use getParameterMemberData<S,T>() for large structs (>64 bytes)
 *    - Cache frequently accessed parameters in member variables
 *
 * 3. **Validation**:
 *    - Check critical parameters for valid ranges
 *    - Validate enum values are not INVALID
 *    - Ensure required parameters are not defaults
 *
 * 4. **Documentation**:
 *    - Log retrieved values for debugging
 *    - Use descriptive variable names
 *    - Comment non-obvious parameter usage
 *
 * @note This method must be overridden in SimTop derivatives
 * @note Called automatically by SimTop::init() after configuration loading
 * @note All parameters reflect final values (default + JSON + CLI)
 *
 * @see TestConfig for parameter definitions and registration
 * @see SimTop::getParameter() for retrieval implementation
 * @see SimTop::getParameterMemberData() for struct member access
 * @see registerConfigs() for configuration object registration
 * @see registerCLIArguments() for CLI binding setup
 */
void TestConfigTop::registerSimulators() {
	CLASS_INFO << "[USER] Command Line Arguments:";
	CLASS_INFO << "	- test_int : " << this->getParameter<int>("TestConfig", "test_int");
	CLASS_INFO << "	- test_float : " << this->getParameter<float>("TestConfig", "test_float");
	CLASS_INFO << "	- test_string : " << this->getParameter<std::string>("TestConfig", "test_string");
	CLASS_INFO << "	- test_tick : " + std::to_string(this->getParameter<Tick>("TestConfig", "test_tick"));
	CLASS_INFO << "	- test_int_enum : " +
	                  TestIntEnumReMap[this->getParameter<TestIntEnum>("TestConfig", "test_int_enum")];
	CLASS_INFO << "	- test_str_enum : " +
	                  TestStrEnumReMap[this->getParameter<TestStrEnum>("TestConfig", "test_str_enum")];

	auto s = this->getParameter<TestStruct>("TestConfig", "test_struct");
	CLASS_INFO << "	- test_struct.test_struct_int accessed by getParameter() : " + std::to_string(s.test_struct_int);
	CLASS_INFO << "	- test_struct.test_struct_float accessed by getParameter() : " +
	                  std::to_string(s.test_struct_float);
	CLASS_INFO << "	- test_struct.test_struct_string accessed by getParameter() : " + s.test_struct_string;
	CLASS_INFO << "	- test_struct.test_struct_tick accessed by getParameter() : " + std::to_string(s.test_struct_tick);
	CLASS_INFO << "	- test_struct.test_struct_int_enum accessed by getParameter() : " +
	                  TestIntEnumReMap[s.test_struct_int_enum];
	CLASS_INFO << "	- test_struct.test_struct_str_enum accessed by getParameter() : " +
	                  TestStrEnumReMap[s.test_struct_str_enum];

	auto test_struct_int =
	    this->getParameterMemberData<TestStruct, int>("TestConfig", "test_struct", "test_struct_int");
	CLASS_INFO << "	- test_struct.test_struct_int accessed by getParameterMemberData() : " +
	                  std::to_string(test_struct_int);
	auto test_struct_float =
	    this->getParameterMemberData<TestStruct, float>("TestConfig", "test_struct", "test_struct_float");
	CLASS_INFO << "	- test_struct.test_struct_float accessed by getParameterMemberData() : " +
	                  std::to_string(test_struct_float);
	auto test_struct_string =
	    this->getParameterMemberData<TestStruct, std::string>("TestConfig", "test_struct", "test_struct_string");
	CLASS_INFO << "	- test_struct.test_struct_string accessed by getParameterMemberData() : " + test_struct_string;
	auto test_struct_tick =
	    this->getParameterMemberData<TestStruct, Tick>("TestConfig", "test_struct", "test_struct_tick");
	CLASS_INFO << "	- test_struct.test_struct_tick accessed by getParameterMemberData() : " +
	                  std::to_string(test_struct_tick);
	auto test_struct_int_enum =
	    this->getParameterMemberData<TestStruct, TestIntEnum>("TestConfig", "test_struct", "test_struct_int_enum");
	CLASS_INFO << "	- test_struct.test_struct_int_enum accessed by getParameterMemberData() : " +
	                  TestIntEnumReMap[test_struct_int_enum];
	auto test_struct_str_enum =
	    this->getParameterMemberData<TestStruct, TestStrEnum>("TestConfig", "test_struct", "test_struct_str_enum");
	CLASS_INFO << "	- test_struct.test_struct_str_enum accessed by getParameterMemberData() : " +
	                  TestStrEnumReMap[test_struct_str_enum];

	CLASS_INFO << "	- test_object : " << this->getParameter<int>("TestConfig2", "test_object");
}
