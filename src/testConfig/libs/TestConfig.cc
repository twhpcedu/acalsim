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
 * @file TestConfig.cc
 * @brief Enum Transformation Maps for Configuration System
 *
 * @details
 * This file implements bidirectional mapping tables (forward and reverse maps) that enable
 * human-readable enum value representation in both CLI arguments and JSON configuration files.
 * These maps are essential for CLI11 transformers and display formatting.
 *
 * # Purpose and Architecture
 *
 * Enum transformation maps serve two critical functions:
 *
 * 1. **Forward Maps (String → Enum)**: Used by CLI11 transformers
 *    - Convert command-line string arguments to enum values
 *    - Enable JSON string-to-enum deserialization
 *    - Provide validation (reject invalid strings)
 *
 * 2. **Reverse Maps (Enum → String)**: Used for display and logging
 *    - Convert enum values back to human-readable strings
 *    - Enable clear logging output
 *    - Support debugging and diagnostics
 *
 * # Enum Type Categories
 *
 * ## Integer-Based Enums (TestIntEnum)
 *
 * Integer-based enums use numeric string representations in configuration:
 *
 * ### Enum Definition (in TestConfig.hh):
 * ```cpp
 * enum class TestIntEnum { INVALID = 0, I_V1, I_V2, I_V3 };
 * ```
 *
 * ### JSON Serialization (in TestConfig.hh):
 * ```cpp
 * NLOHMANN_JSON_SERIALIZE_ENUM(TestIntEnum, {
 *     {TestIntEnum::INVALID, nullptr},
 *     {TestIntEnum::I_V1, 1},
 *     {TestIntEnum::I_V2, 2},
 *     {TestIntEnum::I_V3, 3}
 * })
 * ```
 * - Maps enum values to JSON integers
 * - INVALID maps to null/nullptr
 * - Used during JSON parsing
 *
 * ### Forward Map (String → Enum):
 * ```cpp
 * std::map<std::string, TestIntEnum> TestIntEnumMap
 * ```
 * - Converts CLI string arguments to enum values
 * - Example: "1" → TestIntEnum::I_V1, "2" → TestIntEnum::I_V2
 * - Used in CLI11 CheckedTransformer:
 *   ```cpp
 *   ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 *   ```
 * - Validates input (strings not in map are rejected)
 *
 * ### Reverse Map (Enum → String):
 * ```cpp
 * std::map<TestIntEnum, std::string> TestIntEnumReMap
 * ```
 * - Converts enum values to descriptive strings for display
 * - Example: TestIntEnum::I_V1 → "TestIntEnum::I_V1"
 * - Used in logging:
 *   ```cpp
 *   std::string name = TestIntEnumReMap[enum_value];
 *   CLASS_INFO << "Enum: " << name;
 *   ```
 *
 * ## String-Based Enums (TestStrEnum)
 *
 * String-based enums use string literals in JSON (not integers):
 *
 * ### Enum Definition (in TestConfig.hh):
 * ```cpp
 * enum class TestStrEnum { INVALID = 0, S_V1, S_V2, S_V3 };
 * ```
 *
 * ### JSON Serialization (in TestConfig.hh):
 * ```cpp
 * NLOHMANN_JSON_SERIALIZE_ENUM(TestStrEnum, {
 *     {TestStrEnum::INVALID, nullptr},
 *     {TestStrEnum::S_V1, "1"},
 *     {TestStrEnum::S_V2, "2"},
 *     {TestStrEnum::S_V3, "3"}
 * })
 * ```
 * - Maps enum values to JSON strings (not integers)
 * - Differentiates from integer-based enums
 * - Used during JSON parsing
 *
 * ### Forward Map (String → Enum):
 * ```cpp
 * std::map<std::string, TestStrEnum> TestStrEnumMap
 * ```
 * - Same structure as TestIntEnumMap
 * - Example: "1" → TestStrEnum::S_V1, "2" → TestStrEnum::S_V2
 * - Used in CLI11 transformers (if CLI option registered)
 *
 * ### Reverse Map (Enum → String):
 * ```cpp
 * std::map<TestStrEnum, std::string> TestStrEnumReMap
 * ```
 * - Converts enum to display string
 * - Example: TestStrEnum::S_V1 → "TestStrEnum::S_V1"
 * - Used in logging and output
 *
 * # Usage Patterns
 *
 * ## CLI11 Transformer Integration
 *
 * Forward maps enable CLI argument validation and conversion:
 *
 * ### Registration (in TestConfigTop.cc):
 * ```cpp
 * this->addCLIOption<TestIntEnum>("--test_int_enum", "Enum option description",
 *                                 "TestConfig", "test_int_enum")
 *     ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 * ```
 *
 * ### CLI Usage:
 * ```bash
 * ./testConfig --test_int_enum 1    # Sets TestIntEnum::I_V1
 * ./testConfig --test_int_enum 2    # Sets TestIntEnum::I_V2
 * ./testConfig --test_int_enum 99   # Error: not in TestIntEnumMap
 * ```
 *
 * ### Transformer Behavior:
 * 1. User provides string: `--test_int_enum 2`
 * 2. CheckedTransformer looks up "2" in TestIntEnumMap
 * 3. Finds TestIntEnum::I_V2
 * 4. Sets parameter to I_V2
 * 5. Invalid strings (e.g., "99") cause CLI11 error
 *
 * ## Display and Logging
 *
 * Reverse maps convert enums back to readable strings:
 *
 * ### Retrieval and Display (in TestConfigTop.cc):
 * ```cpp
 * TestIntEnum value = this->getParameter<TestIntEnum>("TestConfig", "test_int_enum");
 * std::string display = TestIntEnumReMap[value];
 * CLASS_INFO << "Current enum: " << display;
 * // Output: "Current enum: TestIntEnum::I_V1"
 * ```
 *
 * ### Benefits:
 * - Human-readable output (not numeric values)
 * - Clear debugging information
 * - Consistent string representation
 *
 * ## JSON Configuration
 *
 * Enums in JSON use NLOHMANN_JSON_SERIALIZE_ENUM mappings:
 *
 * ### Integer-Based Enum in JSON:
 * ```json
 * {
 *   "TestConfig": {
 *     "test_int_enum": {
 *       "type": "TestIntEnum",
 *       "params": 2
 *     }
 *   }
 * }
 * ```
 * - Uses numeric value (2 → TestIntEnum::I_V2)
 * - Parsed via NLOHMANN_JSON_SERIALIZE_ENUM
 *
 * ### String-Based Enum in JSON:
 * ```json
 * {
 *   "TestConfig": {
 *     "test_str_enum": {
 *       "type": "TestStrEnum",
 *       "params": "2"
 *     }
 *   }
 * }
 * ```
 * - Uses string value ("2" → TestStrEnum::S_V2)
 * - Parsed via NLOHMANN_JSON_SERIALIZE_ENUM
 *
 * # Map Construction Details
 *
 * ## Forward Map Structure
 *
 * ```cpp
 * std::map<std::string, TestIntEnum> TestIntEnumMap = {
 *     {"1", TestIntEnum::I_V1},      // CLI: --test_int_enum 1
 *     {"2", TestIntEnum::I_V2},      // CLI: --test_int_enum 2
 *     {"3", TestIntEnum::I_V3},      // CLI: --test_int_enum 3
 *     {"InValid", TestIntEnum::INVALID}  // Explicit invalid mapping
 * };
 * ```
 *
 * Key characteristics:
 * - **String keys**: Must be unique
 * - **Enum values**: Can map multiple strings to same enum (if desired)
 * - **Invalid entry**: Allows explicit "InValid" string input
 * - **Case sensitivity**: Controlled by CLI11's ignore_case option
 *
 * ## Reverse Map Structure
 *
 * ```cpp
 * std::map<TestIntEnum, std::string> TestIntEnumReMap = {
 *     {TestIntEnum::I_V1, "TestIntEnum::I_V1"},       // Fully qualified name
 *     {TestIntEnum::I_V2, "TestIntEnum::I_V2"},
 *     {TestIntEnum::I_V3, "TestIntEnum::I_V3"},
 *     {TestIntEnum::INVALID, "TestIntEnum::INVALID"}
 * };
 * ```
 *
 * Key characteristics:
 * - **Enum keys**: Must cover all enum values
 * - **String values**: Descriptive, fully qualified names
 * - **Uniqueness**: One string per enum value
 * - **Readability**: Optimized for logging output
 *
 * # Synchronization Requirements
 *
 * The three representations must be kept synchronized:
 *
 * 1. **Enum Definition** (TestConfig.hh):
 *    ```cpp
 *    enum class TestIntEnum { INVALID = 0, I_V1, I_V2, I_V3 };
 *    ```
 *
 * 2. **JSON Serialization** (TestConfig.hh):
 *    ```cpp
 *    NLOHMANN_JSON_SERIALIZE_ENUM(TestIntEnum, {
 *        {TestIntEnum::INVALID, nullptr},
 *        {TestIntEnum::I_V1, 1}, ...
 *    })
 *    ```
 *
 * 3. **Forward Map** (TestConfig.cc):
 *    ```cpp
 *    std::map<std::string, TestIntEnum> TestIntEnumMap = { ... };
 *    ```
 *
 * 4. **Reverse Map** (TestConfig.cc):
 *    ```cpp
 *    std::map<TestIntEnum, std::string> TestIntEnumReMap = { ... };
 *    ```
 *
 * **Important**: Adding a new enum value requires updating all four locations.
 *
 * # Best Practices
 *
 * ## 1. Naming Conventions
 *
 * ### Forward Maps:
 * - Name: `<EnumName>Map` (e.g., TestIntEnumMap)
 * - Type: `std::map<std::string, EnumType>`
 * - Purpose: String → Enum conversion
 *
 * ### Reverse Maps:
 * - Name: `<EnumName>ReMap` (e.g., TestIntEnumReMap)
 * - Type: `std::map<EnumType, std::string>`
 * - Purpose: Enum → String conversion
 *
 * ## 2. Forward Map Keys
 *
 * Choose clear, intuitive string representations:
 * - **Good**: "1", "2", "3" (simple, numeric)
 * - **Good**: "fast", "medium", "slow" (descriptive)
 * - **Avoid**: "a", "b", "c" (non-descriptive)
 * - **Avoid**: Long, complex strings (hard to type)
 *
 * ## 3. Reverse Map Values
 *
 * Use fully qualified, descriptive names:
 * - **Good**: "TestIntEnum::I_V1" (fully qualified)
 * - **Good**: "MemoryType::DDR4" (clear context)
 * - **Avoid**: "1" (loses type information)
 * - **Avoid**: "V1" (ambiguous without context)
 *
 * ## 4. Invalid/Default Values
 *
 * Always include INVALID entry:
 * ```cpp
 * {TestIntEnum::INVALID, "TestIntEnum::INVALID"}  // In reverse map
 * {"InValid", TestIntEnum::INVALID}               // In forward map
 * ```
 * - Enables explicit invalid value selection
 * - Prevents undefined behavior
 * - Facilitates validation
 *
 * ## 5. Completeness
 *
 * Ensure all enum values are covered:
 * - **Forward map**: Include all valid input strings
 * - **Reverse map**: Include ALL enum values (including INVALID)
 * - Missing entries cause runtime errors
 *
 * # Example Workflow
 *
 * ## Adding a New Enum Value
 *
 * To add `I_V4` to TestIntEnum:
 *
 * 1. **Update enum definition** (TestConfig.hh):
 *    ```cpp
 *    enum class TestIntEnum { INVALID = 0, I_V1, I_V2, I_V3, I_V4 };
 *    ```
 *
 * 2. **Update JSON serialization** (TestConfig.hh):
 *    ```cpp
 *    NLOHMANN_JSON_SERIALIZE_ENUM(TestIntEnum, {
 *        {TestIntEnum::INVALID, nullptr},
 *        {TestIntEnum::I_V1, 1},
 *        {TestIntEnum::I_V2, 2},
 *        {TestIntEnum::I_V3, 3},
 *        {TestIntEnum::I_V4, 4}  // Add this
 *    })
 *    ```
 *
 * 3. **Update forward map** (TestConfig.cc):
 *    ```cpp
 *    std::map<std::string, TestIntEnum> TestIntEnumMap = {
 *        {"1", TestIntEnum::I_V1},
 *        {"2", TestIntEnum::I_V2},
 *        {"3", TestIntEnum::I_V3},
 *        {"4", TestIntEnum::I_V4},  // Add this
 *        {"InValid", TestIntEnum::INVALID}
 *    };
 *    ```
 *
 * 4. **Update reverse map** (TestConfig.cc):
 *    ```cpp
 *    std::map<TestIntEnum, std::string> TestIntEnumReMap = {
 *        {TestIntEnum::I_V1, "TestIntEnum::I_V1"},
 *        {TestIntEnum::I_V2, "TestIntEnum::I_V2"},
 *        {TestIntEnum::I_V3, "TestIntEnum::I_V3"},
 *        {TestIntEnum::I_V4, "TestIntEnum::I_V4"},  // Add this
 *        {TestIntEnum::INVALID, "TestIntEnum::INVALID"}
 *    };
 *    ```
 *
 * ## Creating a New Enum Type
 *
 * To create a new enum type `CachePolicy`:
 *
 * 1. **Define enum** (in header):
 *    ```cpp
 *    enum class CachePolicy { INVALID = 0, LRU, LFU, FIFO };
 *    ```
 *
 * 2. **Add JSON serialization** (in header):
 *    ```cpp
 *    NLOHMANN_JSON_SERIALIZE_ENUM(CachePolicy, {
 *        {CachePolicy::INVALID, nullptr},
 *        {CachePolicy::LRU, "LRU"},
 *        {CachePolicy::LFU, "LFU"},
 *        {CachePolicy::FIFO, "FIFO"}
 *    })
 *    ```
 *
 * 3. **Create forward map** (in .cc file):
 *    ```cpp
 *    std::map<std::string, CachePolicy> CachePolicyMap = {
 *        {"lru", CachePolicy::LRU},
 *        {"lfu", CachePolicy::LFU},
 *        {"fifo", CachePolicy::FIFO}
 *    };
 *    ```
 *
 * 4. **Create reverse map** (in .cc file):
 *    ```cpp
 *    std::map<CachePolicy, std::string> CachePolicyReMap = {
 *        {CachePolicy::LRU, "CachePolicy::LRU"},
 *        {CachePolicy::LFU, "CachePolicy::LFU"},
 *        {CachePolicy::FIFO, "CachePolicy::FIFO"},
 *        {CachePolicy::INVALID, "CachePolicy::INVALID"}
 *    };
 *    ```
 *
 * 5. **Declare extern** (in header):
 *    ```cpp
 *    extern std::map<std::string, CachePolicy> CachePolicyMap;
 *    extern std::map<CachePolicy, std::string> CachePolicyReMap;
 *    ```
 *
 * @see TestIntEnum for integer-based enum definition
 * @see TestStrEnum for string-based enum definition
 * @see TestConfig.hh for enum declarations and JSON serialization
 * @see TestConfigTop::registerCLIArguments() for CLI transformer usage
 * @see TestConfigTop::registerSimulators() for reverse map usage
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 * @copyright Playlab/ACAL - Apache License 2.0
 */

#include "TestConfig.hh"

/**
 * @brief Forward transformation map for TestIntEnum (String → Enum)
 *
 * @details
 * Maps human-readable string representations to TestIntEnum values. Used by CLI11
 * CheckedTransformer to validate and convert command-line arguments.
 *
 * ## Mapping Table:
 * | String    | Enum Value          | Description                |
 * |-----------|---------------------|----------------------------|
 * | "1"       | TestIntEnum::I_V1   | First valid value          |
 * | "2"       | TestIntEnum::I_V2   | Second valid value         |
 * | "3"       | TestIntEnum::I_V3   | Third valid value          |
 * | "InValid" | TestIntEnum::INVALID| Explicit invalid selection |
 *
 * ## CLI Usage Example:
 * ```bash
 * ./testConfig --test_int_enum 1  # Sets TestIntEnum::I_V1
 * ./testConfig --test_int_enum 2  # Sets TestIntEnum::I_V2
 * ```
 *
 * ## Transformer Usage:
 * ```cpp
 * this->addCLIOption<TestIntEnum>("--test_int_enum", "Description",
 *                                 "TestConfig", "test_int_enum")
 *     ->transform(CLI::CheckedTransformer(TestIntEnumMap, CLI::ignore_case));
 * ```
 *
 * @note Keys must be unique; duplicate strings will cause undefined behavior
 * @note Missing valid strings will make those enum values inaccessible via CLI
 *
 * @see TestIntEnum for enum definition
 * @see TestIntEnumReMap for reverse mapping (Enum → String)
 * @see TestConfigTop::registerCLIArguments() for usage in CLI binding
 */
std::map<std::string, TestIntEnum> TestIntEnumMap = {
    {"1", TestIntEnum::I_V1}, {"2", TestIntEnum::I_V2}, {"3", TestIntEnum::I_V3}, {"InValid", TestIntEnum::INVALID}};

/**
 * @brief Reverse transformation map for TestIntEnum (Enum → String)
 *
 * @details
 * Maps TestIntEnum values to fully qualified string representations for display and logging.
 * Provides human-readable output when retrieving and displaying enum parameter values.
 *
 * ## Mapping Table:
 * | Enum Value          | String                   | Usage                    |
 * |---------------------|--------------------------|--------------------------|
 * | TestIntEnum::I_V1   | "TestIntEnum::I_V1"      | Logging, display output  |
 * | TestIntEnum::I_V2   | "TestIntEnum::I_V2"      | Logging, display output  |
 * | TestIntEnum::I_V3   | "TestIntEnum::I_V3"      | Logging, display output  |
 * | TestIntEnum::INVALID| "TestIntEnum::INVALID"   | Error indication         |
 *
 * ## Display Usage Example:
 * ```cpp
 * TestIntEnum value = this->getParameter<TestIntEnum>("TestConfig", "test_int_enum");
 * std::string display = TestIntEnumReMap[value];
 * CLASS_INFO << "Enum value: " << display;
 * // Output: "Enum value: TestIntEnum::I_V1"
 * ```
 *
 * @note All enum values must be present; missing values cause runtime lookup errors
 * @note Use fully qualified names (EnumType::Value) for clarity in logs
 *
 * @see TestIntEnum for enum definition
 * @see TestIntEnumMap for forward mapping (String → Enum)
 * @see TestConfigTop::registerSimulators() for usage in parameter display
 */
std::map<TestIntEnum, std::string> TestIntEnumReMap = {{TestIntEnum::I_V1, "TestIntEnum::I_V1"},
                                                       {TestIntEnum::I_V2, "TestIntEnum::I_V2"},
                                                       {TestIntEnum::I_V3, "TestIntEnum::I_V3"},
                                                       {TestIntEnum::INVALID, "TestIntEnum::INVALID"}};

/**
 * @brief Forward transformation map for TestStrEnum (String → Enum)
 *
 * @details
 * Maps string representations to TestStrEnum values. Similar to TestIntEnumMap but for
 * string-based enums (enums that serialize to JSON strings instead of integers).
 *
 * ## Mapping Table:
 * | String    | Enum Value          | Description                |
 * |-----------|---------------------|----------------------------|
 * | "1"       | TestStrEnum::S_V1   | First valid value          |
 * | "2"       | TestStrEnum::S_V2   | Second valid value         |
 * | "3"       | TestStrEnum::S_V3   | Third valid value          |
 * | "InValid" | TestStrEnum::INVALID| Explicit invalid selection |
 *
 * ## JSON Representation:
 * TestStrEnum uses string values in JSON (not integers):
 * ```json
 * {
 *   "test_str_enum": {
 *     "type": "TestStrEnum",
 *     "params": "2"
 *   }
 * }
 * ```
 *
 * @note String-based enums differ from integer-based in JSON serialization only
 * @note CLI usage is identical to integer-based enums (same string keys)
 *
 * @see TestStrEnum for enum definition
 * @see TestStrEnumReMap for reverse mapping (Enum → String)
 * @see NLOHMANN_JSON_SERIALIZE_ENUM in TestConfig.hh for JSON mapping
 */
std::map<std::string, TestStrEnum> TestStrEnumMap = {
    {"1", TestStrEnum::S_V1}, {"2", TestStrEnum::S_V2}, {"3", TestStrEnum::S_V3}, {"InValid", TestStrEnum::INVALID}};

/**
 * @brief Reverse transformation map for TestStrEnum (Enum → String)
 *
 * @details
 * Maps TestStrEnum values to fully qualified string representations for display purposes.
 * Provides consistent, readable output for string-based enum values.
 *
 * ## Mapping Table:
 * | Enum Value          | String                   | Usage                    |
 * |---------------------|--------------------------|--------------------------|
 * | TestStrEnum::S_V1   | "TestStrEnum::S_V1"      | Logging, display output  |
 * | TestStrEnum::S_V2   | "TestStrEnum::S_V2"      | Logging, display output  |
 * | TestStrEnum::S_V3   | "TestStrEnum::S_V3"      | Logging, display output  |
 * | TestStrEnum::INVALID| "TestStrEnum::INVALID"   | Error indication         |
 *
 * ## Display Usage Example:
 * ```cpp
 * TestStrEnum value = this->getParameter<TestStrEnum>("TestConfig", "test_str_enum");
 * std::string display = TestStrEnumReMap[value];
 * CLASS_INFO << "String enum: " << display;
 * // Output: "String enum: TestStrEnum::S_V1"
 * ```
 *
 * @note Provides same functionality as TestIntEnumReMap but for string-based enums
 * @note All enum values must be included for complete coverage
 *
 * @see TestStrEnum for enum definition
 * @see TestStrEnumMap for forward mapping (String → Enum)
 * @see TestConfigTop::registerSimulators() for usage in parameter display
 */
std::map<TestStrEnum, std::string> TestStrEnumReMap = {{TestStrEnum::S_V1, "TestStrEnum::S_V1"},
                                                       {TestStrEnum::S_V2, "TestStrEnum::S_V2"},
                                                       {TestStrEnum::S_V3, "TestStrEnum::S_V3"},
                                                       {TestStrEnum::INVALID, "TestStrEnum::INVALID"}};
