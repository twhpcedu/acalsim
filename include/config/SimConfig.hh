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
 * @file SimConfig.hh
 * @brief Type-safe parameter storage with JSON serialization and struct reflection
 *
 * SimConfig provides a flexible parameter management system with compile-time type checking,
 * JSON parsing support, and struct member access via template specialization.
 *
 * **Parameter System Architecture:**
 * ```
 * SimConfig
 *   ├─ Parameter<int>
 *   ├─ Parameter<double>
 *   ├─ Parameter<std::string>
 *   └─ Parameter<CustomStruct>
 *       ├─ getValue<int>("member1")      // Struct member access
 *       └─ setValue<double>("member2", val)
 * ```
 *
 * **Type Safety Model:**
 * - Template-based Parameter<T> ensures compile-time type checking
 * - Runtime type verification with std::is_same_v
 * - Struct member reflection via SPECIALIZE_PARAMETER macro
 *
 * **Usage Example:**
 * ```cpp
 * // Define custom config struct
 * struct CacheConfig {
 *     size_t size;
 *     int associativity;
 *     int latency;
 * };
 *
 * // Register struct members for reflection (once per struct type)
 * SPECIALIZE_PARAMETER(CacheConfig, size_t,
 *     MAKE_MEMBER_PAIR(CacheConfig, size))
 * SPECIALIZE_PARAMETER(CacheConfig, int,
 *     MAKE_MEMBER_PAIR(CacheConfig, associativity),
 *     MAKE_MEMBER_PAIR(CacheConfig, latency))
 *
 * // Create configuration
 * SimConfig config("cache");
 *
 * // Add simple parameters
 * config.addParameter("enabled", true, ParamType::USER_DEFINED);
 * config.addParameter("cores", 4, ParamType::INT);
 * config.addParameter("frequency", 3.5, ParamType::FLOAT);
 *
 * // Add struct parameter
 * CacheConfig l1{32768, 8, 4};
 * config.addParameter("l1", l1, ParamType::USER_DEFINED);
 *
 * // Access simple parameters
 * int cores = config.getParameter<int>("cores");
 * double freq = config.getParameter<double>("frequency");
 *
 * // Access struct members
 * size_t cacheSize = config.getParameterMemberData<CacheConfig, size_t>(
 *     "l1", "size");
 * int assoc = config.getParameterMemberData<CacheConfig, int>(
 *     "l1", "associativity");
 *
 * // Update parameters
 * config.setParameter("cores", 8);
 * config.setParameterMemberData<CacheConfig, size_t>("l1", "size", 65536);
 *
 * // Parse from JSON
 * nlohmann::json j = {
 *     {"cores", 16},
 *     {"frequency", 4.2},
 *     {"l1", {
 *         {"size", 131072},
 *         {"associativity", 16},
 *         {"latency", 3}
 *     }}
 * };
 * config.parseParameters(j);
 * ```
 *
 * @see SimConfigManager For multi-config management
 * @see Parameter<T> For typed parameter container
 * @see CLIManager For command-line integration
 */

#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "utils/HashableType.hh"
#include "utils/Logging.hh"

// Third-Party Library
#include <nlohmann/json.hpp>

namespace acalsim {

// Forward declaration
class SimConfigManager;

/**
 * @enum ParamType
 * @brief Parameter type classification for JSON parsing and validation
 *
 * Categorizes parameters for appropriate JSON deserialization handlers.
 * - **Primitive types** (INT, FLOAT, STRING, TICK): Automatic JSON parsing
 * - **USER_DEFINED**: Custom types requiring parseParametersUserDefined() override
 */
enum class ParamType {
	INT,          ///< Integer parameters (int, uint32_t, size_t, etc.)
	FLOAT,        ///< Floating-point parameters (float, double)
	STRING,       ///< String parameters (std::string)
	TICK,         ///< Simulation tick type
	USER_DEFINED  ///< Custom structs, enums, or complex types
};

/**
 * @class ParameterBase
 * @brief Abstract base class for type-erased parameter storage
 *
 * ParameterBase enables polymorphic parameter storage in SimConfig's unordered_map.
 * Derived template class Parameter<T> implements actual typed storage.
 *
 * **Design Pattern:**
 * - Type erasure via polymorphism
 * - Stores parameter name and type classification
 * - Virtual destructor for proper cleanup
 *
 * @note Not used directly - use Parameter<T> instead
 * @see Parameter<T>
 */
class ParameterBase {
public:
	/** @brief Construct parameter base with name and type classification */
	ParameterBase(std::string _name, ParamType _type) : name(_name), type(_type) {}

	/** @brief Virtual destructor for derived class cleanup */
	virtual ~ParameterBase() = default;

	/** @brief Get parameter name @return Parameter identifier */
	std::string getName() const { return this->name; }

	/** @brief Get parameter type classification @return ParamType category */
	ParamType getType() const { return this->type; }

private:
	std::string name;  ///< Parameter identifier
	ParamType   type;  ///< Type classification for JSON parsing
};

/**
 * @class Parameter<T>
 * @brief Template class for type-safe parameter storage with struct member access
 *
 * Parameter<T> stores a value of type T with compile-time and runtime type checking.
 * Supports struct member access via template specialization using SPECIALIZE_PARAMETER macro.
 *
 * **Key Features:**
 * - Compile-time type safety via templates
 * - Runtime type verification with std::is_same_v
 * - Struct member reflection (requires SPECIALIZE_PARAMETER registration)
 * - Supports any copyable type T
 *
 * **Usage:**
 * ```cpp
 * // Simple types
 * Parameter<int> coresParam("cores", 4, ParamType::INT);
 * int cores = coresParam.getValue<int>("");
 * coresParam.setValue<int>("", 8);
 *
 * // Struct types (after SPECIALIZE_PARAMETER)
 * struct Config { int x; double y; };
 * Parameter<Config> cfgParam("config", {1, 2.5}, ParamType::USER_DEFINED);
 * int x = cfgParam.getValue<int>("x");      // Member access
 * cfgParam.setValue<double>("y", 3.14);     // Member update
 * ```
 *
 * @tparam T Parameter value type (int, double, custom struct, etc.)
 * @see SPECIALIZE_PARAMETER, SimConfig
 */
template <typename T>
class Parameter : public ParameterBase {
public:
	/** @brief Construct typed parameter */
	Parameter(const std::string& _name, const T& _value, ParamType _type)
	    : ParameterBase(_name, _type), value(_value) {}

	/** @brief Destructor */
	~Parameter() override = default;

	/**
	 * @brief Set parameter value or struct member
	 * @tparam TParam Type of value to set (must match T or be member type after specialization)
	 * @param _member_name Struct member name (empty "" for simple types)
	 * @param _value New value
	 * @throws std::runtime_error if type mismatch
	 */
	template <typename TParam>
	void setValue(const std::string& _member_name, const TParam& _value) {
		if constexpr (std::is_same_v<TParam, T>) {
			this->value = _value;
		} else {
			throw std::runtime_error("Type mismatch! Expected " + std::string(typeid(T).name()) + " but got " +
			                         std::string(typeid(TParam).name()) + ".");
		}
	}

	/**
	 * @brief Get parameter value or struct member
	 * @tparam TParam Type to retrieve (must match T or be member type after specialization)
	 * @param _member_name Struct member name (empty "" for simple types)
	 * @return TParam Parameter value or member value
	 * @throws std::runtime_error if type mismatch
	 */
	template <typename TParam>
	TParam getValue(const std::string& _member_name) const {
		if constexpr (std::is_same_v<T, TParam>) {
			return this->value;
		} else {
			throw std::runtime_error("Type mismatch! Expected " + std::string(typeid(T).name()) + " but got " +
			                         std::string(typeid(TParam).name()) + ".");
		}
	}

private:
	T value;  ///< Stored parameter value
};

/**
 * @def FLATTEN
 * @brief Utility macro for variadic argument expansion in SPECIALIZE_PARAMETER
 */
#define FLATTEN(...) __VA_ARGS__

/**
 * @def SPECIALIZE_PARAMETER
 * @brief Register struct member access for Parameter<T> template specialization
 *
 * Generates template specializations enabling getValue<>/setValue<>() to access
 * struct members by name. Must be called once per (struct type, member type) pair.
 *
 * **Usage:**
 * ```cpp
 * struct CacheConfig {
 *     size_t size;
 *     int associativity;
 *     int latency;
 * };
 *
 * // Register size_t member
 * SPECIALIZE_PARAMETER(CacheConfig, size_t,
 *     MAKE_MEMBER_PAIR(CacheConfig, size))
 *
 * // Register int members
 * SPECIALIZE_PARAMETER(CacheConfig, int,
 *     MAKE_MEMBER_PAIR(CacheConfig, associativity),
 *     MAKE_MEMBER_PAIR(CacheConfig, latency))
 * ```
 *
 * @param ParamterStructType Struct type to specialize
 * @param Type Member type (int, double, size_t, etc.)
 * @param ... Comma-separated MAKE_MEMBER_PAIR invocations
 * @see MAKE_MEMBER_PAIR
 */
#define SPECIALIZE_PARAMETER(ParamterStructType, Type, ...)                                                           \
	template <>                                                                                                       \
	template <>                                                                                                       \
	inline Type Parameter<ParamterStructType>::getValue<Type>(const std::string& member_name) const {                 \
		static const std::unordered_map<std::string, Type ParamterStructType::*> member_map = {FLATTEN(__VA_ARGS__)}; \
		auto                                                                     it = member_map.find(member_name);   \
		if (it != member_map.end()) {                                                                                 \
			return this->value.*(it->second);                                                                         \
		} else {                                                                                                      \
			throw std::runtime_error("Member not found: " + member_name);                                             \
		}                                                                                                             \
	}                                                                                                                 \
	template <>                                                                                                       \
	template <>                                                                                                       \
	inline void Parameter<ParamterStructType>::setValue<Type>(const std::string& member_name, const Type& value) {    \
		static const std::unordered_map<std::string, Type ParamterStructType::*> member_map = {FLATTEN(__VA_ARGS__)}; \
		auto                                                                     it = member_map.find(member_name);   \
		if (it != member_map.end()) {                                                                                 \
			this->value.*(it->second) = value;                                                                        \
		} else {                                                                                                      \
			throw std::runtime_error("Member not found: " + member_name);                                             \
		}                                                                                                             \
	}

/**
 * @def MAKE_MEMBER_PAIR
 * @brief Create member name-to-pointer mapping for struct reflection
 *
 * Helper macro used with SPECIALIZE_PARAMETER to create std::pair entries mapping
 * member names (strings) to member pointers for runtime struct field access.
 *
 * **Usage:**
 * ```cpp
 * struct Config { int x; double y; };
 *
 * // Creates: {"x", &Config::x}, {"y", &Config::y}
 * SPECIALIZE_PARAMETER(Config, int,
 *     MAKE_MEMBER_PAIR(Config, x))
 * SPECIALIZE_PARAMETER(Config, double,
 *     MAKE_MEMBER_PAIR(Config, y))
 * ```
 *
 * @param ParamterStructType Struct type containing the member
 * @param Member Member field name (without quotes)
 * @see SPECIALIZE_PARAMETER
 */
#define MAKE_MEMBER_PAIR(ParamterStructType, Member) \
	{ #Member, &ParamterStructType::Member }

/**
 * @class SimConfig
 * @brief Type-safe parameter container with JSON parsing and struct member reflection
 *
 * SimConfig manages a collection of typed parameters with automatic JSON deserialization
 * and support for accessing nested struct members without retrieving entire structs.
 *
 * **Design Pattern:**
 * ```
 * SimConfig
 *   ├─ std::unordered_map<string, ParameterBase*>
 *   │     ├─ "cores" → Parameter<int>
 *   │     ├─ "frequency" → Parameter<double>
 *   │     └─ "cache_config" → Parameter<CacheConfig>
 *   │                           ├─ getValue<size_t>("size")
 *   │                           └─ setValue<int>("associativity", 16)
 *   └─ Friend of SimConfigManager (for access control)
 * ```
 *
 * **Key Features:**
 * - **Type-safe parameter storage**: Template-based compile-time type checking
 * - **JSON integration**: Automatic parsing of primitive types, extensible for custom types
 * - **Struct member reflection**: Direct access to struct fields without full object retrieval
 * - **Virtual inheritance**: Compatible with diamond inheritance patterns (HashableType)
 * - **Memory management**: Automatic cleanup of Parameter objects in destructor
 *
 * **Parameter Lifecycle:**
 * ```
 * 1. Construction: SimConfig config("module_name");
 * 2. Registration: config.addParameter<int>("cores", 4, ParamType::INT);
 * 3. JSON Loading: config.parseParameters(jsonObject);
 * 4. Access: int cores = config.getParameter<int>("cores");
 * 5. Update: config.setParameter<int>("cores", 8);
 * 6. Destruction: Automatic Parameter* cleanup
 * ```
 *
 * **Common Usage Patterns:**
 *
 * **Pattern 1: Simple Parameters**
 * ```cpp
 * SimConfig config("cpu");
 * config.addParameter("cores", 4, ParamType::INT);
 * config.addParameter("frequency", 3.5, ParamType::FLOAT);
 * config.addParameter("vendor", std::string("Intel"), ParamType::STRING);
 *
 * int cores = config.getParameter<int>("cores");
 * config.setParameter("cores", 8);
 * ```
 *
 * **Pattern 2: Struct Parameters with Member Access**
 * ```cpp
 * struct MemoryConfig {
 *     size_t capacity;
 *     int channels;
 *     int latency;
 * };
 *
 * // Register struct members (do once globally)
 * SPECIALIZE_PARAMETER(MemoryConfig, size_t,
 *     MAKE_MEMBER_PAIR(MemoryConfig, capacity))
 * SPECIALIZE_PARAMETER(MemoryConfig, int,
 *     MAKE_MEMBER_PAIR(MemoryConfig, channels),
 *     MAKE_MEMBER_PAIR(MemoryConfig, latency))
 *
 * // Use in configuration
 * SimConfig config("memory");
 * MemoryConfig memCfg{16384, 4, 100};
 * config.addParameter("dram", memCfg, ParamType::USER_DEFINED);
 *
 * // Access individual members
 * size_t cap = config.getParameterMemberData<MemoryConfig, size_t>(
 *     "dram", "capacity");
 * config.setParameterMemberData<MemoryConfig, int>(
 *     "dram", "latency", 80);
 * ```
 *
 * **Pattern 3: JSON Configuration**
 * ```cpp
 * // JSON file: config.json
 * {
 *   "cores": 16,
 *   "frequency": 4.5,
 *   "cache": {
 *     "l1_size": 32768,
 *     "l2_size": 262144
 *   }
 * }
 *
 * // Parse JSON
 * std::ifstream file("config.json");
 * nlohmann::json j = nlohmann::json::parse(file);
 * config.parseParameters(j);
 * ```
 *
 * @note Virtual inheritance from HashableType enables use in multiple inheritance hierarchies
 * @note Friend relationship with SimConfigManager allows controlled internal access
 * @see SimConfigManager For managing multiple SimConfig objects
 * @see Parameter<T> For typed parameter implementation
 * @see CLIManager For command-line integration
 */
class SimConfig : virtual public HashableType {
	friend class SimConfigManager;

public:
	/**
	 * @brief Construct SimConfig parameter container
	 *
	 * Creates an empty parameter container with the given identifier name.
	 * Parameters can be added later via addParameter().
	 *
	 * @param _name Configuration identifier (typically module/component name)
	 *
	 * **Usage:**
	 * ```cpp
	 * SimConfig cpuConfig("cpu");
	 * SimConfig cacheConfig("cache");
	 * SimConfig memoryConfig("memory");
	 * ```
	 *
	 * @note Name is used for logging and identification in SimConfigManager
	 * @see addParameter(), SimConfigManager::addConfig()
	 */
	SimConfig(const std::string& _name) : name(_name) {}

	/**
	 * @brief Destructor - releases all Parameter objects
	 *
	 * Iterates through all registered parameters and deletes them, preventing
	 * memory leaks. Logs deletion in verbose mode for debugging.
	 *
	 * **Memory Management:**
	 * - Automatically cleans up all Parameter<T>* pointers in parameters map
	 * - Called automatically when SimConfig goes out of scope
	 * - Called by SimConfigManager destructor for managed configs
	 *
	 * @note Logs each parameter deletion via VERBOSE_LABELED_INFO()
	 * @see addParameter(), SimConfigManager::~SimConfigManager()
	 */
	~SimConfig() {
		for (auto& it : parameters) {
			VERBOSE_LABELED_INFO(this->name) << "Deleting Parameter objects : " << it.first;
			delete it.second;
		}
	}

	/**
	 * @brief Get configuration name
	 *
	 * Returns the identifier string provided during construction.
	 *
	 * @return std::string Configuration name
	 *
	 * **Usage:**
	 * ```cpp
	 * SimConfig config("cpu");
	 * std::cout << config.getName();  // Prints: "cpu"
	 * ```
	 */
	std::string getName() const { return this->name; }

protected:
	/**
	 * @brief Parse user-defined parameter types from JSON (virtual hook for extension)
	 *
	 * Virtual method called by parseParameters() for parameters with ParamType::USER_DEFINED.
	 * Override in derived classes to handle custom struct/enum deserialization from JSON.
	 *
	 * @param _paramName Name of the user-defined parameter
	 * @param _paramValue JSON object containing the parameter value
	 *
	 * **Extension Pattern:**
	 * ```cpp
	 * class MyConfig : public SimConfig {
	 * protected:
	 *     void parseParametersUserDefined(const std::string& _paramName,
	 *                                     const nlohmann::json& _paramValue) override {
	 *         std::string dataType;
	 *         _paramValue.at("type").get_to(dataType);
	 *
	 *         if (dataType == "CacheConfig") {
	 *             auto cfg = _paramValue.at("value").get<CacheConfig>();
	 *             this->setParameter<CacheConfig>(_paramName, cfg);
	 *         }
	 *         else if (dataType == "PolicyEnum") {
	 *             auto policy = _paramValue.at("value").get<PolicyEnum>();
	 *             this->setParameter<PolicyEnum>(_paramName, policy);
	 *         }
	 *     }
	 * };
	 * ```
	 *
	 * **Integration with nlohmann/json:**
	 *
	 * **Method 1: Custom Structs with from_json()**
	 * ```cpp
	 * struct CacheConfig {
	 *     size_t size;
	 *     int associativity;
	 * };
	 *
	 * // Define in same namespace as CacheConfig
	 * inline void from_json(const nlohmann::json& j, CacheConfig& cfg) {
	 *     j.at("size").get_to(cfg.size);
	 *     j.at("associativity").get_to(cfg.associativity);
	 * }
	 *
	 * // JSON: {"type": "CacheConfig", "value": {"size": 32768, "associativity": 8}}
	 * ```
	 *
	 * **Method 2: Enums with NLOHMANN_JSON_SERIALIZE_ENUM()**
	 * ```cpp
	 * enum class WritePolicy { WriteBack, WriteThrough };
	 *
	 * NLOHMANN_JSON_SERIALIZE_ENUM(WritePolicy, {
	 *     {WritePolicy::WriteBack, "writeback"},
	 *     {WritePolicy::WriteThrough, "writethrough"}
	 * })
	 *
	 * // JSON: {"type": "WritePolicy", "value": "writeback"}
	 * ```
	 *
	 * **Default Implementation:**
	 * Base class implementation is no-op. Only override if you have USER_DEFINED parameters.
	 *
	 * @note Ensure custom types are default-constructible for JSON deserialization
	 * @note Use `inline` for from_json() to avoid multiple definition errors
	 * @see parseParameters(), ParamType::USER_DEFINED
	 * @see ACALSimConfig::parseParametersUserDefined() For framework example
	 * @see https://github.com/nlohmann/json for nlohmann/json documentation
	 */
	virtual void parseParametersUserDefined(const std::string& _paramName, const nlohmann::json& _paramValue) {
		// Default no-op implementation
		// Override in derived classes to handle custom USER_DEFINED parameter types
	}

	/**
	 * @brief Register a parameter with name, default value, and type classification
	 *
	 * Creates a new Parameter<T> and adds it to the internal parameters map.
	 * Each parameter must have a unique name within the SimConfig.
	 *
	 * @tparam T Parameter value type (int, double, std::string, custom struct, etc.)
	 * @param _name Unique parameter identifier (used for get/set operations)
	 * @param _value Default value for the parameter
	 * @param _type Type classification for JSON parsing (INT, FLOAT, STRING, TICK, USER_DEFINED)
	 *
	 * **Usage:**
	 * ```cpp
	 * SimConfig config("cpu");
	 *
	 * // Primitive types
	 * config.addParameter("cores", 4, ParamType::INT);
	 * config.addParameter("frequency", 3.5, ParamType::FLOAT);
	 * config.addParameter("vendor", std::string("Intel"), ParamType::STRING);
	 *
	 * // Custom struct
	 * struct CacheConfig { size_t size; int assoc; };
	 * CacheConfig l1{32768, 8};
	 * config.addParameter("l1_cache", l1, ParamType::USER_DEFINED);
	 * ```
	 *
	 * @note Parameter names must be unique - duplicate names will cause runtime errors
	 * @note ParamType::USER_DEFINED requires parseParametersUserDefined() override for JSON parsing
	 * @see getParameter(), setParameter(), ParamType
	 */
	template <typename T>
	void addParameter(const std::string& _name, const T& _value, ParamType _type);

	/**
	 * @brief Update parameter value by name
	 *
	 * Modifies the value of an existing parameter. The parameter must have been
	 * previously registered via addParameter().
	 *
	 * @tparam T Parameter value type (must match type used in addParameter)
	 * @param _name Parameter identifier
	 * @param _value New value to set
	 *
	 * **Usage:**
	 * ```cpp
	 * config.addParameter("cores", 4, ParamType::INT);
	 * config.setParameter("cores", 8);  // Update to 8 cores
	 *
	 * config.addParameter("frequency", 3.5, ParamType::FLOAT);
	 * config.setParameter("frequency", 4.2);  // Update frequency
	 * ```
	 *
	 * @throws std::runtime_error if parameter not found or type mismatch
	 * @note Type T must match the type used when parameter was added
	 * @see getParameter(), addParameter()
	 */
	template <typename T>
	void setParameter(const std::string& _name, const T& _value);

	/**
	 * @brief Retrieve parameter value by name
	 *
	 * Returns the current value of a registered parameter with compile-time
	 * and runtime type checking.
	 *
	 * @tparam T Expected parameter type (must match registered type)
	 * @param _name Parameter identifier
	 * @return T Current parameter value
	 *
	 * **Usage:**
	 * ```cpp
	 * config.addParameter("cores", 4, ParamType::INT);
	 * int cores = config.getParameter<int>("cores");  // Returns 4
	 *
	 * config.addParameter("frequency", 3.5, ParamType::FLOAT);
	 * double freq = config.getParameter<double>("frequency");  // Returns 3.5
	 * ```
	 *
	 * @throws std::runtime_error if parameter not found or type mismatch
	 * @note Return type must match the type used in addParameter()
	 * @see setParameter(), getParameterMemberData()
	 */
	template <typename T>
	T getParameter(const std::string& _name) const;

	/**
	 * @brief Update struct member field value
	 *
	 * Modifies a specific member field within a struct-type parameter without
	 * retrieving or setting the entire struct. Requires SPECIALIZE_PARAMETER registration.
	 *
	 * @tparam TStruct Struct type (must match type used in addParameter)
	 * @tparam T Member field type
	 * @param _name Parameter identifier (struct parameter name)
	 * @param _member_name Struct member field name (as string)
	 * @param _value New value for the member field
	 *
	 * **Usage:**
	 * ```cpp
	 * struct CacheConfig { size_t size; int assoc; int latency; };
	 *
	 * // Register struct members (once, at global scope)
	 * SPECIALIZE_PARAMETER(CacheConfig, size_t,
	 *     MAKE_MEMBER_PAIR(CacheConfig, size))
	 * SPECIALIZE_PARAMETER(CacheConfig, int,
	 *     MAKE_MEMBER_PAIR(CacheConfig, assoc),
	 *     MAKE_MEMBER_PAIR(CacheConfig, latency))
	 *
	 * // Add struct parameter
	 * CacheConfig l1{32768, 8, 4};
	 * config.addParameter("l1", l1, ParamType::USER_DEFINED);
	 *
	 * // Update individual members
	 * config.setParameterMemberData<CacheConfig, size_t>("l1", "size", 65536);
	 * config.setParameterMemberData<CacheConfig, int>("l1", "latency", 3);
	 * ```
	 *
	 * @throws std::runtime_error if parameter not found, type mismatch, or member not registered
	 * @note Requires SPECIALIZE_PARAMETER registration for the struct type
	 * @see getParameterMemberData(), SPECIALIZE_PARAMETER, MAKE_MEMBER_PAIR
	 */
	template <typename TStruct, typename T>
	void setParameterMemberData(const std::string& _name, const std::string& _member_name, const T& _value);

	/**
	 * @brief Retrieve struct member field value
	 *
	 * Accesses a specific member field within a struct-type parameter without
	 * retrieving the entire struct. Requires SPECIALIZE_PARAMETER registration.
	 *
	 * @tparam TStruct Struct type (must match type used in addParameter)
	 * @tparam T Member field type
	 * @param _name Parameter identifier (struct parameter name)
	 * @param _member_name Struct member field name (as string)
	 * @return T Current value of the member field
	 *
	 * **Usage:**
	 * ```cpp
	 * struct MemoryConfig { size_t capacity; int channels; int latency; };
	 *
	 * // After SPECIALIZE_PARAMETER registration and addParameter...
	 * size_t cap = config.getParameterMemberData<MemoryConfig, size_t>(
	 *     "dram", "capacity");
	 * int channels = config.getParameterMemberData<MemoryConfig, int>(
	 *     "dram", "channels");
	 * ```
	 *
	 * @throws std::runtime_error if parameter not found, type mismatch, or member not registered
	 * @note More efficient than getParameter<TStruct>() when only one field is needed
	 * @see setParameterMemberData(), SPECIALIZE_PARAMETER
	 */
	template <typename TStruct, typename T>
	T getParameterMemberData(const std::string& _name, const std::string& _member_name) const;

	/**
	 * @brief Replace existing parameter with new Parameter object
	 *
	 * Internal method for updating a parameter's storage. Typically used by
	 * SimConfigManager during configuration merging operations.
	 *
	 * @param _name Parameter identifier
	 * @param _param New ParameterBase pointer (takes ownership)
	 *
	 * **Usage (internal):**
	 * ```cpp
	 * // Usually called by framework, not directly by users
	 * auto* newParam = new Parameter<int>("cores", 16, ParamType::INT);
	 * config.updateParameter("cores", newParam);
	 * ```
	 *
	 * @note Transfers ownership - caller should not delete _param afterward
	 * @note Old parameter pointer is deleted before replacement
	 * @warning Internal method - use setParameter() for normal updates
	 * @see setParameter(), SimConfigManager
	 */
	void updateParameter(const std::string& _name, ParameterBase* _param);

	/**
	 * @brief Parse JSON object and update registered parameters
	 *
	 * Deserializes a JSON object to update parameter values. Automatically handles
	 * primitive types (INT, FLOAT, STRING, TICK) and calls parseParametersUserDefined()
	 * for USER_DEFINED types.
	 *
	 * @param _params JSON object containing parameter key-value pairs
	 *
	 * **JSON Format Examples:**
	 *
	 * **Primitive Types:**
	 * ```json
	 * {
	 *   "cores": 16,
	 *   "frequency": 4.5,
	 *   "vendor": "AMD"
	 * }
	 * ```
	 *
	 * **Nested Struct (USER_DEFINED):**
	 * ```json
	 * {
	 *   "cores": 8,
	 *   "cache": {
	 *     "type": "CacheConfig",
	 *     "value": {
	 *       "size": 65536,
	 *       "associativity": 16
	 *     }
	 *   }
	 * }
	 * ```
	 *
	 * **Usage:**
	 * ```cpp
	 * // From file
	 * std::ifstream file("config.json");
	 * nlohmann::json j = nlohmann::json::parse(file);
	 * config.parseParameters(j);
	 *
	 * // From string
	 * nlohmann::json j = nlohmann::json::parse(R"({"cores": 16})");
	 * config.parseParameters(j);
	 * ```
	 *
	 * @note Only updates parameters that were previously registered via addParameter()
	 * @note USER_DEFINED types require parseParametersUserDefined() override
	 * @see parseParametersUserDefined(), ParamType, SimConfigManager::parseConfigFiles()
	 */
	void parseParameters(const nlohmann::json& _params);

private:
	/**
	 * @brief Retrieve typed Parameter pointer by name (internal helper)
	 *
	 * Internal helper method that looks up a parameter in the parameters map,
	 * performs dynamic_cast to the concrete Parameter<T> type, and validates
	 * the cast succeeded. Used by get/set methods.
	 *
	 * @tparam T Parameter value type
	 * @param _name Parameter identifier
	 * @return Parameter<T>* Pointer to typed Parameter object
	 *
	 * **Internal Usage:**
	 * ```cpp
	 * template <typename T>
	 * T SimConfig::getParameter(const std::string& _name) const {
	 *     auto* param = getParameterPtr<T>(_name);
	 *     return param->getValue<T>("");
	 * }
	 * ```
	 *
	 * @throws std::runtime_error if parameter not found or type mismatch
	 * @note Private helper - not for external use
	 * @see getParameter(), setParameter()
	 */
	template <typename T>
	Parameter<T>* getParameterPtr(const std::string& _name) const;

	/**
	 * @brief Parameter storage map
	 *
	 * Maps parameter names to ParameterBase pointers for polymorphic storage.
	 * Actual objects are Parameter<T> instances dynamically allocated on heap.
	 *
	 * **Ownership:**
	 * - SimConfig owns all Parameter* pointers
	 * - Cleaned up in destructor
	 * - Added via addParameter(), accessed via get/set methods
	 */
	std::unordered_map<std::string, ParameterBase*> parameters;

	/**
	 * @brief Configuration identifier
	 *
	 * Immutable name set during construction, used for logging and
	 * identification in SimConfigManager registry.
	 */
	const std::string name;
};

}  // end of namespace acalsim

#include "config/SimConfig.inl"
