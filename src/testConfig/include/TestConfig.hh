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

#pragma once

#include "ACALSim.hh"
using namespace acalsim;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

/****************************************************************************
 * Example #1.1 : User-Defined Parameter : Enum Class(INT)
 ****************************************************************************/
enum class TestIntEnum { INVALID = 0, I_V1, I_V2, I_V3 };
NLOHMANN_JSON_SERIALIZE_ENUM(TestIntEnum, {
                                              {TestIntEnum::INVALID, nullptr},
                                              {TestIntEnum::I_V1, 1},
                                              {TestIntEnum::I_V2, 2},
                                              {TestIntEnum::I_V3, 3},
                                          })

extern std::map<std::string, TestIntEnum> TestIntEnumMap;
extern std::map<TestIntEnum, std::string> TestIntEnumReMap;

/****************************************************************************
 * Example #1.2 : User-Defined Parameter : Enum Class(String)
 ****************************************************************************/
enum class TestStrEnum { INVALID = 0, S_V1, S_V2, S_V3 };
NLOHMANN_JSON_SERIALIZE_ENUM(TestStrEnum, {
                                              {TestStrEnum::INVALID, nullptr},
                                              {TestStrEnum::S_V1, "1"},
                                              {TestStrEnum::S_V2, "2"},
                                              {TestStrEnum::S_V3, "3"},
                                          })

extern std::map<std::string, TestStrEnum> TestStrEnumMap;
extern std::map<TestStrEnum, std::string> TestStrEnumReMap;

/****************************************************************************
 * Place #2 : User-Defined Parameter : Data Structure
 ****************************************************************************/
struct TestStruct {
	int         test_struct_int;
	float       test_struct_float;
	std::string test_struct_string;
	Tick        test_struct_tick;
	TestIntEnum test_struct_int_enum;
	TestStrEnum test_struct_str_enum;

	// default constructor
	TestStruct()
	    : test_struct_int(-1),
	      test_struct_float(-1.0),
	      test_struct_string("This is not TestStruct"),
	      test_struct_tick(-1),
	      test_struct_int_enum(TestIntEnum::INVALID),
	      test_struct_str_enum(TestStrEnum::INVALID) {}
};

// template specialization for TestStruct
SPECIALIZE_PARAMETER(TestStruct, int, MAKE_MEMBER_PAIR(TestStruct, test_struct_int))
SPECIALIZE_PARAMETER(TestStruct, float, MAKE_MEMBER_PAIR(TestStruct, test_struct_float))
SPECIALIZE_PARAMETER(TestStruct, std::string, MAKE_MEMBER_PAIR(TestStruct, test_struct_string))
SPECIALIZE_PARAMETER(TestStruct, Tick, MAKE_MEMBER_PAIR(TestStruct, test_struct_tick))
SPECIALIZE_PARAMETER(TestStruct, TestIntEnum, MAKE_MEMBER_PAIR(TestStruct, test_struct_int_enum))
SPECIALIZE_PARAMETER(TestStruct, TestStrEnum, MAKE_MEMBER_PAIR(TestStruct, test_struct_str_enum))

inline void from_json(const json& j, TestStruct& b) {
	j.at("test_struct_int").get_to(b.test_struct_int);
	j.at("test_struct_float").get_to(b.test_struct_float);
	j.at("test_struct_string").get_to(b.test_struct_string);
	j.at("test_struct_tick").get_to(b.test_struct_tick);
	j.at("test_struct_int_enum").get_to(b.test_struct_int_enum);
	j.at("test_struct_str_enum").get_to(b.test_struct_str_enum);
}

/****************************************************************************
 * Place #2 : User-Defined Parameter : Data Structure
 ****************************************************************************/
class TestConfig : public SimConfig {
public:
	TestConfig(const std::string& _name) : SimConfig(_name) {
		this->addParameter<int>("test_int", -1, ParamType::INT);
		this->addParameter<float>("test_float", -1.0, ParamType::FLOAT);
		this->addParameter<std::string>("test_string", "This is not TestConfig", ParamType::STRING);
		this->addParameter<Tick>("test_tick", 1, ParamType::TICK);
		this->addParameter<TestStruct>("test_struct", TestStruct(), ParamType::USER_DEFINED);
		this->addParameter<TestIntEnum>("test_int_enum", TestIntEnum::INVALID, ParamType::USER_DEFINED);
		this->addParameter<TestStrEnum>("test_str_enum", TestStrEnum::INVALID, ParamType::USER_DEFINED);
		this->setParameterMemberData<TestStruct, int>("test_struct", "test_struct_int", 10);
		this->setParameterMemberData<TestStruct, float>("test_struct", "test_struct_float", 10.0);
		this->setParameterMemberData<TestStruct, std::string>("test_struct", "test_struct_string", "TestStruct");
		this->setParameterMemberData<TestStruct, Tick>("test_struct", "test_struct_tick", 10);
		this->setParameterMemberData<TestStruct, TestIntEnum>("test_struct", "test_struct_int_enum", TestIntEnum::I_V1);
		this->setParameterMemberData<TestStruct, TestStrEnum>("test_struct", "test_struct_str_enum", TestStrEnum::S_V1);
	}

	virtual ~TestConfig() = default;

	void parseParametersUserDefined(const std::string& _param_name, const json& _param_value) override {
		std::string data_type;
		_param_value.at("type").get_to(data_type);

		if (data_type == "TestStruct") {
			auto r = _param_value.at("params").get<TestStruct>();
			this->setParameter<TestStruct>(_param_name, r);
		} else if (data_type == "TestIntEnum") {
			auto r = _param_value.at("params").get<TestIntEnum>();
			this->setParameter<TestIntEnum>(_param_name, r);
		} else if (data_type == "TestStrEnum") {
			auto r = _param_value.at("params").get<TestStrEnum>();
			this->setParameter<TestStrEnum>(_param_name, r);
		} else {
			CLASS_ERROR << "Undefined ParamType in parseParameterUserDefine()!";
		}
	}
};

class TestConfig2 : public SimConfig {
public:
	TestConfig2(const std::string& _name) : SimConfig(_name) {
		this->addParameter<int>("test_object", -1, ParamType::INT);
	}

	virtual ~TestConfig2() = default;
};
