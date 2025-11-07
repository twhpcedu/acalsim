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

#pragma once

#include <string>

#include "ACALSim.hh"
using namespace acalsim;

namespace petile_config {

using json = nlohmann::json;

// user-defined data-type
enum class ReplacementPolicy { LRU, RANDOM, INVALID = -1 };

/* ##########################################
 * ####  Place 1 to be added in order to ####
 * ####    parse user-defined parameter  ####
 * ##########################################
 */
NLOHMANN_JSON_SERIALIZE_ENUM(ReplacementPolicy, {
                                                    {ReplacementPolicy::INVALID, nullptr},
                                                    {ReplacementPolicy::LRU, "lru"},
                                                    {ReplacementPolicy::RANDOM, "random"},
                                                })
// -------------------------------------------

// To print out the data-type of ReplacementPolicy
inline std::string enumToString(enum ReplacementPolicy r) {
	std::string s;
	switch (r) {
		case ReplacementPolicy::INVALID: s = "ReplacementPolicy::INVALID"; break;
		case ReplacementPolicy::LRU: s = "ReplacementPolicy::LRU"; break;
		case ReplacementPolicy::RANDOM: s = "ReplacementPolicy::RANDOM"; break;
		default: ERROR << "Undefined ReplacementPolicy !";
	}

	return s;
}

// user-defined data type
struct CacheStruct {
	int                    associativity;
	int                    mem_size;
	enum ReplacementPolicy replacement_policy;
	std::string            write_policy;

	// default constructor
	CacheStruct() : associativity(0), mem_size(0), replacement_policy(ReplacementPolicy::INVALID), write_policy("") {}
};

/* ##########################################
 * ####  Place 1 to be added in order to ####
 * ####    parse user-defined parameter  ####
 * ##########################################
 */
inline void from_json(const json& j, CacheStruct& c) {
	j.at("associativity").get_to(c.associativity);
	j.at("mem_size").get_to(c.mem_size);
	j.at("replacement_policy").get_to(c.replacement_policy);
	j.at("write_policy").get_to(c.write_policy);
}
// -------------------------------------------

// user-defined data-type
struct BusStruct {
	int         bus_width;
	int         max_outstanding_request;
	std::string architecture;

	// default constructor
	BusStruct() : bus_width(0), max_outstanding_request(0), architecture("") {}
};

/* ##########################################
 * ####  Place 1 to be added in order to ####
 * ####    parse user-defined parameter  ####
 * ##########################################
 */
inline void from_json(const json& j, BusStruct& b) {
	j.at("bus_width").get_to(b.bus_width);
	j.at("max_outstanding_request").get_to(b.max_outstanding_request);
	j.at("architecture").get_to(b.architecture);
}
// -------------------------------------------

// ############################################################################
// ################### minimum requirement to use SimConfig ###################
class PETileConfig : public SimConfig {
public:
	PETileConfig(const std::string& _name) : SimConfig(_name) {
		/* register all parameters into PETileConfig  */
		this->addParameter<int>("mem_width", 256, ParamType::INT);
		this->addParameter<float>("test_for_float", 4.3, ParamType::FLOAT);
		this->addParameter<std::string>("bus_protocol", "AXI4", ParamType::STRING);
		this->addParameter<Tick>("bus_req_delay", 1, ParamType::TICK);
		this->addParameter<Tick>("bus_resp_delay", 1, ParamType::TICK);
		this->addParameter<Tick>("sram_req_delay", 20, ParamType::TICK);
		/* user should define data-type first, and then register the parameters. */
		this->addParameter<CacheStruct>("cache_struct", CacheStruct(), ParamType::USER_DEFINED);
		this->addParameter<BusStruct>("bus_struct", BusStruct(), ParamType::USER_DEFINED);
	}
	~PETileConfig() {}
	// ----------------------------------------------------------------------------

	/* ##########################################
	 * ####  Place 2 to be added in order to ####
	 * ####    parse user-defined parameter  ####
	 * ##########################################
	 */
	void parseParametersUserDefined(const std::string& _param_name, const json& _param_value) override {
		std::string data_type;
		_param_value.at("type").get_to(data_type);

		if (data_type == "CacheStruct") {
			auto c = _param_value.at("value").get<CacheStruct>();
			this->setParameter<CacheStruct>(_param_name, c);
			VERBOSE_CLASS_INFO << "[" + _param_name + ".associativity] " + "setted into " << c.associativity;
			VERBOSE_CLASS_INFO << "[" + _param_name + ".mem_size] " + "setted into " << c.mem_size;
			VERBOSE_CLASS_INFO << "[" + _param_name + ".replacement_policy] " + "setted into " +
			                          enumToString(c.replacement_policy);
			VERBOSE_CLASS_INFO << "[" + _param_name + ".write_policy] " + "setted into " + c.write_policy;
		} else if (data_type == "BusStruct") {
			auto b = _param_value.at("value").get<BusStruct>();
			this->setParameter<BusStruct>(_param_name, b);
			VERBOSE_CLASS_INFO << "[" + _param_name + ".bus_width] " + "setted into " << b.bus_width;
			VERBOSE_CLASS_INFO << "[" + _param_name + ".max_outstanding_request] " + "setted into "
			                   << b.max_outstanding_request;
			VERBOSE_CLASS_INFO << "[" + _param_name + ".architecture] " + "setted into " + b.architecture;
		} else {
			CLASS_ERROR << "Undefined ParamType in parseParameterUserDefine()!";
		}
	}
	// -------------------------------------------
};
}  // namespace petile_config
