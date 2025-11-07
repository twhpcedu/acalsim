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

using json = nlohmann::json;

// User-defined parameter struct
struct TopLevelConfig {
	int gridX;
	int gridY;

	TopLevelConfig() : gridX(4), gridY(4) {}
};

inline void from_json(const json& j, TopLevelConfig& s) {
	j.at("gridX").get_to(s.gridX);
	j.at("gridY").get_to(s.gridY);
}

SPECIALIZE_PARAMETER(TopLevelConfig, int, MAKE_MEMBER_PAIR(TopLevelConfig, gridX),
                     MAKE_MEMBER_PAIR(TopLevelConfig, gridY))

class SystemConfig : public SimConfig {
public:
	SystemConfig(std::string _name) : SimConfig(_name) {
		// register all parameters into SystemConfig
		this->addParameter<std::string>("ModelFileName", "trace.pt", ParamType::STRING);
		this->addParameter<int>("testNo", 0, ParamType::INT);

		// user should define data-type first, and then register the parameters.
		this->addParameter<TopLevelConfig>("top", TopLevelConfig(), ParamType::USER_DEFINED);
	}
	~SystemConfig() {}

	// Define how the user-defined parameters are parsed here
	void parseParametersUserDefined(const std::string& _param_name, const json& _param_value) override {
		std::string data_type;

		_param_value.at("type").get_to(data_type);

		if (data_type == "TopLevelConfig") {
			auto s = _param_value.at("params").get<TopLevelConfig>();
			this->setParameter<TopLevelConfig>(_param_name, s);
		} else {
			CLASS_ERROR << "Undefined ParamType in parseParameterUserDefine()!";
		}
	}
};
