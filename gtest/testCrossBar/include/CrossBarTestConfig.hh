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

namespace testcrossbar {

class CrossBarTestConfig : public acalsim::SimConfig {
public:
	CrossBarTestConfig(const std::string& _name) : acalsim::SimConfig(_name) {
		this->addParameter<int>("n_master", 10, acalsim::ParamType::INT);
		this->addParameter<int>("n_slave", 5, acalsim::ParamType::INT);
		this->addParameter<int>("n_requests", 10, acalsim::ParamType::INT);
	}

	~CrossBarTestConfig() override = default;

	// Parse User Defined Parameter (ParamType::USER_DEFINED)
	// void parseParametersUserDefined(const std::string& _param_name, const json& _param_value) override {}
};
}  // namespace testcrossbar
