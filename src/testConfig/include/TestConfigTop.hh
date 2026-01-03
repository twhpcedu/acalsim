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

#include "TestConfig.hh"

class TestConfigTop : public SimTop {
public:
	TestConfigTop(const std::string _name = "PESTSim", const std::string _configFile = "")
	    : SimTop(_configFile, "trace") {}

	void registerConfigs() override;

	void registerCLIArguments() override;

	void registerSimulators() override;

private:
	int   test_int;
	float test_float;

	TestIntEnum test_int_enum;
};
