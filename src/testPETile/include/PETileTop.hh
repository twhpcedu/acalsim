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

#include "ACALSim.hh"
using namespace acalsim;

#include "PETile.hh"
#include "PETileConfig.hh"

class PETileTop : public STSim<PETile> {
public:
	PETileTop(const std::string _name = "PESTSim", const std::string _configFile = "")
	    : STSim<PETile>(_name, _configFile) {
		this->traceCntr.run(0, &SimTraceContainer::setFilePath, "trace", "src/testPETile/trace");
	}

	void registerConfigs() override {
		/* 1. instantiate "PETileConfig" in constructor of simulator. (Use long name to describe ConfigBase) */
		auto config = new petile_config::PETileConfig("PETile configuration");

		/* 2. add "PETileConfig" into configContainer (Use short name to index ConfigBase) */
		this->addConfig("PETile", config);
	}
};
