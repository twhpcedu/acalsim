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
#include "system/DataMovementManager.hh"

using namespace acalsim;

class PESim : public CPPSimBase, public DataMovementManager {
public:
	PESim(const std::string& name, std::shared_ptr<SimTensorManager> _pTensorManager, int _peID, int _testNum = 0)
	    : CPPSimBase(name + " " + std::to_string(_peID)),
	      DataMovementManager("PEID_" + std::to_string(peID) + "_DMM", _pTensorManager),
	      peID(_peID) {
		CLASS_INFO << "Contructing PESim ID :" + std::to_string(peID);
	}
	~PESim() {}
	void prepareReqList();
	void init() override;
	void step() override;
	void cleanup() override;
	void accept(Tick when, SimPacket& pkt) override;

private:
	uint32_t peID;
};
