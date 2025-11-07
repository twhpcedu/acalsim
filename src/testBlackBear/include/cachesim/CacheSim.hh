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

#include "ACALSim.hh"
#include "system/DataMovementManager.hh"

using namespace acalsim;

class CacheSim : public CPPSimBase, public DataMovementManager {
	int cacheID;

public:
	CacheSim(std::string _name, std::shared_ptr<SimTensorManager> _pTensorManager, int _cacheID)
	    : CPPSimBase(_name),
	      DataMovementManager("CACHEID_" + std::to_string(cacheID) + "_DMM", _pTensorManager),
	      cacheID(_cacheID) {
		CLASS_INFO << "Constructing CacheSim ID : " << cacheID;
	};
	~CacheSim() {}
	void init() override;
	void cleanup() override;
	void accept(Tick when, SimPacket& pkt) override;
};
