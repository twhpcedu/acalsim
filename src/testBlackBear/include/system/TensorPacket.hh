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

#include <memory>
#include <string>

#include "packet/SimPacket.hh"
#include "workloads/tensor/SimTensor.hh"
#include "workloads/tensor/SimTensorManager.hh"

namespace acalsim {

class SimBase;
class SimModule;

class TensorReqPacket : public SimPacket {
	// request description
	uint32_t srcID;
	uint32_t destID;
	int      type;
	uint64_t addr;  // request address
	uint64_t size;  // request size

	// tensor description
	SimTensor* pTensor;

public:
	TensorReqPacket() {}
	TensorReqPacket(uint32_t _srcID, uint32_t _destID, int _type, uint64_t _addr, uint32_t _size, SimTensor* _pTensor) {
		this->renew(_srcID, _destID, _type, _addr, _size, _pTensor);
	}

	void renew(uint32_t _srcID, uint32_t _destID, int _type, uint64_t _addr, uint32_t _size, SimTensor* _pTensor) {
		this->srcID   = _srcID;
		this->destID  = _destID;
		this->type    = _type;
		this->addr    = _addr;
		this->size    = _size;
		this->pTensor = _pTensor;
	}

	void visit(Tick when, SimModule& module) override {}
	void visit(Tick when, SimBase& simulator) override {}
};

class TensorDataPacket : public SimPacket {
	// data movement description
	uint32_t srcID;
	uint32_t destID;
	int      type;
	uint64_t addr;  // dest address
	uint64_t size;  // transfer size

	// tensor description
	SimTensor* pTensor;

public:
	TensorDataPacket() {}
	TensorDataPacket(uint32_t _srcID, uint32_t _destID, int _type, uint64_t _addr, uint32_t _size,
	                 SimTensor* _pTensor) {
		this->renew(_srcID, _destID, _type, _addr, _size, _pTensor);
	}

	void renew(uint32_t _srcID, uint32_t _destID, int _type, uint64_t _addr, uint32_t _size, SimTensor* _pTensor) {
		this->srcID   = _srcID;
		this->destID  = _destID;
		this->type    = _type;
		this->addr    = _addr;
		this->size    = _size;
		this->pTensor = _pTensor;
	}
	void visit(Tick when, SimModule& module) override {}
	void visit(Tick when, SimBase& simulator) override {}
};

}  // end of namespace acalsim
