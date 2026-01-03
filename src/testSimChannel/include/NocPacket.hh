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

#include <stdint.h>

#include <functional>

#include "ACALSim.hh"
using namespace acalsim;

#include "CachePacket.hh"

enum class NocPktTypeEnum { TEST };

class NocRespPacket : public SimPacket {
public:
	NocRespPacket(NocPktTypeEnum _respType, int* _data, int _tid)
	    : SimPacket(PTYPE::MEMRESP), respType(_respType), data(_data), tid(_tid) {}
	~NocRespPacket() {}
	int* getData() { return data; }
	int  getTransactionId() { return this->tid; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	NocPktTypeEnum respType;
	int*           data;
	int            tid;
};

class NocReqPacket : public SimPacket {
public:
	NocReqPacket(NocPktTypeEnum _reqType, int addr, int size, int _tid)
	    : SimPacket(PTYPE::MEMREQ), reqType(_reqType), addr(addr), size(size), tid(_tid) {}
	~NocReqPacket() {}
	int getAddr() { return addr; }
	int getSize() { return size; }
	int getTransactionId() { return this->tid; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	NocPktTypeEnum reqType;
	int            addr;
	int            size;
	int            tid;
};
