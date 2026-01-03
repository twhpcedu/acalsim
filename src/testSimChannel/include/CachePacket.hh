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

enum class CachePktTypeEnum { TEST };

class NocRespPacket;

class CacheRespPacket : public SimPacket {
public:
	CacheRespPacket(CachePktTypeEnum _respType, int* _data, int _tid)
	    : SimPacket(PTYPE::MEMRESP), respType(_respType), data(_data), tid(_tid) {}
	~CacheRespPacket() {}
	int* getData() { return data; }
	void setData(int _data) { *data = _data; }
	int  getTransactionId() { return this->tid; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	CachePktTypeEnum respType;
	int*             data;
	int              tid;
};

class CacheReqPacket : public SimPacket {
public:
	CacheReqPacket(CachePktTypeEnum _reqType, int _addr, int _size, int _tid)
	    : SimPacket(PTYPE::MEMREQ), reqType(_reqType), addr(_addr), size(_size), tid(_tid) {}
	~CacheReqPacket() {}
	int getAddr() { return addr; }
	int getSize() { return size; }
	int getTransactionId() { return this->tid; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	CachePktTypeEnum reqType;
	int              addr;
	int              size;
	int              tid;
};
