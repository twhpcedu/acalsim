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

#include <cstdint>
#include <functional>

#include "ACALSim.hh"
using namespace acalsim;

enum class MemReqTypeEnum { PCU_MEM_READ, PCU_MEM_WRITE, TENSOR_MEM_READ, TENSOR_MEM_WRITE };

class MemRespPacket : public SimPacket {
public:
	MemRespPacket(MemReqTypeEnum _reqType, uint64_t _addr, int _size)
	    : SimPacket(PTYPE::MEMRESP), reqType(_reqType), addr(_addr), size(_size) {}
	~MemRespPacket() {}
	int      getSize() { return size; }
	uint64_t getAddr() { return addr; }
	void     visit(Tick when, SimModule& module) override;
	void     visit(Tick when, SimBase& simulator) override;

private:
	MemReqTypeEnum reqType;
	uint64_t       addr;
	int            size;
};

class MemReqPacket : public SimPacket {
public:
	MemReqPacket(MemReqTypeEnum _reqType, uint64_t _addr, int _size, MemRespPacket* _memRespPkt)
	    : SimPacket(PTYPE::MEMREQ), reqType(_reqType), addr(_addr), size(_size), memRespPkt(_memRespPkt) {}
	~MemReqPacket() {}
	int            getSize() const { return size; }
	uint64_t       getAddr() const { return addr; }
	MemRespPacket* getMemRespPkt() const { return memRespPkt; }

	void setCallback(std::function<void(int, MemRespPacket*)> _callback) { callback = _callback; }
	std::function<void(int, MemRespPacket*)> getCallback() const { return callback; }
	void                                     visit(Tick when, SimModule& module) override;
	void                                     visit(Tick when, SimBase& simulator) override;

public:
	MemReqTypeEnum                           reqType;
	uint64_t                                 addr;
	int                                      size;
	std::function<void(int, MemRespPacket*)> callback;    // The callback function for the caller
	MemRespPacket*                           memRespPkt;  // The packet for the downstream callee to update
};
