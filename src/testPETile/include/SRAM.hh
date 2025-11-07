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
using namespace acalsim;

#include "MemReq.hh"
#include "PETileConfig.hh"

using CacheStruct       = petile_config::CacheStruct;
using ReplacementPolicy = petile_config::ReplacementPolicy;
using BusStruct         = petile_config::BusStruct;

class SRAMRespEvent : public CallbackEvent<void(int, MemRespPacket*)> {
public:
	SRAMRespEvent(int                                      _tid,       // The transaction ID
	              std::function<void(int, MemRespPacket*)> _callback,  // The callback function for the caller
	              MemRespPacket* _memRespPkt                           // The packet for the downstream callee to update
	              )
	    : CallbackEvent(_tid, nullptr, _callback), memRespPkt(_memRespPkt) {}

	void process() override { callerCallback(this->tid, memRespPkt); }

private:
	MemRespPacket* memRespPkt;
};

// SRAM  model
class SRAM : public SimModule {
public:
	SRAM(std::string _name) : SimModule(_name) {}
	~SRAM() {}

	Tick getDelay(MemReqPacket* pkt) {
		return (top->getParameter<Tick>("PETile", "sram_req_delay") + (pkt->getSize() + 1) / 256);
	}
	void memReqPktHandler(Tick when, SimPacket* pkt);
};
