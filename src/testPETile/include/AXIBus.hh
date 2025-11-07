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

class BusRespEvent : public CallbackEvent<void(int, MemRespPacket*)> {
public:
	BusRespEvent(int                                      _tid,        // The transaction ID
	             std::function<void(int, MemRespPacket*)> _callback,   // The callback function for the caller
	             MemRespPacket*                           _memRespPkt  // The packet for the downstream callee to update
	             )
	    : CallbackEvent(_tid, nullptr, _callback), memRespPkt(_memRespPkt) {}

	void process() override {
		CLASS_INFO << "Process BusRespEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();
		callerCallback(this->tid, this->memRespPkt);
	}

private:
	MemRespPacket* memRespPkt;
};

class BusReqEvent : public CallbackEvent<void(int, MemRespPacket*)> {
public:
	BusReqEvent(int                                      _tid,        // The transaction ID to the downstream module
	            SimModule*                               _callee,     // pointer to the callee
	            std::function<void(int, MemRespPacket*)> _callback,   // The callback function for the caller
	            MemReqPacket*                            _memReqPkt,  // The packet for the downstream callee to process
	            SimModule*                               _caller      // pointer to the caller
	            )
	    : CallbackEvent(_tid, _callee, _callback), memReqPkt(_memReqPkt), caller(_caller) {}

	// The callback function associated with this event is CPUReqEvent::CPUReqCallback()
	void busReqCallback(int        _tid, /* transaction ID to the downstream module */
	                    SimModule* module, MemRespPacket* _memRespPkt);

	void process() override;

private:
	MemReqPacket* memReqPkt;
	SimModule*    caller;
};

// AXI Bus  model
class AXIBus : public SimModule {
public:
	// statistics
	struct Stats {
		uint32_t numResp = 0;
	} stats;

	AXIBus(std::string _name) : SimModule(_name), transactionID(0) {}
	~AXIBus() {}

	void init() override { ; }

	Stats* getStats() { return &stats; }

	Tick getRespDelay(int size) {
		return (Tick)(top->getParameter<Tick>("PETile", "bus_resp_delay") + (size + 1) / 32);
	}

	void memReqPktHandler(Tick when, SimPacket* pkt);

private:
	// A unique ID for outbound transactions
	int transactionID;
};
