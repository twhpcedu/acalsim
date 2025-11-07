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

#include <string>

#include "ACALSim.hh"
using namespace acalsim;

#include "CPUReqEvent.hh"
#include "MemReq.hh"
#include "Test.hh"

/**
 * @class A Simple PCU model that generates memory requests to bus
 */
class CPUTraffic : public SimModule {
public:
	// statistics
	struct Stats {
		uint32_t numResp = 0;
	} stats;

	CPUTraffic(std::string _name) : SimModule(_name), transactionID(0) {}
	~CPUTraffic() {}

	void init() override { this->injectMemRequests(); }

	void injectMemRequests() {
		SimModule* bus = this->getDownStream("Bus");
		CLASS_ASSERT_MSG(bus, "Cannot find the Bus module!\n");

		// Inject 5 memory requests every 10 cycles starting from cycle 1, 11, .... , to 51
		for (int i = 0; i < acalsim::gtest::NUM_TEST_REQ; i++) {
			CLASS_INFO << "Schedule cpuReqPacket with transaction id: " << transactionID << " at Tick=" << (1 + i * 10);
			MemRespPacket* memRespPkt = new MemRespPacket(MemReqTypeEnum::TENSOR_MEM_READ, 0x1000 * i, 20 * i);
			std::function<void(int, MemRespPacket*)> callback = [this, memRespPkt](int id, MemRespPacket* pkt) {
				this->MemRespHandler(this->transactionID, memRespPkt);
			};

			MemReqPacket* memReqPkt = new MemReqPacket(MemReqTypeEnum::TENSOR_MEM_READ, 0x1000 * i, 20 * i, memRespPkt);
			CPUReqEvent*  cpuReqEvent = new CPUReqEvent(this->transactionID++, bus, callback, memReqPkt);

			scheduleEvent((SimEvent*)cpuReqEvent, 1 + i * 10);
		}
	}

	void MemRespHandler(int            id,  // Transaction ID
	                    MemRespPacket* pkt) {
		CLASS_INFO << "Receive MemRespPacket with transaction id: " << id;
		respReceived = true;
		//[TODO] should recycle the MemRespPacket objects to speed up simulation
		free(pkt);
		stats.numResp++;

		if (top->isGTestMode() && stats.numResp == acalsim::gtest::NUM_TEST_REQ) {
			top->setGTestBitMask(getSimID(), getID() /* bit 1 for first critiria*/);
		}
	}

	void accept(Tick when, SimPacket& pkt) override {
		CLASS_ERROR << "CPUTraffic::accept(Tick when, SimPacket& pkt) is not implemented yet!!\n";
	}

	bool receivedOrNot() { return respReceived; }

private:
	int  transactionID;
	bool respReceived = false;
};
