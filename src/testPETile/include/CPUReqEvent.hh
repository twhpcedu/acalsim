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

#include <string>

#include "ACALSim.hh"
using namespace acalsim;

#include "MemReq.hh"

class CpuTrafficTraceRecord : public acalsim::SimTraceRecord {
public:
	CpuTrafficTraceRecord(acalsim::Tick _tick, MemReqTypeEnum _req_type, int _transaction_id, uint64_t _addr, int _size)
	    : acalsim::SimTraceRecord(),
	      tick(_tick),
	      req_type(_req_type),
	      transaction_id(_transaction_id),
	      addr(_addr),
	      size(_size) {}

	nlohmann::json toJson() const override {
		nlohmann::json j = nlohmann::json::object();

		j["tick"]           = this->tick;
		j["transaction-id"] = this->transaction_id;
		j["addr"]           = this->addr;
		j["size"]           = this->size;

		switch (this->req_type) {
			case MemReqTypeEnum::PCU_MEM_READ: j["req-type"] = "PCU_MEM_READ"; break;
			case MemReqTypeEnum::PCU_MEM_WRITE: j["req-type"] = "PCU_MEM_WRITE"; break;
			case MemReqTypeEnum::TENSOR_MEM_READ: j["req-type"] = "TENSOR_MEM_READ"; break;
			case MemReqTypeEnum::TENSOR_MEM_WRITE: j["req-type"] = "TENSOR_MEM_WRITE"; break;
			default: j["req-type"] = "UNKNOWN"; break;
		}

		return j;
	}

private:
	acalsim::Tick  tick = -1;
	MemReqTypeEnum req_type;
	int            transaction_id = -1;
	uint64_t       addr;
	int            size;
};

class CPUReqEvent : public CallbackEvent<void(int, MemRespPacket*)> {
public:
	CPUReqEvent(int                                      _tid,       // The transaction ID
	            SimModule*                               _callee,    // pointer of callee
	            std::function<void(int, MemRespPacket*)> _callback,  // The callback function for the caller
	            MemReqPacket*                            _memReqPkt  // The packet for the downstream callee to process
	            )
	    : CallbackEvent<void(int, MemRespPacket*)>(_tid, (void*)_callee, _callback), memReqPkt(_memReqPkt) {}
	~CPUReqEvent() {}

	void cpuReqCallback(int tid, MemRespPacket* _memRespPkt) {
		if (callerCallback) {  // callerCallback = CPUTraffic::MemReqHandler()
			CLASS_INFO << "CPUReqEvent::cpuReqCallback()  transaction id: " << this->tid
			           << " at Tick=" << top->getGlobalTick();
			callerCallback(tid, _memRespPkt);
		}
	}

	void process() override;

private:
	MemReqPacket* memReqPkt;
};
