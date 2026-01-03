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
#include "LimitedObjectContainer.hh"
#include "OutStandingReqQueue.hh"

namespace test_port {

class BaseRspPacket;  // src/testSimPort/include/BasePacket.hh
class BaseReqPacket;  // src/testSimPort/include/BasePacket.hh

class CPUCore : public acalsim::CPPSimBase {
public:
	explicit CPUCore(const std::string& name, size_t max_outstanding_requests, size_t total_requests,
	                 size_t internal_resp_latency, size_t resp_queue_size);

	~CPUCore() override = default;

	void init() final;

	void step() final;

	void cleanup() final;

	void masterPortRetry(acalsim::MasterPort* port) final;

	void handler(BaseRspPacket* packet);

protected:
	// ReqPath
	void           generateRequest();
	void           pushToReqQueue(BaseReqPacket* packet, acalsim::Tick when);
	void           tryIssueRequest();
	void           issueRequest();
	BaseReqPacket* popFromReqQueue();

	// RspPath
	void           acceptPacketFromBus();
	void           pushToRspQueue(BaseRspPacket* packet, acalsim::Tick when);
	void           tryProcessResponse();
	void           processResponse();
	BaseRspPacket* popFromRspQueue();

private:
	// Hardware Connection
	acalsim::MasterPort* m_port;
	acalsim::SlavePort*  s_port;

	static std::atomic<uint32_t> uniqueReqID;

	// System Config for CPU.
	const size_t        kMaxOutstandingRequests = 5;
	const size_t        kTotalRequests          = 100;
	const acalsim::Tick kInteralLatency         = 1;

	// Pending Flag.
	bool has_resp_event_not_process = false;

	// Traffic Counter
	size_t issued_requests   = 0;
	size_t finished_requests = 0;

	LimitedObjectContainer<BaseReqPacket*> req_out_queue;
	LimitedObjectContainer<BaseRspPacket*> rsp_in_queue;

	OutStandingReqQueue outstanding_req_queue;
};

}  // namespace test_port
