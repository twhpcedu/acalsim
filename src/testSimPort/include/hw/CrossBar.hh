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
#include "LimitedObjectContainer.hh"

namespace test_port {

class BaseReqPacket;  // src/testSimPort/include/BasePacket.hh
class BaseRspPacket;  // src/testSimPort/include/BasePacket.hh

class CrossBar : public acalsim::CPPSimBase {
public:
	explicit CrossBar(const std::string& name, size_t internal_latency, size_t queue_size);

	~CrossBar() override = default;

	void init() final;

	void step() final;

	void cleanup() final;

	void masterPortRetry(acalsim::MasterPort* port) final;

	// ReqPath
	void handler(BaseReqPacket* packet);

	// RspPath
	void handler(BaseRspPacket* packet);

protected:
	// ReqPath
	void           acceptPacketFromCPU();
	void           pushToReqQueue(BaseReqPacket* packet, acalsim::Tick when);
	void           tryIssueRequest();
	void           issueRequest();
	BaseReqPacket* popFromReqQueue();

	// RspPath
	void           acceptPacketFromMEM();
	void           pushToRspQueue(BaseRspPacket* packet, acalsim::Tick when);
	void           tryIssueResponse();
	void           issueResponse();
	BaseRspPacket* popFromRspQueue();

private:
	// Hardware Connection
	acalsim::MasterPort* m_port_to_cpu;
	acalsim::MasterPort* m_port_to_mem;
	acalsim::SlavePort*  s_port_from_cpu;
	acalsim::SlavePort*  s_port_from_mem;

	// System Config for CPU.
	const acalsim::Tick kInteralLatency = 2;

	// Pending Flag.
	bool has_req_event_not_process  = false;  // Req Path: CPU -> BUS -> MEM
	bool has_resp_event_not_process = false;  // Resp Path: MEM -> BUS -> CPU

	LimitedObjectContainer<BaseReqPacket*> req_queue;
	LimitedObjectContainer<BaseRspPacket*> rsp_queue;
};

}  // namespace test_port
