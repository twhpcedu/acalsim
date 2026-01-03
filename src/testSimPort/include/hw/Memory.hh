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

#include "ACALSim.hh"
#include "LimitedObjectContainer.hh"

namespace test_port {

class BaseReqPacket;  // src/testSimPort/include/BasePacket.hh
class BaseRspPacket;  // src/testSimPort/include/BasePacket.hh

class Memory : public acalsim::CPPSimBase {
public:
	explicit Memory(const std::string& name, size_t internal_resp_latency, size_t queue_size);

	~Memory() override = default;

	void init() final;

	void step() final;

	void cleanup() final;

	void masterPortRetry(acalsim::MasterPort* port) final;

	void handler(BaseReqPacket* packet);

protected:
	// ReqPath
	void           acceptPacketFromBUS();
	void           pushToProcessQueue(BaseReqPacket* packet, acalsim::Tick when);
	void           tryProcessRequest();
	void           processRequest();
	BaseReqPacket* popFromProcessQueue();

	// RspPath
	void issueResponse(BaseRspPacket* rsp_packet);

private:
	// Hardware Connection
	acalsim::MasterPort* m_port;
	acalsim::SlavePort*  s_port;

	// System Config for CPU.
	const acalsim::Tick kInteralLatency = 5;

	// Pending Flag.
	bool has_resp_event_not_process = false;

	LimitedObjectContainer<BaseReqPacket*> process_queue;
};
}  // namespace test_port
