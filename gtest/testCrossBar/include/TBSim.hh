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

#include <ACALSim.hh>

namespace testcrossbar {

class TBPacket : public acalsim::crossbar::CrossBarPacket {
public:
	TBPacket(size_t _src_idx = 0, size_t _dst_idx = 0) : CrossBarPacket(_src_idx, _dst_idx) {}
	virtual ~TBPacket() {}
	void renew(size_t _src_idx = 0, size_t _dst_idx = 0) { CrossBarPacket::renew(_src_idx, _dst_idx); }
	void visit(acalsim::Tick when, acalsim::SimModule& module) override;
	void visit(acalsim::Tick when, acalsim::SimBase& simulator) override;
};

class TBSim : public acalsim::CPPSimBase {
public:
	explicit TBSim(int idx, const std::string& name)
	    : acalsim::CPPSimBase(name + ":" + std::to_string(idx)), device_idx_(idx) {
		registerSimPort();
	}
	virtual ~TBSim() override = default;

	void init() override {
		this->m_reg = this->getPipeRegister("bus-m");
		LABELED_INFO(name) << "TBSim::init() - is connected to PR:" << this->m_reg->getName();
	}

	void                registerSimPort() { this->s_port = this->addSlavePort("bus-s", 1); }
	size_t              getDeviceIdx() const { return this->device_idx_; }
	acalsim::SlavePort* getSlavePort() { return s_port; }

protected:
	acalsim::SimPipeRegister* m_reg;
	acalsim::SlavePort*       s_port;
	size_t                    device_idx_;
};

class MasterTBSim : public TBSim {
public:
	explicit MasterTBSim(int idx) : TBSim(idx, "MDevice") {}
	virtual ~MasterTBSim() override = default;

	void init() final;
	void step() final;
	void cleanup() final;
	void masterPortRetry(acalsim::MasterPort* port) final;

	void handle(TBPacket* packet);

private:
	/// map w/ key: TestCROSSBarPacket:TransactionID
	std::map<size_t, bool> transaction_id_map;

	static std::atomic<size_t> transaction_id;

	void issueRequest();

	size_t num_slave_device_;

	size_t num_requests      = 0;
	size_t issued_requests   = 0;
	size_t finished_requests = 0;
};

class SlaveTBSim : public TBSim {
public:
	explicit SlaveTBSim(int idx) : TBSim(idx, "SDevice") {}
	virtual ~SlaveTBSim() override = default;

	void init() final;
	void step() final;
	void cleanup() final;
	void masterPortRetry(acalsim::MasterPort* port) final;

	void handle(TBPacket* packet);

private:
	void tryAcceptResponse();

	void issueResponse(size_t master_device_id, size_t _transaction_id);
};

}  // namespace testcrossbar
