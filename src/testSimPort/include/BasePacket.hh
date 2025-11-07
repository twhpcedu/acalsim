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

namespace test_port {

class BasePacket : public acalsim::SimPacket {
public:
	explicit BasePacket(uint32_t _req_id = UINT32_MAX) : acalsim::SimPacket(), req_id_(_req_id) {}
	~BasePacket() override = default;

	void renew(uint32_t _req_id) {
		this->acalsim::SimPacket::renew();
		this->req_id_ = _req_id;
	}

	uint32_t getReqId() const { return this->req_id_; }

	void visit(acalsim::Tick when, acalsim::SimModule& module) final;

private:
	uint32_t req_id_;
};

class BaseReqPacket : public BasePacket {
public:
	explicit BaseReqPacket(uint32_t _req_id = UINT32_MAX) : BasePacket(_req_id) {}
	~BaseReqPacket() override = default;
	void visit(acalsim::Tick when, acalsim::SimBase& simulator) final;
	void renew(uint32_t _req_id) { this->BasePacket::renew(_req_id); }
};

class BaseRspPacket : public BasePacket {
public:
	explicit BaseRspPacket(uint32_t _req_id = UINT32_MAX) : BasePacket(_req_id) {}
	~BaseRspPacket() override = default;
	void visit(acalsim::Tick when, acalsim::SimBase& simulator) final;
	void renew(uint32_t _req_id) { this->BasePacket::renew(_req_id); }
};

}  // namespace test_port
