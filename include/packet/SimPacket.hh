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

#include <atomic>
#include <string>

#include "container/RecycleContainer/RecyclableObject.hh"
#include "external/gem5/Event.hh"
#include "utils/TypeDef.hh"

namespace acalsim {

class SimBase;
class SimModule;

class SimPacket : public RecyclableObject, virtual public HashableType {
public:
	enum class PTYPE { MEMREQ, MEMRESP, PEREQ, PERESP, EVENT, DATA, MAX, SYSTEMC };
	SimPacket(PTYPE _pktType = PTYPE::MEMREQ) : id(SimPacket::uniquePktId++), pktType(_pktType) {}

	virtual ~SimPacket() {}

	void renew(const PTYPE& _pktType = PTYPE::MEMREQ);

	system_id_t getID() const { return this->id; }
	std::string getIDStr() const { return std::to_string(this->id); }

	virtual void visit(Tick when, SimModule& module)  = 0;
	virtual void visit(Tick when, SimBase& simulator) = 0;

	virtual std::string getName() const { return std::string("Packet-") + this->getIDStr(); }

protected:
	PTYPE pktType;

private:
	system_id_t id;

	static std::atomic<system_id_t> uniquePktId;
};

}  // namespace acalsim
