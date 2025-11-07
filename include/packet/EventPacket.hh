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

#include "event/SimEvent.hh"
#include "packet/SimPacket.hh"

namespace acalsim {

class SimBase;
class SimModule;

class EventPacket : public SimPacket {
public:
	EventPacket(SimEvent* _event = nullptr, Tick _when = 0) : SimPacket(PTYPE::EVENT), event(_event), when(_when) {}
	~EventPacket() {}
	SimEvent*    getEvent() const { return event; }
	Tick         getWhen() const { return when; }
	void         renew(SimEvent* _event, Tick _when);
	virtual void visit(Tick when, SimModule& module) override;
	virtual void visit(Tick when, SimBase& simulator) override;

	virtual std::string getName() const override { return std::string("EventPacket-") + this->getIDStr(); }

private:
	SimEvent* event;
	Tick      when;
};

}  // namespace acalsim
