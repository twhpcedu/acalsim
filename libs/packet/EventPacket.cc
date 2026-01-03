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

#include "packet/EventPacket.hh"

#include "container/RecycleContainer/RecycleContainer.hh"
#include "sim/SimBase.hh"
#include "sim/SimTop.hh"

namespace acalsim {

void EventPacket::renew(SimEvent* _event, Tick _when) {
	SimPacket::renew(PTYPE::EVENT);
	this->event = _event;
	this->when  = _when;
}

// visit function when a EventPacket is sent to a SimModule object
void EventPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void EventPacket::visit(SimBase& module) is not implemented yet!";
}

// visit function when a EventPacket is sent to a simulator
void EventPacket::visit(Tick when, SimBase& simulator) {
	// test whether the popped pointer is an `EventPacket` object
	// insert the event to the eventQ
	simulator._scheduleEvent(this->event, this->when);
	top->getRecycleContainer()->recycle(this);
}

}  // end of namespace acalsim
