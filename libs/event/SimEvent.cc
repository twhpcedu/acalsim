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

#include "event/SimEvent.hh"

#include "container/RecycleContainer/RecycleContainer.hh"
#include "sim/SimTop.hh"

namespace acalsim {

std::atomic<system_id_t> SimEvent::uniqueEventId = 0;

SimEvent::SimEvent(std::string _name) : name(_name), id(SimEvent::uniqueEventId++) {
	this->setFlags(EventBase::Managed);
}

void SimEvent::renew(std::string _name) {
	name     = _name;
	this->id = SimEvent::uniqueEventId++;
}

void SimEvent::releaseImpl() {
	if (!scheduled()) { top->getRecycleContainer()->recycle(this); }
}

}  // namespace acalsim
