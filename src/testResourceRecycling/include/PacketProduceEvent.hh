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
#include "PacketConsumeEvent.hh"

using SimBase   = acalsim::SimBase;
using SimEvent  = acalsim::SimEvent;
using SimPacket = acalsim::SimPacket;
using Tick      = acalsim::Tick;

template <typename T>
class PacketProduceEvent : public SimEvent {
public:
	PacketProduceEvent() : SimEvent() { ; }
	PacketProduceEvent(SimBase* _sim, size_t _size, Tick _latency)
	    : SimEvent(), sim(_sim), size(_size), latency(_latency) {
		;
	}

	void renew(SimBase* _sim, const size_t& _size, const Tick& _latency) {
		this->SimEvent::renew();

		this->sim     = _sim;
		this->size    = _size;
		this->latency = _latency;
	}

	void process() override {
		auto event = acalsim::top->getRecycleContainer()->acquire<PacketConsumeEvent<T>>(&PacketConsumeEvent<T>::renew,
		                                                                                 this->sim, this->size);
		// auto event = new PacketConsumeEvent<T>(this->sim, this->size);
		this->sim->scheduleEvent(/* event */ event,
		                         /* when */ acalsim::top->getGlobalTick() + this->latency);
	}

private:
	SimBase* sim     = nullptr;
	size_t   size    = 0;
	Tick     latency = 5;
};
