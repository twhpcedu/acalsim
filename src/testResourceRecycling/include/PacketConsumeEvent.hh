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

#include <memory>

#include "ACALSim.hh"

using SimBase   = acalsim::SimBase;
using SimEvent  = acalsim::SimEvent;
using SimPacket = acalsim::SimPacket;

template <typename T>
class PacketConsumeEvent : public SimEvent {
public:
	PacketConsumeEvent() : SimEvent(), sim(nullptr) { ; }
	PacketConsumeEvent(SimBase* _sim, size_t _size) : SimEvent(), sim(_sim) { this->initPacketVec(_size); }

	void renew(SimBase* _sim, size_t _size);

	void initPacketVec(size_t _size);

	void process() override;

private:
	SimBase*                        sim;
	std::vector<std::shared_ptr<T>> packetVec;
};
