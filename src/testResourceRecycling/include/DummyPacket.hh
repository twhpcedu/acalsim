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

#include <vector>

#include "ACALSim.hh"
#include "ResourceUser.hh"

using SimBase   = acalsim::SimBase;
using SimModule = acalsim::SimModule;
using Tick      = acalsim::Tick;
using SimPacket = acalsim::SimPacket;

class DummyPacket : public SimPacket {
public:
	DummyPacket() : SimPacket() {
		for (size_t i = 0; i < 48; ++i) { this->vec.push_back(i); }
	}

	void visit(Tick when, SimModule& module) override {}

	void visit(Tick when, SimBase& simulator) override {
		if (auto sim = dynamic_cast<ResourceUser*>(&simulator)) { sim->dummyPacketHandler(this); }
	}

private:
	std::vector<int> vec;
};
