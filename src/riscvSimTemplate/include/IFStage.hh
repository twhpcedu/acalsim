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

#include <string>

#include "ACALSim.hh"
#include "Emulator.hh"
#include "InstPacket.hh"

class IFStage : public acalsim::CPPSimBase {
public:
	IFStage(std::string name) : acalsim::CPPSimBase(name) {}
	~IFStage() {}

	void init() override {}
	void step() override;
	void cleanup() override {}
	void instPacketHandler(Tick when, SimPacket* pkt);

private:
};
