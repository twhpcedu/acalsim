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
#include "DataStruct.hh"

using namespace acalsim;

class InstPacket : public SimPacket {
public:
	InstPacket() {}
	InstPacket(const instr& _i) : SimPacket(), isTakenBranch(false) { inst = _i; }
	virtual ~InstPacket() {}

	void visit(Tick _when, SimModule& _module) override;
	void visit(Tick _when, SimBase& _simulator) override;

	void renew(const instr& _i) {
		inst          = _i;
		isTakenBranch = false;
	}

	// static data (instruction encoding)
	instr       inst;
	std::string str;
	uint32_t    pc;
	bool        isTakenBranch;
};
