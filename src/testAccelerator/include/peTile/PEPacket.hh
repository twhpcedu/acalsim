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
#include <string>

#include "dataLinkLayer/DLLPacket.hh"
#include "peTile/PESim.hh"

class PERNocPacket : public DLLPacket<DLLRNocFrame> {
public:
	PERNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLRNocFrame>> data)
	    : DLLPacket<DLLRNocFrame>(when, data) {}

	void visit(Tick when, SimModule& module) override {
		CLASS_ERROR << "void PERNocPacket::visit(SimModule& module) is not implemented yet!";
	}
	void visit(Tick when, SimBase& simulator) override {
		auto peSim = dynamic_cast<PESim*>(&simulator);
		if (peSim) {
			peSim->RNocPacketHandler(when, this);
		} else {
			CLASS_ERROR << "Invalid simulator type";
		}
	}
};

class PEDNocPacket : public DLLPacket<DLLDNocFrame> {
public:
	PEDNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLDNocFrame>> data)
	    : DLLPacket<DLLDNocFrame>(when, data) {}

	void visit(Tick when, SimModule& module) override {}
	void visit(Tick when, SimBase& simulator) override {}
};
