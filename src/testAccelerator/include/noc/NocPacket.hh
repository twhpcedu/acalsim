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

#include "ACALSim.hh"
using namespace acalsim;

#include "dataLinkLayer/DLLPacket.hh"
#include "noc/NocSim.hh"

class RNocPacket : public DLLPacket<DLLRNocFrame> {
public:
	RNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLRNocFrame>> data) : DLLPacket<DLLRNocFrame>(when, data) {}
	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;
};

class DNocPacket : public DLLPacket<DLLDNocFrame> {
public:
	DNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLDNocFrame>> data) : DLLPacket<DLLDNocFrame>(when, data) {}
	void visit(Tick when, SimModule& module) override {}
	void visit(Tick when, SimBase& simulator) override {}
};
