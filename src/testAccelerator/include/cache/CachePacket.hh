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

#include <memory>
#include <string>

#include "cache/CacheSim.hh"
#include "dataLinkLayer/DLLPacket.hh"

class CacheRNocPacket : public DLLPacket<DLLRNocFrame> {
public:
	bool blackhole;
	CacheRNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLRNocFrame>> data, bool _blackhole = false)
	    : DLLPacket<DLLRNocFrame>(when, data), blackhole(_blackhole) {}

	void visit(Tick when, SimModule& module) override {
		CLASS_ERROR << "void CacheRNocPacket::visit(SimModule& module) is not implemented yet!";
	}
	void visit(Tick when, SimBase& simulator) override {
		auto cacheSim = dynamic_cast<CacheSim*>(&simulator);
		if (cacheSim) {
			cacheSim->RNocPacketHandler(when, this);
		} else {
			CLASS_ERROR << "Invalid simulator type";
		}
	}
};

class CacheDNocPacket : public DLLPacket<DLLDNocFrame> {
public:
	CacheDNocPacket(Tick when, std::shared_ptr<SharedContainer<DLLDNocFrame>> data)
	    : DLLPacket<DLLDNocFrame>(when, data) {}

	void visit(Tick when, SimModule& module) override {}
	void visit(Tick when, SimBase& simulator) override {}
};
