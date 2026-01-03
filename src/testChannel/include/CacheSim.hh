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

#include <string>

#include "ACALSim.hh"
using namespace acalsim;

#include "CachePacket.hh"
#include "NocPacket.hh"

class CacheSim : public CPPSimBase {
public:
	CacheSim(std::string name) : CPPSimBase(name) { CLASS_INFO << "Constructing CacheSim ..."; }
	~CacheSim() {}
	// Todo : need more information to calculate more accurate latency
	Tick getRespDelay(int size) { return (Tick)(cacheMemoryLatency + (size + 1) / cacheMemoryBandwidth); }
	void init() override;
	void cleanup() override;
	void step() override { CLASS_INFO << "CacheSim executes on thread " << std::this_thread::get_id(); }
	void handleNOCRequest(CacheReqPacket* cacheReqPkt, int id, Tick when);

private:
	static const int cacheMemoryLatency = 1;

	static const int cacheMemoryBandwidth = 32;
};
