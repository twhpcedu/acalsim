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
using namespace acalsim;

#include "CachePacket.hh"
#include "NocPacket.hh"

class NocSim : public CPPSimBase {
public:
	NocSim(std::string name) : CPPSimBase(name), reqQueue(new UnorderedRequestQueue<SimPacket*>()) {
		CLASS_INFO << "Contructing NocSim...";
	}
	~NocSim() {}

	void init() override {}
	void cleanup() override {}
	void step() override { VERBOSE_CLASS_INFO << "NocSim executes on thread " << std::this_thread::get_id(); }

	void handleTGRequest(NocReqPacket* _nocReqPkt);
	void handleCacheRespond(CacheRespPacket* _cacheRespPkt);
	Tick getRespDelay() { return (Tick)nocRespDelay; }

private:
	static const int                   nocRespDelay = 1;
	static const int                   nocBandwidth = 32;
	UnorderedRequestQueue<SimPacket*>* reqQueue;
};
