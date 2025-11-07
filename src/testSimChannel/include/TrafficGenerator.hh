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
using namespace acalsim;

#include "CachePacket.hh"
#include "NocPacket.hh"

class TrafficGenerator : public CPPSimBase {
public:
	TrafficGenerator(std::string name) : CPPSimBase(name) {}
	~TrafficGenerator() {}

	void init() override;
	void cleanup() override {}
	void step() override { VERBOSE_CLASS_INFO << "TrafficGenerator executes on thread " << std::this_thread::get_id(); }

	void sendNoCRequest(int _tid);
	void handleNoCRespond(NocRespPacket* _nocRespPkt);
	Tick getRemoteDelay() { return (Tick)tgRemoteDelay; }

protected:
	static const int tgRemoteDelay = 10;
};
