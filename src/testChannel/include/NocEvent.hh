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

#include <cstdint>
#include <functional>
#include <iostream>

#include "ACALSim.hh"
using namespace acalsim;

#include "CachePacket.hh"
#include "NocPacket.hh"
#include "NocSim.hh"

using namespace acalsim;

class NocReqEvent : public CallbackEvent<void(Tick, int, NocRespPacket*, SimBase*)> {
public:
	NocReqEvent(int _tid, std::string _name, std::function<void(Tick, int, NocRespPacket*, SimBase*)> _callback,
	            SimBase* _callee, NocReqPacket* _nocReqPkt)
	    : CallbackEvent(_tid, _callee, _callback), _name("NocReqEvent_" + _name), nocReqPkt(_nocReqPkt) {}
	~NocReqEvent() {}

	const std::string name() const override { return _name; }
	void              process() override;

private:
	std::string   _name;
	NocReqPacket* nocReqPkt;  // noc request packet including data
};

class NocRespEvent : public CallbackEvent<void(Tick, int, NocRespPacket*, SimBase*)> {
public:
	NocRespEvent(
	    int                                                      _tid,  // The transaction ID
	    std::string                                              _name,
	    std::function<void(Tick, int, NocRespPacket*, SimBase*)> _callback,  // The callback function for the caller
	    NocRespPacket* _nocRespPkt,  // The packet for the downstream callee to update
	    SimBase*       _callee)
	    : CallbackEvent(_tid, _callee, _callback), _name("NocRespEvent_" + _name), nocRespPkt(_nocRespPkt) {}

	const std::string name() const override { return _name; }
	void              process() override {
        CLASS_INFO << "Process NocRespEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();

        // callerCallback = TrafficEvent::NocRespHandler
        callerCallback(top->getGlobalTick(), this->tid, this->nocRespPkt, ((NocSim*)callee));
	}

private:
	std::string    _name;
	NocRespPacket* nocRespPkt;
};
