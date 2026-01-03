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

#include <functional>
#include <iostream>
#include <string>

#include "ACALSim.hh"
using namespace acalsim;

#include "NocPacket.hh"
#include "UserDefinedArgs.hh"

class TrafficEvent : public SimEvent {
public:
	TrafficEvent(SimBase* _sim, int _id, std::string _name)
	    : SimEvent(), sim(_sim), id(_id), _name("TrafficEvent_" + _name) {
		this->clearFlags(EventBase::Managed);
	}
	~TrafficEvent() {}

	const std::string name() const override { return _name; }
	void              process() override;
	void              callback();

	void NocRespHandler(Tick when, int id, NocRespPacket* pkt, SimBase* sim);
	void sendSharedData();
	void sendNoCEvent();

private:
	std::string _name;

	int id;

	// Simulator pointer
	SimBase* sim;
};

class TrafficRespEvent : public SimEvent {
public:
	TrafficRespEvent(int _id, std::string _name, NocRespPacket* _nocRespPkt)
	    : SimEvent(), id(_id), _name("TrafficRespEvent_" + _name), nocRespPkt(_nocRespPkt) {}
	~TrafficRespEvent() {}

	NocRespPacket* getNocRespPkt() { return nocRespPkt; }

	const std::string name() const override { return _name; }
	void              process() override {
        UserDefinedArgs* args = new UserDefinedArgs(this->getNocRespPkt()->getData(), "example");
        auto             record = acalsim::ChromeTraceRecord::createInstantEvent("System", "Transaction Finish",
		                                                                                      top->getGlobalTick(), "g", "", "", args);
        top->addChromeTraceRecord(record);
        CLASS_INFO << "Transaction Finish ! Data = " + std::to_string(this->getNocRespPkt()->getData());
	};

private:
	std::string _name;

	int id;

	NocRespPacket* nocRespPkt;
};
