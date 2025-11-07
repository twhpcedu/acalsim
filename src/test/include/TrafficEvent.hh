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

#include <functional>
#include <iostream>
#include <string>

#include "ACALSim.hh"
using namespace acalsim;

class TrafficEvent : public SimEvent {
public:
	TrafficEvent(SimBase* _sim, int _id, std::string name)
	    : SimEvent(), sim(_sim), id(_id), _name("TrafficEvent_" + name) {
		this->clearFlags(this->Managed);
	}
	~TrafficEvent() {}

	const std::string name() const override { return _name; }
	void              process() override;
	void              callback();

private:
	std::string _name;
	// PE ID
	int id;

	// Simulator pointer
	SimBase* sim;
};
