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

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "ACALSim.hh"
using namespace acalsim;

class NocEvent : public SimEvent {
public:
	NocEvent(int _id, std::string name, std::function<void(void)> _callback = nullptr)
	    : SimEvent(), id(_id), _name("NocEvent_" + name), callback(_callback) {}
	~NocEvent() {}

	const std::string name() const override { return _name; }
	void              process() override;

private:
	std::string _name;
	// PE ID
	int id;

	// callback function pointer
	std::function<void(void)> callback;
};
