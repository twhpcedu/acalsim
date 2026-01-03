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
#include <vector>

// ACALSim
#include "external/gem5/Event.hh"

// Third-party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace acalsim {

class JsonEvent {
protected:
	// category e.g. "MessagePacketEvent" ,
	std::string cat;

	// message name e.g. "SW_INTERRUPT.TENSOR_READY"
	std::string name;

	// ph is the phase type.
	// B and E for begin and end for duration events
	// X for complete events. An extra parameter dur has to be added for the duration of the event.
	// i for instant events. They don’t look very nice on the visualization though… : "X"
	std::string ph;

	// ts is the timestamp for the tracing clock.
	Tick ts;

	// pid and tid are the process and thread ID’s.
	uint64_t pid;  // reserved for the time being
	uint64_t tid;  // reserved for the time being
	Tick     dur;  // latency

public:
	JsonEvent(std::string _name, std::string _ph, Tick _ts) : name(_name), ph(_ph), ts(_ts) {}
	JsonEvent(std::string _name, Tick _ts, Tick _dur) : name(_name), ph("X"), ts(_ts), dur(_dur) {}
	~JsonEvent() {}
};

class MessagePacketJsonEvent : public JsonEvent {
	/*
	 * the following private members will be written into the args filed in Json
	 * e.g. "args" : {
	 *           "source" : "MCPU" ,
	 *           "dest" : "PETile-0" ,
	 *           "tensor-id" : 0 ,
	 *           "message-packet-id" : 0
	 *       }
	 */
	std::string source;
	std::string dest;
	uint64_t    tensorId;
	uint64_t    packetId;

public:
	MessagePacketJsonEvent(std::string _source, std::string _dest, uint64_t _tensorId, uint64_t _packetId,
	                       std::string _name, std::string _ph, Tick _ts)
	    : source(_source),
	      dest(_dest),
	      tensorId(_tensorId),
	      packetId(_packetId),
	      JsonEvent(_name, _ph, _ts) {  // add B or E events
	}
	MessagePacketJsonEvent(std::string _source, std::string _dest, uint64_t _tensorId, uint64_t _packetId,
	                       std::string _name, Tick _ts, Tick _dur)
	    : source(_source),
	      dest(_dest),
	      tensorId(_tensorId),
	      packetId(_packetId),
	      JsonEvent(_name, _ts, _dur) {  // add X events
	}

	~MessagePacketJsonEvent() {}
};

template <class T>
class JsonContainer {
private:
	std::vector<std::shared_ptr<T>> elements;

public:
	JsonContainer() {}
	~JsonContainer() {}

	template <typename... Args>
	void add(Args... args) {
		std::shared_ptr<T> t = std::make_shared<T>(args...);
		elements.push_back(t);
	}

	std::shared_ptr<T> get(int which) const { return elements[which]; }

	int  size() const { return elements.size(); }
	void writeToJsonFile(json& file) {}
};

}  // end of namespace acalsim
