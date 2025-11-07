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

#include "ACALSim.hh"
using namespace acalsim;

#include "noc/NocPacket.hh"

template <typename TCallback, typename TPacket>
class NocEvent : public CallbackEvent<void(void)> {
public:
	NocEvent(int _tid, std::string _name, TPacket* _pkt, void* _callee = nullptr,
	         std::function<void(void)> _callback = nullptr)
	    : CallbackEvent<void(void)>(_tid, _callee, _callback), _name("NocEvent_" + _name), pkt(_pkt) {}
	~NocEvent() {}

	const std::string name() const override { return _name; }
	TPacket*          getPacket() { return pkt; }

	virtual void process() = 0;

private:
	std::string _name;
	TPacket*    pkt;
};

class RNocEvent : public NocEvent<void(void), RNocPacket> {
public:
	RNocEvent(int _id, std::string _name, RNocPacket* _pkt, void* _callee = nullptr,
	          std::function<void(void)> _callback = nullptr)
	    : NocEvent(_id, _name, _pkt, _callee, _callback) {}
	~RNocEvent() {}

	void process() override;
};

class DNocEvent : public NocEvent<DLLDNocFrame, DNocPacket> {
public:
	DNocEvent(int _id, std::string _name, DNocPacket* _pkt, void* _callee = nullptr,
	          std::function<void(void)> _callback = nullptr)
	    : NocEvent(_id, _name, _pkt, _callee, _callback) {}
	~DNocEvent() {}

	void process() override {}
};
