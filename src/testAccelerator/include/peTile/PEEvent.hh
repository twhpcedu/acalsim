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

#include "peTile/PEPacket.hh"

template <typename TCallback, typename TPacket>
class PEEvent : public CallbackEvent<TCallback> {
public:
	PEEvent(int _tid, std::string _name, TPacket* _pkt, void* _callee = nullptr,
	        std::function<TCallback> _callback = nullptr)
	    : CallbackEvent<TCallback>(_tid, _callee, _callback), _name("PEEvent_" + _name), pkt(_pkt) {}
	~PEEvent() {}

	const std::string name() const override { return _name; }
	TPacket*          getPacket() { return pkt; }

	virtual void process() = 0;

private:
	std::string _name;
	TPacket*    pkt;
};

class PERNocEvent : public PEEvent<void(void), PERNocPacket> {
public:
	PERNocEvent(int _id, std::string _name, PERNocPacket* _pkt, void* _callee = nullptr,
	            std::function<void(void)> _callback = nullptr)
	    : PEEvent(_id, _name, _pkt, _callee, _callback) {}
	~PERNocEvent() {}

	void process() override;
};

class PEDNocEvent : public PEEvent<void(void), PEDNocPacket> {
public:
	PEDNocEvent(int _id, std::string _name, PEDNocPacket* _pkt, void* _callee = nullptr,
	            std::function<void(void)> _callback = nullptr)
	    : PEEvent(_id, _name, _pkt, _callee, _callback) {}
	~PEDNocEvent() {}

	void process() override {}
};
