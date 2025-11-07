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

#include "cache/CachePacket.hh"

template <typename TCallback, typename TPacket>
class CacheEvent : public CallbackEvent<TCallback> {
public:
	CacheEvent(int _tid, std::string _name, TPacket* _pkt, void* _callee = nullptr,
	           std::function<TCallback> _callback = nullptr)
	    : CallbackEvent<TCallback>(_tid, _callee, _callback), _name("CacheEvent_" + _name), pkt(_pkt) {}
	~CacheEvent() {}

	const std::string name() const override { return _name; }
	TPacket*          getPacket() { return pkt; }

	virtual void process() = 0;

private:
	std::string _name;
	TPacket*    pkt;
};

class CacheRNocEvent : public CacheEvent<void(void), CacheRNocPacket> {
public:
	CacheRNocEvent(int _id, std::string _name, CacheRNocPacket* _pkt, void* _callee = nullptr,
	               std::function<void(void)> _callback = nullptr)
	    : CacheEvent(_id, _name, _pkt, _callee, _callback) {}
	~CacheRNocEvent() {}

	void process() override;
};

class CacheDNocEvent : public CacheEvent<void(void), CacheDNocPacket> {
public:
	CacheDNocEvent(int _id, std::string _name, CacheDNocPacket* _pkt, void* _callee = nullptr,
	               std::function<void(void)> _callback = nullptr)
	    : CacheEvent(_id, _name, _pkt, _callee, _callback) {}
	~CacheDNocEvent() {}

	void process() override {}
};
