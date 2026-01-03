
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

#include "ACALSim.hh"
using namespace acalsim;

enum class PEReqTypeEnum { TEST };

class PERespPacket : public SimPacket {
public:
	PERespPacket(PEReqTypeEnum _reqType, int* _d) : SimPacket(PTYPE::PERESP), reqType(_reqType), d(_d) {}
	~PERespPacket() {}
	int  getResult() { return *d; }
	void setResult(int _d) { *d = _d; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	PEReqTypeEnum reqType;
	int*          d;
};

class PEReqPacket : public SimPacket {
public:
	PEReqPacket(PEReqTypeEnum _reqType, int _a, int _b, int _c, PERespPacket* _PERespPkt)
	    : SimPacket(PTYPE::PEREQ), reqType(_reqType), a(_a), b(_b), c(_c), PERespPkt(_PERespPkt) {}
	~PEReqPacket() {}
	int           getA() { return a; }
	int           getB() { return b; }
	int           getC() { return c; }
	PERespPacket* getPERespPkt() { return PERespPkt; }
	void          setCallback(std::function<void(int, PERespPacket*)> _callback) { callback = _callback; }
	std::function<void(int, PERespPacket*)> getCallback() { return callback; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	PEReqTypeEnum                           reqType;
	int                                     a;
	int                                     b;
	int                                     c;
	std::function<void(int, PERespPacket*)> callback;   // The callback function for the caller
	PERespPacket*                           PERespPkt;  // The packet for the downstream callee to update
};
