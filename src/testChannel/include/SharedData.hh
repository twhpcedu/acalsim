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

class TestSharedData {
public:
	TestSharedData() {}
	~TestSharedData() {}

	void init() {
		// initFunc(vInt, vLong, pInt);*/
		for (int i = 0; i < nElements; i++) {
			vInt[i]  = i;
			vLong[i] = i * 10000;
			pInt[i]  = vInt + i;
		}
	}

	void print() {
		for (int i = 0; i < nElements; i++) {
			char msg[256];
			sprintf(msg, "vInt[%d]: %d vLong[%d]: %lld pInt[%d]: 0x%p", i, vInt[i], i, vLong[i], i, pInt[i]);
			INFO << std::string(msg);
			vLong[i] = i * 10000;
			pInt[i]  = vInt + i;
		}
	}

	void set(int which, int seed, int pindex) {
		// A setter to test whether we can modify the data member correctly
		vInt[which]  = seed;
		vLong[which] = seed * 10000;
		pInt[which]  = vInt + pindex;
	}

private:
	static const int nElements = 5;
	int              vInt[nElements];
	long long        vLong[nElements];
	int*             pInt[nElements];
};

class SharedDataPacket : public SimPacket, virtual public HashableType {
public:
	SharedDataPacket(Tick _when, std::shared_ptr<acalsim::SharedContainer<TestSharedData>> _data)
	    : SimPacket(PTYPE::DATA), when(_when) {
		data = _data;
	}
	~SharedDataPacket() {}
	void* getData() { return data.get(); }
	Tick  getWhen() { return when; }

	void visit(Tick when, SimModule& module) override;
	void visit(Tick when, SimBase& simulator) override;

private:
	Tick                                                      when;
	std::shared_ptr<acalsim::SharedContainer<TestSharedData>> data;
};
