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

#include <memory>

#include "ACALSimSC.hh"
using namespace acalsim;

class InBoundData {
public:
	InBoundData() {}
	InBoundData(int _id, int a, int b, int c) : id(_id), A(a), B(b), C(c) {}

	void set(int _id, int _a, int _b, int _c) {
		this->id = _id;
		this->A  = _a;
		this->B  = _b;
		this->C  = _c;
	}

	int A;
	int B;
	int C;
	int id;
};

class OutBoundData {
public:
	OutBoundData() {}
	OutBoundData(int _id, int d) : id(_id), D(d) {}
	void set(int _id, int _d) {
		this->id = _id;
		this->D  = _d;
	}
	int id;
	int D;
};

template <typename T>
class MacPacket : public SCSimPacket {
public:
	MacPacket(std::shared_ptr<SharedContainer<T>> _data = nullptr) : SCSimPacket(), data(_data) {}

	virtual void visit(Tick when, SimModule& module) override {}
	virtual void visit(Tick when, SimBase& simulator) override {}

	std::shared_ptr<SharedContainer<T>> getData() { return this->data; }

	void renew(std::shared_ptr<SharedContainer<T>> _data) { this->data = _data; }

protected:
	std::shared_ptr<SharedContainer<T>> data;
};

class MacInPacket : public MacPacket<InBoundData> {
public:
	MacInPacket(std::shared_ptr<SharedContainer<InBoundData>> _data = nullptr) : MacPacket<InBoundData>(_data) {}

	void renew(std::shared_ptr<SharedContainer<InBoundData>> _data) { this->MacPacket<InBoundData>::renew(_data); }
};

class MacOutPacket : public MacPacket<OutBoundData> {
public:
	MacOutPacket(std::shared_ptr<SharedContainer<OutBoundData>> _data = nullptr) : MacPacket<OutBoundData>(_data) {}

	void renew(std::shared_ptr<SharedContainer<OutBoundData>> _data) { this->MacPacket<OutBoundData>::renew(_data); }
	void visit(Tick when, SimBase& simulator) override;
};
