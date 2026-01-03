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

#include "external/gem5/Event.hh"
#include "packet/SimPacket.hh"

namespace acalsim {

class SimBase;
class SimModule;

class DataPacket : public SimPacket {
private:
	Tick  when = 0;
	void* data = nullptr;

public:
	DataPacket() : SimPacket(PTYPE::DATA) {}
	DataPacket(Tick _when, void* _data) : SimPacket(PTYPE::DATA), when(_when) { data = _data; }
	~DataPacket() {}
	void*        getData() const { return data; }
	Tick         getWhen() const { return when; }
	void         renew(Tick _when, void* _data);
	virtual void visit(Tick when, SimModule& module) override;
	virtual void visit(Tick when, SimBase& simulator) override;

	virtual std::string getName() const override { return std::string("DataPacket-") + this->getIDStr(); }
};

}  // namespace acalsim
