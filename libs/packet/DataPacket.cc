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

#include "packet/DataPacket.hh"

namespace acalsim {

void DataPacket::renew(Tick _when, void* _data) {
	SimPacket::renew(PTYPE::DATA);
	this->when = _when;
	this->data = _data;
}

// visit function when a DataPacket is sent to a SimModule object
void DataPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void DataPacket::visit(SimBase& module) is not implemented yet!";
}

// visit function when a DataPacket is sent to a simulator
void DataPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void DataPacket::visit(SimBase& simulator) is not implemented yet!";
}

}  // namespace acalsim
