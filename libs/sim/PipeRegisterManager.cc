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

#include "sim/PipeRegisterManager.hh"

#include "utils/Logging.hh"

namespace acalsim {

void PipeRegisterManagerBase::addPipeRegister(SimPipeRegister* _reg) {
	auto name = _reg->getName();

	auto existing = this->registers.contains(name);
	CLASS_ASSERT_MSG(!existing, "PipeRegisterManager `" + name + "` already exists!");
	this->registers.insert(std::make_pair(name, _reg));
}

SimPipeRegister* PipeRegisterManagerBase::getPipeRegister(const std::string& _name) const {
	auto iter = this->registers.find(_name);
	CLASS_ASSERT_MSG(iter != this->registers.end(), "The PipeRegister \'" + _name + "\' does not exist.");
	return iter->second;
}

void PipeRegisterManager::runSyncPipeRegister() {
	for (auto& [_, reg] : this->registers) { reg->sync(); }
}

}  // namespace acalsim
