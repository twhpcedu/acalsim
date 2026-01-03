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
#include <unordered_map>
#include <vector>

#include "ACALSim.hh"

/**
 * @class TopPipeRegisterManager
 * @brief Concrete implementation of the pipe register manager
 */
class TopPipeRegisterManager : public acalsim::PipeRegisterManagerBase, virtual public acalsim::HashableType {
public:
	/**
	 * @brief Constructor
	 * @param name Manager name
	 */
	TopPipeRegisterManager(const std::string& name) : acalsim::PipeRegisterManagerBase(name) {}

	/**
	 * @brief Destructor
	 */
	virtual ~TopPipeRegisterManager() {
		// Clean up all registered pipe registers
		for (auto& pair : registers) { delete pair.second; }
		registers.clear();
	}

	/**
	 * @brief Implementation of the pipe register synchronization
	 */
	void runSyncPipeRegister();
};
