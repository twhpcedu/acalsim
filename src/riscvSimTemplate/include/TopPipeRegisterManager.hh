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
	virtual void runSyncPipeRegister();

	/**
	 * @brief Set stall control flags for specified pipeline registers
	 * @param stalled_pipe_names Vector of pipeline register names to stall
	 *
	 * This method activates stall flags on the specified pipeline registers.
	 * If a register name is not found, a warning is logged and that name is skipped.
	 */
	void setPipeStallControl(std::vector<std::string> stalled_pipe_names) {
		for (const auto& name : stalled_pipe_names) {
			// Using find() method first to check if key exists
			auto it = registers.find(name);
			if (it != registers.end()) {
				// Key exists, safe to set stall flag
				it->second->setStallFlag();
			} else {
				// Log warning about missing register
				CLASS_ASSERT_MSG("Pipeline register '%s' not found when attempting to set stall", name.c_str());
			}
		}
	}
};
