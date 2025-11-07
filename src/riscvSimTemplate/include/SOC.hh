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

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>

#include "ACALSim.hh"
#include "CPU.hh"
#include "DataMemory.hh"
#include "DataStruct.hh"
#include "Emulator.hh"
#include "port/MasterPort.hh"

/**
 * @class SOC
 * @brief System-on-Chip (SOC) module integrating CPU, memory, and ISA emulator
 * @details Represents the top-level hardware system that combines:
 *          - An ISA behavior emulator
 *          - A single-cycle CPU model
 *          - A data memory subsystem
 *          Inherits from STSimBase to provide simulation functionality
 */
class SOC : public acalsim::CPPSimBase {
public:
	/**
	 * @brief Constructor for the SOC class
	 * @param _name Name identifier for the SOC instance (default: "top-level SOC")
	 */
	SOC(std::string _name = "top-level SOC");

	/**
	 * @brief Virtual destructor
	 */
	virtual ~SOC() {}

	void init() override {
		this->registerModules();
		simInit();
	}

	/**
	 * @brief Registers all hardware modules in the system
	 * @details Sets up the connections between CPU, memory, and emulator
	 * @override Overrides base class method
	 */
	void registerModules() override;

	/**
	 * @brief Initializes the simulation environment
	 * @details Performs necessary setup before simulation start
	 * @override Overrides base class method
	 */
	void simInit();

	/**
	 * @brief Performs cleanup after simulation
	 * @details Handles resource deallocation and final state management
	 * @override Overrides base class method
	 */
	void cleanup() override;

	void masterPortRetry(acalsim::MasterPort* _port) override;

private:
	Emulator*   isaEmulator;  ///< ISA behavior model for instruction emulation
	CPU*        cpu;          ///< Single-cycle CPU hardware model
	DataMemory* dmem;         ///< Data memory subsystem model
};
