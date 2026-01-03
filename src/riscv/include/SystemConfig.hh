
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

#include "ACALSim.hh"

using json = nlohmann::json;

/**
 * @class EmulatorConfig
 * @brief Configuration class for the CPU emulator settings
 * @details Inherits from SimConfig and defines parameters for memory layout,
 *          assembly file handling, and emulator constraints
 */
class EmulatorConfig : public acalsim::SimConfig {
public:
	/**
	 * @brief Constructor that initializes emulator configuration parameters
	 * @param _name Name identifier for the configuration instance
	 * @details Sets up the following parameters:
	 *          - memory_size: Total memory size in bytes (default: 65536)
	 *          - data_offset: Starting offset for data segment (default: 8192)
	 *          - text_offset: Starting offset for text/code segment (default: 0)
	 *          - max_label_count: Maximum number of labels supported (default: 128)
	 *          - max_src_len: Maximum source code length in bytes (default: 1048576)
	 *          - asm_file_path: Path to the assembly source file (default: empty)
	 */
	EmulatorConfig(const std::string& _name) : acalsim::SimConfig(_name) {
		this->addParameter<int>("memory_size", 65536, acalsim::ParamType::INT);
		this->addParameter<int>("data_offset", 8192, acalsim::ParamType::INT);
		this->addParameter<int>("text_offset", 0, acalsim::ParamType::INT);
		this->addParameter<int>("max_label_count", 128, acalsim::ParamType::INT);
		this->addParameter<int>("max_src_len", 1048576, acalsim::ParamType::INT);
		this->addParameter<std::string>("asm_file_path", "", acalsim::ParamType::STRING);
	}

	/**
	 * @brief Default destructor
	 */
	~EmulatorConfig() {}
};

/**
 * @class SOCConfig
 * @brief Configuration class for System-on-Chip (SOC) timing parameters
 * @details Inherits from SimConfig and defines latency parameters for
 *          memory operations in the system
 */
class SOCConfig : public acalsim::SimConfig {
public:
	/**
	 * @brief Constructor that initializes SOC timing parameters
	 * @param _name Name identifier for the configuration instance
	 * @details Sets up the following parameters:
	 *          - memory_read_latency: Clock cycles for memory read operations (default: 1)
	 *          - memory_write_latency: Clock cycles for memory write operations (default: 1)
	 */
	SOCConfig(const std::string& _name) : acalsim::SimConfig(_name) {
		this->addParameter<acalsim::Tick>("memory_read_latency", 1, acalsim::ParamType::TICK);
		this->addParameter<acalsim::Tick>("memory_write_latency", 1, acalsim::ParamType::TICK);
	}

	/**
	 * @brief Default destructor
	 */
	~SOCConfig() {}
};
