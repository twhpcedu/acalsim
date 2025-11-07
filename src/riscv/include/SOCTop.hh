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

#include "ACALSim.hh"
#include "EXEStage.hh"
#include "Emulator.hh"
#include "IFStage.hh"
#include "SOC.hh"
#include "SystemConfig.hh"
#include "TopPipeRegisterManager.hh"
#include "WBStage.hh"

/**
 * @class SOCTop
 * @brief Top-level System-on-Chip simulation class
 * @details Manages the complete SoC simulation environment, including configuration
 *          registration, CLI argument handling, and trace control. Inherits from
 *          STSim template class specialized for SOC type.
 */
class SOCTop : public acalsim::SimTop {
public:
	/**
	 * @brief Constructor for SOCTop simulation environment
	 * @param _name Name of the simulation instance (default: "SOCTop")
	 * @param _configFile Path to configuration file (default: empty string)
	 * @details Initializes the simulation and sets up trace container with
	 *          default path "soc/trace"
	 */
	SOCTop(const std::string _name = "SOCTop", const std::string _configFile = "")
	    : acalsim::SimTop(_name, _configFile) {
		this->traceCntr.run(0, &acalsim::SimTraceContainer::setFilePath, "trace", "soc/trace");
	}
	virtual ~SOCTop() {}

	/**
	 * @brief Registers configuration objects for the simulation
	 * @details Creates and registers two configuration objects:
	 *          1. EmulatorConfig: Configuration for the CPU emulator
	 *          2. SOCConfig: Configuration for SOC timing parameters
	 * @override Overrides base class method
	 */
	void registerConfigs() override {
		auto emuConfig = new EmulatorConfig("Emulator configuration");
		this->addConfig("Emulator", emuConfig);
		auto socConfig = new SOCConfig("SOC configuration");
		this->addConfig("SOC", socConfig);
	}

	/**
	 * @brief Registers command-line interface arguments
	 * @details Sets up CLI options for the simulation:
	 *          - --asm_file_path: Path to the assembly code file
	 * @override Overrides base class method
	 */
	void registerCLIArguments() override {
		this->addCLIOption<std::string>("--asm_file_path",                    // Option name
		                                "The file path of an assembly code",  // Description
		                                "Emulator",                           // Config section
		                                "asm_file_path"                       // Parameter name
		);
	}

	void registerSimulators() override {
		this->soc  = new SOC("top-level soc");
		this->sIF  = new IFStage("IF stage model");
		this->sEXE = new EXEStage("EXE stage model");
		this->sWB  = new WBStage("WB stage model");

		this->addSimulator(this->soc);
		this->addSimulator(this->sIF);
		this->addSimulator(this->sEXE);
		this->addSimulator(this->sWB);

		// Create SimPort connection between SOC(functional modeling) and sIF(timing model)
		// SOC only sends an instruction to the IF stage only when there is no backpressue
		/* SOC -> sIF */
		this->soc->addMasterPort("sIF-m");
		this->sIF->addSlavePort("soc-s", 1);
		// connect SimPort
		acalsim::SimPortManager::ConnectPort(this->soc, this->sIF, "sIF-m", "soc-s");
	}

	void registerPipeRegisters() override {
		// SimPipeRegister Setup
		// IF ->prIF2EXE->EXE->prEXE2WB->WB

		acalsim::SimPipeRegister* prIF2EXE = new acalsim::SimPipeRegister("prIF2EXE");
		acalsim::SimPipeRegister* prEXE2WB = new acalsim::SimPipeRegister("prEXE2WB");

		this->pipeRegisterManager = new TopPipeRegisterManager("Top-Level Pipe Register Manager");
		this->pipeRegisterManager->addPipeRegister(prIF2EXE);
		this->pipeRegisterManager->addPipeRegister(prEXE2WB);

		this->sIF->addPRMasterPort("prIF2EXE-in", prIF2EXE);
		this->sEXE->addPRSlavePort("prIF2EXE-out", prIF2EXE);
		this->sEXE->addPRMasterPort("prEXE2WB-in", prEXE2WB);
		this->sWB->addPRSlavePort("prEXE2WB-out", prEXE2WB);
	}

private:
	SOC*      soc;
	Emulator* isaEmulator;
	IFStage*  sIF;
	EXEStage* sEXE;
	WBStage*  sWB;
};
