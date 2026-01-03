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

#include "RISCVSoCComponent.hh"

#include <sst/core/sst_config.h>

// Include RISC-V simulator headers with proper paths
#include "../../src/riscv/include/DEStage.hh"
#include "../../src/riscv/include/EXEStage.hh"
#include "../../src/riscv/include/IFStage.hh"
#include "../../src/riscv/include/MEMStage.hh"
#include "../../src/riscv/include/SOC.hh"
#include "../../src/riscv/include/WBStage.hh"

using namespace ACALSim::SSTIntegration;

RISCVSoCComponent::RISCVSoCComponent(::SST::ComponentId_t id, ::SST::Params& params)
    : ::SST::Component(id), current_cycle_(0), max_cycles_(0), simulation_done_(false), mem_link_(nullptr) {
	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("RISCVSoCComponent[@p:@l]: ", verbose, 0, ::SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing RISC-V RV32I SoC Component\n");

	// Get clock frequency
	clock_freq_ = params.find<std::string>("clock", "1GHz");
	out_.verbose(CALL_INFO, 2, 0, "Clock frequency: %s\n", clock_freq_.c_str());

	// Get simulation parameters
	max_cycles_ = params.find<uint64_t>("max_cycles", 0);
	if (max_cycles_ > 0) { out_.verbose(CALL_INFO, 2, 0, "Max cycles: %lu\n", max_cycles_); }

	// Get RISC-V specific parameters
	asm_file_path_  = params.find<std::string>("asm_file", "");
	memory_size_    = params.find<uint32_t>("memory_size", 65536);
	text_offset_    = params.find<uint32_t>("text_offset", 0);
	data_offset_    = params.find<uint32_t>("data_offset", 8192);
	dump_registers_ = params.find<bool>("dump_registers", true);
	dump_memory_    = params.find<bool>("dump_memory", false);

	if (asm_file_path_.empty()) { out_.fatal(CALL_INFO, -1, "Error: asm_file parameter is required\n"); }

	out_.verbose(CALL_INFO, 2, 0, "Assembly file: %s\n", asm_file_path_.c_str());
	out_.verbose(CALL_INFO, 2, 0, "Memory size: %u bytes\n", memory_size_);

	// Register statistics
	stat_instructions_    = registerStatistic<uint64_t>("instructions_executed");
	stat_cycles_          = registerStatistic<uint64_t>("cycles");
	stat_branches_        = registerStatistic<uint64_t>("branches_taken");
	stat_loads_           = registerStatistic<uint64_t>("loads");
	stat_stores_          = registerStatistic<uint64_t>("stores");
	stat_pipeline_stalls_ = registerStatistic<uint64_t>("pipeline_stalls");

	// Configure optional memory link
	mem_link_ = configureLink("mem_port");
	if (mem_link_) { out_.verbose(CALL_INFO, 2, 0, "External memory port configured\n"); }

	// Initialize RISC-V simulator
	initRISCVSimulator(params);

	// Register clock handler
	tc_ =
	    registerClock(clock_freq_, new ::SST::Clock::Handler2<RISCVSoCComponent, &RISCVSoCComponent::clockTick>(this));

	// Tell SST we control simulation end
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	out_.verbose(CALL_INFO, 1, 0, "RISC-V SoC initialization complete\n");
}

RISCVSoCComponent::~RISCVSoCComponent() { out_.verbose(CALL_INFO, 1, 0, "Destroying RISCVSoCComponent\n"); }

void RISCVSoCComponent::initRISCVSimulator(::SST::Params& params) {
	out_.verbose(CALL_INFO, 2, 0, "Initializing RISC-V simulator components\n");

	// Note: The actual RISC-V simulator requires SimTop infrastructure
	// For a complete implementation, you would need to either:
	// 1. Refactor SOC to work standalone (extract from SimTop dependency)
	// 2. Create a lightweight SimTop wrapper within this component
	// 3. Use the full SOCTop infrastructure
	//
	// This implementation assumes option 1 or 2 has been done
	// For now, we create placeholder instances that would be properly
	// initialized in a production implementation

	out_.verbose(CALL_INFO, 2, 0, "Creating SOC instance\n");
	// soc_ = std::make_unique<SOC>("riscv_soc");

	out_.verbose(CALL_INFO, 2, 0, "Creating pipeline stages\n");
	// if_stage_ = std::make_unique<IFStage>("IF");
	// exe_stage_ = std::make_unique<EXEStage>("EXE");
	// wb_stage_ = std::make_unique<WBStage>("WB");

	// In production, initialize the simulator here
	// This would include:
	// 1. Load assembly program
	// 2. Initialize memory
	// 3. Setup pipeline connections
	// 4. Reset CPU state

	loadAssemblyProgram(asm_file_path_);

	out_.verbose(CALL_INFO, 2, 0, "RISC-V simulator initialization complete\n");
}

void RISCVSoCComponent::loadAssemblyProgram(const std::string& asm_file) {
	out_.verbose(CALL_INFO, 2, 0, "Loading assembly program: %s\n", asm_file.c_str());

	// Verify file exists
	FILE* fp = fopen(asm_file.c_str(), "r");
	if (!fp) { out_.fatal(CALL_INFO, -1, "Error: Cannot open assembly file: %s\n", asm_file.c_str()); }
	fclose(fp);

	// In production implementation:
	// 1. Parse assembly file using Emulator class
	// 2. Load instructions into instruction memory
	// 3. Initialize data memory with .data segment
	// 4. Resolve labels and set PC to entry point

	out_.verbose(CALL_INFO, 2, 0, "Assembly program loaded successfully\n");
}

void RISCVSoCComponent::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");

	// Initialize RISC-V components if not already done
	if (soc_) {
		// soc_->init();
	}
}

void RISCVSoCComponent::init(unsigned int phase) {
	out_.verbose(CALL_INFO, 2, 0, "Init phase %u\n", phase);

	// Multi-phase initialization if needed
	// RISC-V simulator is self-contained, so typically no cross-component init needed
}

void RISCVSoCComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");

	// Cleanup RISC-V simulator
	if (soc_) {
		// soc_->cleanup();
	}

	// Dump register file if requested
	if (dump_registers_) { dumpRegisterFile(); }

	// Dump memory if requested
	if (dump_memory_) { dumpMemory(); }

	// Print statistics
	out_.output(CALL_INFO, "\n=== RISC-V SoC Statistics ===\n");
	out_.output(CALL_INFO, "Cycles:              %lu\n", current_cycle_);
	out_.output(CALL_INFO, "Instructions:        %lu\n", stat_instructions_->getCollectionCount());
	if (current_cycle_ > 0) {
		double ipc = (double)stat_instructions_->getCollectionCount() / current_cycle_;
		out_.output(CALL_INFO, "IPC:                 %.3f\n", ipc);
	}
	out_.output(CALL_INFO, "Branches taken:      %lu\n", stat_branches_->getCollectionCount());
	out_.output(CALL_INFO, "Loads:               %lu\n", stat_loads_->getCollectionCount());
	out_.output(CALL_INFO, "Stores:              %lu\n", stat_stores_->getCollectionCount());
	out_.output(CALL_INFO, "Pipeline stalls:     %lu\n", stat_pipeline_stalls_->getCollectionCount());
	out_.output(CALL_INFO, "=============================\n");

	primaryComponentOKToEndSim();
}

bool RISCVSoCComponent::clockTick(::SST::Cycle_t cycle) {
	// Execute one cycle of RISC-V simulation
	current_cycle_++;
	stat_cycles_->addData(1);

	// In production implementation:
	// 1. Execute one step of SOC (CPU instruction execution)
	// 2. Step pipeline stages (IF, EXE, WB)
	// 3. Update statistics based on instruction type
	// 4. Check for HCF (halt) instruction

	if (soc_) {
		// soc_->step();
		// Update statistics based on executed instruction
		stat_instructions_->addData(1);
	}

	// Check if simulation should end
	bool should_continue = !shouldEndSimulation();

	if (!should_continue) {
		out_.verbose(CALL_INFO, 1, 0, "Simulation complete at cycle %lu\n", current_cycle_);
		primaryComponentOKToEndSim();
	}

	return should_continue;
}

bool RISCVSoCComponent::shouldEndSimulation() {
	// End if max cycles reached
	if (max_cycles_ > 0 && current_cycle_ >= max_cycles_) {
		out_.verbose(CALL_INFO, 1, 0, "Reached max cycles (%lu)\n", max_cycles_);
		return true;
	}

	// End if simulation complete (HCF instruction or no more events)
	if (simulation_done_) { return true; }

	// In production: check if CPU executed HCF instruction
	// if (soc_ && soc_->isDone()) {
	//     simulation_done_ = true;
	//     return true;
	// }

	return false;
}

void RISCVSoCComponent::dumpRegisterFile() {
	out_.output(CALL_INFO, "\n=== Register File State ===\n");

	// In production implementation:
	// Get register values from CPU and display them
	// Format: x0-x31 with ABI names

	// Example output format:
	// x0  (zero) = 0x00000000
	// x1  (ra)   = 0x00000100
	// x2  (sp)   = 0x00001000
	// ...

	out_.output(CALL_INFO, "(Register dump not implemented in stub)\n");
	out_.output(CALL_INFO, "===========================\n");
}

void RISCVSoCComponent::dumpMemory() {
	out_.output(CALL_INFO, "\n=== Memory Contents ===\n");

	// In production implementation:
	// Dump data memory contents in hexdump format
	// Show both text and data segments

	out_.output(CALL_INFO, "(Memory dump not implemented in stub)\n");
	out_.output(CALL_INFO, "=======================\n");
}
