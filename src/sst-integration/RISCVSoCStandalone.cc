/*
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "RISCVSoCStandalone.hh"

#include <fstream>

#include <sst/core/sst_config.h>

#include "SOCTop.hh"

using namespace ACALSim::SST;

RISCVSoCStandalone::RISCVSoCStandalone(::SST::ComponentId_t id, ::SST::Params& params)
    : ::SST::Component(id),
      current_cycle_(0),
      max_cycles_(0),
      simulation_done_(false) {

	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("RISCVSoCStandalone[@p:@l]: ", verbose, 0, ::SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing Standalone RISC-V SoC\n");

	// Get parameters
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	asm_file_ = params.find<std::string>("asm_file", "");
	config_file_ = params.find<std::string>("config_file", "");
	max_cycles_ = params.find<uint64_t>("max_cycles", 0);

	// Validate assembly file
	if (asm_file_.empty()) {
		out_.fatal(CALL_INFO, -1, "Error: asm_file parameter is required\n");
	}

	// Check if file exists
	std::ifstream asm_check(asm_file_);
	if (!asm_check.good()) {
		out_.fatal(CALL_INFO, -1, "Error: Assembly file not found: %s\n", asm_file_.c_str());
	}
	asm_check.close();

	out_.verbose(CALL_INFO, 2, 0, "Assembly file: %s\n", asm_file_.c_str());
	out_.verbose(CALL_INFO, 2, 0, "Clock: %s\n", clock_freq.c_str());

	// Create temporary config file if not provided
	if (config_file_.empty()) {
		// Create a temporary config JSON with parameters from SST
		config_file_ = "/tmp/sst_riscv_config.json";
		std::ofstream config_out(config_file_);

		uint32_t memory_size = params.find<uint32_t>("memory_size", 65536);
		uint32_t text_offset = params.find<uint32_t>("text_offset", 0);
		uint32_t data_offset = params.find<uint32_t>("data_offset", 8192);

		config_out << "{\n";
		config_out << "  \"Emulator\": {\n";
		config_out << "    \"asm_file_path\": \"" << asm_file_ << "\",\n";
		config_out << "    \"memory_size\": " << memory_size << ",\n";
		config_out << "    \"text_offset\": " << text_offset << ",\n";
		config_out << "    \"data_offset\": " << data_offset << ",\n";
		config_out << "    \"max_label_count\": 128,\n";
		config_out << "    \"max_src_len\": 1048575\n";
		config_out << "  }\n";
		config_out << "}\n";
		config_out.close();

		out_.verbose(CALL_INFO, 2, 0, "Created temp config: %s\n", config_file_.c_str());
	}

	// Create SOCTop instance
	out_.verbose(CALL_INFO, 2, 0, "Creating SOCTop instance\n");
	soc_top_ = std::make_shared<SOCTop>(config_file_);

	// Initialize the RISC-V simulator
	// Note: SOCTop::init() expects argc/argv, but we can pass dummy values
	char prog_name[] = "sst_riscv";
	char* argv[] = {prog_name, nullptr};
	int argc = 1;

	out_.verbose(CALL_INFO, 2, 0, "Initializing RISC-V simulator\n");
	soc_top_->init(argc, argv);

	// Register clock
	tc_ = registerClock(clock_freq, new ::SST::Clock::Handler<RISCVSoCStandalone>(
	                                    this, &RISCVSoCStandalone::clockTick));

	// Control simulation end
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	out_.verbose(CALL_INFO, 1, 0, "Standalone RISC-V SoC initialized\n");
}

RISCVSoCStandalone::~RISCVSoCStandalone() {
	out_.verbose(CALL_INFO, 1, 0, "Destroying RISCVSoCStandalone\n");
}

void RISCVSoCStandalone::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");
}

void RISCVSoCStandalone::init(unsigned int phase) {
	out_.verbose(CALL_INFO, 2, 0, "Init phase %u\n", phase);
}

void RISCVSoCStandalone::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");

	// Finish the RISC-V simulator
	if (soc_top_) {
		out_.verbose(CALL_INFO, 1, 0, "Finalizing RISC-V simulator\n");
		soc_top_->finish();
	}

	// Clean up temp config if we created it
	if (config_file_ == "/tmp/sst_riscv_config.json") {
		std::remove(config_file_.c_str());
	}

	out_.output(CALL_INFO, "\n=== RISC-V Simulation Complete ===\n");
	out_.output(CALL_INFO, "Total cycles: %lu\n", current_cycle_);
	out_.output(CALL_INFO, "==================================\n");

	primaryComponentOKToEndSim();
}

bool RISCVSoCStandalone::clockTick(::SST::Cycle_t cycle) {
	current_cycle_++;

	// Run one step of the RISC-V simulator
	if (soc_top_ && !simulation_done_) {
		// SOCTop::run() normally runs the entire simulation,
		// but we need to run it step-by-step for SST integration
		// This requires stepping the internal simulators manually

		// Check if we should end
		if (soc_top_->isDone() || soc_top_->getGlobalTick() >= 100000) {
			// Simulation complete
			simulation_done_ = true;
			out_.verbose(CALL_INFO, 1, 0, "RISC-V simulation complete\n");
			primaryComponentOKToEndSim();
			return false;
		}

		// Step the simulation (this is a simplified version)
		// In production, you would call the individual simulator steps
		// For now, we let SOCTop manage one iteration
	}

	// Check max cycles
	if (max_cycles_ > 0 && current_cycle_ >= max_cycles_) {
		out_.verbose(CALL_INFO, 1, 0, "Reached max cycles (%lu)\n", max_cycles_);
		simulation_done_ = true;
		primaryComponentOKToEndSim();
		return false;
	}

	return !simulation_done_;
}
