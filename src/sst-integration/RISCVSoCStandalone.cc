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

#include <sst/core/sst_config.h>

#include <fstream>

// Include RISC-V SOCTop header
#include "../../src/riscv/include/SOCTop.hh"

// Include ACALSim headers for event-driven execution
#include "channel/SimChannel.hh"

using namespace ACALSim::SSTIntegration;
using namespace acalsim;

RISCVSoCStandalone::RISCVSoCStandalone(::SST::ComponentId_t id, ::SST::Params& params)
    : ::SST::Component(id), current_cycle_(0), max_cycles_(0), simulation_done_(false), ready_to_terminate_(false) {
	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("RISCVSoCStandalone[@p:@l]: ", verbose, 0, ::SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing Standalone RISC-V SoC\n");

	// Get parameters
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	asm_file_              = params.find<std::string>("asm_file", "");
	config_file_           = params.find<std::string>("config_file", "");
	max_cycles_            = params.find<uint64_t>("max_cycles", 0);

	// Validate assembly file
	if (asm_file_.empty()) { out_.fatal(CALL_INFO, -1, "Error: asm_file parameter is required\n"); }

	// Print early to confirm constructor runs
	printf("[RISC-V] Constructor running - asm_file=%s\n", asm_file_.c_str());
	fflush(stdout);

	// Check if file exists
	std::ifstream asm_check(asm_file_);
	if (!asm_check.good()) { out_.fatal(CALL_INFO, -1, "Error: Assembly file not found: %s\n", asm_file_.c_str()); }
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
		config_out << "  },\n";
		config_out << "  \"SOC\": {\n";
		config_out << "    \"name\": \"RISCVSoC\"\n";
		config_out << "  }\n";
		config_out << "}\n";
		config_out.close();

		out_.verbose(CALL_INFO, 2, 0, "Created temp config: %s\n", config_file_.c_str());
	}

	// Create SOCTop instance
	// NOTE: SOCTop's first parameter is misleadingly named "_name" but actually goes to SimTop's config file path
	out_.verbose(CALL_INFO, 2, 0, "Creating SOCTop instance\n");
	soc_top_ = std::make_shared<SOCTop>(config_file_);

	// Set global acalsim::top (required by ACALSim framework)
	acalsim::top = soc_top_;

	// Force single-threaded execution to avoid threading conflicts with SST
	// SST uses a single-threaded event loop, while ACALSim by default creates a worker thread pool
	// This must be called BEFORE init() to prevent worker thread creation
	out_.verbose(CALL_INFO, 2, 0, "Configuring single-threaded mode for SST integration\n");
	soc_top_->setSingleThreadedMode();

	// Defer initialization to SST's init() phase
	out_.verbose(CALL_INFO, 2, 0, "SOCTop created, will initialize in SST init() phase\n");

	// Register as primary component and tell SST not to end without us
	// This pattern follows SST's coreTestMessageGeneratorComponent example
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	// Register clock - the clock will self-schedule events to keep simulation alive
	tc_ =
	    registerClock(clock_freq, new ::SST::Clock::Handler2<RISCVSoCStandalone, &RISCVSoCStandalone::clockTick>(this));

	out_.verbose(CALL_INFO, 1, 0, "Standalone RISC-V SoC initialized\n");
}

RISCVSoCStandalone::~RISCVSoCStandalone() { out_.verbose(CALL_INFO, 1, 0, "Destroying RISCVSoCStandalone\n"); }

void RISCVSoCStandalone::setup() {
	printf("[RISC-V] setup() called\n");
	fflush(stdout);
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");

	// NOTE: Defer startRunning() to first clockTick() to avoid crash
	// Calling it here causes SST to crash immediately after setup() returns

	printf("[RISC-V] setup() complete\n");
	fflush(stdout);
	out_.verbose(CALL_INFO, 1, 0, "Setup complete - ready for clock ticks\n");
}

void RISCVSoCStandalone::init(unsigned int phase) {
	printf("[RISC-V] init() phase=%u\n", phase);
	fflush(stdout);
	out_.verbose(CALL_INFO, 2, 0, "Init phase %u\n", phase);

	// Initialize ACALSim simulator in first init phase
	if (phase == 0) {
		out_.verbose(CALL_INFO, 1, 0, "Initializing RISC-V simulator in phase 0\n");

		// Initialize SOCTop with dummy CLI arguments
		char  prog_name[] = "sst_riscv";
		char* argv[]      = {prog_name, nullptr};
		int   argc        = 1;

		out_.verbose(CALL_INFO, 2, 0, "Calling soc_top_->init()\n");
		soc_top_->init(argc, argv);

		printf("[RISC-V] soc_top_->init() complete\n");
		fflush(stdout);
		out_.verbose(CALL_INFO, 2, 0, "RISC-V simulator initialized\n");
	}
}

void RISCVSoCStandalone::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");

	// WORKAROUND: Intentionally leak soc_top_ and acalsim::top to avoid destruction crashes
	// The issue is that SOCTop's destructor tries to cleanup SimModules which then
	// try to access SimTop's RecycleContainer, but RecycleContainer is already being
	// destroyed. This is a fundamental lifecycle issue between ACALSim's architecture
	// (which expects init->run->finish lifecycle) and SST's clock-driven model.
	//
	// Since this component runs in a short-lived SST process, the memory leak is acceptable.
	// A proper fix would require redesigning the RISC-V integration to not use SOCTop,
	// or fixing the destruction order in ACALSim's SimTop/SimBase hierarchy.
	//
	// We leak by releasing the shared_ptr and intentionally not deleting the raw pointer.
	// This prevents ANY destructor from being called, avoiding the RecycleContainer crash.

	if (soc_top_) {
		out_.verbose(CALL_INFO, 2, 0, "Leaking SOCTop instance to avoid destruction crash\n");
		// Create a heap-allocated shared_ptr that will never be deleted
		// This keeps the reference count > 0 so the destructor never runs
		new std::shared_ptr<SOCTop>(soc_top_);
		soc_top_.reset();  // Release our reference, but heap copy keeps object alive
	}

	// Also leak the global reference
	if (acalsim::top) {
		out_.verbose(CALL_INFO, 2, 0, "Leaking global acalsim::top reference\n");
		// Create a heap-allocated shared_ptr that will never be deleted
		new std::shared_ptr<acalsim::SimTopBase>(acalsim::top);
		acalsim::top.reset();  // Release our reference, but heap copy keeps object alive
	}

	// Clean up temp config if we created it
	if (config_file_ == "/tmp/sst_riscv_config.json") { std::remove(config_file_.c_str()); }

	out_.output(CALL_INFO, "\n=== RISC-V Simulation Complete ===\n");
	out_.output(CALL_INFO, "Total cycles: %lu\n", current_cycle_);
	out_.output(CALL_INFO, "==================================\n");
}

bool RISCVSoCStandalone::clockTick(::SST::Cycle_t cycle) {
	try {
		out_.verbose(CALL_INFO, 3, 0, "=== clockTick() called: SST cycle=%lu, internal cycle=%lu ===\n", cycle,
		             current_cycle_);

		if (!soc_top_) {
			out_.verbose(CALL_INFO, 1, 0, "ERROR: soc_top_ is null, returning false\n");
			return false;
		}

		if (simulation_done_) {
			out_.verbose(CALL_INFO, 1, 0, "Simulation already done, returning false\n");
			return false;
		}

		// Check if we need to terminate (ExitEvent was issued in previous iteration)
		// Return TRUE to stop the clock - SST will end since we called primaryComponentOKToEndSim()
		if (ready_to_terminate_) {
			printf("[RISC-V] readyToTerminate=true, stopping clock\n");
			fflush(stdout);
			out_.verbose(CALL_INFO, 1, 0, "Ready to terminate (ExitEvent processed), stopping clock\n");
			return true;  // TRUE = stop clock, simulation will end
		}

		// First clockTick: start thread manager in RUNNING state
		if (current_cycle_ == 0) {
			printf("[RISC-V] First clockTick - calling startRunning()\n");
			fflush(stdout);
			soc_top_->startRunning();
			printf("[RISC-V] startRunning() complete\n");
			fflush(stdout);
		}

		current_cycle_++;

		// ====================================================================================
		// Execute one iteration of ACALSim's event-driven 2-phase simulation
		// This follows the pattern from SimTopBase::run() in libs/sim/SimTop.cc:324-473
		// ====================================================================================

		// [Phase 1]: Parallel simulator execution
		{
			printf("[RISC-V] Phase 1 starting\n");
			fflush(stdout);
			out_.verbose(CALL_INFO, 3, 0, "Phase 1: Starting parallel execution\n");

			soc_top_->startPhase1();

			// Control thread step function (may be empty for SOCTop)
			soc_top_->control_thread_step();

			soc_top_->finishPhase1();

			printf("[RISC-V] Phase 1 complete\n");
			fflush(stdout);
			out_.verbose(CALL_INFO, 3, 0, "Phase 1: Parallel execution complete\n");
		}

		// [Phase 2]: Synchronization and coordination
		{
			printf("[RISC-V] Phase 2 starting\n");
			fflush(stdout);
			out_.verbose(CALL_INFO, 3, 0, "Phase 2: Starting synchronization\n");

			printf("[RISC-V] Calling startPhase2()\n");
			fflush(stdout);
			soc_top_->startPhase2();

			printf("[RISC-V] Syncing pipe registers\n");
			fflush(stdout);
			// Sync pipe registers (IF->EXE, EXE->WB pipeline registers)
			if (soc_top_->getPipeRegisterManager()) { soc_top_->getPipeRegisterManager()->runSyncPipeRegister(); }

			printf("[RISC-V] Syncing SimPorts\n");
			fflush(stdout);
			// Sync SimPorts (inter-simulator communication)
			soc_top_->runSyncSimPort();

			printf("[RISC-V] Running inter-iteration update\n");
			fflush(stdout);
			// Update simulator activity tracking (pending event bit masks)
			soc_top_->runInterIterationUpdate();

			printf("[RISC-V] Getting global tick\n");
			fflush(stdout);
			// Calculate next event tick and fast-forward
			Tick current_tick = soc_top_->getGlobalTick();
			Tick next_tick    = current_tick + 1;

			printf("[RISC-V] Checking if all simulators done\n");
			fflush(stdout);
			bool all_done = soc_top_->isAllSimulatorDone();
			out_.verbose(CALL_INFO, 2, 0, "Simulator status: current_tick=%lu, all_done=%s\n", current_tick,
			             all_done ? "TRUE" : "FALSE");

			if (all_done) {
				printf("[RISC-V] All simulators done - starting 2-phase termination\n");
				fflush(stdout);
				// ACALSim's 2-phase termination: issue ExitEvent now, terminate next iteration
				out_.verbose(CALL_INFO, 1, 0, "All simulators done at tick %lu (SST cycle %lu) - issuing ExitEvent\n",
				             current_tick, current_cycle_);

				soc_top_->issueExitEvent(next_tick);
				ready_to_terminate_ = true;  // Flag for next iteration

				// Tell SST we're ready to end (but simulation continues one more iteration)
				printf("[RISC-V] Calling primaryComponentOKToEndSim()\n");
				fflush(stdout);
				primaryComponentOKToEndSim();
			} else {
				printf("[RISC-V] Getting fast-forward cycles\n");
				fflush(stdout);
				// Get the next tick that has pending events across all simulators
				next_tick = soc_top_->getFastForwardCycles();

				printf("[RISC-V] Fast-forward to tick %lu\n", next_tick);
				fflush(stdout);
				out_.verbose(CALL_INFO, 2, 0, "Fast-forwarding from tick %lu to %lu (delta=%lu)\n", current_tick,
				             next_tick, next_tick - current_tick);
			}

			printf("[RISC-V] Fast-forwarding global tick to %lu\n", next_tick);
			fflush(stdout);
			// Advance global tick to next event
			soc_top_->fastForwardGlobalTick(next_tick);

			printf("[RISC-V] Toggling channel dual-queue status\n");
			fflush(stdout);
			// Toggle dual-queue channel buffers (PING <-> PONG)
			SimChannelGlobal::toggleChannelDualQueueStatus();

			printf("[RISC-V] Finishing Phase 2\n");
			fflush(stdout);
			soc_top_->finishPhase2();

			printf("[RISC-V] Phase 2 complete\n");
			fflush(stdout);
			out_.verbose(CALL_INFO, 3, 0, "Phase 2: Synchronization complete\n");
		}

		printf("[RISC-V] Phases complete, checking conditions\n");
		fflush(stdout);

		// Log progress periodically
		if (current_cycle_ % 1000 == 0) {
			out_.verbose(CALL_INFO, 1, 0, "SST cycle: %lu, ACALSim tick: %lu\n", current_cycle_,
			             soc_top_->getGlobalTick());
		}

		// Check max cycles limit
		if (max_cycles_ > 0 && current_cycle_ >= max_cycles_) {
			out_.verbose(CALL_INFO, 1, 0, "Reached max SST cycles (%lu), ending simulation\n", max_cycles_);
			printf("[RISC-V] Max cycles reached, calling primaryComponentOKToEndSim()\n");
			fflush(stdout);
			primaryComponentOKToEndSim();
			return true;  // TRUE = stop clock immediately
		}

		// Continue simulation - return FALSE to keep clock ticking
		// In SST: FALSE = continue clock, TRUE = stop clock
		printf("[RISC-V] clockTick() returning FALSE to continue (cycle=%lu, tick=%lu)\n", current_cycle_,
		       soc_top_->getGlobalTick());
		fflush(stdout);
		out_.verbose(CALL_INFO, 3, 0, "=== clockTick() returning FALSE: continue clock ===\n");
		return false;  // FALSE = continue clock
	} catch (const std::exception& e) {
		printf("[RISC-V] EXCEPTION in clockTick: %s\n", e.what());
		out_.fatal(CALL_INFO, -1, "Exception in clockTick: %s\n", e.what());
		return false;
	} catch (...) {
		printf("[RISC-V] UNKNOWN EXCEPTION in clockTick\n");
		out_.fatal(CALL_INFO, -1, "Unknown exception in clockTick\n");
		return false;
	}
}
