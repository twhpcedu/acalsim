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

/**
 * @file SOC.cc
 * @brief System-on-Chip Integration for RISC-V Simulator Template
 *
 * @details
 * This file implements the SOC (System-on-Chip) class, which serves as the central
 * integration point for all hardware components in the simplified RISC-V simulator template.
 *
 * **TEMPLATE SIMPLIFICATION:**
 * Unlike the full riscv simulator (src/riscv/), this template version uses a simplified
 * architecture where the CPU directly executes instructions without separate pipeline stages.
 * This makes it ideal for:
 * - Learning computer architecture fundamentals
 * - Understanding event-driven simulation
 * - Building a foundation before tackling pipelined designs
 *
 * **Component Architecture:**
 * @code
 *                    ┌─────────────────────────────────┐
 *                    │          SOC Module              │
 *                    │  (System-on-Chip Integrator)     │
 *                    └─────────────────────────────────┘
 *                               │
 *              ┌────────────────┼────────────────┐
 *              │                │                │
 *              ▼                ▼                ▼
 *      ┌──────────────┐  ┌──────────┐   ┌────────────┐
 *      │     CPU      │  │ Emulator │   │  DataMemory│
 *      │   (Timing)   │  │(Behavior)│   │  (Storage) │
 *      └──────┬───────┘  └──────────┘   └─────┬──────┘
 *             │                                │
 *             │   MasterPort "DSDmem"          │
 *             └────────────────────────────────┘
 *                        (Port Connection)
 * @endcode
 *
 * **Key Components:**
 * 1. **CPU (Timing Model):**
 *    - Executes instructions from instruction memory
 *    - Maintains register file and program counter
 *    - Handles instruction fetch, decode, and execution
 *    - Single-cycle execution model (no pipeline stages)
 *
 * 2. **Emulator (Functional Model):**
 *    - Parses RISC-V assembly files
 *    - Converts assembly to internal instruction representation
 *    - Initializes instruction and data memory
 *    - Resolves labels and pseudo-instructions
 *
 * 3. **DataMemory (Storage Model):**
 *    - Provides byte-addressable memory space
 *    - Supports load/store operations (LB, LH, LW, SB, SH, SW)
 *    - Configurable size via configs.json
 *    - Shared between instruction and data segments
 *
 * **Initialization Sequence:**
 * @code
 * SOC Constructor
 *    │
 *    ├─> registerModules()
 *    │   ├─> Create DataMemory (configurable size)
 *    │   ├─> Create Emulator (ISA functional model)
 *    │   ├─> Create CPU (single-cycle timing model)
 *    │   ├─> Add modules to simulation
 *    │   └─> Connect CPU to DataMemory via ports
 *    │
 *    └─> simInit()
 *        ├─> Parse assembly file via Emulator
 *        ├─> Initialize instruction memory in CPU
 *        ├─> Initialize data memory
 *        ├─> Resolve labels and pseudo-instructions
 *        ├─> Initialize all child modules
 *        └─> Schedule first ExecOneInstrEvent at tick=1
 * @endcode
 *
 * **Event-Driven Execution:**
 * The SOC coordinates event-driven simulation:
 * - ExecOneInstrEvent: Triggers CPU to execute one instruction
 * - Each instruction completion schedules the next ExecOneInstrEvent
 * - HCF (Halt and Catch Fire) instruction stops event scheduling
 * - Global tick counter tracks simulation time
 *
 * **Memory Organization:**
 * @code
 * Address Space (Configurable):
 * ┌──────────────────┐ 0x00000000
 * │  Text Segment    │ (Instructions)
 * │  (text_offset)   │
 * ├──────────────────┤ 0x00010000 (data_offset)
 * │  Data Segment    │ (Initialized data)
 * │                  │
 * ├──────────────────┤
 * │  Stack/Heap      │ (Dynamic allocation)
 * │                  │
 * └──────────────────┘ memory_size
 * @endcode
 *
 * **Differences from Full Simulator:**
 * | Feature              | Template (riscvSimTemplate) | Full (src/riscv/)    |
 * |----------------------|----------------------------|----------------------|
 * | Pipeline Stages      | None (single-cycle)        | IF, EX separate      |
 * | Instruction Exec     | Direct in CPU              | Staged through IF/EX |
 * | Complexity           | Simplified                 | Production-grade     |
 * | Educational Use      | Beginner-friendly          | Advanced learning    |
 * | Extension Path       | Add stages incrementally   | Already complete     |
 *
 * **Port Connections:**
 * - CPU MasterPort "DSDmem" -> DataMemory SlavePort "USCPU"
 * - Enables memory read/write operations from CPU
 * - Backpressure mechanism for flow control
 *
 * @see CPU Single-cycle CPU implementation
 * @see Emulator Assembly parser and ISA functional model
 * @see DataMemory Memory subsystem implementation
 * @see ExecOneInstrEvent Event that triggers instruction execution
 * @see main.cc Entry point and simulation flow
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @version 1.0
 * @copyright Apache License 2.0
 */

#include "SOC.hh"

#include "event/ExecOneInstrEvent.hh"

/**
 * @brief Constructor for the SOC class
 *
 * @details
 * Initializes the System-on-Chip base class. The actual hardware component
 * creation is deferred to registerModules() which is called during init().
 *
 * @param _name Identifier for this SOC instance (for logging/debugging)
 */
SOC::SOC(std::string _name) : acalsim::CPPSimBase(_name) {}

/**
 * @brief Registers and connects all hardware modules in the system
 *
 * @details
 * Creates and interconnects the three main components of the simulator:
 * 1. DataMemory: Byte-addressable memory for instructions and data
 * 2. Emulator: ISA functional model for assembly parsing
 * 3. CPU: Timing model for instruction execution
 *
 * **Port Connections:**
 * - CPU.MasterPort("DSDmem") -> DataMemory.SlavePort("USCPU")
 * - Enables CPU to send memory read/write requests to DataMemory
 * - Bi-directional communication for data transfer and backpressure
 *
 * **Configuration:**
 * Memory size is read from configs.json:
 * @code{.json}
 * {
 *   "Emulator": {
 *     "memory_size": 1048576  // 1MB default
 *   }
 * }
 * @endcode
 *
 * @note This function is called during SOC::init() after configuration is loaded
 * @note Module order matters: modules are initialized in registration order
 *
 * @see DataMemory Memory subsystem implementation
 * @see CPU Single-cycle CPU implementation
 * @see Emulator Assembly parser and ISA model
 */
void SOC::registerModules() {
	// Get the maximal memory footprint size in the Emulator Configuration
	size_t mem_size = acalsim::top->getParameter<int>("Emulator", "memory_size");

	// Data Memory Timing Model
	this->dmem = new DataMemory("Data Memory", mem_size);

	// Instruction Set Architecture Emulator (Functional Model)
	this->isaEmulator = new Emulator("RISCV RV32I Emulator");

	// CPU Timing Model
	this->cpu = new CPU("Single-Cycle CPU Model", this);

	// register modules
	this->addModule(this->cpu);
	this->addModule(this->dmem);

	// connect modules (connected_module, master port name, slave port name)
	this->cpu->addDownStream(this->dmem, "DSDmem");
	this->dmem->addUpStream(this->cpu, "USCPU");
}

/**
 * @brief Initializes the simulation environment
 *
 * @details
 * Performs critical initialization steps before simulation begins:
 * 1. Parses the RISC-V assembly file via Emulator
 * 2. Populates instruction memory in CPU
 * 3. Initializes data memory with .data segment values
 * 4. Resolves all labels to absolute addresses
 * 5. Initializes all registered modules (CPU, DataMemory)
 * 6. Schedules the first instruction execution event
 *
 * **Assembly Parsing Flow:**
 * @code
 * simInit()
 *    │
 *    ├─> Read asm_file_path from config
 *    │
 *    ├─> isaEmulator->parse()
 *    │   ├─> Parse .text section -> CPU instruction memory
 *    │   ├─> Parse .data section -> DataMemory
 *    │   ├─> Handle pseudo-instructions (li, la, mv, j, ret, etc.)
 *    │   └─> Store label locations
 *    │
 *    ├─> isaEmulator->normalize_labels()
 *    │   ├─> Convert label references to addresses
 *    │   ├─> Resolve branch/jump targets
 *    │   └─> Finalize instruction operands
 *    │
 *    ├─> Initialize child modules
 *    │   ├─> CPU::init()
 *    │   └─> DataMemory::init()
 *    │
 *    └─> Schedule first ExecOneInstrEvent at tick=1
 *        └─> Kick-starts event-driven simulation
 * @endcode
 *
 * **Event Recycling:**
 * Uses RecycleContainer for efficient event object reuse:
 * - Events are acquired from pool instead of new/delete
 * - Reduces memory allocation overhead
 * - Events returned to pool after processing
 *
 * **Configuration Requirements:**
 * Expects "asm_file_path" in configs.json:
 * @code{.json}
 * {
 *   "Emulator": {
 *     "asm_file_path": "path/to/program.s"
 *   }
 * }
 * @endcode
 *
 * @note Called automatically by SOC::init() after registerModules()
 * @note The first event is scheduled at tick=1, not tick=0
 * @note All modules must be registered before calling simInit()
 *
 * @see Emulator::parse() Assembly file parsing implementation
 * @see Emulator::normalize_labels() Label resolution
 * @see ExecOneInstrEvent Event that drives instruction execution
 */
void SOC::simInit() {
	CLASS_INFO << name + " SOC::simInit()!";

	// Initialize the ISA Emulator
	// Parse assmebly file and initialize data memory and instruction memory
	std::string asm_file_path = acalsim::top->getParameter<std::string>("Emulator", "asm_file_path");

	this->isaEmulator->parse(asm_file_path, ((uint8_t*)this->dmem->getMemPtr()), this->cpu->getIMemPtr());
	this->isaEmulator->normalize_labels(this->cpu->getIMemPtr());

	// Initialize all child modules
	for (auto& module : this->modules) { module->init(); }

	// Inject trigger event
	auto               rc    = acalsim::top->getRecycleContainer();
	ExecOneInstrEvent* event = rc->acquire<ExecOneInstrEvent>(&ExecOneInstrEvent::renew, 1 /*id*/, this->cpu);
	this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
}

/**
 * @brief Performs cleanup operations after simulation completion
 *
 * @details
 * Called at the end of simulation to:
 * - Print final register file state for verification
 * - Log cleanup completion
 * - Prepare for graceful shutdown
 *
 * **Register File Output:**
 * Prints all 32 RISC-V registers (x0-x31) in hexadecimal format,
 * showing the final architectural state after program execution.
 *
 * @note Called automatically by SOCTop::finish()
 * @note Does not deallocate memory (handled by destructors)
 *
 * @see CPU::printRegfile() Register file printing implementation
 */
void SOC::cleanup() {
	this->cpu->printRegfile();
	CLASS_INFO << "SOC::cleanup() ";
}

/**
 * @brief Handles retry requests from master ports when backpressure is released
 *
 * @details
 * Implements the backpressure handling mechanism in the simulator. When a master
 * port fails to push data due to downstream congestion, this callback is invoked
 * when the path becomes available again.
 *
 * **Backpressure Flow:**
 * @code
 * CPU tries to send InstPacket
 *    │
 *    ├─> IF MasterPort full (backpressure)
 *    │   ├─> Store packet in CPU::pendingInstPacket
 *    │   └─> Wait for retry callback
 *    │
 *    └─> When IF port pops entry
 *        ├─> SlavePort signals retry
 *        ├─> SOC::masterPortRetry() called
 *        └─> CPU::retrySendInstPacket() resends pending packet
 * @endcode
 *
 * **Port Identification:**
 * Currently handles:
 * - "sIF-m": Master port connecting SOC to IFStage
 *
 * @param _port Pointer to the master port requesting retry
 *
 * @note In this simplified template, the IF stage is minimal
 * @note Full simulator (src/riscv/) has more complex retry logic
 *
 * @see CPU::retrySendInstPacket() Retry handler in CPU
 * @see MasterPort Port abstraction for inter-module communication
 */
void SOC::masterPortRetry(acalsim::MasterPort* _port) {
	if (_port->getName() == "sIF-m") { this->cpu->retrySendInstPacket(_port); }
}
