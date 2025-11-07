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
 * @brief System-on-Chip (SOC) Integration and Orchestration Module
 *
 * @details
 * This file implements the SOC class, which serves as the central orchestrator
 * for the RISC-V simulator. The SOC integrates three major components:
 * - CPU timing model (instruction execution)
 * - Data memory timing model (load/store operations)
 * - ISA emulator (assembly parsing and functional model)
 *
 * **Component Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                         SOC                                 │
 * │  (System-on-Chip - Top-Level Hardware Integration)         │
 * │                                                             │
 * │  ┌───────────────────────────────────────────────────┐     │
 * │  │              ISA Emulator                         │     │
 * │  │  (Functional Model - Pre-Simulation)             │     │
 * │  │                                                   │     │
 * │  │  1. Parse assembly file (.s)                     │     │
 * │  │  2. Resolve labels and symbols                   │     │
 * │  │  3. Expand pseudo-instructions                   │     │
 * │  │  4. Initialize IMEM with parsed instructions     │     │
 * │  │  5. Initialize DMEM with .data section          │     │
 * │  └───────────────────────────────────────────────────┘     │
 * │                                                             │
 * │  ┌───────────────────────────────────────────────────┐     │
 * │  │              CPU                                  │     │
 * │  │  (Timing Model - Event-Driven Execution)         │     │
 * │  │                                                   │     │
 * │  │  - Fetch instructions from IMEM                  │     │
 * │  │  - Decode and execute operations                 │     │
 * │  │  - Manage 32 general-purpose registers          │     │
 * │  │  - Generate memory requests                      │     │
 * │  │  - Send InstPackets to pipeline stages          │     │
 * │  │                                                   │     │
 * │  │  Master Port: "sIF-m" ──────┐                    │     │
 * │  │  (connects to IFStage)       │                    │     │
 * │  │                              │                    │     │
 * │  │  Downstream Port: "DSDmem" ──┼────┐              │     │
 * │  └──────────────────────────────┼────┼──────────────┘     │
 * │                                 │    │                     │
 * │                                 │    │                     │
 * │  ┌──────────────────────────────┼────▼──────────────┐     │
 * │  │              Data Memory     │                    │     │
 * │  │  (Timing Model - Memory Subsystem)               │     │
 * │  │                                                   │     │
 * │  │  - Handles load/store operations                 │     │
 * │  │  - Byte-addressable memory                       │     │
 * │  │  - Supports byte, half-word, word access         │     │
 * │  │  - Configurable memory size                      │     │
 * │  │                                                   │     │
 * │  │  Upstream Port: "USCPU" (from CPU)               │     │
 * │  └───────────────────────────────────────────────────┘     │
 * │                                                             │
 * │                                                             │
 * │  ┌─────────────── Backpressure Flow ──────────────┐       │
 * │  │                                                  │       │
 * │  │  IFStage (slave port full)                     │       │
 * │  │     │                                            │       │
 * │  │     ▼                                            │       │
 * │  │  SOC::masterPortRetry()                         │       │
 * │  │     │                                            │       │
 * │  │     ▼                                            │       │
 * │  │  CPU::retrySendInstPacket()                     │       │
 * │  │     │                                            │       │
 * │  │     ▼                                            │       │
 * │  │  Resend pendingInstPacket                       │       │
 * │  │     │                                            │       │
 * │  │     ▼                                            │       │
 * │  │  Schedule next ExecOneInstrEvent                │       │
 * │  └──────────────────────────────────────────────────┘       │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Initialization Sequence:**
 *
 * The SOC follows a multi-phase initialization process:
 *
 * ```
 * Phase 1: Module Registration (SOC::registerModules)
 * ──────────────────────────────────────────────────────
 *   1. Query memory size from config
 *   2. Create DataMemory instance
 *   3. Create Emulator instance
 *   4. Create CPU instance
 *   5. Register CPU and DataMemory as child modules
 *   6. Establish port connections:
 *      - CPU.DSDmem → DataMemory
 *      - DataMemory.USCPU → CPU
 *
 * Phase 2: Simulation Initialization (SOC::simInit)
 * ──────────────────────────────────────────────────────
 *   1. Load assembly file path from config
 *   2. Parse assembly file:
 *      - Extract .text section → IMEM
 *      - Extract .data section → DMEM
 *      - Resolve all labels
 *      - Expand pseudo-instructions
 *   3. Normalize labels (convert to absolute addresses)
 *   4. Initialize all child modules (CPU, DataMemory)
 *   5. Create initial ExecOneInstrEvent
 *   6. Schedule event at tick T=1
 *
 * Phase 3: Execution (Event-Driven Loop)
 * ──────────────────────────────────────────────────────
 *   - Managed by SOCTop::run()
 *   - Processes events in chronological order
 *   - CPU executes one instruction per event
 *   - Pipeline stages advance packets
 *
 * Phase 4: Cleanup (SOC::cleanup)
 * ──────────────────────────────────────────────────────
 *   1. Print final register file state
 *   2. Display execution statistics
 * ```
 *
 * **Port Connection Model:**
 *
 * The SOC establishes hierarchical port connections between modules:
 *
 * ```
 * CPU (Master) ─── DSDmem ───▶ DataMemory (Slave)
 *     Sends:                   Receives:
 *     - MemReadReqPacket        - Load requests
 *     - MemWriteReqPacket       - Store requests
 *
 * DataMemory ─── USCPU ───▶ CPU
 *     Sends:               Receives:
 *     - Data responses      - Load data
 *
 * CPU (Master) ─── sIF-m ───▶ IFStage (Slave)
 *     Sends:                  Receives:
 *     - InstPacket             - Completed instructions
 *                               - For pipeline visualization
 * ```
 *
 * **Event Model:**
 *
 * The SOC manages event scheduling and execution:
 *
 * 1. **ExecOneInstrEvent**:
 *    - Triggered at each cycle
 *    - Causes CPU to execute one instruction
 *    - Self-scheduling (creates next event after completion)
 *    - Terminates on HCF instruction
 *
 * 2. **Event Flow**:
 * ```
 * T=1: ExecOneInstrEvent
 *      └─ CPU::execOneInstr()
 *         ├─ Fetch from IMEM
 *         ├─ Execute operation
 *         ├─ Access DMEM (if load/store)
 *         ├─ Commit instruction
 *         └─ Schedule next ExecOneInstrEvent @ T=2
 *
 * T=2: ExecOneInstrEvent (next instruction)
 *      └─ Repeat cycle...
 * ```
 *
 * **Backpressure Handling:**
 *
 * The SOC implements a retry mechanism for handling backpressure:
 *
 * 1. **Normal Flow** (No Backpressure):
 *    - CPU sends InstPacket to IFStage
 *    - IFStage accepts packet
 *    - CPU schedules next ExecOneInstrEvent
 *
 * 2. **Backpressure Flow**:
 *    - CPU tries to send InstPacket to IFStage
 *    - IFStage slave port is full → returns false
 *    - CPU stores packet in pendingInstPacket
 *    - CPU does NOT schedule next event (stalls)
 *
 * 3. **Retry Flow**:
 *    - IFStage processes packet, frees space
 *    - IFStage calls masterPortRetry() on its slave port
 *    - Triggers SOC::masterPortRetry()
 *    - SOC identifies port by name ("sIF-m")
 *    - SOC calls CPU::retrySendInstPacket()
 *    - CPU resends pendingInstPacket
 *    - On success, CPU schedules next ExecOneInstrEvent
 *
 * **Memory Configuration:**
 *
 * Memory layout is controlled by configuration parameters:
 *
 * ```
 * Config Parameter        Default    Description
 * ────────────────────────────────────────────────────
 * memory_size            65536      Total DMEM size (bytes)
 * text_offset            0          Start of .text section
 * data_offset            32768      Start of .data section
 *
 * Memory Map:
 * ┌──────────────────┐ 0x00000000
 * │  .text (IMEM)    │ ← Instructions
 * │                  │
 * ├──────────────────┤ 0x00008000 (data_offset)
 * │  .data (DMEM)    │ ← Data variables
 * │                  │
 * │  Heap/Stack      │
 * │  (if used)       │
 * └──────────────────┘ 0x00010000 (65536)
 * ```
 *
 * **Usage Example:**
 *
 * ```cpp
 * // SOC is instantiated by SOCTop
 * auto soc = new SOC("RISC-V SOC");
 *
 * // Register modules and establish connections
 * soc->registerModules();
 *
 * // Initialize with assembly file
 * soc->simInit();
 * // → Parses assembly
 * // → Initializes IMEM/DMEM
 * // → Schedules first instruction
 *
 * // Simulation runs (managed by SOCTop)
 * // ...
 *
 * // Cleanup after simulation
 * soc->cleanup();
 * // → Prints register file
 * // → Displays statistics
 * ```
 *
 * **Key Responsibilities:**
 * - Module instantiation and lifecycle management
 * - Port connection establishment
 * - Assembly file parsing orchestration
 * - Initial event scheduling
 * - Backpressure/retry coordination
 * - Final state reporting
 *
 * @see CPU for instruction execution timing model
 * @see DataMemory for memory subsystem timing model
 * @see Emulator for assembly parsing functional model
 * @see IFStage for instruction fetch pipeline stage
 * @see ExecOneInstrEvent for instruction execution events
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "SOC.hh"

#include "event/ExecOneInstrEvent.hh"

/**
 * @brief SOC constructor
 *
 * @param _name Name identifier for this SOC instance
 *
 * @details
 * Constructs the System-on-Chip module as a CPPSimBase instance.
 * The actual component instantiation occurs in registerModules().
 */
SOC::SOC(std::string _name) : acalsim::CPPSimBase(_name) {}

/**
 * @brief Register and connect all hardware modules
 *
 * @details
 * This method creates and configures the three major components of the SOC:
 *
 * **1. Data Memory (DMEM)**:
 *    - Timing model for load/store operations
 *    - Size configured via "memory_size" parameter
 *    - Byte-addressable memory array
 *
 * **2. ISA Emulator**:
 *    - Functional model for assembly parsing
 *    - Converts assembly text → instruction structures
 *    - Resolves labels and pseudo-instructions
 *    - Not added as a child module (pre-simulation only)
 *
 * **3. CPU**:
 *    - Single-cycle timing model
 *    - Executes instructions from IMEM
 *    - Accesses DMEM for load/store operations
 *    - Sends InstPackets to pipeline stages
 *
 * **Port Connections Established:**
 * ```
 * CPU.DSDmem ────▶ DataMemory (downstream connection)
 *                  - CPU can send memory requests
 *
 * DataMemory.USCPU ────▶ CPU (upstream connection)
 *                        - DataMemory can respond to CPU
 * ```
 *
 * **Module Registration Order:**
 * 1. Create DataMemory with configured size
 * 2. Create Emulator for assembly parsing
 * 3. Create CPU with reference to this SOC
 * 4. Add CPU to child modules list
 * 5. Add DataMemory to child modules list
 * 6. Connect CPU → DataMemory ports
 *
 * @note The ISA Emulator is NOT added as a child module because it only
 *       performs pre-simulation parsing and is not part of the timing model
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
 * @brief Initialize simulation state and schedule first event
 *
 * @details
 * This method performs critical pre-simulation initialization:
 *
 * **Phase 1: Assembly Parsing**
 * - Loads assembly file path from configuration
 * - Calls Emulator::parse() to process assembly file:
 *   ```
 *   Input:  program.s (assembly text)
 *   Output: IMEM (parsed instruction structures)
 *           DMEM (initialized data section)
 *   ```
 * - Parsing steps:
 *   1. Read assembly file line by line
 *   2. Identify .text and .data sections
 *   3. Parse instructions into instr structures
 *   4. Store instructions in CPU's IMEM
 *   5. Copy .data section into DMEM
 *   6. Record label locations
 *
 * **Phase 2: Label Normalization**
 * - Converts symbolic labels → absolute addresses
 * - Updates branch/jump target addresses
 * - Resolves pseudo-instruction expansions
 * - Example:
 *   ```
 *   Before: beq x1, x2, loop    (symbolic)
 *   After:  beq x1, x2, 0x0020  (absolute address)
 *   ```
 *
 * **Phase 3: Module Initialization**
 * - Calls init() on all registered child modules:
 *   - CPU::init() - Initialize PC, registers
 *   - DataMemory::init() - Prepare memory subsystem
 *
 * **Phase 4: Initial Event Scheduling**
 * - Creates the first ExecOneInstrEvent
 * - Assigns event ID = 1 (first instruction)
 * - Associates event with CPU instance
 * - Schedules event at tick T=1 (next cycle)
 * - This kickstarts the event-driven simulation loop
 *
 * **Event Scheduling Mechanism:**
 * ```
 * T=0: simInit() executes
 *      └─ Parse assembly → IMEM/DMEM
 *      └─ Schedule ExecOneInstrEvent @ T=1
 *
 * T=1: ExecOneInstrEvent triggers
 *      └─ CPU::execOneInstr() executes
 *      └─ Schedules next event @ T=2 (if not HCF)
 *
 * T=2: Next ExecOneInstrEvent triggers...
 * ```
 *
 * @note This method is called during SOCTop::init() before run() starts
 * @see Emulator::parse() for assembly parsing details
 * @see Emulator::normalize_labels() for label resolution
 * @see ExecOneInstrEvent for instruction execution event
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
 * @brief Cleanup and finalization after simulation completes
 *
 * @details
 * Performs post-simulation tasks:
 *
 * **1. Register File Dump**:
 * - Calls CPU::printRegfile() to display final register state
 * - Shows all 32 general-purpose registers (x0-x31)
 * - Format: hexadecimal values, 8 registers per row
 * - Example output:
 *   ```
 *   Register File Snapshot:
 *   x00:0x00000000 x01:0x00000005 x02:0x0000000a ...
 *   x08:0x00000000 x09:0x00000000 x10:0x0000000f ...
 *   ```
 *
 * **2. Statistics Logging**:
 * - Logs cleanup message via CLASS_INFO macro
 * - Can be extended to include:
 *   - Total instruction count
 *   - Execution cycles
 *   - Memory access statistics
 *   - Branch prediction accuracy
 *
 * @note Called by SOCTop::finish() after simulation loop completes
 * @see CPU::printRegfile() for register display implementation
 */
void SOC::cleanup() {
	this->cpu->printRegfile();
	CLASS_INFO << "SOC::cleanup() ";
}

/**
 * @brief Handle backpressure retry callbacks from downstream modules
 *
 * @param port The master port that can now accept packets after being stalled
 *
 * @details
 * This callback is invoked when a previously-full downstream port becomes
 * available to accept new packets. It implements the retry mechanism for
 * handling backpressure in the simulation.
 *
 * **Backpressure Scenario:**
 * ```
 * T=N: CPU tries to send InstPacket to IFStage
 *      └─ IFStage slave port full → returns false
 *      └─ CPU stores packet in pendingInstPacket
 *      └─ CPU stalls (no new event scheduled)
 *
 * T=N+k: IFStage processes packet, frees space
 *        └─ IFStage calls retry() on slave port
 *        └─ Triggers SOC::masterPortRetry()
 *        └─ SOC identifies port "sIF-m"
 *        └─ SOC calls CPU::retrySendInstPacket()
 *        └─ CPU resends pendingInstPacket
 *        └─ On success, CPU schedules next event
 * ```
 *
 * **Port Identification:**
 * - "sIF-m": Master port from SOC to IFStage
 *   - Carries InstPackets for pipeline visualization
 *   - When this port retries, forward to CPU
 *
 * **Retry Flow:**
 * 1. Receive retry callback with port reference
 * 2. Check port name to identify which connection
 * 3. Forward retry to appropriate module (CPU)
 * 4. Module attempts to resend pending packet
 * 5. If successful, resume normal execution flow
 *
 * **Why Retry Mechanism is Needed:**
 * - Models realistic hardware backpressure
 * - Prevents data loss when buffers are full
 * - Enables timing-accurate stall modeling
 * - Demonstrates producer-consumer synchronization
 *
 * @note Currently only handles "sIF-m" port, but can be extended for other ports
 * @see CPU::retrySendInstPacket() for retry implementation
 */
void SOC::masterPortRetry(acalsim::MasterPort* port) {
	if (port->getName() == "sIF-m") { this->cpu->retrySendInstPacket(port); }
}
