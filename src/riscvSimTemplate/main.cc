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

/**
 * @file main.cc
 * @brief RISC-V RV32I ISA Simulator Template - Simplified Event-Driven Model
 *
 * @details
 * This is a **SIMPLIFIED TEMPLATE VERSION** of the RISC-V simulator designed for educational purposes.
 * Unlike the full simulator in src/riscv/, this template provides a streamlined architecture that omits
 * separate pipeline stages (like EXEStage) for easier comprehension and learning.
 *
 * **Key Differences from Full riscv Simulator:**
 * - Simplified: No separate EXEStage component - execution happens directly in CPU
 * - Direct execution: CPU executes instructions immediately without pipeline stages
 * - Educational focus: Easier to understand for beginners learning computer architecture
 * - Extensible: Can be extended to add more pipeline stages as learning progresses
 * - Single-cycle model: Each instruction completes in one logical cycle
 *
 * **System Architecture:**
 * @code
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        RISC-V Simulator Template                 │
 * │                     (Event-Driven Architecture)                  │
 * └─────────────────────────────────────────────────────────────────┘
 *
 *          ┌───────────────┐
 *          │   main.cc     │  Entry point, configures and runs simulation
 *          └───────┬───────┘
 *                  │
 *                  ▼
 *          ┌───────────────┐
 *          │    SOCTop     │  Top-level simulation controller
 *          └───────┬───────┘
 *                  │
 *                  ▼
 *    ┌─────────────────────────┐
 *    │         SOC             │  System-on-Chip integrator
 *    └────┬──────────────┬─────┘
 *         │              │
 *         ▼              ▼
 *    ┌─────────┐   ┌──────────┐
 *    │   CPU   │   │ Emulator │  ISA functional model
 *    │ (Timing)│   │(Behavior)│
 *    └────┬────┘   └──────────┘
 *         │
 *         ▼
 *    ┌─────────┐
 *    │ DataMem │   Memory subsystem
 *    └─────────┘
 * @endcode
 *
 * **Execution Flow:**
 * 1. main() creates SOCTop with configuration file path
 * 2. SOCTop::init() parses command-line arguments and initializes system
 * 3. SOCTop::run() starts event-driven simulation loop
 * 4. CPU executes instructions via ExecOneInstrEvent
 * 5. Each instruction: Fetch -> Execute -> Commit (in single cycle)
 * 6. SOCTop::finish() collects statistics and cleans up
 *
 * **Event-Driven Model:**
 * The simulator uses a discrete event simulation approach:
 * - ExecOneInstrEvent: Triggered to execute next instruction
 * - MemReqEvent: Handles memory read/write operations
 * - Events scheduled via global tick counter
 * - Each instruction schedules the next instruction's execution
 *
 * **RISC-V RV32I ISA Support:**
 * This template implements the complete RV32I base integer instruction set:
 * - Arithmetic: ADD, ADDI, SUB
 * - Logical: AND, ANDI, OR, ORI, XOR, XORI
 * - Shifts: SLL, SLLI, SRL, SRLI, SRA, SRAI
 * - Comparisons: SLT, SLTI, SLTU, SLTIU
 * - Branches: BEQ, BNE, BLT, BLTU, BGE, BGEU
 * - Jumps: JAL, JALR
 * - Memory: LB, LBU, LH, LHU, LW, SB, SH, SW
 * - Upper Immediate: LUI, AUIPC
 * - Pseudo-instructions: li, la, mv, j, ret, beqz, bnez
 *
 * **Configuration File:**
 * The simulator reads configuration from configs.json:
 * @code{.json}
 * {
 *   "Emulator": {
 *     "asm_file_path": "path/to/assembly.s",
 *     "memory_size": 1048576,
 *     "data_offset": 65536,
 *     "text_offset": 0
 *   }
 * }
 * @endcode
 *
 * **Usage Example:**
 * @code{.sh}
 * # Compile the simulator
 * make riscvSimTemplate
 *
 * # Run with assembly program
 * ./build/riscvSimTemplate input_program.s
 *
 * # The simulator will:
 * # 1. Parse the assembly file
 * # 2. Initialize memory and register file
 * # 3. Execute instructions event-by-event
 * # 4. Print register file state at completion
 * @endcode
 *
 * **Extension Possibilities:**
 * This template can be extended to add:
 * - Pipeline stages (IF, ID, EX, MEM, WB)
 * - Hazard detection and forwarding
 * - Branch prediction
 * - Cache hierarchies
 * - Multi-cycle instructions
 * - Performance counters
 *
 * @see SOC Main system-on-chip implementation
 * @see SOCTop Top-level simulation controller
 * @see CPU CPU timing model with instruction execution
 * @see Emulator ISA functional model for assembly parsing
 * @see src/riscv/ Full RISC-V simulator with complete pipeline
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @version 1.0
 * @copyright Apache License 2.0
 */

#include "ACALSim.hh"
#include "SOCTop.hh"

/**
 * @brief Main entry point for the RISC-V simulator template
 *
 * @details
 * Initializes and runs the event-driven RISC-V RV32I simulator. This function:
 * 1. Creates the top-level SOCTop object with configuration file
 * 2. Initializes the simulation environment (parses assembly, sets up memory)
 * 3. Runs the event-driven simulation loop until completion
 * 4. Finalizes and prints statistics
 *
 * **Execution Sequence:**
 * @code
 * main() Entry
 *    │
 *    ├─> Create SOCTop("configs.json")
 *    │   └─> Load simulation parameters
 *    │
 *    ├─> init(argc, argv)
 *    │   ├─> Parse command-line arguments
 *    │   ├─> Create SOC (CPU + Memory + Emulator)
 *    │   ├─> Parse assembly file
 *    │   └─> Schedule first ExecOneInstrEvent
 *    │
 *    ├─> run()
 *    │   └─> Execute event queue until empty
 *    │       ├─> ExecOneInstrEvent: Fetch & Execute instruction
 *    │       ├─> MemReqEvent: Handle memory operations
 *    │       └─> Schedule next events
 *    │
 *    └─> finish()
 *        ├─> Print register file state
 *        ├─> Collect performance statistics
 *        └─> Cleanup resources
 * @endcode
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *             argv[0]: Program name
 *             argv[1]: (Optional) Path to assembly file to execute
 *
 * @return 0 on successful completion
 *
 * @note The configuration file path is hardcoded to "src/riscvSimTemplate/configs.json"
 * @note Assembly file path can be overridden via command-line or configs.json
 *
 * @see SOCTop Top-level simulation controller class
 * @see SOC System-on-chip integration class
 * @see CPU Instruction execution engine
 *
 * @example
 * @code{.cpp}
 * // Typical invocation:
 * // ./riscvSimTemplate examples/fibonacci.s
 *
 * // Creates simulator -> Initializes -> Runs -> Reports results
 * @endcode
 */
int main(int argc, char** argv) {
	acalsim::top = std::make_shared<SOCTop>("src/riscvSimTemplate/configs.json");
	acalsim::top->init(argc, argv);
	acalsim::top->run();
	acalsim::top->finish();
	return 0;
}
