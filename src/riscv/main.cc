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
 * @brief RISC-V RV32I ISA Simulator - Event-Driven Timing Model Entry Point
 *
 * @details
 * This file implements the main entry point for a complete RISC-V RV32I instruction set
 * architecture (ISA) simulator built on the ACALSim event-driven simulation framework.
 * The simulator demonstrates advanced computer architecture concepts including:
 * - Event-driven timing-accurate simulation
 * - Single-cycle CPU execution model with pipeline stage visualization
 * - Separate instruction fetch (IF) and execute (EXE) stages
 * - Backpressure handling and retry mechanisms
 * - Memory hierarchy modeling (instruction and data memory)
 *
 * **System Architecture Overview:**
 * ```
 * ┌────────────────────────────────────────────────────────────────┐
 * │                        SOCTop                                  │
 * │  (Top-Level Simulation Controller)                            │
 * │                                                                │
 * │  ┌──────────────────────────────────────────────────────┐     │
 * │  │                    SOC                               │     │
 * │  │  (System-on-Chip Orchestrator)                      │     │
 * │  │                                                      │     │
 * │  │  ┌────────────────────────────────────────────┐     │     │
 * │  │  │            CPU                             │     │     │
 * │  │  │  (Single-Cycle Timing Model)              │     │     │
 * │  │  │                                            │     │     │
 * │  │  │  ┌──────────────────────────────────┐    │     │     │
 * │  │  │  │  Register File (RF)              │    │     │     │
 * │  │  │  │  x0  (zero) - always 0           │    │     │     │
 * │  │  │  │  x1  (ra)   - return address     │    │     │     │
 * │  │  │  │  x2  (sp)   - stack pointer      │    │     │     │
 * │  │  │  │  ...                              │    │     │     │
 * │  │  │  │  x31 (t6)   - temporary          │    │     │     │
 * │  │  │  └──────────────────────────────────┘    │     │     │
 * │  │  │                                            │     │     │
 * │  │  │  ┌──────────────────────────────────┐    │     │     │
 * │  │  │  │  Instruction Memory (IMEM)       │    │     │     │
 * │  │  │  │  - Parsed assembly instructions  │    │     │     │
 * │  │  │  │  - Read-only after init          │    │     │     │
 * │  │  │  └──────────────────────────────────┘    │     │     │
 * │  │  │                                            │     │     │
 * │  │  │  PC (Program Counter) ──────────┐        │     │     │
 * │  │  └──────────────────────────────────│────────┘     │     │
 * │  │                                      │              │     │
 * │  │  ┌───────────────────────────────────▼────────┐    │     │
 * │  │  │         Data Memory (DMEM)                 │    │     │
 * │  │  │  - Timing model for load/store            │    │     │
 * │  │  │  - Supports byte, half-word, word access  │    │     │
 * │  │  └────────────────────────────────────────────┘    │     │
 * │  │                                                      │     │
 * │  │  ┌────────────────────────────────────────────┐    │     │
 * │  │  │         ISA Emulator                       │    │     │
 * │  │  │  (Functional Model - Assembly Parser)     │    │     │
 * │  │  │  - Parses .s assembly files               │    │     │
 * │  │  │  - Resolves labels and pseudo-instructions│    │     │
 * │  │  │  - Initializes IMEM and DMEM              │    │     │
 * │  │  └────────────────────────────────────────────┘    │     │
 * │  └──────────────────────────────────────────────────────┘     │
 * │                                                                │
 * │  ┌──────────────────────────────────────────────────────┐     │
 * │  │         Pipeline Visualization Stages                │     │
 * │  │                                                      │     │
 * │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │     │
 * │  │  │ IFStage  │───▶│ EXEStage │───▶│ WBStage  │      │     │
 * │  │  │          │    │          │    │          │      │     │
 * │  │  │ - Hazard │    │ - Process│    │ - Retire │      │     │
 * │  │  │   detect │    │   packet │    │   inst   │      │     │
 * │  │  │ - Stall  │    │ - Control│    │          │      │     │
 * │  │  │   logic  │    │   hazard │    │          │      │     │
 * │  │  └──────────┘    └──────────┘    └──────────┘      │     │
 * │  │       ▲                                             │     │
 * │  │       │ Backpressure/Retry                         │     │
 * │  │       └─────────────────────────────────────────    │     │
 * │  └──────────────────────────────────────────────────────┘     │
 * └────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **RISC-V RV32I Instruction Set Support:**
 *
 * This simulator implements the complete RV32I base integer instruction set:
 *
 * 1. **R-Type Instructions** (Register-Register Operations):
 *    - Arithmetic: ADD, SUB
 *    - Logical: AND, OR, XOR
 *    - Shift: SLL, SRL, SRA
 *    - Comparison: SLT, SLTU
 *    - Format: op rd, rs1, rs2
 *
 * 2. **I-Type Instructions** (Immediate Operations):
 *    - Arithmetic: ADDI
 *    - Logical: ANDI, ORI, XORI
 *    - Shift: SLLI, SRLI, SRAI
 *    - Comparison: SLTI, SLTIU
 *    - Load: LB, LBU, LH, LHU, LW
 *    - Jump: JALR
 *    - Format: op rd, rs1, imm
 *
 * 3. **S-Type Instructions** (Store Operations):
 *    - SB (store byte), SH (store half-word), SW (store word)
 *    - Format: op rs2, offset(rs1)
 *
 * 4. **B-Type Instructions** (Branch Operations):
 *    - BEQ, BNE, BLT, BLTU, BGE, BGEU
 *    - Format: op rs1, rs2, label
 *
 * 5. **U-Type Instructions** (Upper Immediate):
 *    - LUI (load upper immediate)
 *    - AUIPC (add upper immediate to PC)
 *    - Format: op rd, imm
 *
 * 6. **J-Type Instructions** (Jump):
 *    - JAL (jump and link)
 *    - Format: op rd, label
 *
 * 7. **Special Instructions**:
 *    - HCF (Halt and Catch Fire) - terminates simulation
 *
 * **Event-Driven Execution Model:**
 *
 * The simulator uses an event-driven approach where instruction execution is
 * modeled as a series of timestamped events:
 *
 * ```
 * Time (Ticks)  Event                      Component
 * ─────────────────────────────────────────────────────────
 *   T=0         System Init                SOCTop
 *   T=1         ExecOneInstrEvent          CPU
 *               ├─ Fetch instruction       CPU::fetchInstr()
 *               ├─ Execute instruction     CPU::processInstr()
 *               ├─ Memory access (if needed) CPU::memRead/Write()
 *               └─ Commit instruction      CPU::commitInstr()
 *   T=2         InstPacket → IFStage       IFStage::step()
 *               ├─ Hazard detection        (data/control)
 *               └─ Forward to EXEStage     (if no hazard)
 *   T=3         InstPacket → EXEStage      EXEStage::step()
 *   T=4         InstPacket → WBStage       WBStage::step()
 * ```
 *
 * **Instruction Execution Flow:**
 *
 * ```
 *  ┌─────────────┐
 *  │   Start     │
 *  └──────┬──────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  1. FETCH (CPU::fetchInstr)             │
 *  │     - PC → Instruction Memory           │
 *  │     - Retrieve 32-bit instruction       │
 *  │     - Parse into instr structure        │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  2. DECODE & EXECUTE (CPU::processInstr)│
 *  │     - Decode opcode and operands        │
 *  │     - Read register file (if needed)    │
 *  │     - Perform ALU operation OR          │
 *  │     - Generate memory address           │
 *  │     - Update PC (branches/jumps)        │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  3. MEMORY ACCESS (if Load/Store)       │
 *  │     - CPU::memRead() or memWrite()      │
 *  │     - DataMemory timing model           │
 *  │     - Byte/Half/Word alignment          │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  4. WRITEBACK (implicit in processInstr)│
 *  │     - Update destination register       │
 *  │     - rf[rd] ← result                   │
 *  │     - x0 always remains 0               │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  5. COMMIT (CPU::commitInstr)           │
 *  │     - Send InstPacket to IFStage        │
 *  │     - Handle backpressure (if any)      │
 *  │     - Schedule next instruction event   │
 *  │     - OR halt (HCF instruction)         │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *  ┌─────────────────────────────────────────┐
 *  │  6. PIPELINE VISUALIZATION              │
 *  │     IF → EXE → WB stages                │
 *  │     Hazard detection & handling         │
 *  └──────┬──────────────────────────────────┘
 *         │
 *         ▼
 *    ┌────────┐    Yes    ┌──────────────┐
 *    │ HCF?   │──────────▶│ End Simulation│
 *    └────┬───┘           └──────────────┘
 *         │ No
 *         └────▶ Loop to FETCH (next instruction)
 * ```
 *
 * **Backpressure and Retry Mechanism:**
 *
 * The simulator implements a realistic backpressure mechanism to model
 * pipeline stalls and resource contention:
 *
 * 1. **Backpressure Detection:**
 *    - When CPU tries to send InstPacket to IFStage
 *    - If IFStage's slave port is full → backpressure
 *    - CPU stores packet in pendingInstPacket
 *
 * 2. **Retry Callback:**
 *    - IFStage calls SOC::masterPortRetry() when space available
 *    - SOC forwards retry to CPU::retrySendInstPacket()
 *    - CPU resends pendingInstPacket
 *    - On success, schedules next ExecOneInstrEvent
 *
 * 3. **Hazard Handling:**
 *    - Data hazards: RAW (Read-After-Write) dependencies detected
 *    - Control hazards: Taken branches cause pipeline flush
 *    - IFStage stalls when hazards detected
 *
 * **Memory Access Patterns:**
 *
 * - **Instruction Memory (IMEM):**
 *   - Read-only after initialization
 *   - Accessed via PC (4-byte aligned)
 *   - Contains parsed assembly instructions
 *
 * - **Data Memory (DMEM):**
 *   - Supports load/store operations
 *   - Byte-addressable
 *   - Access types: byte (8-bit), half-word (16-bit), word (32-bit)
 *   - Sign extension for LB, LH
 *   - Zero extension for LBU, LHU
 *
 * **Register File Management:**
 *
 * The simulator maintains a 32-register file following RISC-V ABI:
 * - x0 (zero): Hardwired to 0 (writes ignored)
 * - x1 (ra): Return address
 * - x2 (sp): Stack pointer
 * - x3 (gp): Global pointer
 * - x4 (tp): Thread pointer
 * - x5-x7 (t0-t2): Temporaries
 * - x8-x9 (s0-s1): Saved registers
 * - x10-x17 (a0-a7): Function arguments/return values
 * - x18-x27 (s2-s11): Saved registers
 * - x28-x31 (t3-t6): Temporaries
 *
 * **Usage Example:**
 *
 * ```bash
 * # Run the simulator with an assembly file
 * ./riscv_simulator
 *
 * # Configuration file (configs.json) specifies:
 * # - Assembly file path
 * # - Memory size
 * # - Data/text segment offsets
 * ```
 *
 * **Sample Assembly Program:**
 * ```assembly
 * .text
 * main:
 *     addi x1, x0, 5      # x1 = 5
 *     addi x2, x0, 10     # x2 = 10
 *     add  x3, x1, x2     # x3 = x1 + x2 = 15
 *     sw   x3, 0(x2)      # Store x3 to memory[x2]
 *     lw   x4, 0(x2)      # Load from memory[x2] to x4
 *     hcf                 # Halt simulation
 * ```
 *
 * **Key Features:**
 * - Complete RV32I ISA implementation
 * - Event-driven timing accuracy
 * - Pipeline stage visualization (IF, EXE, WB)
 * - Data and control hazard detection
 * - Backpressure handling with retry mechanism
 * - Assembly file parsing with label resolution
 * - Pseudo-instruction support (li, la, j, ret, mv, etc.)
 * - Register file state visualization
 *
 * @see SOCTop for simulation orchestration
 * @see SOC for system component integration
 * @see CPU for instruction execution model
 * @see Emulator for assembly parsing
 * @see IFStage for instruction fetch stage
 * @see EXEStage for execution stage
 * @see DataMemory for memory timing model
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "ACALSim.hh"
#include "SOCTop.hh"

/**
 * @brief Main entry point for RISC-V RV32I simulator
 *
 * @details
 * Initializes and runs the complete event-driven simulation system:
 *
 * 1. **SOCTop Creation**: Instantiates the top-level simulation controller
 *    with configuration from "src/riscv/configs.json"
 *
 * 2. **Initialization**: Calls SOCTop::init() which:
 *    - Parses command-line arguments
 *    - Loads configuration parameters
 *    - Creates SOC, CPU, pipeline stages
 *    - Parses assembly file into IMEM
 *    - Initializes data memory
 *    - Schedules initial ExecOneInstrEvent
 *
 * 3. **Simulation Loop**: Executes SOCTop::run() which:
 *    - Processes events in timestamp order
 *    - Executes instructions one per cycle
 *    - Handles pipeline stage transitions
 *    - Continues until HCF instruction or event queue empty
 *
 * 4. **Cleanup**: Calls SOCTop::finish() which:
 *    - Prints final register file state
 *    - Displays simulation statistics
 *    - Frees allocated resources
 *
 * **Execution Timeline:**
 * ```
 * T=0: System initialization
 *      └─ Parse assembly → IMEM
 *      └─ Initialize DMEM and registers
 *      └─ Schedule first ExecOneInstrEvent @ T=1
 *
 * T=1: Execute first instruction
 *      └─ Fetch, decode, execute
 *      └─ Send to IFStage
 *      └─ Schedule next ExecOneInstrEvent @ T=2
 *
 * T=2: Execute second instruction + Pipeline advance
 *      └─ IFStage → EXEStage (first instruction)
 *      └─ Execute second instruction
 *      ...
 *
 * T=N: HCF instruction encountered
 *      └─ Stop scheduling new events
 *      └─ Drain pipeline
 *      └─ Print register file
 *      └─ Exit simulation
 * ```
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on successful simulation completion
 *
 * @note The configuration file path is hardcoded to "src/riscv/configs.json"
 *       Modify this path if running from a different directory
 */
int main(int argc, char** argv) {
	acalsim::top = std::make_shared<SOCTop>("src/riscv/configs.json");
	acalsim::top->init(argc, argv);
	acalsim::top->run();
	acalsim::top->finish();
	return 0;
}
