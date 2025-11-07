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
 * @file CPU.cc
 * @brief RISC-V RV32I Single-Cycle CPU Timing Model Implementation
 *
 * @details
 * This file implements a complete single-cycle CPU timing model for the RISC-V
 * RV32I instruction set architecture. The CPU executes one instruction per cycle
 * in an event-driven simulation framework, demonstrating:
 * - Complete RV32I instruction decode and execution
 * - 32-register general-purpose register file
 * - Memory load/store operations with DataMemory timing model
 * - Program counter management and control flow
 * - Backpressure handling for pipeline stage communication
 *
 * **CPU Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────┐
 * │                         CPU                                │
 * │  (Single-Cycle RISC-V RV32I Processor)                    │
 * │                                                            │
 * │  ┌──────────────────────────────────────────────────┐     │
 * │  │  Program Counter (PC)                            │     │
 * │  │  - 32-bit address register                       │     │
 * │  │  - Points to current instruction in IMEM         │     │
 * │  │  - Updated each cycle:                           │     │
 * │  │    • PC + 4 (sequential)                         │     │
 * │  │    • Branch target (conditional)                 │     │
 * │  │    • Jump target (JAL/JALR)                      │     │
 * │  └──────────────┬───────────────────────────────────┘     │
 * │                 │                                          │
 * │                 ▼                                          │
 * │  ┌──────────────────────────────────────────────────┐     │
 * │  │  Instruction Memory (IMEM)                       │     │
 * │  │  - Array of parsed instruction structures        │     │
 * │  │  - Read-only after initialization                │     │
 * │  │  - Size: data_offset / 4 instructions            │     │
 * │  │  - Access: PC / 4 = instruction index            │     │
 * │  └──────────────┬───────────────────────────────────┘     │
 * │                 │                                          │
 * │                 ▼                                          │
 * │  ┌──────────────────────────────────────────────────┐     │
 * │  │  Instruction Decode & Execute                    │     │
 * │  │                                                   │     │
 * │  │  ┌──────────────────────────────────────┐        │     │
 * │  │  │  ALU Operations                      │        │     │
 * │  │  │  - ADD, SUB, AND, OR, XOR           │        │     │
 * │  │  │  - SLL, SRL, SRA (shifts)           │        │     │
 * │  │  │  - SLT, SLTU (comparisons)          │        │     │
 * │  │  └──────────────────────────────────────┘        │     │
 * │  │                                                   │     │
 * │  │  ┌──────────────────────────────────────┐        │     │
 * │  │  │  Branch/Jump Logic                   │        │     │
 * │  │  │  - BEQ, BNE, BLT, BGE, BLTU, BGEU   │        │     │
 * │  │  │  - JAL, JALR (unconditional)        │        │     │
 * │  │  │  - Compute target address            │        │     │
 * │  │  │  - Update PC conditionally           │        │     │
 * │  │  └──────────────────────────────────────┘        │     │
 * │  │                                                   │     │
 * │  │  ┌──────────────────────────────────────┐        │     │
 * │  │  │  Load/Store Unit                     │        │     │
 * │  │  │  - LB, LBU, LH, LHU, LW             │        │     │
 * │  │  │  - SB, SH, SW                        │        │     │
 * │  │  │  - Address calculation: rs1 + imm    │        │     │
 * │  │  │  - Interface with DataMemory         │        │     │
 * │  │  └──────────────────────────────────────┘        │     │
 * │  └───────────────────────────────────────────────────┘     │
 * │                                                            │
 * │  ┌──────────────────────────────────────────────────┐     │
 * │  │  Register File (RF)                              │     │
 * │  │  - 32 general-purpose registers (x0-x31)         │     │
 * │  │  - x0 hardwired to 0 (writes ignored)            │     │
 * │  │  - 32-bit integer values                         │     │
 * │  │                                                   │     │
 * │  │  ABI Names:                                      │     │
 * │  │  x0  = zero  (constant 0)                        │     │
 * │  │  x1  = ra    (return address)                    │     │
 * │  │  x2  = sp    (stack pointer)                     │     │
 * │  │  x3  = gp    (global pointer)                    │     │
 * │  │  x4  = tp    (thread pointer)                    │     │
 * │  │  x5-7  = t0-t2  (temporaries)                    │     │
 * │  │  x8-9  = s0-s1  (saved registers)                │     │
 * │  │  x10-17 = a0-a7  (arguments/return values)       │     │
 * │  │  x18-27 = s2-s11 (saved registers)               │     │
 * │  │  x28-31 = t3-t6  (temporaries)                   │     │
 * │  └──────────────────────────────────────────────────┘     │
 * │                                                            │
 * │  Port Connections:                                        │
 * │  ┌──────────────────────────────────────────────────┐     │
 * │  │  DSDmem ────▶ DataMemory                         │     │
 * │  │  (Load/Store requests)                           │     │
 * │  │                                                   │     │
 * │  │  sIF-m ────▶ IFStage                             │     │
 * │  │  (InstPacket for pipeline visualization)         │     │
 * │  └──────────────────────────────────────────────────┘     │
 * └────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Instruction Execution Cycle:**
 *
 * Each ExecOneInstrEvent triggers this execution sequence:
 *
 * ```
 * ┌─────────────────────────────────────────────────────────┐
 * │  1. FETCH (CPU::fetchInstr)                            │
 * │     ────────────────────────────────────────────────    │
 * │     Input:  PC (program counter)                        │
 * │     Action: index = PC / 4                              │
 * │             instruction = IMEM[index]                   │
 * │     Output: instr structure with opcode & operands      │
 * └─────────────┬───────────────────────────────────────────┘
 *               │
 *               ▼
 * ┌─────────────────────────────────────────────────────────┐
 * │  2. DECODE & EXECUTE (CPU::processInstr)               │
 * │     ────────────────────────────────────────────────    │
 * │     Switch on opcode:                                   │
 * │                                                          │
 * │     R-Type (reg-reg):                                   │
 * │       rd ← rs1 OP rs2                                   │
 * │       Examples: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA  │
 * │                                                          │
 * │     I-Type (reg-imm):                                   │
 * │       rd ← rs1 OP imm                                   │
 * │       Examples: ADDI, ANDI, ORI, XORI, SLLI, SRLI      │
 * │                                                          │
 * │     Load:                                               │
 * │       addr ← rs1 + imm                                  │
 * │       rd ← MEM[addr]  (call memRead)                    │
 * │                                                          │
 * │     Store:                                              │
 * │       addr ← rs1 + imm                                  │
 * │       MEM[addr] ← rs2  (call memWrite)                  │
 * │                                                          │
 * │     Branch:                                             │
 * │       if (rs1 CMP rs2)                                  │
 * │         PC ← label_addr                                 │
 * │       else                                              │
 * │         PC ← PC + 4                                     │
 * │                                                          │
 * │     JAL:                                                │
 * │       rd ← PC + 4                                       │
 * │       PC ← label_addr                                   │
 * │                                                          │
 * │     JALR:                                               │
 * │       rd ← PC + 4                                       │
 * │       PC ← rs1 + imm                                    │
 * │                                                          │
 * │     LUI:                                                │
 * │       rd ← imm << 12                                    │
 * │                                                          │
 * │     AUIPC:                                              │
 * │       rd ← PC + (imm << 12)                             │
 * └─────────────┬───────────────────────────────────────────┘
 *               │
 *               ▼
 * ┌─────────────────────────────────────────────────────────┐
 * │  3. MEMORY ACCESS (if Load/Store)                      │
 * │     ────────────────────────────────────────────────    │
 * │     CPU::memRead() or CPU::memWrite()                   │
 * │     - Create MemReadReqPacket or MemWriteReqPacket     │
 * │     - Send to DataMemory via DSDmem port               │
 * │     - DataMemory processes request                      │
 * │     - For loads: receive data, write to rd             │
 * │     - Handle byte/half/word and sign extension         │
 * └─────────────┬───────────────────────────────────────────┘
 *               │
 *               ▼
 * ┌─────────────────────────────────────────────────────────┐
 * │  4. COMMIT (CPU::commitInstr)                          │
 * │     ────────────────────────────────────────────────    │
 * │     Create InstPacket with executed instruction         │
 * │     Try to send to IFStage via sIF-m port:             │
 * │                                                          │
 * │     SUCCESS:                                            │
 * │       - InstPacket accepted by IFStage                  │
 * │       - Schedule next ExecOneInstrEvent @ T+1           │
 * │       - Continue execution                              │
 * │                                                          │
 * │     BACKPRESSURE:                                       │
 * │       - IFStage slave port full                         │
 * │       - Store in pendingInstPacket                      │
 * │       - Do NOT schedule next event (STALL)              │
 * │       - Wait for retry callback                         │
 * │                                                          │
 * │     HCF INSTRUCTION:                                    │
 * │       - Send InstPacket to IFStage                      │
 * │       - Do NOT schedule next event                      │
 * │       - Simulation will terminate                       │
 * └─────────────┬───────────────────────────────────────────┘
 *               │
 *               ▼
 * ┌─────────────────────────────────────────────────────────┐
 * │  5. RETRY (if backpressure occurred)                   │
 * │     ────────────────────────────────────────────────    │
 * │     CPU::retrySendInstPacket()                          │
 * │     - Called by SOC::masterPortRetry()                  │
 * │     - Resend pendingInstPacket to IFStage              │
 * │     - On success:                                       │
 * │       • Clear pendingInstPacket                         │
 * │       • Schedule next ExecOneInstrEvent @ T+1           │
 * │       • Resume normal execution                         │
 * └─────────────────────────────────────────────────────────┘
 * ```
 *
 * **RISC-V RV32I Instruction Format Details:**
 *
 * 1. **R-Type** (Register-Register):
 *    ```
 *    Format: op rd, rs1, rs2
 *    Execution: rf[rd] = rf[rs1] OP rf[rs2]
 *    Examples:
 *      add  x3, x1, x2   # x3 = x1 + x2
 *      sub  x3, x1, x2   # x3 = x1 - x2
 *      and  x3, x1, x2   # x3 = x1 & x2
 *      sll  x3, x1, x2   # x3 = x1 << x2
 *    ```
 *
 * 2. **I-Type** (Register-Immediate):
 *    ```
 *    Format: op rd, rs1, imm
 *    Execution: rf[rd] = rf[rs1] OP imm
 *    Examples:
 *      addi x1, x0, 5    # x1 = x0 + 5 = 5
 *      andi x3, x1, 0xF  # x3 = x1 & 0xF
 *      slli x3, x1, 2    # x3 = x1 << 2
 *    ```
 *
 * 3. **Load Instructions**:
 *    ```
 *    Format: op rd, offset(rs1)
 *    Address: addr = rf[rs1] + sign_extend(offset)
 *
 *    LW  (load word):       rf[rd] = MEM[addr][31:0]
 *    LH  (load half):       rf[rd] = sign_extend(MEM[addr][15:0])
 *    LHU (load half unsigned): rf[rd] = zero_extend(MEM[addr][15:0])
 *    LB  (load byte):       rf[rd] = sign_extend(MEM[addr][7:0])
 *    LBU (load byte unsigned): rf[rd] = zero_extend(MEM[addr][7:0])
 *
 *    Examples:
 *      lw  x1, 0(x2)     # x1 = MEM[x2 + 0] (word)
 *      lh  x1, 4(x2)     # x1 = sign_extend(MEM[x2 + 4][15:0])
 *      lbu x1, 8(x2)     # x1 = zero_extend(MEM[x2 + 8][7:0])
 *    ```
 *
 * 4. **Store Instructions**:
 *    ```
 *    Format: op rs2, offset(rs1)
 *    Address: addr = rf[rs1] + sign_extend(offset)
 *
 *    SW (store word): MEM[addr][31:0] = rf[rs2][31:0]
 *    SH (store half): MEM[addr][15:0] = rf[rs2][15:0]
 *    SB (store byte): MEM[addr][7:0]  = rf[rs2][7:0]
 *
 *    Examples:
 *      sw x1, 0(x2)      # MEM[x2 + 0] = x1 (word)
 *      sh x1, 4(x2)      # MEM[x2 + 4][15:0] = x1[15:0]
 *      sb x1, 8(x2)      # MEM[x2 + 8][7:0] = x1[7:0]
 *    ```
 *
 * 5. **Branch Instructions**:
 *    ```
 *    Format: op rs1, rs2, label
 *    Target: PC = label_addr (if condition true)
 *            PC = PC + 4     (if condition false)
 *
 *    BEQ:  if (rs1 == rs2) take branch
 *    BNE:  if (rs1 != rs2) take branch
 *    BLT:  if (rs1 <  rs2) take branch (signed)
 *    BGE:  if (rs1 >= rs2) take branch (signed)
 *    BLTU: if (rs1 <  rs2) take branch (unsigned)
 *    BGEU: if (rs1 >= rs2) take branch (unsigned)
 *
 *    Examples:
 *      beq  x1, x2, loop # if (x1 == x2) goto loop
 *      bne  x1, x0, done # if (x1 != 0) goto done
 *      blt  x1, x2, less # if (x1 < x2) goto less
 *    ```
 *
 * 6. **Jump Instructions**:
 *    ```
 *    JAL (Jump and Link):
 *      Format: jal rd, label
 *      Execution: rf[rd] = PC + 4
 *                 PC = label_addr
 *      Example: jal ra, function  # call function
 *
 *    JALR (Jump and Link Register):
 *      Format: jalr rd, rs1, offset
 *      Execution: rf[rd] = PC + 4
 *                 PC = rf[rs1] + offset
 *      Example: jalr zero, ra, 0  # return from function
 *    ```
 *
 * 7. **Upper Immediate**:
 *    ```
 *    LUI (Load Upper Immediate):
 *      Format: lui rd, imm
 *      Execution: rf[rd] = imm << 12
 *      Example: lui x1, 0x12345  # x1 = 0x12345000
 *
 *    AUIPC (Add Upper Immediate to PC):
 *      Format: auipc rd, imm
 *      Execution: rf[rd] = PC + (imm << 12)
 *      Example: auipc x1, 0x100  # x1 = PC + 0x100000
 *    ```
 *
 * **Backpressure Handling:**
 *
 * The CPU implements backpressure to model realistic pipeline stalls:
 *
 * ```
 * Normal Flow (No Backpressure):
 * ────────────────────────────────
 * T=N: ExecOneInstrEvent
 *      └─ execOneInstr()
 *         └─ processInstr()
 *            └─ commitInstr()
 *               └─ push(InstPacket) → IFStage  [SUCCESS]
 *               └─ schedule(ExecOneInstrEvent @ T=N+1)
 *
 * T=N+1: Next instruction executes...
 *
 * Backpressure Flow:
 * ──────────────────
 * T=N: ExecOneInstrEvent
 *      └─ execOneInstr()
 *         └─ processInstr()
 *            └─ commitInstr()
 *               └─ push(InstPacket) → IFStage  [FAIL - port full]
 *               └─ pendingInstPacket = InstPacket
 *               └─ NO event scheduled (CPU STALLS)
 *
 * T=N+1: CPU idle (no event)
 * T=N+2: IFStage processes packet, frees space
 *        └─ IFStage triggers retry()
 *           └─ SOC::masterPortRetry("sIF-m")
 *              └─ CPU::retrySendInstPacket()
 *                 └─ push(pendingInstPacket) → IFStage  [SUCCESS]
 *                 └─ schedule(ExecOneInstrEvent @ T=N+3)
 *                 └─ pendingInstPacket = nullptr
 *
 * T=N+3: CPU resumes execution...
 * ```
 *
 * **Register File Management:**
 *
 * - Register x0 is special: always reads as 0, writes are ignored
 * - All other registers (x1-x31) are general-purpose
 * - No special handling for ABI names (handled by assembler)
 * - Register values persist across instructions
 *
 * **Program Counter Behavior:**
 *
 * ```
 * Sequential:  PC ← PC + 4           (most instructions)
 * Branch:      PC ← target           (if condition true)
 * JAL:         PC ← target           (unconditional)
 * JALR:        PC ← rs1 + offset     (computed target)
 * HCF:         PC unchanged          (halts execution)
 * ```
 *
 * **Key Design Decisions:**
 *
 * 1. **Single-Cycle Model**: All instructions complete in one cycle
 *    - Simplifies timing model
 *    - Memory accesses complete immediately
 *    - Real hardware would have multi-cycle memory
 *
 * 2. **Event-Driven**: Uses ExecOneInstrEvent for scheduling
 *    - Enables accurate timing simulation
 *    - Natural backpressure handling
 *    - Easy integration with pipeline visualization
 *
 * 3. **Separate IMEM/DMEM**: Instructions and data in different memories
 *    - Harvard architecture style
 *    - IMEM read-only after init
 *    - DMEM supports read/write
 *
 * 4. **Backpressure Support**: Handles downstream stalls gracefully
 *    - Models realistic pipeline behavior
 *    - Prevents packet loss
 *    - Demonstrates producer-consumer synchronization
 *
 * @see CPU.hh for class declaration
 * @see Emulator for assembly parsing
 * @see DataMemory for load/store timing model
 * @see IFStage for pipeline visualization
 * @see ExecOneInstrEvent for instruction execution events
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "CPU.hh"

#include <iomanip>
#include <sstream>

#include "DataMemory.hh"
#include "InstPacket.hh"
#include "SOC.hh"
#include "event/ExecOneInstrEvent.hh"
#include "event/MemReqEvent.hh"

CPU::CPU(std::string _name, SOC* _soc)
    : acalsim::SimModule(_name), pc(0), inst_cnt(0), soc(_soc), pendingInstPacket(nullptr) {
	auto data_offset = acalsim::top->getParameter<int>("Emulator", "data_offset");
	this->imem       = new instr[data_offset / 4];
	for (int i = 0; i < data_offset / 4; i++) {
		this->imem[i].op      = UNIMPL;
		this->imem[i].a1.type = OPTYPE_NONE;
		this->imem[i].a2.type = OPTYPE_NONE;
		this->imem[i].a3.type = OPTYPE_NONE;
	}
	for (int i = 0; i < 32; i++) { this->rf[i] = 0; }
}

void CPU::execOneInstr() {
	// This lab models a single-CPU cycle as shown in Lab7
	// Fetch instrucion
	instr i = this->fetchInstr(this->pc);

	// Prepare instruction packet
	auto        rc         = top->getRecycleContainer();
	InstPacket* instPacket = rc->acquire<InstPacket>(&InstPacket::renew, i);
	instPacket->pc         = this->pc;

	// Execute the instruction in the same cycle
	processInstr(i, instPacket);
}

void CPU::processInstr(const instr& _i, InstPacket* instPacket) {
	bool  done   = false;
	auto& rf_ref = this->rf;
	this->incrementInstCount();
	int pc_next = this->pc + 4;

	switch (_i.op) {
		case ADD: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] + rf_ref[_i.a3.reg]; break;
		case SUB: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] - rf_ref[_i.a3.reg]; break;
		case SLT: rf_ref[_i.a1.reg] = (*(int32_t*)&rf_ref[_i.a2.reg]) < (*(int32_t*)&rf_ref[_i.a3.reg]) ? 1 : 0; break;
		case SLTU: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] + rf_ref[_i.a3.reg]; break;
		case AND: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] & rf_ref[_i.a3.reg]; break;
		case OR: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] | rf_ref[_i.a3.reg]; break;
		case XOR: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] ^ rf_ref[_i.a3.reg]; break;
		case SLL: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] << rf_ref[_i.a3.reg]; break;
		case SRL: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] >> rf_ref[_i.a3.reg]; break;
		case SRA: rf_ref[_i.a1.reg] = (*(int32_t*)&rf_ref[_i.a2.reg]) >> rf_ref[_i.a3.reg]; break;

		case ADDI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] + _i.a3.imm; break;
		case SLTI: rf_ref[_i.a1.reg] = (*(int32_t*)&rf_ref[_i.a2.reg]) < (*(int32_t*)&(_i.a3.imm)) ? 1 : 0; break;
		case SLTIU: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] < _i.a3.imm ? 1 : 0; break;
		case ANDI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] & _i.a3.imm; break;
		case ORI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] | _i.a3.imm; break;
		case XORI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] ^ _i.a3.imm; break;
		case SLLI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] << _i.a3.imm; break;
		case SRLI: rf_ref[_i.a1.reg] = rf_ref[_i.a2.reg] >> _i.a3.imm; break;
		case SRAI: rf_ref[_i.a1.reg] = (*(int32_t*)&rf_ref[_i.a2.reg]) >> _i.a3.imm; break;

		case BEQ:
			if (rf_ref[_i.a1.reg] == rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;
		case BGE:
			if (*(int32_t*)&rf_ref[_i.a1.reg] >= *(int32_t*)&rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;
		case BGEU:
			if (rf_ref[_i.a1.reg] >= rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;
		case BLT:
			if (*(int32_t*)&rf_ref[_i.a1.reg] < *(int32_t*)&rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;
		case BLTU:
			if (rf_ref[_i.a1.reg] < rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;
		case BNE:
			if (rf_ref[_i.a1.reg] != rf_ref[_i.a2.reg]) pc_next = _i.a3.imm;
			break;

		case JAL:
			rf_ref[_i.a1.reg] = this->pc + 4;
			pc_next           = _i.a2.imm;
			break;
		case JALR:
			rf_ref[_i.a1.reg] = this->pc + 4;
			pc_next           = rf_ref[_i.a2.reg] + _i.a3.imm;
			break;
		case AUIPC: rf_ref[_i.a1.reg] = this->pc + (_i.a2.imm << 12); break;
		case LUI: rf_ref[_i.a1.reg] = (_i.a2.imm << 12); break;

		case LB:
		case LBU:
		case LH:
		case LHU:
		case LW: this->memRead(_i, _i.op, this->rf[_i.a2.reg] + _i.a3.imm, _i.a1, instPacket); break;
		case SB:
		case SH:
		case SW: this->memWrite(_i, _i.op, this->rf[_i.a2.reg] + _i.a3.imm, this->rf[_i.a1.reg], instPacket); break;

		case HCF: break;
		case UNIMPL:
		default:
			CLASS_INFO << "Reached an unimplemented instruction!";
			if (_i.psrc) printf("Instruction: %s\n", _i.psrc);
			break;
	}

	this->commitInstr(_i, instPacket);
	if (pc_next != pc + 4) instPacket->isTakenBranch = true;
	this->pc = pc_next;
}

void CPU::commitInstr(const instr& _i, InstPacket* instPacket) {
	if (_i.op == HCF) {
		// end of simulation.
		// Stop scheduling new events to process instructions.
		// There might be pending events in the simulator.
		if (!this->soc->getMasterPort("sIF-m")->push(instPacket)) {
			pendingInstPacket = instPacket;
		} else {
			CLASS_INFO << "Instruction " << this->instrToString(_i.op)
			           << " is completed at Tick = " << acalsim::top->getGlobalTick() << " | PC = " << this->pc;
		}
		return;
	}

	// send the packet to the IF stage
	if (this->soc->getMasterPort("sIF-m")->push(instPacket)) {
		// send the instruction packet to the IF stage successfully
		// schedule the next trigger event
		CLASS_INFO << "Instruction " << this->instrToString(_i.op)
		           << " is completed at Tick = " << acalsim::top->getGlobalTick() << " | PC = " << this->pc;
		CLASS_INFO << "send " + this->instrToString(instPacket->inst.op) << "@ PC=" << instPacket->pc
		           << " to IFStage successfully";
		auto               rc = acalsim::top->getRecycleContainer();
		ExecOneInstrEvent* event =
		    rc->acquire<ExecOneInstrEvent>(&ExecOneInstrEvent::renew, this->getInstCount() /*id*/, this);
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	} else {
		// get backpressure from the IF stage
		// Wait until the master port pops out the entry and retry
		// This case, we need to store the instruction packet locally
		pendingInstPacket = instPacket;
		CLASS_INFO << "send " + this->instrToString(instPacket->inst.op) << "@ PC=" << instPacket->pc
		           << ", Got backpressure";
	}
}

void CPU::retrySendInstPacket(MasterPort* mp) {
	if (!pendingInstPacket) return;
	if (mp->push(pendingInstPacket)) {
		CLASS_INFO << "resend " + this->instrToString(pendingInstPacket->inst.op) << "@ PC=" << pendingInstPacket->pc
		           << " to IFStage successfully";
		CLASS_INFO << "Instruction " << this->instrToString(pendingInstPacket->inst.op)
		           << " is completed at Tick = " << acalsim::top->getGlobalTick()
		           << " | PC = " << pendingInstPacket->pc;

		if (pendingInstPacket->inst.op == HCF) return;

		// send the instruction packet to the IF stage successfully
		// schedule the next trigger event
		auto               rc = acalsim::top->getRecycleContainer();
		ExecOneInstrEvent* event =
		    rc->acquire<ExecOneInstrEvent>(&ExecOneInstrEvent::renew, this->getInstCount() /*id*/, this);
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
		pendingInstPacket = nullptr;
	} else {
		CLASS_ERROR << " CPU::retrySendInstPacket() failed!";
	}
}

bool CPU::memRead(const instr& _i, instr_type _op, uint32_t _addr, operand _a1, InstPacket* instPacket) {
	// If latency is larger than 1, e.g. cache miss or multi-cycle SRAM reads

	auto              rc  = acalsim::top->getRecycleContainer();
	MemReadReqPacket* pkt = rc->acquire<MemReadReqPacket>(&MemReadReqPacket::renew, nullptr, _i, _op, _addr, _a1);
	auto data = ((DataMemory*)this->getDownStream("DSDmem"))->memReadReqHandler(acalsim::top->getGlobalTick(), pkt);
	this->rf[_i.a1.reg] = data;
	CLASS_INFO << "handle memRead for " << this->instrToString(instPacket->inst.op) << " @ PC=" << instPacket->pc;
	return true;
}

bool CPU::memWrite(const instr& _i, instr_type _op, uint32_t _addr, uint32_t _data, InstPacket* instPacket) {
	auto rc = acalsim::top->getRecycleContainer();

	MemWriteReqPacket* pkt = rc->acquire<MemWriteReqPacket>(&MemWriteReqPacket::renew, nullptr, _i, _op, _addr, _data);
	this->getDownStream("DSDmem")->accept(acalsim::top->getGlobalTick(), *((acalsim::SimPacket*)pkt));
	CLASS_INFO << "handle memWrite for " << this->instrToString(instPacket->inst.op) << " @ PC=" << instPacket->pc;

	return true;
}

void CPU::printRegfile() const {
	std::ostringstream oss;

	oss << "Register File Snapshot:\n\n";
	for (int i = 0; i < 32; i++) {
		oss << "x" << std::setw(2) << std::setfill('0') << std::dec << i << ":0x";

		oss << std::setw(8) << std::setfill('0') << std::hex << this->rf[i] << " ";

		if ((i + 1) % 8 == 0) { oss << "\n"; }
	}

	oss << '\n';

	CLASS_INFO << oss.str();
}

instr CPU::fetchInstr(uint32_t _pc) const {
	uint32_t iid = _pc / 4;
	return this->imem[iid];
}

std::string CPU::instrToString(instr_type _op) const {
	switch (_op) {
		case UNIMPL: return "UNIMPL";

		// R-type
		case ADD: return "ADD";
		case AND: return "AND";
		case OR: return "OR";
		case XOR: return "XOR";
		case SUB: return "SUB";
		case SLL: return "SLL";
		case SRL: return "SRL";
		case SRA: return "SRA";
		case SLT: return "SLT";
		case SLTU: return "SLTU";

		// I-type
		case ADDI: return "ADDI";
		case ANDI: return "ANDI";
		case ORI: return "ORI";
		case XORI: return "XORI";
		case SLLI: return "SLLI";
		case SRLI: return "SRLI";
		case SRAI: return "SRAI";
		case SLTI: return "SLTI";
		case SLTIU: return "SLTIU";

		// Load
		case LB: return "LB";
		case LBU: return "LBU";
		case LH: return "LH";
		case LHU: return "LHU";
		case LW: return "LW";

		// Store
		case SB: return "SB";
		case SH: return "SH";
		case SW: return "SW";

		// Branch
		case BEQ: return "BEQ";
		case BNE: return "BNE";
		case BGE: return "BGE";
		case BGEU: return "BGEU";
		case BLT: return "BLT";
		case BLTU: return "BLTU";

		// Jump
		case JAL: return "JAL";
		case JALR: return "JALR";

		// Upper / Immediate
		case AUIPC: return "AUIPC";
		case LUI: return "LUI";

		// Special
		case HCF: return "HCF";

		default: return "UNKNOWN";
	}
}
