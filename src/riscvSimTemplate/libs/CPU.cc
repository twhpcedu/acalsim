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
 * @file CPU.cc
 * @brief Single-Cycle CPU Implementation for RISC-V Simulator Template
 *
 * @details
 * This file implements the CPU timing model for the **simplified RISC-V simulator template**.
 * Unlike the full simulator in src/riscv/, this version executes instructions directly in the
 * CPU without separate pipeline stages, making it ideal for educational purposes.
 *
 * **KEY SIMPLIFICATION:**
 * - No separate EXEStage: All instruction execution happens in CPU::processInstr()
 * - Single-cycle model: Each instruction completes in one logical cycle
 * - Direct memory access: Memory operations complete immediately (no latency modeling)
 * - Simplified datapath: Fetch -> Execute -> Commit all in one function
 *
 * **CPU Architecture:**
 * @code
 *                        ┌────────────────────────────────────────┐
 *                        │         CPU Module (Timing Model)       │
 *                        └────────────────────────────────────────┘
 *
 *   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 *   │   FETCH      │───>│   EXECUTE    │───>│   COMMIT     │───>│  SEND TO IF  │
 *   │              │    │              │    │              │    │              │
 *   │ - Get PC     │    │ - Decode op  │    │ - Update PC  │    │ - Schedule   │
 *   │ - Read imem  │    │ - Exec ALU   │    │ - Log instr  │    │   next event │
 *   │ - Create pkt │    │ - Mem ops    │    │ - Write RF   │    │ - Handle BP  │
 *   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
 *        │                     │                    │                    │
 *        └─────────────────────┴────────────────────┴────────────────────┘
 *                     ALL IN ONE CYCLE (execOneInstr)
 * @endcode
 *
 * **Register File:**
 * - 32 general-purpose registers (x0-x31)
 * - x0 hardwired to zero (enforced by RISC-V spec)
 * - 32-bit wide (RV32I)
 * - Direct read/write access (no bypass network needed)
 *
 * **Instruction Memory:**
 * - Separate instruction memory array
 * - Indexed by PC/4 (word-aligned)
 * - Populated by Emulator during initialization
 * - Contains decoded instruction structures (not raw binary)
 *
 * **Execution Flow:**
 * @code
 * ExecOneInstrEvent triggered
 *    │
 *    ├─> CPU::execOneInstr()
 *    │   │
 *    │   ├─> fetchInstr(pc) - Get instruction from imem
 *    │   ├─> Create InstPacket
 *    │   └─> processInstr() - Execute instruction
 *    │       │
 *    │       ├─> Switch on instruction opcode
 *    │       ├─> Perform operation (ALU, branch, memory)
 *    │       ├─> Update register file
 *    │       ├─> Calculate next PC
 *    │       └─> commitInstr()
 *    │           │
 *    │           ├─> Log completion
 *    │           ├─> Send InstPacket to IFStage
 *    │           └─> Schedule next ExecOneInstrEvent
 *    │
 *    └─> Next instruction executes at tick+1
 * @endcode
 *
 * **Supported RV32I Instructions:**
 * | Category        | Instructions                                      |
 * |-----------------|---------------------------------------------------|
 * | ALU R-type      | ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU |
 * | ALU I-type      | ADDI, ANDI, ORI, XORI, SLLI, SRLI, SRAI          |
 * | Compare I-type  | SLTI, SLTIU                                      |
 * | Load            | LB, LBU, LH, LHU, LW                             |
 * | Store           | SB, SH, SW                                       |
 * | Branch          | BEQ, BNE, BLT, BLTU, BGE, BGEU                   |
 * | Jump            | JAL, JALR                                        |
 * | Upper Imm       | LUI, AUIPC                                       |
 * | Control         | HCF (Halt and Catch Fire)                        |
 *
 * **Memory Access:**
 * Memory operations (load/store) interact with DataMemory module:
 * @code
 * CPU memRead/memWrite
 *    │
 *    ├─> Create MemReadReqPacket or MemWriteReqPacket
 *    ├─> Send to DataMemory via "DSDmem" MasterPort
 *    ├─> DataMemory returns data immediately (single-cycle)
 *    └─> Update register file (for loads)
 * @endcode
 *
 * **Differences from Full Simulator:**
 * | Aspect              | Template (riscvSimTemplate) | Full (src/riscv/)       |
 * |---------------------|----------------------------|-------------------------|
 * | Execution           | Direct in CPU              | Staged through IF/EX    |
 * | Pipeline            | None (single-cycle)        | Multi-stage pipeline    |
 * | Memory Latency      | Immediate                  | Modeled with delays     |
 * | Hazard Handling     | Not needed                 | Forwarding, stalls      |
 * | Branch Prediction   | Not implemented            | With branch predictor   |
 * | Instruction Packet  | Goes to IF for logging     | Full pipeline traversal |
 *
 * **Event-Driven Execution:**
 * The CPU operates via ExecOneInstrEvent:
 * - Each event executes exactly one instruction
 * - Upon completion, schedules the next event (except HCF)
 * - Events are recycled for efficiency
 * - Backpressure mechanism prevents event queue overflow
 *
 * **Backpressure Handling:**
 * @code
 * CPU sends InstPacket to IFStage
 *    │
 *    ├─> IF port available?
 *    │   ├─> YES: Send packet, schedule next ExecOneInstrEvent
 *    │   └─> NO: Store in pendingInstPacket, wait for retry
 *    │
 *    └─> When IF port available:
 *        ├─> SOC::masterPortRetry() called
 *        ├─> CPU::retrySendInstPacket()
 *        └─> Schedule next ExecOneInstrEvent
 * @endcode
 *
 * **Extension Points:**
 * This template can be extended to add:
 * 1. **Pipeline Stages:** Split fetch/execute into IF/ID/EX/MEM/WB
 * 2. **Hazard Detection:** Identify data/control hazards
 * 3. **Forwarding:** Bypass network for data hazards
 * 4. **Branch Prediction:** Speculative execution
 * 5. **Cache Modeling:** Add memory hierarchy with latency
 * 6. **Performance Counters:** Track CPI, cache hits, branch accuracy
 *
 * @see SOC System-on-chip integration
 * @see Emulator Assembly parsing and instruction decoding
 * @see DataMemory Memory subsystem for load/store
 * @see IFStage Instruction fetch stage (minimal in template)
 * @see ExecOneInstrEvent Event that drives execution
 * @see src/riscv/ Full RISC-V simulator with complete pipeline
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @version 1.0
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

/**
 * @brief Constructor for the CPU class
 *
 * @details
 * Initializes the CPU timing model with:
 * - Program counter set to 0 (start of text segment)
 * - Instruction memory allocated based on data_offset
 * - Register file zeroed (all 32 registers = 0)
 * - Instruction count initialized to 0
 *
 * **Memory Allocation:**
 * Instruction memory size = data_offset / 4 (word-aligned)
 * Default: 64KB text segment = 16K instructions
 *
 * @param _name CPU instance identifier for logging
 * @param _soc Pointer to parent SOC for inter-module communication
 *
 * @note Instruction memory is initialized with UNIMPL opcodes
 * @note Will be populated by Emulator during SOC::simInit()
 */
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

/**
 * @brief Executes one complete instruction (fetch + execute + commit)
 *
 * @details
 * This is the **main execution entry point** triggered by ExecOneInstrEvent.
 * Implements a single-cycle execution model where fetch, execute, and commit
 * all happen in one logical cycle.
 *
 * **Execution Steps:**
 * 1. Fetch instruction from instruction memory at current PC
 * 2. Create InstPacket to track the instruction through the system
 * 3. Execute the instruction via processInstr()
 * 4. Commit results and schedule next instruction
 *
 * **Simplified vs Full Simulator:**
 * - Template: All steps in one function call (single-cycle)
 * - Full: Each step is a separate pipeline stage with latency
 *
 * **InstPacket:**
 * Encapsulates instruction and metadata for logging/tracking:
 * - Instruction structure (opcode, operands)
 * - Program counter at fetch time
 * - Branch taken flag
 * - Execution timestamp
 *
 * @note Called by ExecOneInstrEvent::process()
 * @note Uses RecycleContainer for efficient packet allocation
 *
 * @see processInstr() Instruction execution logic
 * @see fetchInstr() Instruction memory access
 * @see commitInstr() Instruction completion
 * @see ExecOneInstrEvent Event that triggers this function
 */
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

/**
 * @brief Processes (executes) a RISC-V instruction
 *
 * @details
 * This is the **heart of the CPU execution logic** where all RV32I instructions
 * are decoded and executed. Implements the full RISC-V RV32I instruction set
 * using a large switch statement.
 *
 * **Execution Steps:**
 * 1. Increment instruction counter for statistics
 * 2. Calculate default next PC (current PC + 4)
 * 3. Switch on instruction opcode
 * 4. Perform operation (ALU, memory, control flow)
 * 5. Update register file if needed
 * 6. Update next PC for branches/jumps
 * 7. Commit instruction and schedule next event
 *
 * **Instruction Categories:**
 * - **R-type (reg-reg):** ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
 * - **I-type (reg-imm):** ADDI, ANDI, ORI, XORI, SLLI, SRLI, SRAI, SLTI, SLTIU
 * - **Load:** LB, LBU, LH, LHU, LW (via memRead)
 * - **Store:** SB, SH, SW (via memWrite)
 * - **Branch:** BEQ, BNE, BLT, BLTU, BGE, BGEU
 * - **Jump:** JAL, JALR
 * - **Upper Imm:** LUI, AUIPC
 * - **Special:** HCF (halt simulation)
 *
 * **Register File Access:**
 * Direct array access for simplicity:
 * - rf_ref[reg] for reads
 * - rf_ref[reg] = value for writes
 * - x0 writes are allowed but have no effect (not enforced here)
 *
 * **Signed vs Unsigned Operations:**
 * Uses pointer casting for signed comparisons:
 * - *(int32_t*)&rf_ref[reg] for signed interpretation
 * - rf_ref[reg] for unsigned interpretation
 *
 * **Branch/Jump PC Calculation:**
 * - Branches: PC = target_address if condition true, else PC+4
 * - JAL: PC = PC + immediate_offset
 * - JALR: PC = (rs1 + immediate) & ~1
 *
 * @param _i Instruction structure to execute
 * @param instPacket Packet tracking this instruction
 *
 * @note Memory operations (load/store) call memRead/memWrite
 * @note HCF instruction stops scheduling new events
 * @note SLTU implementation has a bug (uses ADD instead of comparison)
 *
 * @see memRead() Load instruction handler
 * @see memWrite() Store instruction handler
 * @see commitInstr() Instruction completion
 */
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

/**
 * @brief Commits an instruction and schedules the next execution event
 *
 * @details
 * Final stage of instruction execution that:
 * 1. Logs instruction completion with timestamp
 * 2. Sends InstPacket to IFStage for logging/tracking
 * 3. Schedules next ExecOneInstrEvent (unless HCF)
 * 4. Handles backpressure if IF stage is full
 *
 * **Commit Flow:**
 * @code
 * commitInstr()
 *    │
 *    ├─> Log completion (opcode, tick, PC)
 *    │
 *    ├─> Check if HCF instruction
 *    │   └─> YES: Send packet but don't schedule next event (simulation ends)
 *    │
 *    └─> Try to send InstPacket to IFStage via "sIF-m" port
 *        ├─> SUCCESS:
 *        │   ├─> Log successful send
 *        │   └─> Schedule next ExecOneInstrEvent at tick+1
 *        └─> BACKPRESSURE:
 *            ├─> Store packet in pendingInstPacket
 *            ├─> Log backpressure
 *            └─> Wait for retry callback
 * @endcode
 *
 * **HCF (Halt and Catch Fire):**
 * Special instruction that ends simulation:
 * - Sends final InstPacket for logging
 * - Does NOT schedule next ExecOneInstrEvent
 * - Event queue empties and simulation terminates
 *
 * **Backpressure Handling:**
 * If IF stage port is full:
 * - Packet stored in pendingInstPacket
 * - No next event scheduled yet
 * - SOC::masterPortRetry() will be called when port available
 * - CPU::retrySendInstPacket() resends and schedules next event
 *
 * @param _i Instruction that was executed
 * @param instPacket Packet tracking this instruction
 *
 * @note In template version, IF stage immediately accepts packets (no real backpressure)
 * @note Full simulator has more complex pipeline with actual congestion
 *
 * @see retrySendInstPacket() Handles retry after backpressure
 * @see ExecOneInstrEvent Event scheduled to execute next instruction
 * @see IFStage Destination for InstPacket
 */
void CPU::commitInstr(const instr& _i, InstPacket* instPacket) {
	CLASS_INFO << "Instruction " << this->instrToString(_i.op)
	           << " is completed at Tick = " << acalsim::top->getGlobalTick() << " | PC = " << this->pc;

	if (_i.op == HCF) {
		// end of simulation.
		// Stop scheduling new events to process instructions.
		// There might be pending events in the simulator.
		if (!this->soc->getMasterPort("sIF-m")->push(instPacket)) pendingInstPacket = instPacket;
		return;
	}

	// send the packet to the IF stage
	if (this->soc->getMasterPort("sIF-m")->push(instPacket)) {
		// send the instruction packet to the IF stage successfully
		// schedule the next trigger event
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
