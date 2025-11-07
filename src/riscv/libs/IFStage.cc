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
 * @file IFStage.cc
 * @brief Instruction Fetch (IF) Pipeline Stage with Hazard Detection
 *
 * @details
 * This file implements the Instruction Fetch (IF) pipeline stage, which serves
 * as the first stage in the pipeline visualization of the RISC-V simulator.
 * The IF stage receives completed instruction packets from the CPU and forwards
 * them to the EXE stage, while performing critical hazard detection to prevent
 * data and control hazards.
 *
 * **Pipeline Stage Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────┐
 * │                       IFStage                              │
 * │  (Instruction Fetch & Hazard Detection)                   │
 * │                                                            │
 * │  Inputs:                                                   │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  soc-s (Slave Port from CPU)         │                 │
 * │  │  - Receives InstPacket               │                 │
 * │  │  - Contains executed instruction     │                 │
 * │  │  - PC, opcode, operands, metadata    │                 │
 * │  └──────────────────────────────────────┘                 │
 * │                │                                           │
 * │                ▼                                           │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  Hazard Detection Logic              │                 │
 * │  │                                       │                 │
 * │  │  Check for:                           │                 │
 * │  │  • Data Hazards (RAW dependencies)   │                 │
 * │  │  • Control Hazards (taken branches)  │                 │
 * │  │                                       │                 │
 * │  │  Compare:                             │                 │
 * │  │  • Current instruction operands       │                 │
 * │  │  • EXE stage destination register     │                 │
 * │  │  • WB stage destination register      │                 │
 * │  └──────────┬────────────────────────────┘                 │
 * │             │                                              │
 * │             ▼                                              │
 * │  ┌────────────────────────────┐                           │
 * │  │  Decision                  │                           │
 * │  └──────┬─────────────────────┘                           │
 * │         │                                                  │
 * │    ┌────┴─────┐                                           │
 * │    │          │                                           │
 * │    ▼          ▼                                           │
 * │  HAZARD     NO HAZARD                                     │
 * │    │          │                                           │
 * │    │          ▼                                           │
 * │    │    ┌─────────────────────────────────┐              │
 * │    │    │  Forward to EXE Stage           │              │
 * │    │    │  via prIF2EXE pipeline register │              │
 * │    │    └─────────────────────────────────┘              │
 * │    │                                                      │
 * │    ▼                                                      │
 * │  ┌─────────────────────────────────┐                     │
 * │  │  STALL Pipeline                 │                     │
 * │  │  - Keep packet in slave port    │                     │
 * │  │  - Advance EXE → WB             │                     │
 * │  │  - Insert bubble in EXE         │                     │
 * │  │  - Retry next cycle             │                     │
 * │  └─────────────────────────────────┘                     │
 * │                                                            │
 * │  Outputs:                                                  │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  prIF2EXE-in (Pipeline Register)     │                 │
 * │  │  - Forwards InstPacket to EXE stage  │                 │
 * │  └──────────────────────────────────────┘                 │
 * │                                                            │
 * │  State Tracking:                                           │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  EXEInstPacket                       │                 │
 * │  │  - Instruction currently in EXE stage│                 │
 * │  │  - Used for data hazard detection    │                 │
 * │  │                                       │                 │
 * │  │  WBInstPacket                        │                 │
 * │  │  - Instruction currently in WB stage │                 │
 * │  │  - Used for data hazard detection    │                 │
 * │  └──────────────────────────────────────┘                 │
 * └────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Hazard Detection:**
 *
 * The IF stage implements two types of hazard detection:
 *
 * **1. Data Hazards (Read-After-Write - RAW):**
 *
 * A data hazard occurs when an instruction depends on the result of a
 * previous instruction that hasn't been written back yet.
 *
 * ```
 * Example RAW Hazard:
 * ───────────────────────────────────────────────────────
 * T=1: add x3, x1, x2    # x3 will be written
 * T=2: sub x5, x3, x4    # x5 depends on x3 (HAZARD!)
 *
 * Without Stall:
 *   T=1: add executes, x3 not yet updated
 *   T=2: sub reads old value of x3 → WRONG!
 *
 * With Stall:
 *   T=1: add executes
 *   T=2: sub stalled (waits for x3)
 *   T=3: add writes back x3
 *   T=4: sub reads correct value of x3 → CORRECT!
 * ```
 *
 * **Detection Logic:**
 * ```cpp
 * // IF-EXE Hazard: Current inst uses register that EXE inst will write
 * dataHazard = (current.rs1 == EXE.rd || current.rs2 == EXE.rd)
 *
 * // IF-WB Hazard: Current inst uses register that WB inst is writing
 * dataHazard |= (current.rs1 == WB.rd || current.rs2 == WB.rd)
 * ```
 *
 * **2. Control Hazards (Branch/Jump):**
 *
 * A control hazard occurs when a branch or jump changes the PC, potentially
 * invalidating the next instruction.
 *
 * ```
 * Example Control Hazard:
 * ───────────────────────────────────────────────────────
 * T=1: beq x1, x2, target  # Conditional branch
 * T=2: add x3, x1, x2      # Next sequential instruction
 *
 * If branch is taken:
 *   - PC jumps to 'target'
 *   - 'add' instruction is invalid
 *   - Must flush pipeline (insert bubble)
 *
 * Detection:
 *   controlHazard = EXEInstPacket->isTakenBranch
 * ```
 *
 * **Pipeline Register Flow:**
 *
 * ```
 * prIF2EXE Pipeline Register
 * ──────────────────────────────────────────────────────
 *
 * Producer (IFStage):                 Consumer (EXEStage):
 * ─────────────────────              ──────────────────────
 * 1. Receive InstPacket              1. Read prIF2EXE-out
 *    from CPU via soc-s                 (automatically available)
 *
 * 2. Check hazards                   2. Process instruction
 *    (data, control)
 *                                    3. Forward to prEXE2WB
 * 3. If no hazard:
 *    push to prIF2EXE-in             4. Advance to WB stage
 *    (becomes prIF2EXE-out
 *     next cycle)
 *
 * 4. If hazard:
 *    STALL (don't push)
 *    forceStepInNextIteration()
 * ```
 *
 * **Stall Mechanism:**
 *
 * When a hazard is detected, the IF stage stalls the pipeline:
 *
 * ```
 * Normal Flow (No Hazard):
 * ────────────────────────────────────────
 * T=N: IF receives packet from CPU
 *      └─ Check hazards → NONE
 *      └─ pop() from soc-s
 *      └─ push() to prIF2EXE
 *      └─ Update EXEInstPacket tracker
 *
 * T=N+1: EXE processes packet...
 *
 * Stall Flow (Hazard Detected):
 * ────────────────────────────────────────
 * T=N: IF receives packet from CPU
 *      └─ Check hazards → DATA HAZARD!
 *      └─ Do NOT pop() from soc-s
 *      └─ Do NOT push() to prIF2EXE
 *      └─ Advance WB ← EXE (clear hazard)
 *      └─ Clear EXEInstPacket (bubble)
 *      └─ forceStepInNextIteration()
 *
 * T=N+1: IF retries same packet
 *        └─ Check hazards → Likely cleared
 *        └─ pop() and push() if clear
 * ```
 *
 * **Step Function Behavior:**
 *
 * The `step()` function is the main entry point called each cycle:
 *
 * ```cpp
 * void IFStage::step() {
 *     // 1. Check if packet available on slave port
 *     if (!soc-s->isPopValid())
 *         return;  // No packet, nothing to do
 *
 *     // 2. Peek at packet (don't pop yet)
 *     InstPacket* packet = soc-s->front();
 *
 *     // 3. Detect data hazards
 *     dataHazard = check_RAW_dependencies(packet);
 *
 *     // 4. Detect control hazards
 *     controlHazard = EXEInstPacket->isTakenBranch;
 *
 *     // 5. Make decision
 *     if (no hazards) {
 *         pop() packet from soc-s;
 *         push() to prIF2EXE;
 *         Update trackers;
 *     } else {
 *         Stall (don't pop);
 *         Advance pipeline to clear hazard;
 *         Force retry next cycle;
 *     }
 * }
 * ```
 *
 * **Interaction with CPU:**
 *
 * The IF stage communicates with the CPU through the port mechanism:
 *
 * ```
 * CPU Side:                          IFStage Side:
 * ─────────────────                  ──────────────────────
 * 1. Execute instruction             1. Wait for packet on soc-s
 *
 * 2. Try push(InstPacket)            2. Packet arrives in queue
 *    to sIF-m port
 *                                    3. step() called by simulator
 * 3. If port full:
 *    - Store pendingInstPacket       4. Check hazards
 *    - Wait for retry
 *                                    5a. No hazard:
 * 4. On retry callback:                  - pop() accepts packet
 *    - Resend pendingInstPacket          - CPU's push() succeeds
 *    - Schedule next event
 *                                    5b. Hazard:
 *                                        - Don't pop()
 *                                        - Retry next cycle
 * ```
 *
 * **Forwarding Logic:**
 *
 * When no hazards are detected, the IF stage forwards the packet:
 *
 * ```
 * instPacketHandler(Tick when, SimPacket* pkt):
 *     // 1. Push to pipeline register
 *     prIF2EXE->push(pkt)
 *
 *     // 2. Update state trackers
 *     WBInstPacket = EXEInstPacket    // Previous EXE → WB
 *     EXEInstPacket = pkt             // Current IF → EXE
 *
 *     // 3. EXE stage automatically sees packet
 *     //    via prIF2EXE-out on next step()
 * ```
 *
 * **Example Execution Timeline:**
 *
 * ```
 * Assembly Program:
 *   add  x3, x1, x2    # I1
 *   sub  x5, x3, x4    # I2 (uses x3 - HAZARD!)
 *   or   x6, x5, x7    # I3 (uses x5 - HAZARD!)
 *   xor  x8, x6, x9    # I4 (uses x6 - HAZARD!)
 *
 * Execution with Stalls:
 * ────────────────────────────────────────────────────────
 * T=1: CPU executes I1 (add)
 *      IF receives I1
 *      └─ No hazard (first instruction)
 *      └─ Forward to EXE
 *      EXE: empty
 *      WB: empty
 *
 * T=2: CPU executes I2 (sub)
 *      IF receives I2
 *      └─ Data hazard! I2.rs1(x3) == I1.rd(x3)
 *      └─ STALL (don't forward)
 *      └─ Advance: WB ← I1, EXE ← bubble
 *      EXE: I1
 *      WB: empty
 *
 * T=3: IF retries I2
 *      └─ No hazard (I1 in WB, written back)
 *      └─ Forward to EXE
 *      EXE: bubble
 *      WB: I1 (writes x3)
 *
 * T=4: CPU executes I3 (or)
 *      IF receives I3
 *      └─ Data hazard! I3.rs1(x5) == I2.rd(x5)
 *      └─ STALL
 *      EXE: I2
 *      WB: bubble
 *
 * T=5: IF retries I3
 *      └─ No hazard
 *      └─ Forward to EXE
 *      EXE: bubble
 *      WB: I2 (writes x5)
 *
 * ... (pattern continues)
 * ```
 *
 * **Key Design Decisions:**
 *
 * 1. **Conservative Hazard Detection**: Detects all possible RAW hazards,
 *    even if forwarding could resolve some
 *
 * 2. **Two-Stage Look-Ahead**: Checks both EXE and WB stages for dependencies
 *
 * 3. **Bubble Insertion**: On stall, advances pipeline but inserts bubble
 *    (nullptr) to maintain timing
 *
 * 4. **Forced Retry**: Uses forceStepInNextIteration() to ensure stalled
 *    instructions are retried
 *
 * @see EXEStage for execution stage implementation
 * @see CPU for instruction execution and packet generation
 * @see InstPacket for instruction packet structure
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "IFStage.hh"

void IFStage::step() {
	// Only move forward when
	// 1. the incoming slave port has instruction ready
	// 2. the downstream pipeline register is available

	// check hazards
	bool dataHazard = false;
	if (this->getSlavePort("soc-s")->isPopValid()) {
		InstPacket* instPacket = ((InstPacket*)this->getSlavePort("soc-s")->front());

		// IF, EXE hazard
		dataHazard =
		    EXEInstPacket && (instPacket->inst.a2.type == OPTYPE_REG && EXEInstPacket->inst.a1.type == OPTYPE_REG &&
		                          instPacket->inst.a2.reg == EXEInstPacket->inst.a1.reg ||
		                      instPacket->inst.a3.type == OPTYPE_REG && EXEInstPacket->inst.a1.type == OPTYPE_REG &&
		                          instPacket->inst.a3.reg == EXEInstPacket->inst.a1.reg);

		// IF, WB hazard
		dataHazard |=
		    WBInstPacket && (instPacket->inst.a2.type == OPTYPE_REG && WBInstPacket->inst.a1.type == OPTYPE_REG &&
		                         instPacket->inst.a2.reg == WBInstPacket->inst.a1.reg ||
		                     instPacket->inst.a3.type == OPTYPE_REG && WBInstPacket->inst.a1.type == OPTYPE_REG &&
		                         instPacket->inst.a3.reg == WBInstPacket->inst.a1.reg);
	}
	bool controlHazard = false;
	if (EXEInstPacket) { controlHazard = EXEInstPacket->isTakenBranch; }

	Tick currTick = top->getGlobalTick();
	if (this->getSlavePort("soc-s")->isPopValid()) {
		CLASS_INFO << "   IFStage step() : has an inbound  InstPacket availble ";

		if (!dataHazard && !controlHazard) {
			CLASS_INFO << "   IFStage step() :  popped an InstPacket";
			SimPacket* pkt = this->getSlavePort("soc-s")->pop();
			this->accept(currTick, *pkt);

		} else {
			WBInstPacket  = EXEInstPacket;
			EXEInstPacket = nullptr;
			// There are still pending request but no new input in the next cycle
			this->forceStepInNextIteration();
			if (dataHazard) CLASS_INFO << "   IFStage step() :  data Hazard detected. Stall IFStage";
			if (controlHazard) CLASS_INFO << "   IFStage step() :  control Hazard detected. Stall IFStage";
		}
	}
}

void IFStage::instPacketHandler(Tick when, SimPacket* pkt) {
	CLASS_INFO << "   IFStage::instPacketHandler() has received InstPacket @PC=" << ((InstPacket*)pkt)->pc
	           << " from soc-s and push it to prIF2EXE-in";

	// push to the prIF2EXE register
	if (!this->getPipeRegister("prIF2EXE-in")->push(pkt)) { CLASS_ERROR << "IFStage failed to handle an InstPacket!"; }
	WBInstPacket  = EXEInstPacket;
	EXEInstPacket = (InstPacket*)pkt;
}
