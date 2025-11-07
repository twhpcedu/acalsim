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
 * @file EXEStage.cc
 * @brief Execution (EXE) Pipeline Stage Implementation
 *
 * @details
 * This file implements the Execution (EXE) pipeline stage, which serves as the
 * second stage in the pipeline visualization of the RISC-V simulator. The EXE
 * stage receives instruction packets from the IF stage, checks for control
 * hazards, and forwards them to the WB (writeback) stage.
 *
 * **Pipeline Stage Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────┐
 * │                       EXEStage                             │
 * │  (Execution & Control Hazard Detection)                   │
 * │                                                            │
 * │  Inputs:                                                   │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  prIF2EXE-out (Pipeline Register)    │                 │
 * │  │  - Receives InstPacket from IF stage │                 │
 * │  │  - Contains executed instruction     │                 │
 * │  │  - PC, opcode, operands, metadata    │                 │
 * │  │  - isTakenBranch flag                │                 │
 * │  └──────────────────────────────────────┘                 │
 * │                │                                           │
 * │                ▼                                           │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  Control Hazard Detection            │                 │
 * │  │                                       │                 │
 * │  │  Check:                               │                 │
 * │  │  • isTakenBranch flag                │                 │
 * │  │                                       │                 │
 * │  │  If branch taken:                     │                 │
 * │  │  • Pipeline flush may be needed       │                 │
 * │  │  • Logged for debugging               │                 │
 * │  └──────────┬────────────────────────────┘                 │
 * │             │                                              │
 * │             ▼                                              │
 * │  ┌────────────────────────────┐                           │
 * │  │  Forward to WB Stage       │                           │
 * │  │  via prEXE2WB-in           │                           │
 * │  └────────────────────────────┘                           │
 * │                                                            │
 * │  Outputs:                                                  │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  prEXE2WB-in (Pipeline Register)     │                 │
 * │  │  - Forwards InstPacket to WB stage   │                 │
 * │  └──────────────────────────────────────┘                 │
 * │                                                            │
 * │  State Tracking:                                           │
 * │  ┌──────────────────────────────────────┐                 │
 * │  │  WBInstPacket                        │                 │
 * │  │  - Instruction currently in WB stage │                 │
 * │  │  - Tracked for pipeline visibility   │                 │
 * │  └──────────────────────────────────────┘                 │
 * └────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Three-Stage Pipeline Visualization:**
 *
 * The complete pipeline consists of three stages:
 *
 * ```
 * ┌──────────┐      ┌──────────┐      ┌──────────┐
 * │ IFStage  │ ───▶ │ EXEStage │ ───▶ │ WBStage  │
 * └──────────┘      └──────────┘      └──────────┘
 *      │                  │                 │
 *      │                  │                 │
 *  Hazard            Control           Retire
 *  Detection         Hazard            Instruction
 *  (Data)            Detection
 *
 * Pipeline Registers:
 *  prIF2EXE           prEXE2WB
 *     │                  │
 *     ▼                  ▼
 *  IF → EXE           EXE → WB
 * ```
 *
 * **EXE Stage Responsibilities:**
 *
 * 1. **Receive Packets from IF Stage:**
 *    - Check if prIF2EXE-out has a valid packet
 *    - Retrieve InstPacket from pipeline register
 *
 * 2. **Control Hazard Detection:**
 *    - Check isTakenBranch flag
 *    - Log control hazard information
 *    - Note: Actual instruction has already executed in CPU
 *
 * 3. **Forward to WB Stage:**
 *    - Check if prEXE2WB-in is available (not stalled)
 *    - Push packet to prEXE2WB-in
 *    - Update WBInstPacket tracker
 *
 * 4. **Pipeline Register Management:**
 *    - Pop packet from prIF2EXE-out
 *    - Push packet to prEXE2WB-in
 *    - Maintain pipeline flow
 *
 * **Step Function Behavior:**
 *
 * ```cpp
 * void EXEStage::step() {
 *     // 1. Check if packet available from IF stage
 *     if (!prIF2EXE-out->isValid())
 *         return;  // No packet, nothing to do
 *
 *     // 2. Get packet from pipeline register
 *     InstPacket* packet = prIF2EXE-out->value();
 *
 *     // 3. Check for control hazard
 *     controlHazard = packet->isTakenBranch;
 *     if (controlHazard)
 *         LOG("Control hazard detected");
 *
 *     // 4. Check if WB stage is ready
 *     if (prEXE2WB-in->isStalled())
 *         return;  // WB not ready, wait
 *
 *     // 5. Forward packet to WB stage
 *     pop() from prIF2EXE-out;
 *     accept() packet (calls instPacketHandler);
 *     // instPacketHandler pushes to prEXE2WB-in
 * }
 * ```
 *
 * **Control Hazard Detection:**
 *
 * Control hazards arise from branch and jump instructions:
 *
 * ```
 * Example Control Hazard:
 * ──────────────────────────────────────────────────────
 * Program:
 *   0x00: beq x1, x2, target   # Conditional branch
 *   0x04: add x3, x1, x2       # Next sequential instruction
 *   0x08: sub x4, x3, x5       # Another instruction
 *   ...
 *   0x20: target: or x6, x7, x8  # Branch target
 *
 * If branch is taken (x1 == x2):
 *   1. CPU sets isTakenBranch = true
 *   2. PC jumps to 0x20 (target)
 *   3. EXE stage detects control hazard
 *   4. Instructions at 0x04, 0x08 should be flushed
 *      (if they were fetched)
 *
 * Detection in EXE stage:
 *   if (instPacket->isTakenBranch) {
 *       CLASS_INFO << "Control hazard detected";
 *       // In this implementation, hazard is noted but
 *       // no explicit flush needed (single-cycle model)
 *   }
 * ```
 *
 * **Pipeline Register Flow:**
 *
 * ```
 * prIF2EXE (from IF)          prEXE2WB (to WB)
 * ──────────────────          ────────────────
 *
 * Producer: IFStage           Producer: EXEStage
 * Consumer: EXEStage          Consumer: WBStage
 *
 * IFStage:                    EXEStage:
 *   push(packet)                push(packet)
 *   ↓                           ↓
 * prIF2EXE-in                 prEXE2WB-in
 *   ↓ (next cycle)              ↓ (next cycle)
 * prIF2EXE-out                prEXE2WB-out
 *   ↓                           ↓
 * EXEStage:                   WBStage:
 *   pop() & process             pop() & retire
 * ```
 *
 * **Interaction with Pipeline Registers:**
 *
 * ```
 * Read from prIF2EXE:
 * ────────────────────────────────────────
 * 1. Check: prIF2EXE-out->isValid()
 *    - true: Packet available
 *    - false: No packet, return
 *
 * 2. Peek: packet = prIF2EXE-out->value()
 *    - Get packet without removing
 *    - Check for control hazards
 *
 * 3. Pop: prIF2EXE-out->pop()
 *    - Remove packet from register
 *    - Frees space for IF stage
 *
 * Write to prEXE2WB:
 * ────────────────────────────────────────
 * 1. Check: !prEXE2WB-in->isStalled()
 *    - true: Can accept packet
 *    - false: WB stage busy, wait
 *
 * 2. Push: prEXE2WB-in->push(packet)
 *    - Store packet in register
 *    - Will be available to WB next cycle
 *
 * 3. Update: WBInstPacket = packet
 *    - Track for pipeline visibility
 * ```
 *
 * **Example Execution Timeline:**
 *
 * ```
 * Assembly Program:
 *   I1: add  x3, x1, x2
 *   I2: beq  x1, x2, target  (branch taken)
 *   I3: sub  x4, x3, x5      (should be flushed)
 *   I4: or   x6, x7, x8      (should be flushed)
 *   ...
 *   target: and x9, x10, x11
 *
 * Timeline:
 * ────────────────────────────────────────────────────────
 * T=1: I1 enters pipeline
 *      IF: I1
 *      EXE: empty
 *      WB: empty
 *
 * T=2: I2 enters pipeline, I1 advances
 *      IF: I2
 *      EXE: I1 (no control hazard)
 *      WB: empty
 *
 * T=3: I3 enters pipeline, I2 advances, I1 retires
 *      IF: I3
 *      EXE: I2 (control hazard detected! isTakenBranch=true)
 *      WB: I1
 *
 * T=4: I4 would enter, but branch taken
 *      IF: target instruction (and x9, x10, x11)
 *      EXE: I3 (stalled/flushed due to control hazard)
 *      WB: I2
 *
 * T=5: Pipeline continues from target
 *      IF: next instruction after target
 *      EXE: target instruction
 *      WB: empty (I3 was flushed)
 * ```
 *
 * **instPacketHandler Method:**
 *
 * This method is called when a packet is accepted from prIF2EXE:
 *
 * ```cpp
 * void EXEStage::instPacketHandler(Tick when, SimPacket* pkt) {
 *     // Log receipt
 *     CLASS_INFO << "EXE received InstPacket @ PC="
 *                << ((InstPacket*)pkt)->pc;
 *
 *     // Push to next stage
 *     if (!prEXE2WB->push(pkt)) {
 *         CLASS_ERROR << "EXE failed to push to WB!";
 *     }
 *
 *     // Update tracker
 *     WBInstPacket = (InstPacket*)pkt;
 * }
 * ```
 *
 * **Key Design Decisions:**
 *
 * 1. **Simplified Execution Model:**
 *    - Actual execution happens in CPU, not EXE stage
 *    - EXE stage is primarily for pipeline visualization
 *    - Demonstrates pipeline concept without full complexity
 *
 * 2. **Control Hazard Detection:**
 *    - Detects taken branches via isTakenBranch flag
 *    - Logs hazard for debugging/analysis
 *    - In single-cycle model, flush is implicit
 *
 * 3. **Minimal Stall Logic:**
 *    - Only stalls if WB stage is full
 *    - Simpler than full multi-cycle pipeline
 *
 * 4. **Pipeline Register Interface:**
 *    - Uses standard push/pop interface
 *    - Maintains pipeline flow
 *    - Enables timing-accurate simulation
 *
 * **Differences from Real Hardware:**
 *
 * In a real pipelined RISC-V processor:
 *
 * ```
 * Real Hardware EXE Stage:
 * ────────────────────────────────────────
 * • Performs actual ALU operations
 * • Computes branch targets
 * • Evaluates branch conditions
 * • Generates memory addresses
 * • Detects exceptions
 *
 * This Simulator's EXE Stage:
 * ────────────────────────────────────────
 * • Receives already-executed instructions
 * • Detects control hazards
 * • Forwards packets to WB
 * • Demonstrates pipeline concept
 * • Enables timing visualization
 * ```
 *
 * **Why This Design?**
 *
 * This simplified EXE stage design serves educational purposes:
 *
 * 1. **Separation of Concerns:**
 *    - CPU handles functional execution
 *    - Pipeline stages handle timing visualization
 *
 * 2. **Easier Understanding:**
 *    - Students see pipeline stages clearly
 *    - Hazard detection is explicit and visible
 *
 * 3. **Incremental Complexity:**
 *    - Start with single-cycle (CPU)
 *    - Add pipeline visualization (IF/EXE/WB)
 *    - Can extend to full multi-cycle later
 *
 * 4. **Event-Driven Friendly:**
 *    - Fits naturally into event-driven framework
 *    - Easy to track packet flow
 *    - Simple debugging
 *
 * @see IFStage for instruction fetch and data hazard detection
 * @see WBStage for writeback stage
 * @see CPU for actual instruction execution
 * @see InstPacket for instruction packet structure
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "EXEStage.hh"

void EXEStage::step() {
	// Only move forward when
	// 1. the incoming slave port has instruction ready
	// 2. the downstream pipeline register is available
	Tick currTick = top->getGlobalTick();

	InstPacket* inboundPacket = nullptr;

	// check hazards
	bool controlHazard = false;
	if (this->getPipeRegister("prIF2EXE-out")->isValid()) {
		InstPacket* instPacket = ((InstPacket*)this->getPipeRegister("prIF2EXE-out")->value());
		controlHazard          = instPacket->isTakenBranch;
		inboundPacket          = instPacket;
	}

	if (inboundPacket)
		CLASS_INFO << "   EXEStage step() an InstPacket @PC=" << inboundPacket->pc
		           << " controlHazard: " << (controlHazard ? "Yes" : "No");
	else
		CLASS_INFO << "   EXEStage step(), no inbound packet";

	if (this->getPipeRegister("prIF2EXE-out")->isValid() && !this->getPipeRegister("prEXE2WB-in")->isStalled()) {
		SimPacket* pkt = this->getPipeRegister("prIF2EXE-out")->pop();
		// process tht packet regardless whether it has control hazard or not
		this->accept(currTick, *pkt);
	}
}

void EXEStage::instPacketHandler(Tick when, SimPacket* pkt) {
	CLASS_INFO << "   EXEStage::instPacketHandler()  has received and an InstPacket @PC=" << ((InstPacket*)pkt)->pc
	           << " from prIF2EXE-out and push it to prEXE2WB-in";

	// push to the prEXE2WB register
	if (!this->getPipeRegister("prEXE2WB-in")->push(pkt)) { CLASS_ERROR << "EXEStage failed to handle an InstPacket!"; }
	WBInstPacket = (InstPacket*)pkt;
}
