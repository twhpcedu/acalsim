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
 * @file IFStage.cc
 * @brief Instruction Fetch Stage (Minimal Template Version)
 *
 * @details
 * This file implements the IFStage (Instruction Fetch Stage) for the **simplified
 * RISC-V simulator template**. Unlike the full simulator in src/riscv/, this version
 * of IFStage is **minimal and primarily for logging purposes**.
 *
 * **KEY SIMPLIFICATION IN TEMPLATE:**
 * In this template, instruction fetching happens directly in CPU::fetchInstr(), not here.
 * The IFStage exists primarily to:
 * - Receive completed InstPackets from CPU for logging
 * - Demonstrate the port-based communication mechanism
 * - Provide a placeholder for future pipeline expansion
 * - Recycle instruction packets back to the object pool
 *
 * **Contrast with Full Simulator:**
 * | Feature           | Template (riscvSimTemplate)     | Full (src/riscv/)            |
 * |-------------------|---------------------------------|------------------------------|
 * | Instruction Fetch | Done in CPU::fetchInstr()       | Done in IFStage::step()      |
 * | Role              | Packet logging & recycling      | Active fetch from I-cache    |
 * | Pipeline Register | Exists but unused (prIF2EXE)    | Active (forwards to EXEStage)|
 * | Complexity        | Minimal (2 functions)           | Full fetch logic             |
 * | Educational Use   | Shows port mechanism            | Shows real pipeline stage    |
 *
 * **Architecture in Template:**
 * @code
 *                    ┌────────────────────────────┐
 *                    │      CPU (executes)        │
 *                    └──────────┬─────────────────┘
 *                               │
 *                               │ InstPacket
 *                               │ (via "sIF-m" MasterPort)
 *                               ▼
 *                    ┌────────────────────────────┐
 *                    │     IFStage (logs)         │
 *                    │  - Receives InstPacket     │
 *                    │  - Logs completion         │
 *                    │  - Recycles packet         │
 *                    └────────────────────────────┘
 *                               │
 *                               ▼
 *                    ┌────────────────────────────┐
 *                    │   RecycleContainer         │
 *                    │  (Object pool)             │
 *                    └────────────────────────────┘
 * @endcode
 *
 * **Architecture in Full Simulator:**
 * @code
 *                    ┌────────────────────────────┐
 *                    │    IFStage (fetches)       │
 *                    │  - Read PC                 │
 *                    │  - Access I-cache/memory   │
 *                    │  - Create InstPacket       │
 *                    │  - Handle cache miss       │
 *                    └──────────┬─────────────────┘
 *                               │
 *                               │ InstPacket
 *                               │ (via prIF2EXE)
 *                               ▼
 *                    ┌────────────────────────────┐
 *                    │    EXEStage (executes)     │
 *                    │  - Decode instruction      │
 *                    │  - Execute ALU/branch      │
 *                    │  - Memory operations       │
 *                    └────────────────────────────┘
 * @endcode
 *
 * **Port Communication:**
 * IFStage uses ACALSim's port mechanism for inter-module communication:
 * - **SlavePort "soc-s":** Receives InstPackets from SOC/CPU
 * - **Pipeline Register "prIF2EXE-in":** Would forward to EXEStage (unused in template)
 *
 * **Step Function:**
 * The step() function is called each simulation cycle but does minimal work:
 * 1. Check if SlavePort has an InstPacket ready
 * 2. Check if downstream pipeline register is available (always true in template)
 * 3. Pop packet from port
 * 4. Call accept() which invokes instPacketHandler()
 * 5. Recycle packet back to pool
 *
 * **Execution Flow:**
 * @code
 * CPU::commitInstr()
 *    │
 *    ├─> Push InstPacket to "sIF-m" MasterPort
 *    │
 *    ▼
 * IFStage::step()  (called next cycle)
 *    │
 *    ├─> Check if SlavePort "soc-s" has packet
 *    ├─> Check if "prIF2EXE-in" pipeline register available
 *    ├─> Pop InstPacket from port
 *    ├─> accept(tick, packet)
 *    │   └─> instPacketHandler(tick, packet)
 *    │       ├─> Log packet receipt
 *    │       └─> Recycle packet to pool
 *    │
 *    └─> (Packet recycled, memory freed)
 * @endcode
 *
 * **Pipeline Register Mechanism:**
 * The template includes "prIF2EXE-in" pipeline register to demonstrate the concept,
 * but it's not actively used:
 * - In full simulator: Stores instruction between IF and EX stages
 * - In template: Placeholder for future expansion
 * - Always reports "not stalled" (no backpressure)
 *
 * **Extension Path:**
 * To convert this template into a pipelined simulator:
 * 1. Move CPU::fetchInstr() logic to IFStage::step()
 * 2. Have IFStage create InstPackets from instruction memory
 * 3. Write InstPackets to prIF2EXE pipeline register
 * 4. Create EXEStage to read from prIF2EXE and execute
 * 5. Add hazard detection and forwarding logic
 * 6. Implement branch prediction in IFStage
 *
 * **Object Recycling:**
 * InstPackets are recycled for memory efficiency:
 * - Allocated from RecycleContainer pool (not new/malloc)
 * - Returned to pool in instPacketHandler() (not delete/free)
 * - Reduces allocation overhead in event-driven simulation
 * - Critical for performance with millions of instructions
 *
 * **Why This Design?**
 * The minimal IFStage serves pedagogical purposes:
 * - **Simplicity:** Students learn CPU execution without pipeline complexity
 * - **Gradual Learning:** Can add pipeline stages incrementally
 * - **Port Mechanism:** Demonstrates inter-module communication
 * - **Extensibility:** Clear path to add full IF functionality
 *
 * **Usage in Simulation:**
 * @code
 * // During SOC initialization:
 * SOCTop creates IFStage
 * IFStage registered as module
 * CPU's "sIF-m" MasterPort connected to IFStage's "soc-s" SlavePort
 *
 * // During simulation:
 * CPU executes instruction -> sends InstPacket to IFStage
 * IFStage::step() called -> pops packet -> recycles it
 * @endcode
 *
 * @see CPU CPU timing model with direct instruction fetch
 * @see SOC System-on-chip that connects CPU and IFStage
 * @see InstPacket Instruction packet structure
 * @see src/riscv/libs/IFStage.cc Full IFStage implementation
 *
 * @note This is a TEMPLATE version - much simpler than src/riscv/
 * @note Real instruction fetching happens in CPU, not here
 * @note Primarily for logging and demonstrating port communication
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @version 1.0
 * @copyright Apache License 2.0
 */

#include "IFStage.hh"

/**
 * @brief Step function called each simulation cycle
 *
 * @details
 * In the template version, this function simply:
 * 1. Checks if an InstPacket is available from SOC/CPU
 * 2. Checks if downstream pipeline register is available (always true)
 * 3. Pops the packet and processes it
 * 4. Recycles the packet back to the object pool
 *
 * **Minimal Template Behavior:**
 * Unlike the full simulator where IFStage actively fetches instructions,
 * here we just receive already-executed instructions for logging.
 *
 * **Port Checking:**
 * - SlavePort "soc-s" connects to CPU's MasterPort "sIF-m"
 * - isPopValid() checks if packet is available
 * - pop() removes packet from port queue
 *
 * **Pipeline Register:**
 * - "prIF2EXE-in" would store instruction for next stage
 * - In template, it's unused (no EXEStage)
 * - isStalled() always returns false
 *
 * @note Called automatically by simulation loop each tick
 * @note Does NOT fetch instructions (that's CPU::fetchInstr())
 *
 * @see instPacketHandler() Processes the received packet
 * @see CPU::commitInstr() Sends packets to this stage
 */
void IFStage::step() {
	// Only move forward when
	// 1. the incoming slave port has instruction ready
	// 2. the downstream pipeline register is available
	Tick currTick = top->getGlobalTick();
	if (this->getSlavePort("soc-s")->isPopValid()) {
		CLASS_INFO << "IFStage has an inbound  InstPacket availble ";
		if (!this->getPipeRegister("prIF2EXE-in")->isStalled()) {
			CLASS_INFO << "IFStage has popped an InstPacket";
			SimPacket* pkt = this->getSlavePort("soc-s")->pop();
			this->accept(currTick, *pkt);
		}
	}
}

/**
 * @brief Handles received InstPacket (logging and recycling)
 *
 * @details
 * This is the packet processing callback invoked by accept().
 * In the template version, it simply:
 * 1. Logs that the packet was received
 * 2. Recycles the packet back to the RecycleContainer
 *
 * **Object Recycling:**
 * InstPackets are allocated from RecycleContainer (object pool):
 * - CPU acquires packet: rc->acquire<InstPacket>()
 * - IFStage recycles packet: rc->recycle(pkt)
 * - Avoids new/delete overhead in hot path
 *
 * **In Full Simulator:**
 * This function would:
 * - Decode instruction fields
 * - Write to prIF2EXE pipeline register
 * - Signal EXEStage that instruction is ready
 *
 * @param when Simulation tick when packet was received
 * @param pkt Pointer to the InstPacket to process
 *
 * @note Packet memory is returned to pool, not freed
 * @note In full simulator, packet would be forwarded, not recycled here
 *
 * @see RecycleContainer Object pool for efficient allocation
 * @see InstPacket Instruction packet structure
 */
void IFStage::instPacketHandler(Tick when, SimPacket* pkt) {
	CLASS_INFO << "IFStage has received and recycled an InstPacket";
	acalsim::top->getRecycleContainer()->recycle(pkt);
}
