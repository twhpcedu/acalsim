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
 * @file DataReq.cc
 * @brief Data request packet implementation using visitor pattern
 *
 * This file implements the DataReqPacket class, which demonstrates the visitor
 * pattern for type-safe packet handling in ACALSim. DataReqPacket represents
 * a simple data transfer packet that can carry arbitrary payload data.
 *
 * **Visitor Pattern Overview:**
 *
 * The visitor pattern enables double-dispatch for packet handling:
 * ```
 * Packet Type × Receiver Type → Specific Handler
 *
 * DataReqPacket × PE         → PE::dataPacketHandler()
 * DataReqPacket × SimModule  → (not implemented)
 * PEReqPacket   × PE         → PE::peReqPacketHandler()
 * ...
 * ```
 *
 * **How Visitor Pattern Works:**
 *
 * Traditional approach (without visitor):
 * ```cpp
 * void PE::handlePacket(SimPacket* pkt) {
 *     if (auto dataPkt = dynamic_cast<DataReqPacket*>(pkt)) {
 *         // Handle data packet
 *     } else if (auto peReqPkt = dynamic_cast<PEReqPacket*>(pkt)) {
 *         // Handle PE request
 *     }
 *     // Problem: PE needs to know about all packet types
 * }
 * ```
 *
 * Visitor approach (this implementation):
 * ```cpp
 * // Packet knows how to visit different simulator types
 * void DataReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto pe = dynamic_cast<PE*>(&simulator)) {
 *         pe->dataPacketHandler(when, this);
 *     }
 * }
 *
 * // Usage:
 * packet->visit(currentTick, *simulator);
 * // Automatically calls correct handler based on both packet and simulator type
 * ```
 *
 * **Benefits of Visitor Pattern:**
 *
 * 1. **Extensibility**: New packet types don't require changes to simulators
 * 2. **Type Safety**: Compile-time checking of handler signatures
 * 3. **Separation of Concerns**: Packet routing logic in packet classes
 * 4. **Open-Closed Principle**: Open for extension, closed for modification
 *
 * **DataReqPacket Usage Pattern:**
 *
 * Creating and sending:
 * ```cpp
 * // In TrafficGenerator or other source:
 * int* data = new int(42);  // Allocate payload
 * DataReqPacket* pkt = new DataReqPacket(tick + 10, data);
 *
 * // Option 1: Direct visit (immediate processing)
 * pkt->visit(currentTick, *peSimulator);
 *
 * // Option 2: Send via channel (delayed processing)
 * // Would need EventPacket wrapper for channel transmission
 * ```
 *
 * Receiving and handling:
 * ```cpp
 * // In PE::dataPacketHandler():
 * int* payload = (int*)dataPkt->getData();
 * int value = *payload;  // Extract data
 * // Process value...
 * ```
 *
 * **Packet Hierarchy:**
 * ```
 * SimPacket (abstract base)
 *   ├─ DataPacket (base for data transfers)
 *   │   └─ DataReqPacket (this class)
 *   ├─ PEReqPacket (PE computation requests)
 *   ├─ PERespPacket (PE computation responses)
 *   └─ EventPacket (wraps SimEvents for channels)
 * ```
 *
 * **Design Notes:**
 *
 * 1. **Why Two visit() Overloads?**
 *    - visit(SimModule&): For module-level handling
 *    - visit(SimBase&): For simulator-level handling
 *    - Allows different handling at different hierarchical levels
 *
 * 2. **Why Not Used in Main Example?**
 *    - testCommunication focuses on PEReqPacket
 *    - DataReqPacket demonstrates alternative packet type
 *    - Shows extensibility of the pattern
 *    - Could be used for raw data transfers vs computations
 *
 * 3. **Memory Management:**
 *    - DataReqPacket stores void* to payload
 *    - Caller responsible for payload allocation
 *    - Handler or packet destructor must free payload
 *    - Packet itself managed by sender/framework
 *
 * **Comparison with PEReqPacket:**
 *
 * DataReqPacket:
 * - Simple data transfer
 * - Generic void* payload
 * - No built-in callback mechanism
 * - No response packet association
 * - Lightweight and flexible
 *
 * PEReqPacket:
 * - Structured computation request
 * - Typed parameters (a, b, c)
 * - Callback for response handling
 * - Associated response packet
 * - More complex but feature-rich
 *
 * **Extension Examples:**
 *
 * Custom packet type:
 * ```cpp
 * class MemoryReqPacket : public SimPacket {
 * public:
 *     MemoryReqPacket(uint64_t addr, size_t size)
 *         : SimPacket(PTYPE::MEMREQ), address(addr), size(size) {}
 *
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto memCtrl = dynamic_cast<MemoryController*>(&simulator)) {
 *             memCtrl->handleMemoryRequest(when, this);
 *         }
 *     }
 *
 * private:
 *     uint64_t address;
 *     size_t size;
 * };
 * ```
 *
 * @see DataReq.hh for DataReqPacket class definition
 * @see PE.cc for dataPacketHandler() implementation
 * @see PEReq.cc for comparison with PEReqPacket
 */

#include "DataReq.hh"

#include "PE.hh"

/**
 * @brief Visit a SimModule with this DataReqPacket
 *
 * This visitor method handles the case where a DataReqPacket is sent to
 * a SimModule object (a sub-component within a simulator). Currently
 * unimplemented as the testCommunication example works at simulator level.
 *
 * **SimModule vs SimBase:**
 *
 * - SimBase: Top-level simulator (e.g., PE, TrafficGenerator)
 * - SimModule: Sub-component within simulator (e.g., cache bank, ALU pipeline)
 *
 * **When This Would Be Used:**
 *
 * If PE were decomposed into modules:
 * ```
 * PE (SimBase)
 *   ├─ PECore (SimModule)
 *   ├─ LocalCache (SimModule)
 *   └─ RouterInterface (SimModule)
 * ```
 *
 * Then DataReqPacket could be routed to specific modules:
 * ```cpp
 * void DataReqPacket::visit(Tick when, SimModule& module) {
 *     if (auto cache = dynamic_cast<LocalCache*>(&module)) {
 *         cache->handleDataRequest(when, this);
 *     } else if (auto core = dynamic_cast<PECore*>(&module)) {
 *         core->handleDataRequest(when, this);
 *     } else {
 *         CLASS_ERROR << "Unsupported module type for DataReqPacket";
 *     }
 * }
 * ```
 *
 * **Implementation Strategy:**
 *
 * When implementing this method:
 * 1. Determine which module types can receive DataReqPackets
 * 2. Use dynamic_cast to identify module type
 * 3. Call appropriate handler method on module
 * 4. Handle error case (unknown module type)
 *
 * **Current Status:**
 *
 * Not implemented because:
 * - Example uses flat simulator structure (no internal modules)
 * - All packets handled at SimBase level
 * - Simplifies the example for learning purposes
 *
 * @param when Simulation tick when packet arrived
 * @param module Reference to target SimModule
 *
 * @note Logs error if called - implementation needed for module-level routing
 *
 * @see visit(Tick, SimBase&) for simulator-level implementation
 */
void DataReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void DataReqPacket::visit(SimModule& module)is not implemented yet!";
}

/**
 * @brief Visit a SimBase (simulator) with this DataReqPacket
 *
 * This is the primary visitor method for DataReqPacket, implementing the
 * visitor pattern's double-dispatch mechanism. It routes the packet to
 * the appropriate handler based on the simulator type.
 *
 * **Dispatch Flow:**
 * ```
 * 1. Packet arrives at simulator (SimBase&)
 * 2. Attempt to downcast to known simulator types
 * 3. If cast succeeds, call simulator's specific handler
 * 4. Handler processes packet with full type information
 * ```
 *
 * **Implementation Pattern:**
 * ```cpp
 * void DataReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto pe = dynamic_cast<PE*>(&simulator)) {
 *         pe->dataPacketHandler(when, this);  // Type-safe call
 *         return;
 *     }
 *     // Could add more simulator types:
 *     // if (auto cache = dynamic_cast<CacheSim*>(&simulator)) { ... }
 *     // if (auto noc = dynamic_cast<NocSim*>(&simulator)) { ... }
 *
 *     CLASS_ERROR << "Unsupported simulator type";
 * }
 * ```
 *
 * **Why dynamic_cast?**
 *
 * - Runtime type identification (RTTI)
 * - Safe downcasting with nullptr on failure
 * - Enables visitor pattern without virtual dispatch table
 * - Performance cost negligible compared to simulation work
 *
 * **Alternative Implementations:**
 *
 * 1. **Type enum (faster but less extensible):**
 * ```cpp
 * switch (simulator.getType()) {
 *     case SimType::PE:
 *         static_cast<PE*>(&simulator)->dataPacketHandler(when, this);
 *         break;
 *     case SimType::CACHE:
 *         static_cast<CacheSim*>(&simulator)->dataPacketHandler(when, this);
 *         break;
 * }
 * ```
 *
 * 2. **Virtual handler method (couples packet to simulator):**
 * ```cpp
 * class SimBase {
 *     virtual void handleDataReq(Tick when, DataReqPacket* pkt) = 0;
 * };
 * // Problem: Every simulator must implement every packet type handler
 * ```
 *
 * **Error Handling:**
 *
 * Current implementation logs error if simulator type unsupported:
 * - Helps catch routing mistakes
 * - Indicates missing handler implementation
 * - Could be extended to throw exception or return error code
 *
 * **Usage Example:**
 *
 * From sender:
 * ```cpp
 * DataReqPacket* pkt = new DataReqPacket(tick, myData);
 * SimBase* destination = getDownStream("DSPE");
 * pkt->visit(currentTick, *destination);
 * // Automatically routes to PE::dataPacketHandler() if destination is PE
 * ```
 *
 * From framework (after receiving via channel):
 * ```cpp
 * // EventPacket contains DataReqPacket
 * // Framework extracts packet and calls visit
 * dataReqPacket->visit(scheduledTick, *receivingSimulator);
 * ```
 *
 * **Extending to Multiple Simulators:**
 *
 * To support DataReqPacket in more simulators:
 * ```cpp
 * void DataReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto pe = dynamic_cast<PE*>(&simulator)) {
 *         pe->dataPacketHandler(when, this);
 *     } else if (auto cache = dynamic_cast<CacheSim*>(&simulator)) {
 *         cache->handleDataRequest(when, this);
 *     } else if (auto mem = dynamic_cast<MemorySim*>(&simulator)) {
 *         mem->processDataRequest(when, this);
 *     } else {
 *         CLASS_ERROR << "Unsupported simulator type: "
 *                     << simulator.getName();
 *     }
 * }
 * ```
 *
 * **Performance Considerations:**
 *
 * - dynamic_cast overhead: ~5-20 cycles (negligible in simulation)
 * - Could optimize with simulator type hints if profiling shows bottleneck
 * - Alternative: maintain static type ID for fast dispatch
 *
 * @param when Simulation tick when packet should be processed
 * @param simulator Reference to target simulator
 *
 * @note Currently only supports PE simulator type. Add more cases
 *       to support DataReqPacket in other simulator types.
 *
 * @see PE::dataPacketHandler() for actual packet processing
 * @see PEReqPacket::visit() for comparison with different packet type
 */
void DataReqPacket::visit(Tick when, SimBase& simulator) {
	auto pe = dynamic_cast<PE*>(&simulator);
	if (pe) {
		pe->dataPacketHandler(when, this);
	} else {
		CLASS_ERROR << "Invalid simulator type";
	}
}
