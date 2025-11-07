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
 * @file MacPacket.cc
 * @brief Packet type definitions for MAC simulation with visitor pattern implementation
 *
 * @details
 * This file implements the packet classes and visitor pattern dispatch mechanism for MAC
 * (Multiply-Accumulate) simulation. It demonstrates how to create custom packet types that
 * carry typed data between simulators in the ACALSim framework, and how to use the visitor
 * design pattern for type-safe packet handling.
 *
 * ## Overview
 *
 * The MacPacket hierarchy provides:
 * - **InBoundData**: Input data structure (A, B, C operands)
 * - **OutBoundData**: Output data structure (D result)
 * - **MacPacket<T>**: Template base for typed packet containers
 * - **MacInPacket**: Input packet (TGSim → MacSim)
 * - **MacOutPacket**: Output packet (MacSim → TGSim) with visitor dispatch
 *
 * This file implements the visitor pattern for MacOutPacket, enabling type-safe,
 * extensible packet handling without switch statements or type checking overhead.
 *
 * ## Packet Type Hierarchy
 *
 * ```
 * SimPacket (ACALSim base)
 *     |
 *     +-- SCSimPacket (SystemC packet base)
 *         |
 *         +-- MacPacket<T> (Template container)
 *             |
 *             +-- MacInPacket (MacPacket<InBoundData>)
 *             |   - Contains: transaction ID, A, B, C
 *             |   - Direction: TGSim → MacSim
 *             |   - Purpose: Carry MAC operation inputs
 *             |
 *             +-- MacOutPacket (MacPacket<OutBoundData>)
 *                 - Contains: transaction ID, D (result)
 *                 - Direction: MacSim → TGSim
 *                 - Purpose: Carry MAC operation result
 *                 - Implements: visit() for visitor dispatch
 * ```
 *
 * ## Data Structures
 *
 * ### InBoundData
 *
 * Input data for MAC operation (A × B + C):
 *
 * ```cpp
 * class InBoundData {
 * public:
 *     int A;   // First multiplicand (4-bit: 0-15)
 *     int B;   // Second multiplicand (4-bit: 0-15)
 *     int C;   // Addend (8-bit: 0-255)
 *     int id;  // Transaction identifier
 *
 *     InBoundData(int _id, int a, int b, int c);
 *     void set(int _id, int _a, int _b, int _c);
 * };
 * ```
 *
 * **Purpose**:
 * - Encapsulates MAC input parameters
 * - Provides transaction tracking via id field
 * - Used for golden model verification
 * - Stored in TGSim's verification queue
 *
 * **Field Constraints**:
 * - A, B: Limited to 0-14 in practice (4-bit hardware)
 * - C: Limited to 0-14 in practice (could be 8-bit)
 * - id: Unique per transaction, monotonically increasing
 *
 * ### OutBoundData
 *
 * Output data from MAC operation:
 *
 * ```cpp
 * class OutBoundData {
 * public:
 *     int D;   // Result of A × B + C (9-bit: 0-511)
 *     int id;  // Transaction identifier (matches InBoundData)
 *
 *     OutBoundData(int _id, int d);
 *     void set(int _id, int _d);
 * };
 * ```
 *
 * **Purpose**:
 * - Carries MAC computation result
 * - Correlates with input via matching id
 * - Enables verification by comparing expected vs. actual
 *
 * **Field Constraints**:
 * - D: Up to 9-bit value (max 15×15+15 = 240 in practice)
 * - id: Must match corresponding InBoundData id
 *
 * ## MacPacket<T> Template
 *
 * Generic packet container using template parameterization:
 *
 * ```cpp
 * template <typename T>
 * class MacPacket : public SCSimPacket {
 * protected:
 *     std::shared_ptr<SharedContainer<T>> data;
 *
 * public:
 *     MacPacket(std::shared_ptr<SharedContainer<T>> _data);
 *     std::shared_ptr<SharedContainer<T>> getData();
 *     void renew(std::shared_ptr<SharedContainer<T>> _data);
 *     virtual void visit(Tick when, SimModule& module) override;
 *     virtual void visit(Tick when, SimBase& simulator) override;
 * };
 * ```
 *
 * **Template Parameter**:
 * - T: Data type (InBoundData or OutBoundData)
 * - Enables type-safe container access
 * - Compile-time type checking
 *
 * **SharedContainer**:
 * - Reference-counted container (shared_ptr)
 * - Holds multiple data elements (though typically one)
 * - Avoids data copying across simulator boundaries
 * - Automatic memory management
 *
 * **Virtual Methods**:
 * - visit(SimModule&): Dispatch to module handlers (default: no-op)
 * - visit(SimBase&): Dispatch to simulator handlers (overridden in MacOutPacket)
 *
 * ## MacInPacket
 *
 * Input packet carrying operands from TGSim to MacSim:
 *
 * ```cpp
 * class MacInPacket : public MacPacket<InBoundData> {
 * public:
 *     MacInPacket(std::shared_ptr<SharedContainer<InBoundData>> _data);
 *     void renew(std::shared_ptr<SharedContainer<InBoundData>> _data);
 * };
 * ```
 *
 * **Usage Example** (in TGSim::injectTraffic()):
 * ```cpp
 * auto container = std::make_shared<SharedContainer<InBoundData>>();
 * container->add(transactionID, A, B, C);
 * auto packet = new MacInPacket(container);
 * this->pushToMasterPort("sc_top1-m", packet);
 * ```
 *
 * **Lifecycle**:
 * 1. Created in TGSim with input data
 * 2. Pushed to master port
 * 3. Delivered to MacSim slave port
 * 4. Extracted in MacInterface::setInputs()
 * 5. Freed after data extraction
 *
 * ## MacOutPacket
 *
 * Output packet carrying result from MacSim to TGSim:
 *
 * ```cpp
 * class MacOutPacket : public MacPacket<OutBoundData> {
 * public:
 *     MacOutPacket(std::shared_ptr<SharedContainer<OutBoundData>> _data);
 *     void renew(std::shared_ptr<SharedContainer<OutBoundData>> _data);
 *     void visit(Tick when, SimBase& simulator) override;
 * };
 * ```
 *
 * **Key Difference**:
 * - Overrides visit(SimBase&) for visitor pattern dispatch
 * - Enables type-safe handler invocation in TGSim
 *
 * **Usage Example** (in MacInterface::getOutputs()):
 * ```cpp
 * auto container = std::make_shared<SharedContainer<OutBoundData>>();
 * container->add(transactionID, D.read().to_int());
 * return new MacOutPacket(container);
 * ```
 *
 * **Lifecycle**:
 * 1. Created in MacInterface with result data
 * 2. Returned from getOutputs()
 * 3. Pushed to master port by framework
 * 4. Delivered to TGSim slave port
 * 5. visit() called to dispatch to handler
 * 6. Freed in TGSim::macOutPacketHandler()
 *
 * ## Visitor Pattern Implementation
 *
 * The visitor pattern enables type-safe packet handling without runtime type checking:
 *
 * ### Pattern Overview
 *
 * Traditional approach (inflexible):
 * ```cpp
 * void handlePacket(SimPacket* pkt) {
 *     if (auto mac_pkt = dynamic_cast<MacOutPacket*>(pkt)) {
 *         // Handle MacOutPacket
 *     } else if (auto other_pkt = dynamic_cast<OtherPacket*>(pkt)) {
 *         // Handle OtherPacket
 *     }
 *     // Requires modification for each new packet type
 * }
 * ```
 *
 * Visitor pattern approach (extensible):
 * ```cpp
 * // Packet dispatches to appropriate handler based on type
 * pkt->visit(tick, simulator);
 * ```
 *
 * ### visit() Implementation
 *
 * MacOutPacket::visit() implementation (this file):
 *
 * ```cpp
 * void MacOutPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto tg = dynamic_cast<TGSim*>(&simulator)) {
 *         tg->macOutPacketHandler(-1, *this);
 *     }
 * }
 * ```
 *
 * **How It Works**:
 * 1. Framework calls accept() on receiving simulator
 * 2. accept() calls packet->visit(tick, *this)
 * 3. visit() receives simulator reference
 * 4. dynamic_cast checks if simulator is TGSim
 * 5. If match, calls specific handler method
 * 6. Handler extracts data and processes result
 *
 * ### Visitor Pattern Benefits
 *
 * **Type Safety**:
 * - Compile-time enforcement of handler signatures
 * - No missing case warnings from compiler
 * - Clear contract between packet and handler
 *
 * **Extensibility**:
 * - New packet types don't require modifying existing code
 * - Each packet defines its own dispatch logic
 * - Handlers only handle packets they care about
 *
 * **Separation of Concerns**:
 * - Packet defines routing (visit)
 * - Simulator defines handling (macOutPacketHandler)
 * - Clear responsibility boundaries
 *
 * **Single Dispatch vs. Double Dispatch**:
 * - Single: Method called based on runtime type of one object
 * - Double: Method called based on runtime types of two objects
 * - Visitor implements double dispatch (packet type + simulator type)
 *
 * ## Complete Transaction Flow
 *
 * End-to-end packet flow with visitor pattern:
 *
 * ```
 * 1. TGSim::injectTraffic()
 *    |
 *    +-- Create InBoundData(id, A, B, C)
 *    +-- Store in inDataQ for verification
 *    +-- Wrap in SharedContainer<InBoundData>
 *    +-- Create MacInPacket(container)
 *    +-- pushToMasterPort("sc_top1-m", packet)
 *    |
 *    v
 * 2. Port Delivery
 *    |
 *    +-- Packet queued in MacSim slave port
 *    +-- Framework detects packet arrival
 *    |
 *    v
 * 3. MacInterface::setInputs(MacInPacket*)
 *    |
 *    +-- Extract InBoundData from container
 *    +-- Write A, B, C to SystemC signals
 *    +-- Assert enable signal
 *    +-- Free MacInPacket
 *    |
 *    v
 * 4. SC_MAC Computation (SystemC)
 *    |
 *    +-- Cycle 1: MUL_Out = A × B
 *    +-- Cycle 2: ADD_Out = MUL_Out + C
 *    +-- Cycle 3: D = ADD_Out, assert done
 *    |
 *    v
 * 5. MacInterface::getOutputs()
 *    |
 *    +-- Read D signal value
 *    +-- Create OutBoundData(id, D)
 *    +-- Wrap in SharedContainer<OutBoundData>
 *    +-- Create MacOutPacket(container)
 *    +-- Return packet to framework
 *    |
 *    v
 * 6. Port Delivery
 *    |
 *    +-- Packet queued in TGSim slave port
 *    +-- Framework calls TGSim::step()
 *    |
 *    v
 * 7. TGSim::step()
 *    |
 *    +-- Detect packet in slave port
 *    +-- Call this->accept(tick, *packet)
 *    |
 *    v
 * 8. Framework: packet->visit(tick, *TGSim)
 *    |
 *    v
 * 9. MacOutPacket::visit(tick, simulator)  [THIS FILE]
 *    |
 *    +-- dynamic_cast<TGSim*>(&simulator)
 *    +-- Call tg->macOutPacketHandler(-1, *this)
 *    |
 *    v
 * 10. TGSim::macOutPacketHandler(tick, MacOutPacket&)
 *     |
 *     +-- Extract OutBoundData from packet
 *     +-- Call checkAns() to verify result
 *     +-- Free MacOutPacket
 *     +-- Inject next traffic if not done
 * ```
 *
 * ## SharedContainer Pattern
 *
 * Efficient data management using reference counting:
 *
 * **Creation**:
 * ```cpp
 * auto container = std::make_shared<SharedContainer<InBoundData>>();
 * container->add(id, A, B, C);
 * ```
 *
 * **Benefits**:
 * - Reference-counted via shared_ptr
 * - No data copying when passing packets
 * - Automatic cleanup when last reference released
 * - Type-safe container for homogeneous data
 *
 * **Operations**:
 * - add(args...): Append element (forwarded to constructor)
 * - get(index): Retrieve element as shared_ptr<T>
 * - size(): Get element count
 *
 * **Typical Pattern**:
 * - One element per container in this example
 * - Could contain multiple elements for batch operations
 * - Accessed via get(0).get() to obtain raw pointer
 *
 * ## Extending Packet Types
 *
 * To create custom packet types with visitor pattern:
 *
 * **1. Define Data Structures**:
 * ```cpp
 * class MyInputData {
 * public:
 *     int field1;
 *     std::string field2;
 *     MyInputData(int f1, std::string f2) : field1(f1), field2(f2) {}
 * };
 * ```
 *
 * **2. Create Packet Classes**:
 * ```cpp
 * class MyInPacket : public MacPacket<MyInputData> {
 * public:
 *     MyInPacket(std::shared_ptr<SharedContainer<MyInputData>> data)
 *         : MacPacket<MyInputData>(data) {}
 * };
 *
 * class MyOutPacket : public MacPacket<MyOutputData> {
 * public:
 *     MyOutPacket(std::shared_ptr<SharedContainer<MyOutputData>> data)
 *         : MacPacket<MyOutputData>(data) {}
 *
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto my_sim = dynamic_cast<MySimulator*>(&simulator)) {
 *             my_sim->handleMyPacket(when, *this);
 *         }
 *     }
 * };
 * ```
 *
 * **3. Implement Handler**:
 * ```cpp
 * class MySimulator : public CPPSimBase {
 * public:
 *     void handleMyPacket(Tick when, MyOutPacket& pkt) {
 *         auto data = pkt.getData()->get(0).get();
 *         // Process data
 *         free(&pkt);
 *     }
 * };
 * ```
 *
 * ## Memory Management Summary
 *
 * **Packet Ownership**:
 * - Sender creates: new MacInPacket(container)
 * - Ownership transfers through port
 * - Receiver frees: free(packet)
 *
 * **Container Lifecycle**:
 * - Created with shared_ptr (reference counted)
 * - Shared between packet and data accessors
 * - Freed automatically when packet deleted
 *
 * **Data Lifetime**:
 * - InBoundData: Allocated separately in TGSim, freed after verification
 * - OutBoundData: Created in container, freed with container
 *
 * ## Why This Design
 *
 * **Template Parameterization**:
 * - Type safety at compile time
 * - Code reuse for different data types
 * - Clear data structure contracts
 *
 * **Visitor Pattern**:
 * - Extensible packet handling
 * - Type-safe dispatch
 * - No central switch statement
 *
 * **SharedContainer**:
 * - Efficient zero-copy data transfer
 * - Automatic memory management
 * - Supports variable-length payloads
 *
 * **Transaction IDs**:
 * - Enables request-response correlation
 * - Supports out-of-order completion
 * - Facilitates verification
 *
 * @see MacInterface.cc for packet creation and extraction
 * @see TGSim.cc for packet generation and handling
 * @see MacSim.cc for SystemC simulator integration
 * @see testSTSystemC.cc for complete system example
 * @see SimPacket for base packet class
 * @see SCSimPacket for SystemC packet base class
 */

#include "MacPacket.hh"

#include "TGSim.hh"

void MacOutPacket::visit(Tick when, SimBase& simulator) {
	if (auto tg = dynamic_cast<TGSim*>(&simulator)) { tg->macOutPacketHandler(-1, *this); }
}
