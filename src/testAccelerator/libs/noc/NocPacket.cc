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
 * @file NocPacket.cc
 * @brief NoC packet visitor pattern implementation for type-safe routing dispatch
 *
 * This file implements the **visitor pattern** for RNocPacket, enabling type-safe dispatch
 * to NocSim's routing handler. The visitor pattern decouples packet types from simulator
 * logic, allowing packets to invoke the appropriate handler method on their target simulator.
 *
 * **Visitor Pattern Role:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                   Visitor Pattern Dispatch Flow                    │
 * │                                                                    │
 * │  Problem Solved:                                                   │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ How does a generic accept() call know which specific         │ │
 * │  │ handler to invoke for a particular packet type?              │ │
 * │  │                                                              │ │
 * │  │ Without Visitor Pattern (Broken):                            │ │
 * │  │   void NocSim::accept(Tick when, SimPacket& pkt) {           │ │
 * │  │       // How do we know which handler to call?               │ │
 * │  │       // if (???) RNocPacketHandler(...)                     │ │
 * │  │       // Requires RTTI and ugly casting                      │ │
 * │  │   }                                                          │ │
 * │  │                                                              │ │
 * │  │ With Visitor Pattern (Elegant):                              │ │
 * │  │   void NocSim::accept(Tick when, SimPacket& pkt) {           │ │
 * │  │       pkt.visit(when, *this);  // Packet knows its handler! │ │
 * │  │   }                                                          │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Dispatch Mechanism:                                               │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. NocSim::accept(tick, packet) called                       │ │
 * │  │    └─ packet.visit(tick, *nocSim)                            │ │
 * │  │                                                              │ │
 * │  │ 2. RNocPacket::visit(tick, simulator) invoked                │ │
 * │  │    ├─ dynamic_cast simulator to NocSim*                      │ │
 * │  │    ├─ Validate cast succeeded                                │ │
 * │  │    └─ nocSim->RNocPacketHandler(tick, this)                  │ │
 * │  │                                                              │ │
 * │  │ 3. NocSim::RNocPacketHandler(tick, packet)                   │ │
 * │  │    └─ Type-safe packet processing                            │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Complete Visitor Pattern Flow:**
 * ```
 * Step 1: Event Delivery
 *   RNocEvent::process()
 *   └─ nocSim->accept(tick, *packet)
 *
 * Step 2: Accept Triggers Visitor
 *   NocSim::accept(Tick when, SimPacket& pkt)
 *   └─ pkt.visit(when, *this)
 *       // Polymorphic call - dispatches to RNocPacket::visit()
 *
 * Step 3: Visitor Casts and Validates (THIS FILE)
 *   RNocPacket::visit(Tick when, SimBase& simulator)
 *   ├─ NocSim* nocSim = dynamic_cast<NocSim*>(&simulator)
 *   ├─ if (nocSim) {
 *   │    └─ nocSim->RNocPacketHandler(when, this)  // Type-safe!
 *   └─ } else {
 *        CLASS_ERROR << "Invalid simulator type"
 *      }
 *
 * Step 4: Handler Processes Packet
 *   NocSim::RNocPacketHandler(Tick when, SimPacket* pkt)
 *   └─ Extract routing info, route to destination
 * ```
 *
 * **Why Visitor Pattern?**
 * ```
 * Benefits:
 *   ✓ Type Safety: Compile-time polymorphism + runtime validation
 *   ✓ Decoupling: Packet doesn't know NocSim details, just interface
 *   ✓ Extensibility: Easy to add new packet/simulator types
 *   ✓ Single Dispatch Point: accept() is uniform across all packets
 *
 * Alternative Approaches (and why they're worse):
 *   ✗ Dynamic Cast in accept():
 *       void accept(SimPacket& pkt) {
 *           if (auto* rPkt = dynamic_cast<RNocPacket*>(&pkt))
 *               RNocPacketHandler(rPkt);
 *           else if (auto* pPkt = dynamic_cast<PERNocPacket*>(&pkt))
 *               PEPacketHandler(pPkt);
 *           // Becomes unmaintainable with many packet types!
 *       }
 *
 *   ✗ Switch on Packet Type:
 *       void accept(SimPacket& pkt) {
 *           switch (pkt.getType()) {
 *               case RNOC: RNocPacketHandler(...); break;
 *               case PE: PEPacketHandler(...); break;
 *               // Fragile - easy to forget cases
 *           }
 *       }
 *
 *   ✓ Visitor Pattern (used here):
 *       void accept(SimPacket& pkt) {
 *           pkt.visit(*this);  // Packet knows which handler to call!
 *       }
 * ```
 *
 * **Implementation Details:**
 *
 * **Method 1: visit(Tick, SimModule&) - Module-Level Dispatch**
 * ```cpp
 * void RNocPacket::visit(Tick when, SimModule& module) {
 *     CLASS_ERROR << "Not implemented yet!";
 * }
 * ```
 * - SimModule is a lower-level abstraction (not used in this example)
 * - Would handle sub-simulator components
 * - Currently not implemented - raises error if called
 * - Reserved for future fine-grained dispatch
 *
 * **Method 2: visit(Tick, SimBase&) - Simulator-Level Dispatch (USED)**
 * ```cpp
 * void RNocPacket::visit(Tick when, SimBase& simulator) {
 *     auto nocSim = dynamic_cast<NocSim*>(&simulator);
 *     if (nocSim) {
 *         nocSim->RNocPacketHandler(when, this);
 *     } else {
 *         CLASS_ERROR << "Invalid simulator type";
 *     }
 * }
 * ```
 * - Validates simulator is actually NocSim
 * - Type-safe cast using dynamic_cast
 * - Invokes specific handler: RNocPacketHandler()
 * - Error if wrong simulator type (safety check)
 *
 * **Type Safety Validation:**
 * ```
 * Scenario: Correct Usage
 *   RNocPacket delivered to NocSim
 *   ├─ dynamic_cast<NocSim*> succeeds
 *   ├─ nocSim != nullptr
 *   └─ RNocPacketHandler() invoked
 *
 * Scenario: Programming Error (Misrouted Packet)
 *   RNocPacket mistakenly delivered to PESim
 *   ├─ dynamic_cast<NocSim*>(&peSim) fails
 *   ├─ nocSim == nullptr
 *   └─ CLASS_ERROR: "Invalid simulator type"
 *       → Prevents undefined behavior
 *       → Helps catch routing bugs
 * ```
 *
 * **Execution Example:**
 * ```
 * Tick 5: MCPU sends task to PE #0
 *
 * 1. Event Processing:
 *    RNocEvent::process()
 *    └─ ((NocSim*)callee)->accept(5, *packet)
 *
 * 2. Accept Method:
 *    NocSim::accept(5, RNocPacket&)
 *    └─ packet.visit(5, *this)  // 'this' is NocSim*
 *
 * 3. Visitor Dispatch (THIS FILE):
 *    RNocPacket::visit(5, SimBase& simulator)
 *    ├─ auto nocSim = dynamic_cast<NocSim*>(&simulator)
 *    ├─ if (nocSim)  // TRUE - cast succeeded
 *    └─ nocSim->RNocPacketHandler(5, this)
 *
 * 4. Handler Execution:
 *    NocSim::RNocPacketHandler(5, RNocPacket*)
 *    ├─ Extract routing info
 *    ├─ Determine destination: PE #0
 *    └─ Route packet: execUnicast()
 * ```
 *
 * **Class Structure:**
 * ```cpp
 * // Base packet class
 * class SimPacket {
 * public:
 *     virtual void visit(Tick when, SimModule& module) = 0;
 *     virtual void visit(Tick when, SimBase& simulator) = 0;
 * };
 *
 * // NoC routing packet
 * class RNocPacket : public SimPacket {
 * private:
 *     Tick when;
 *     SharedContainer<DLLRNocFrame>* sharedContainer;
 *
 * public:
 *     // Visitor pattern methods (implemented in this file)
 *     void visit(Tick when, SimModule& module) override;
 *     void visit(Tick when, SimBase& simulator) override;
 *
 *     // Accessors
 *     SharedContainer<DLLRNocFrame>* getSharedContainer();
 * };
 * ```
 *
 * **Comparison with Other Packet Types:**
 * ```
 * Packet Type       Target Simulator    Handler Method
 * ──────────────────────────────────────────────────────────
 * RNocPacket    →   NocSim             RNocPacketHandler()
 * PERNocPacket  →   PESim              RNocPacketHandler()
 * CacheRNocPkt  →   CacheSim           RNocPacketHandler()
 * MCPUPacket    →   MCPUSim            (direct accept)
 * ```
 *
 * **Related Files:**
 * - NocPacket.hh: RNocPacket class definition
 * - NocSim.cc: Target simulator with RNocPacketHandler()
 * - NocEvent.cc: Triggers accept() which invokes visitor
 * - testAccelerator.cc: System architecture overview
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see NocSim.cc for packet handler implementation
 * @see NocEvent.cc for event-driven visitor invocation
 * @see SimPacket.hh for visitor pattern base interface
 */

#include "noc/NocPacket.hh"

#include "noc/NocSim.hh"

using namespace acalsim;

/**
 * @brief Module-level visitor (not implemented for NoC packets)
 *
 * This visitor method would handle dispatch to SimModule-level components,
 * which are sub-simulator abstractions. Currently not used in the testAccelerator
 * example, as all routing is handled at the SimBase (simulator) level.
 *
 * @param when Current simulation tick
 * @param module Target module (sub-component of simulator)
 * @throws CLASS_ERROR Always - not implemented
 */
void RNocPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void RNocPacket::visit(SimModule& module) is not implemented yet!";
}

/**
 * @brief Simulator-level visitor implementing type-safe dispatch to NocSim
 *
 * This method implements the visitor pattern for RNocPacket, performing a type-safe
 * cast to NocSim and invoking the appropriate routing handler. This enables polymorphic
 * dispatch without requiring the caller to know the specific packet type.
 *
 * **Dispatch Logic:**
 * 1. Attempt dynamic_cast of simulator reference to NocSim*
 * 2. If cast succeeds (simulator is NocSim):
 *    - Invoke nocSim->RNocPacketHandler(when, this)
 *    - Handler processes routing and forwards packet
 * 3. If cast fails (programming error):
 *    - Raise CLASS_ERROR with diagnostic message
 *    - Prevents undefined behavior from wrong simulator type
 *
 * **Type Safety:**
 * - dynamic_cast ensures runtime type checking
 * - Nullptr check validates cast succeeded
 * - Error handling catches misrouted packets
 *
 * **Example Execution:**
 * ```
 * // Correct usage
 * NocSim* nocSim = ...;
 * RNocPacket packet(...);
 * packet.visit(5, *nocSim);
 * // → dynamic_cast succeeds → RNocPacketHandler(5, &packet) called
 *
 * // Programming error (would be caught)
 * PESim* peSim = ...;
 * RNocPacket packet(...);
 * packet.visit(5, *peSim);
 * // → dynamic_cast fails → CLASS_ERROR: "Invalid simulator type"
 * ```
 *
 * @param when Current simulation tick (passed to handler)
 * @param simulator Target simulator (should be NocSim)
 */
void RNocPacket::visit(Tick when, SimBase& simulator) {
	auto nocSim = dynamic_cast<NocSim*>(&simulator);
	if (nocSim) {
		nocSim->RNocPacketHandler(when, this);
	} else {
		CLASS_ERROR << "Invalid simulator type";
	}
}
