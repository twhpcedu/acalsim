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
 * @file MemReq.cc
 * @brief Memory request/response packet visitor pattern implementation
 *
 * This file implements the visitor pattern for MemReqPacket and MemRespPacket classes,
 * enabling type-safe packet routing to different module types (AXI Bus, SRAM) without
 * explicit type checking at every call site. This demonstrates the double-dispatch pattern
 * for polymorphic packet handling in event-driven simulation.
 *
 * **Visitor Pattern Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                    Visitor Pattern Flow                                │
 * │                  (Packet → Module Routing)                             │
 * │                                                                        │
 * │  Caller Side (e.g., CPUReqEvent::process()):                           │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ AXIBus::accept(tick, memReqPkt)                                  │ │
 * │  │   └─> SimModule::accept(when, pkt) [base class]                 │ │
 * │  │       └─> pkt.visit(when, *this)  // Polymorphic dispatch       │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Visitor Implementation (MemReqPacket::visit()):                       │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ if (auto bus = dynamic_cast<AXIBus*>(&module)) {                 │ │
 * │  │     bus->memReqPktHandler(when, this);  // Route to bus handler │ │
 * │  │ } else if (auto sram = dynamic_cast<SRAM*>(&module)) {           │ │
 * │  │     sram->memReqPktHandler(when, this);  // Route to SRAM       │ │
 * │  │ } else {                                                         │ │
 * │  │     CLASS_ERROR << "Invalid module type";  // Fail for unknown  │ │
 * │  │ }                                                                │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Handler Invocation:                                                   │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ AXIBus::memReqPktHandler(when, pkt)                              │ │
 * │  │   - Process request packet                                       │ │
 * │  │   - Create BusReqEvent for downstream SRAM                       │ │
 * │  │   - Schedule with bus latency                                    │ │
 * │  │                                                                  │ │
 * │  │ SRAM::memReqPktHandler(when, pkt)                                │ │
 * │  │   - Process memory access                                        │ │
 * │  │   - Create SRAMRespEvent with response                           │ │
 * │  │   - Schedule with memory latency                                 │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Packet Class Hierarchy:**
 * ```
 * SimPacket (base class)
 *   │
 *   ├─> MemReqPacket (memory request)
 *   │   ├─ reqType: TENSOR_MEM_READ/WRITE, PCU_MEM_READ/WRITE
 *   │   ├─ addr: Target memory address
 *   │   ├─ size: Transfer size in bytes
 *   │   ├─ callback: Response callback function
 *   │   ├─ memRespPkt: Embedded response packet
 *   │   └─ visit(): Routes to AXIBus or SRAM
 *   │
 *   └─> MemRespPacket (memory response)
 *       ├─ reqType: Original request type
 *       ├─ addr: Original address
 *       ├─ size: Transfer size
 *       └─ visit(): Not used (delivered via callbacks)
 * ```
 *
 * **Double Dispatch Mechanism:**
 * ```
 * Why Visitor Pattern?
 * ════════════════════════════════════════════
 *
 * Problem: Type-safe routing without explicit casts
 *   - accept() receives SimPacket* (base class pointer)
 *   - Need to invoke correct handler based on:
 *     1. Packet type (MemReqPacket vs MemRespPacket)
 *     2. Module type (AXIBus vs SRAM)
 *   - Traditional approach: Manual dynamic_cast at every call site
 *   - Visitor approach: Centralized type dispatch
 *
 * Double Dispatch:
 *   First Dispatch (polymorphic packet):
 *     pkt.visit(when, module)
 *       └─> Resolves packet type (MemReqPacket or MemRespPacket)
 *
 *   Second Dispatch (dynamic cast):
 *     if (auto bus = dynamic_cast<AXIBus*>(&module))
 *       └─> Resolves module type (AXIBus or SRAM)
 *
 * Result: Correct handler invoked based on both types
 *   MemReqPacket + AXIBus  → AXIBus::memReqPktHandler()
 *   MemReqPacket + SRAM    → SRAM::memReqPktHandler()
 *   MemRespPacket + Module → Error (not implemented)
 * ```
 *
 * **Routing Examples:**
 * ```
 * Example 1: CPUReqEvent → AXI Bus
 * ═══════════════════════════════════════════════
 *
 * CPUReqEvent::process():
 *   AXIBus::accept(tick, memReqPkt)
 *     │
 *     └─> SimModule::accept(when, pkt)
 *         └─> pkt.visit(when, *this)
 *             └─> MemReqPacket::visit(when, module)
 *                 ├─ module type is AXIBus
 *                 ├─ dynamic_cast<AXIBus*>(&module) succeeds
 *                 └─> bus->memReqPktHandler(when, this)
 *
 * Example 2: BusReqEvent → SRAM
 * ═══════════════════════════════════════════════
 *
 * BusReqEvent::process():
 *   SRAM::accept(tick, memReqPkt)
 *     │
 *     └─> SimModule::accept(when, pkt)
 *         └─> pkt.visit(when, *this)
 *             └─> MemReqPacket::visit(when, module)
 *                 ├─ module type is SRAM
 *                 ├─ dynamic_cast<SRAM*>(&module) succeeds
 *                 └─> sram->memReqPktHandler(when, this)
 * ```
 *
 * **MemReqPacket Structure:**
 * ```
 * MemReqPacket Fields:
 * ════════════════════════════════════════════
 *
 * reqType (MemReqTypeEnum):
 *   - TENSOR_MEM_READ:  Read from tensor memory
 *   - TENSOR_MEM_WRITE: Write to tensor memory
 *   - PCU_MEM_READ:     Read from PCU private memory
 *   - PCU_MEM_WRITE:    Write to PCU private memory
 *
 * addr (uint64_t):
 *   - Target memory address
 *   - Examples: 0x0000, 0x1000, 0x2000
 *
 * size (int):
 *   - Transfer size in bytes
 *   - Examples: 0, 20, 40, 60, 80
 *
 * callback (std::function<void(int, MemRespPacket*)>):
 *   - Response callback function
 *   - Set by sender (CPUReqEvent, BusReqEvent)
 *   - Invoked when response ready
 *
 * memRespPkt (MemRespPacket*):
 *   - Embedded response packet
 *   - Pre-allocated for efficiency
 *   - Delivered via callback chain
 * ```
 *
 * **MemRespPacket Usage:**
 * ```
 * Current Status: Callback-Based Delivery
 * ════════════════════════════════════════════
 *
 * MemRespPacket is NOT routed via visitor pattern.
 * Instead, responses are delivered through callbacks:
 *
 * Response Flow:
 *   SRAM creates SRAMRespEvent
 *     └─> Event invokes callback (BusReqEvent::busReqCallback)
 *         └─> Callback creates BusRespEvent
 *             └─> Event invokes callback (CPUReqEvent::cpuReqCallback)
 *                 └─> Callback invokes CPUTraffic::MemRespHandler
 *
 * Why Callbacks Instead of Visitor?
 *   - Responses follow reverse path of requests
 *   - Callback chain preserves routing context
 *   - No need for explicit module addressing
 *   - Automatic response correlation
 *
 * visit() Methods for MemRespPacket:
 *   - Implemented but report errors if called
 *   - Serve as safety net for misuse
 *   - Future extension point if needed
 * ```
 *
 * **Error Handling:**
 * ```
 * Invalid Module Type:
 * ════════════════════════════════════════════
 *
 * if (dynamic_cast<AXIBus*>(...) fails &&
 *     dynamic_cast<SRAM*>(...) fails) {
 *     CLASS_ERROR << "Invalid module type";
 * }
 *
 * Causes:
 *   - Packet sent to unsupported module
 *   - New module type added but visitor not updated
 *   - Module type mismatch in connection
 *
 * Unimplemented Methods:
 * ════════════════════════════════════════════
 *
 * MemReqPacket::visit(SimBase&):
 *   - Not used in current architecture
 *   - Requests always sent to modules, not simulators
 *   - Reports error if called
 *
 * MemRespPacket::visit(SimModule&) & visit(SimBase&):
 *   - Responses use callback delivery
 *   - Not routed through visitor pattern
 *   - Report errors if called
 * ```
 *
 * **Extension for Multi-Tile Mesh:**
 * ```
 * Future: Mesh Router Support
 * ════════════════════════════════════════════
 *
 * Additional Module Types:
 *   void MemReqPacket::visit(Tick when, SimModule& module) {
 *     if (auto bus = dynamic_cast<AXIBus*>(&module)) {
 *       bus->memReqPktHandler(when, this);
 *     } else if (auto sram = dynamic_cast<SRAM*>(&module)) {
 *       sram->memReqPktHandler(when, this);
 *     } else if (auto router = dynamic_cast<MeshRouter*>(&module)) {
 *       router->memReqPktHandler(when, this);
 *     } else if (auto nic = dynamic_cast<NetworkInterface*>(&module)) {
 *       nic->memReqPktHandler(when, this);
 *     } else {
 *       CLASS_ERROR << "Invalid module type";
 *     }
 *   }
 *
 * Mesh Routing Packet:
 *   class MeshMemReqPacket : public MemReqPacket {
 *     int dst_x, dst_y;  // Destination tile coordinates
 *     int src_x, src_y;  // Source tile coordinates
 *     int hop_count;     // Number of hops traversed
 *   };
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Visitor Pattern:**
 *    - Double dispatch for type-safe routing
 *    - Centralizes packet handling logic
 *    - Extensible for new module types
 *    - Avoids manual type checks everywhere
 *
 * 2. **Packet Embedding:**
 *    - MemRespPacket embedded in MemReqPacket
 *    - Reduces memory allocation overhead
 *    - Maintains request-response correlation
 *    - Simplifies memory management
 *
 * 3. **Callback-Based Responses:**
 *    - Responses delivered via callback chain
 *    - Visitor pattern only for requests
 *    - Asymmetric request/response routing
 *    - Automatic reverse path following
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/MemReq.hh
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - SRAM: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/SRAM.cc
 * - CPU Req Event: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUReqEvent.cc
 * - CPU Traffic: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUTraffic.cc
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "MemReq.hh"

#include "AXIBus.hh"
#include "SRAM.hh"

/**
 * @brief Visitor method for routing MemReqPacket to appropriate module handler
 *
 * This method implements the visitor pattern's accept-side logic, performing dynamic
 * type checking to route the memory request packet to the correct module-specific
 * handler. It supports both AXI Bus and SRAM modules in the current PE tile architecture.
 *
 * **Routing Logic:**
 * ```
 * Module Type Detection:
 *   │
 *   ├─ Is module an AXIBus?
 *   │  └─ Yes: bus->memReqPktHandler(when, this)
 *   │         └─> AXI Bus processes request, creates BusReqEvent
 *   │
 *   ├─ Is module an SRAM?
 *   │  └─ Yes: sram->memReqPktHandler(when, this)
 *   │         └─> SRAM processes request, creates SRAMRespEvent
 *   │
 *   └─ Else: Error - Invalid module type
 *      └─ CLASS_ERROR logged
 *      └─ Indicates misconfiguration or unsupported module
 * ```
 *
 * **Usage Examples:**
 * ```
 * Example 1: CPUReqEvent → AXI Bus
 *   CPUReqEvent::process() {
 *     AXIBus::accept(tick, memReqPkt);
 *       └─> SimModule::accept(when, pkt)
 *           └─> pkt.visit(when, *this)
 *               └─> MemReqPacket::visit(when, module)
 *                   ├─ module is AXIBus*
 *                   └─> bus->memReqPktHandler(when, this)
 *   }
 *
 * Example 2: BusReqEvent → SRAM
 *   BusReqEvent::process() {
 *     SRAM::accept(tick, memReqPkt);
 *       └─> SimModule::accept(when, pkt)
 *           └─> pkt.visit(when, *this)
 *               └─> MemReqPacket::visit(when, module)
 *                   ├─ module is SRAM*
 *                   └─> sram->memReqPktHandler(when, this)
 *   }
 * ```
 *
 * @param when Simulation tick when packet should be processed
 * @param module Reference to the module receiving this packet (AXIBus or SRAM)
 *
 * @note Dynamic cast used for type-safe module identification
 * @note Error logged if module type is not supported
 * @note This is the "accept" side of the visitor pattern
 *
 * @see AXIBus::memReqPktHandler() Handler for bus module
 * @see SRAM::memReqPktHandler() Handler for memory module
 * @see SimModule::accept() Base class method that invokes visit()
 */
void MemReqPacket::visit(Tick when, SimModule& module) {
	if (auto bus = dynamic_cast<AXIBus*>(&module)) {
		bus->memReqPktHandler(when, this);
	} else if (auto sram = dynamic_cast<SRAM*>(&module)) {
		sram->memReqPktHandler(when, this);
	} else {
		CLASS_ERROR << "Invalid module type";
	}
}

/**
 * @brief Visitor method for routing MemReqPacket to simulator (NOT IMPLEMENTED)
 *
 * This method is not used in the current PE tile architecture. Memory requests
 * are always sent to modules (AXI Bus, SRAM), never directly to the simulator.
 * This method exists for interface completeness but reports an error if called.
 *
 * @param when Simulation tick when packet should be processed
 * @param simulator Reference to the simulator receiving this packet
 *
 * @note This method is not implemented and will log an error if called
 * @note Future extension point if simulator-level packet handling is needed
 */
void MemReqPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void MemReqPacket::visit(SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visitor method for routing MemRespPacket to module (NOT USED)
 *
 * This method is not used in the current architecture. Memory responses are delivered
 * via callback chains rather than visitor pattern routing. This method exists for
 * interface completeness but reports an error if called.
 *
 * **Why Not Used:**
 * - Responses follow callback-based delivery
 * - Callback chain preserves request path in reverse
 * - No need for explicit module addressing
 * - Automatic response correlation with requests
 *
 * @param when Simulation tick when packet should be processed
 * @param module Reference to the module receiving this packet
 *
 * @note This method is not implemented and will log an error if called
 * @note Responses use callback delivery, not visitor pattern
 *
 * @see SRAMRespEvent Response event that uses callbacks
 * @see BusRespEvent Response event that uses callbacks
 */
void MemRespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void MemReqPacket::visit(SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor method for routing MemRespPacket to simulator (NOT USED)
 *
 * This method is not used in the current architecture. Memory responses are delivered
 * via callback chains rather than visitor pattern routing. This method exists for
 * interface completeness but reports an error if called.
 *
 * @param when Simulation tick when packet should be processed
 * @param simulator Reference to the simulator receiving this packet
 *
 * @note This method is not implemented and will log an error if called
 * @note Responses use callback delivery, not visitor pattern
 */
void MemRespPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void MemReqPacket::visit(SimBase& simulator) is not implemented yet!";
}
