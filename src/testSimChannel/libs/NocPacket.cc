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
 * @file NocPacket.cc
 * @brief Network-on-Chip Packet Visitor Implementation
 *
 * @details
 * This file implements the visitor pattern methods for NoC packet types (NocReqPacket
 * and NocRespPacket). The visitor pattern enables type-safe, polymorphic packet routing
 * where packets "know" how to handle themselves when they arrive at different simulator
 * components. This eliminates the need for manual type casting and centralizes packet
 * handling logic.
 *
 * # Visitor Pattern Overview
 *
 * The visitor pattern in ACALSim works as follows:
 *
 * @code{.unparsed}
 *   SimChannel Framework
 *         │
 *         │ 1. Packet arrives at destination
 *         ↓
 *   packet->visit(when, destination)
 *         │
 *         │ 2. Packet determines its type
 *         ↓
 *   dynamic_cast<CorrectSimulator*>(&destination)
 *         │
 *         │ 3. Call appropriate handler
 *         ↓
 *   simulator->handlePacket(this)
 * @endcode
 *
 * ## Benefits of This Pattern:
 * 1. **Type Safety**: Compile-time checking of packet-handler compatibility
 * 2. **Extensibility**: Easy to add new packet types without modifying simulators
 * 3. **Decoupling**: Packets don't need to know simulator internals
 * 4. **Error Detection**: Clear error messages for routing mistakes
 *
 * # Packet Type Hierarchy
 *
 * @code{.unparsed}
 *              SimPacket (base)
 *                   │
 *         ┌─────────┴─────────┐
 *         │                   │
 *    PTYPE::MEMREQ       PTYPE::MEMRESP
 *         │                   │
 *    ┌────┴────┐         ┌────┴────┐
 *    │         │         │         │
 * NocReq   CacheReq   NocResp  CacheResp
 * Packet   Packet     Packet   Packet
 * @endcode
 *
 * # NocReqPacket Visitor Implementation
 *
 * ## Purpose:
 * Routes memory requests from TrafficGenerator to NocSim.
 *
 * ## Packet Structure:
 * @code
 * class NocReqPacket : public SimPacket {
 * private:
 *     NocPktTypeEnum reqType;  // Operation type (TEST in this example)
 *     int addr;                // Memory address (0x0000)
 *     int size;                // Transfer size (256 bytes)
 *     int tid;                 // Transaction ID (0, 1, ...)
 * };
 * @endcode
 *
 * ## visit(Tick when, SimBase& simulator) Implementation:
 *
 * This method is called when the packet arrives at a SimBase-derived component.
 *
 * @code
 * void NocReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<NocSim*>((SimBase*)(&simulator))) {
 *         // Packet is at the correct destination (NocSim)
 *         dynamic_cast<NocSim*>((SimBase*)(&simulator))->handleTGRequest(this);
 *     } else {
 *         // Packet arrived at wrong simulator type
 *         CLASS_ERROR << "Invalid simulator type!";
 *     }
 * }
 * @endcode
 *
 * ### Execution Flow:
 * 1. **Type Check**: `dynamic_cast<NocSim*>()` verifies destination is a NocSim
 * 2. **Call Handler**: If match, calls `NocSim::handleTGRequest(this)`
 * 3. **Error Logging**: If mismatch, logs error (routing bug detected)
 *
 * ### Example Routing:
 * @code{.unparsed}
 * Tick 11: NocReqPacket arrives at NocSim slave port "TG2NOC-s"
 *          │
 *          ↓ Framework calls visit(11, nocSimInstance)
 *          │
 *          ↓ dynamic_cast<NocSim*>(...) succeeds
 *          │
 *          ↓ Calls nocSimInstance.handleTGRequest(this)
 *          │
 *          ↓ NocSim processes request:
 *            - Stores in reqQueue
 *            - Creates CacheReqPacket
 *            - Forwards to CacheSim
 * @endcode
 *
 * ## visit(Tick when, SimModule& module) Implementation:
 *
 * This method would be called if the packet arrives at a SimModule-derived component
 * (not used in this example).
 *
 * @code
 * void NocReqPacket::visit(Tick when, SimModule& module) {
 *     CLASS_ERROR << "void NocReqPacket::visit (SimModule& module) is not implemented yet!";
 * }
 * @endcode
 *
 * This indicates NocReqPacket is **only intended for SimBase simulators**, not modules.
 *
 * # NocRespPacket Visitor Implementation
 *
 * ## Purpose:
 * Routes memory responses from NocSim back to TrafficGenerator.
 *
 * ## Packet Structure:
 * @code
 * class NocRespPacket : public SimPacket {
 * private:
 *     NocPktTypeEnum respType;  // Response type (TEST)
 *     int* data;                // Data payload (e.g., 111)
 *     int tid;                  // Transaction ID (matches request)
 * };
 * @endcode
 *
 * ## visit(Tick when, SimBase& simulator) Implementation:
 *
 * @code
 * void NocRespPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<TrafficGenerator*>((SimBase*)(&simulator))) {
 *         // Packet is at the correct destination (TrafficGenerator)
 *         dynamic_cast<TrafficGenerator*>((SimBase*)(&simulator))->handleNoCRespond(this);
 *     } else {
 *         // Packet arrived at wrong simulator type
 *         CLASS_ERROR << "Invalid simulator type!";
 *     }
 * }
 * @endcode
 *
 * ### Execution Flow:
 * 1. **Type Check**: `dynamic_cast<TrafficGenerator*>()` verifies destination
 * 2. **Call Handler**: If match, calls `TrafficGenerator::handleNoCRespond(this)`
 * 3. **Error Logging**: If mismatch, logs error
 *
 * ### Example Routing:
 * @code{.unparsed}
 * Tick 20: NocRespPacket arrives at TrafficGenerator slave port "NOC2TG-s"
 *          │
 *          ↓ Framework calls visit(20, trafficGenInstance)
 *          │
 *          ↓ dynamic_cast<TrafficGenerator*>(...) succeeds
 *          │
 *          ↓ Calls trafficGenInstance.handleNoCRespond(this)
 *          │
 *          ↓ TrafficGenerator processes response:
 *            - Extracts transaction ID
 *            - Retrieves data (111)
 *            - Logs completion
 * @endcode
 *
 * # Packet Routing in Full System
 *
 * ## Request Path (NocReqPacket):
 * @code{.unparsed}
 * [1] TrafficGenerator sends NocReqPacket
 *     │ sendPacketViaChannel("TG2NOC-m", ...)
 *     ↓
 * [2] SimChannel queues packet with delays
 *     │ local_delay = 1, remote_delay = 10
 *     ↓
 * [3] Tick 11: Packet arrives at NocSim
 *     │ Framework calls: nocReqPkt->visit(11, nocSim)
 *     ↓
 * [4] NocReqPacket::visit() executes
 *     │ dynamic_cast<NocSim*>() succeeds
 *     │ Calls nocSim.handleTGRequest(nocReqPkt)
 *     ↓
 * [5] NocSim processes request
 *     │ Stores in reqQueue
 *     │ Creates CacheReqPacket (different type!)
 *     │ Forwards to CacheSim
 * @endcode
 *
 * ## Response Path (NocRespPacket):
 * @code{.unparsed}
 * [1] CacheSim sends CacheRespPacket
 *     │ (CacheRespPacket, not NocRespPacket yet)
 *     ↓
 * [2] NocSim receives CacheRespPacket
 *     │ handleCacheRespond() called
 *     ↓
 * [3] NocSim creates NocRespPacket
 *     │ Extracts data from CacheRespPacket
 *     │ Creates new NocRespPacket with same tid
 *     │ sendPacketViaChannel("NOC2TG-m", ...)
 *     ↓
 * [4] Tick 20: Packet arrives at TrafficGenerator
 *     │ Framework calls: nocRespPkt->visit(20, trafficGen)
 *     ↓
 * [5] NocRespPacket::visit() executes
 *     │ dynamic_cast<TrafficGenerator*>() succeeds
 *     │ Calls trafficGen.handleNoCRespond(nocRespPkt)
 *     ↓
 * [6] TrafficGenerator processes response
 *     │ Logs "get data = 111"
 * @endcode
 *
 * # Error Handling and Debugging
 *
 * ## Invalid Destination Errors:
 *
 * If a packet is routed to the wrong simulator type:
 * @code
 * // Example: NocRespPacket accidentally sent to CacheSim
 * void NocRespPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<TrafficGenerator*>(&simulator)) {
 *         // This check FAILS (simulator is CacheSim, not TrafficGenerator)
 *     } else {
 *         CLASS_ERROR << "Invalid simulator type!";
 *         // Error logged: NocRespPacket expected TrafficGenerator but got CacheSim
 *     }
 * }
 * @endcode
 *
 * ## Common Routing Errors:
 * 1. **Wrong Port Connection**: Connected master port to wrong slave port
 * 2. **Missing Port**: Forgot to call ChannelPortManager::ConnectPort()
 * 3. **Type Mismatch**: Sent CachePacket to component expecting NocPacket
 *
 * ## Debugging Tips:
 * - Check CLASS_ERROR messages for "Invalid simulator type!"
 * - Verify port names match between ConnectPort() and sendPacketViaChannel()
 * - Ensure packet types match expected handler signatures
 *
 * # Extending with New Packet Types
 *
 * ## Adding a New NoC Packet Type:
 *
 * 1. **Define packet class in NocPacket.hh**:
 * @code
 * enum class NocPktTypeEnum { TEST, READ, WRITE };  // Add new types
 *
 * class NocWritePacket : public SimPacket {
 * public:
 *     NocWritePacket(int addr, int* data, int tid)
 *         : SimPacket(PTYPE::MEMREQ), addr(addr), data(data), tid(tid) {}
 *
 *     void visit(Tick when, SimModule& module) override;
 *     void visit(Tick when, SimBase& simulator) override;
 *
 * private:
 *     int addr;
 *     int* data;
 *     int tid;
 * };
 * @endcode
 *
 * 2. **Implement visitor in this file**:
 * @code
 * void NocWritePacket::visit(Tick when, SimBase& simulator) {
 *     if (auto* nocSim = dynamic_cast<NocSim*>(&simulator)) {
 *         nocSim->handleWriteRequest(this);
 *     } else {
 *         CLASS_ERROR << "Invalid simulator type for NocWritePacket!";
 *     }
 * }
 * @endcode
 *
 * 3. **Add handler to NocSim.hh**:
 * @code
 * class NocSim : public CPPSimBase {
 * public:
 *     void handleWriteRequest(NocWritePacket* pkt);  // New handler
 * };
 * @endcode
 *
 * # Design Patterns and Best Practices
 *
 * ## 1. Double Dispatch Pattern:
 * The visitor pattern implements double dispatch:
 * - First dispatch: Framework calls `packet->visit(when, simulator)`
 * - Second dispatch: Packet calls `simulator->handlePacket(this)`
 *
 * This resolves both packet type AND simulator type at runtime.
 *
 * ## 2. Error-First Design:
 * Always include the `else` clause with error logging:
 * @code
 * if (correctType) {
 *     // Handle packet
 * } else {
 *     CLASS_ERROR << "Invalid type!";  // ALWAYS include this
 * }
 * @endcode
 *
 * ## 3. Consistent Casting Pattern:
 * Use the same casting approach throughout:
 * @code
 * if (dynamic_cast<TargetType*>((SimBase*)(&simulator))) {
 *     dynamic_cast<TargetType*>((SimBase*)(&simulator))->handleMethod(this);
 * }
 * @endcode
 *
 * ## 4. SimBase vs. SimModule Overloads:
 * Provide both overloads even if one isn't used:
 * @code
 * void visit(Tick when, SimBase& simulator) override;    // For CPPSimBase
 * void visit(Tick when, SimModule& module) override;     // For SimModule
 * @endcode
 *
 * # Performance Considerations
 *
 * ## dynamic_cast Cost:
 * - Involves RTTI lookup (~few CPU cycles)
 * - Negligible compared to packet processing time
 * - Only performed once per packet arrival
 *
 * ## Alternative Approach (Not Recommended):
 * Could use static_cast with type tags:
 * @code
 * // DON'T DO THIS - less safe, more error-prone
 * if (simulator.getType() == SimType::NOC) {
 *     static_cast<NocSim*>(&simulator)->handleTGRequest(this);
 * }
 * @endcode
 *
 * The dynamic_cast approach is preferred for safety and maintainability.
 *
 * @see NocPacket.hh For packet class declarations
 * @see TrafficGenerator For NocRespPacket handler (handleNoCRespond)
 * @see NocSim For NocReqPacket handler (handleTGRequest)
 * @see CachePacket.cc For similar visitor implementations for cache packets
 * @see testSimChannel.cc For complete system integration
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note The visitor pattern is a key design pattern in ACALSim for type-safe packet
 *       routing. Understanding this file is essential for adding new packet types
 *       or simulator components.
 *
 * @warning Do NOT modify the dynamic_cast logic unless you fully understand the
 *          implications. Incorrect casting can cause silent routing failures or
 *          undefined behavior.
 */

#include "NocPacket.hh"

#include "CachePacket.hh"
#include "NocSim.hh"
#include "TrafficGenerator.hh"

using namespace acalsim;

void NocRespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void NocRespPacket::visit (SimModule& module) is not implemented yet!";
}

void NocRespPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<TrafficGenerator*>((SimBase*)(&simulator))) {
		dynamic_cast<TrafficGenerator*>((SimBase*)(&simulator))->handleNoCRespond(this);
	} else {
		CLASS_ERROR << "Invalid simulator type!";
	}
}

void NocReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void NocReqPacket::visit (SimModule& module) is not implemented yet!";
}

// When NocSim visit this packet, it will depacket this packet and forward payload to downstream CacheSim
void NocReqPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<NocSim*>((SimBase*)(&simulator))) {
		dynamic_cast<NocSim*>((SimBase*)(&simulator))->handleTGRequest(this);
	} else {
		CLASS_ERROR << "Invalid simulator type!";
	}
}
