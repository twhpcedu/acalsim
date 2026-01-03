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
 * @file CachePacket.cc
 * @brief Cache Memory Packet Visitor Implementation
 *
 * @details
 * This file implements the visitor pattern methods for cache packet types (CacheReqPacket
 * and CacheRespPacket). These packets represent the memory hierarchy protocol layer and
 * are used for communication between the Network-on-Chip (NocSim) and the cache memory
 * (CacheSim). The visitor pattern ensures type-safe routing and handler dispatch.
 *
 * # Relationship to NoC Packets
 *
 * The system uses two protocol layers:
 *
 * @code{.unparsed}
 *   Network Layer (NoC Protocol):
 *     NocReqPacket, NocRespPacket
 *     ↓ Protocol Translation at NocSim
 *   Memory Layer (Cache Protocol):
 *     CacheReqPacket, CacheRespPacket
 *
 *   Packet Flow:
 *   TrafficGenerator → [NocReq] → NocSim → [CacheReq] → CacheSim
 *   TrafficGenerator ← [NocResp] ← NocSim ← [CacheResp] ← CacheSim
 * @endcode
 *
 * This layered approach allows:
 * - Independent evolution of network and memory protocols
 * - Protocol-specific optimizations (e.g., NoC routing vs. cache coherence)
 * - Realistic modeling of hardware abstraction layers
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
 * NocReq   CacheReq   NocResp  CacheResp  ← Implemented here
 * Packet   Packet     Packet   Packet
 * @endcode
 *
 * # CacheReqPacket Visitor Implementation
 *
 * ## Purpose:
 * Routes cache memory requests from NocSim to CacheSim.
 *
 * ## Packet Structure:
 * @code
 * class CacheReqPacket : public SimPacket {
 * private:
 *     CachePktTypeEnum reqType;  // Operation type (TEST)
 *     int addr;                  // Memory address (0x0000)
 *     int size;                  // Transfer size (256 bytes)
 *     int tid;                   // Transaction ID (0, 1, ...)
 * };
 * @endcode
 *
 * ## visit(Tick when, SimBase& simulator) Implementation:
 *
 * This is the critical method that routes cache requests to the cache simulator.
 *
 * @code
 * void CacheReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<CacheSim*>((SimBase*)(&simulator))) {
 *         // Packet is at the correct destination (CacheSim)
 *         dynamic_cast<CacheSim*>((SimBase*)(&simulator))->handleNOCRequest(this, when);
 *         // Note: 'when' parameter is passed to enable timing-dependent behavior
 *     } else {
 *         // Packet arrived at wrong simulator type
 *         CLASS_ERROR << "Invalid simulator type!";
 *     }
 * }
 * @endcode
 *
 * ### Key Feature: 'when' Parameter Propagation
 * Unlike NoC packets, CacheReqPacket passes the **arrival time** to the handler:
 * @code
 * handleNOCRequest(this, when)
 *                       ^^^^
 * @endcode
 *
 * This allows CacheSim to:
 * - Generate timing-dependent data: `data = 100 + when`
 * - Implement time-aware cache policies
 * - Track request arrival patterns
 *
 * ### Execution Flow:
 * @code{.unparsed}
 * Tick 11: CacheReqPacket arrives at CacheSim slave port "NOC2Cache-s"
 *          │
 *          ↓ Framework calls visit(11, cacheSimInstance)
 *          │
 *          ↓ dynamic_cast<CacheSim*>(...) succeeds
 *          │
 *          ↓ Calls cacheSimInstance.handleNOCRequest(this, 11)
 *          │                                              ^^
 *          ↓ CacheSim processes request at tick 11:
 *            - Generates data = 100 + 11 = 111
 *            - Calculates delay = 9 ticks
 *            - Creates CacheRespPacket
 *            - Sends response (arrives at tick 20)
 * @endcode
 *
 * ## visit(Tick when, SimModule& module) Implementation:
 *
 * @code
 * void CacheReqPacket::visit(Tick when, SimModule& module) {
 *     CLASS_ERROR << "void CacheReqPacket::visit (SimModule& module) is not implemented yet!";
 * }
 * @endcode
 *
 * This indicates CacheReqPacket is **only intended for SimBase simulators** (CacheSim),
 * not for SimModule-derived components.
 *
 * # CacheRespPacket Visitor Implementation
 *
 * ## Purpose:
 * Routes cache memory responses from CacheSim back to NocSim.
 *
 * ## Packet Structure:
 * @code
 * class CacheRespPacket : public SimPacket {
 * private:
 *     CachePktTypeEnum respType;  // Response type (TEST)
 *     int* data;                  // Data payload (e.g., 111)
 *     int tid;                    // Transaction ID (matches request)
 * };
 * @endcode
 *
 * ## visit(Tick when, SimBase& simulator) Implementation:
 *
 * @code
 * void CacheRespPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<NocSim*>((SimBase*)(&simulator))) {
 *         // Packet is at the correct destination (NocSim)
 *         dynamic_cast<NocSim*>((SimBase*)(&simulator))->handleCacheRespond(this);
 *     } else {
 *         // Packet arrived at wrong simulator type
 *         CLASS_ERROR << "Invalid simulator type!";
 *     }
 * }
 * @endcode
 *
 * ### Execution Flow:
 * @code{.unparsed}
 * Tick 20: CacheRespPacket arrives at NocSim slave port "Cache2NOC-s"
 *          │
 *          ↓ Framework calls visit(20, nocSimInstance)
 *          │
 *          ↓ dynamic_cast<NocSim*>(...) succeeds
 *          │
 *          ↓ Calls nocSimInstance.handleCacheRespond(this)
 *          │
 *          ↓ NocSim processes response:
 *            - Extracts transaction ID (tid=0)
 *            - Looks up original NocReqPacket in reqQueue
 *            - Extracts data pointer (111)
 *            - Creates NocRespPacket
 *            - Routes back to TrafficGenerator
 * @endcode
 *
 * # Complete Packet Lifecycle
 *
 * ## Request Path (Forward):
 * @code{.unparsed}
 * [1] TrafficGenerator creates NocReqPacket
 *     │ Type: NocReqPacket
 *     │ Content: {type=TEST, addr=0, size=256, tid=0}
 *     ↓
 * [2] NocSim receives NocReqPacket (via visit)
 *     │ NocReqPacket::visit() → NocSim::handleTGRequest()
 *     ↓
 * [3] NocSim transforms to CacheReqPacket (This File's Domain)
 *     │ Type: NocReqPacket → CacheReqPacket (protocol translation)
 *     │ Content: {type=TEST, addr=0, size=256, tid=0}
 *     ↓
 * [4] CacheSim receives CacheReqPacket (via visit - THIS FILE)
 *     │ CacheReqPacket::visit(when=11, cacheSimInstance)
 *     │ → CacheSim::handleNOCRequest(this, 11)
 *     ↓
 * [5] CacheSim processes request
 *     │ Generates data = 111
 *     │ Calculates delay = 9 ticks
 *     │ Creates CacheRespPacket
 * @endcode
 *
 * ## Response Path (Backward):
 * @code{.unparsed}
 * [5] CacheSim creates CacheRespPacket (This File's Domain)
 *     │ Type: CacheRespPacket
 *     │ Content: {type=TEST, data=111, tid=0}
 *     ↓
 * [6] NocSim receives CacheRespPacket (via visit - THIS FILE)
 *     │ CacheRespPacket::visit(when=20, nocSimInstance)
 *     │ → NocSim::handleCacheRespond(this)
 *     ↓
 * [7] NocSim transforms to NocRespPacket
 *     │ Type: CacheRespPacket → NocRespPacket (protocol translation)
 *     │ Content: {type=TEST, data=111, tid=0}
 *     ↓
 * [8] TrafficGenerator receives NocRespPacket (via visit)
 *     │ NocRespPacket::visit() → TrafficGenerator::handleNoCRespond()
 *     │ Logs: "get data = 111"
 * @endcode
 *
 * # Protocol Translation Details
 *
 * ## Why Two Packet Types?
 *
 * NocReqPacket and CacheReqPacket contain the **same information** but represent
 * different protocol layers:
 *
 * @code
 * // Network Layer Packet
 * NocReqPacket {
 *     NocPktTypeEnum reqType;    // NoC-specific type enum
 *     int addr, size, tid;       // Shared fields
 * };
 *
 * // Memory Layer Packet
 * CacheReqPacket {
 *     CachePktTypeEnum reqType;  // Cache-specific type enum
 *     int addr, size, tid;       // Same fields, different context
 * };
 * @endcode
 *
 * Benefits of protocol separation:
 * 1. **Extensibility**: Add NoC routing info without affecting cache protocol
 * 2. **Type Safety**: Cannot accidentally route NocPacket to CacheSim
 * 3. **Realistic Modeling**: Mirrors real hardware protocol stacks
 *
 * ## Transformation at NocSim:
 * @code
 * // In NocSim::handleTGRequest()
 * void handleTGRequest(NocReqPacket* nocReqPkt) {
 *     // Extract common fields
 *     int addr = nocReqPkt->getAddr();
 *     int size = nocReqPkt->getSize();
 *     int tid = nocReqPkt->getTransactionId();
 *
 *     // Create cache-layer packet
 *     auto cacheReqPkt = new CacheReqPacket(
 *         CachePktTypeEnum::TEST,  // Map NocPktTypeEnum → CachePktTypeEnum
 *         addr, size, tid          // Copy shared fields
 *     );
 *
 *     // Send to cache
 *     pushToMasterChannelPort("NOC2Cache-m", cacheReqPkt);
 * }
 * @endcode
 *
 * # Error Handling and Debugging
 *
 * ## Invalid Destination Errors:
 *
 * @code
 * // Example: CacheReqPacket accidentally routed to TrafficGenerator
 * void CacheReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (dynamic_cast<CacheSim*>(&simulator)) {
 *         // This check FAILS (simulator is TrafficGenerator, not CacheSim)
 *     } else {
 *         CLASS_ERROR << "Invalid simulator type!";
 *         // Error: CacheReqPacket expected CacheSim but got TrafficGenerator
 *     }
 * }
 * @endcode
 *
 * ## Common Errors:
 * 1. **Wrong Port**: `NOC2Cache-m` connected to TrafficGenerator instead of CacheSim
 * 2. **Missing Transformation**: NocSim tries to send NocReqPacket directly to CacheSim
 * 3. **Type Confusion**: Using NocPacket handler for CachePacket
 *
 * ## Debugging Checklist:
 * - Verify port connections: `NOC2Cache-m` ↔ `NOC2Cache-s`
 * - Check packet transformations in NocSim handlers
 * - Ensure visit() implementations match expected destinations
 * - Look for CLASS_ERROR messages indicating type mismatches
 *
 * # Extending for Advanced Cache Protocols
 *
 * ## 1. Adding Read/Write Operations:
 * @code
 * enum class CachePktTypeEnum { TEST, READ, WRITE };
 *
 * class CacheWriteReqPacket : public SimPacket {
 * public:
 *     CacheWriteReqPacket(int addr, int* data, int size, int tid)
 *         : SimPacket(PTYPE::MEMREQ), addr(addr), data(data), size(size), tid(tid) {}
 *
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto* cache = dynamic_cast<CacheSim*>(&simulator)) {
 *             cache->handleWriteRequest(this, when);
 *         } else {
 *             CLASS_ERROR << "Invalid simulator for CacheWriteReqPacket!";
 *         }
 *     }
 *
 * private:
 *     int addr;
 *     int* data;  // Write data
 *     int size;
 *     int tid;
 * };
 * @endcode
 *
 * ## 2. Cache Coherence Messages:
 * @code
 * enum class CoherenceType { INVALIDATE, WRITEBACK, SHARE };
 *
 * class CoherencePacket : public SimPacket {
 * public:
 *     CoherencePacket(CoherenceType type, int addr, int tid)
 *         : SimPacket(PTYPE::COHERENCE), type(type), addr(addr), tid(tid) {}
 *
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto* cache = dynamic_cast<CacheSim*>(&simulator)) {
 *             cache->handleCoherenceMessage(this, when);
 *         } else {
 *             CLASS_ERROR << "Invalid simulator for CoherencePacket!";
 *         }
 *     }
 *
 * private:
 *     CoherenceType type;
 *     int addr;
 *     int tid;
 * };
 * @endcode
 *
 * ## 3. Multi-Level Cache Packets:
 * @code
 * class L2CacheReqPacket : public SimPacket {
 * public:
 *     L2CacheReqPacket(int addr, int size, bool fromL1, int tid)
 *         : SimPacket(PTYPE::MEMREQ), addr(addr), size(size), fromL1(fromL1), tid(tid) {}
 *
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto* l2Cache = dynamic_cast<L2CacheSim*>(&simulator)) {
 *             l2Cache->handleL1Request(this, when);
 *         } else if (auto* dram = dynamic_cast<DRAMSim*>(&simulator)) {
 *             dram->handleCacheRequest(this, when);
 *         } else {
 *             CLASS_ERROR << "Invalid simulator for L2CacheReqPacket!";
 *         }
 *     }
 *
 * private:
 *     int addr;
 *     int size;
 *     bool fromL1;  // Track packet origin
 *     int tid;
 * };
 * @endcode
 *
 * # Design Patterns and Best Practices
 *
 * ## 1. Pass Timing Information When Needed:
 * CacheReqPacket passes `when` to handleNOCRequest():
 * @code
 * cache->handleNOCRequest(this, when);  // Pass arrival time
 * @endcode
 *
 * NoC packets don't need this (stateless routing):
 * @code
 * noc->handleTGRequest(this);  // No timing parameter
 * @endcode
 *
 * ## 2. Protocol-Specific Type Enums:
 * Don't share enums across protocol layers:
 * @code
 * // Good: Separate enums
 * enum class NocPktTypeEnum { TEST, UNICAST, MULTICAST };
 * enum class CachePktTypeEnum { TEST, READ, WRITE };
 *
 * // Bad: Shared enum creates coupling
 * enum class PacketTypeEnum { TEST, UNICAST, MULTICAST, READ, WRITE };
 * @endcode
 *
 * ## 3. Consistent Error Messages:
 * Include packet type in error messages:
 * @code
 * CLASS_ERROR << "Invalid simulator type for CacheReqPacket!";
 * // Not just: "Invalid simulator type!"
 * @endcode
 *
 * ## 4. Data Ownership Transfer:
 * When packets carry heap-allocated data:
 * @code
 * // CacheSim creates data
 * int* data = new int(111);
 * auto resp = new CacheRespPacket(..., data, ...);
 *
 * // Ownership transfers through packet chain:
 * // CacheSim → CacheRespPacket → NocSim → NocRespPacket → TrafficGenerator
 * // Framework handles final cleanup
 * @endcode
 *
 * @see CachePacket.hh For packet class declarations
 * @see CacheSim For CacheReqPacket handler (handleNOCRequest)
 * @see NocSim For CacheRespPacket handler (handleCacheRespond)
 * @see NocPacket.cc For companion NoC packet visitor implementations
 * @see testSimChannel.cc For complete system integration
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note Cache packets represent the memory hierarchy protocol layer. They should
 *       contain cache-specific semantics (coherence, replacement, prefetch hints)
 *       separate from network routing concerns (handled by NoC packets).
 *
 * @warning When adding new cache packet types, ensure both visit() overloads are
 *          implemented (SimBase and SimModule) even if one just logs an error.
 *          This prevents undefined behavior from pure virtual function calls.
 */

#include "CachePacket.hh"

#include "CacheSim.hh"
#include "NocPacket.hh"
#include "NocSim.hh"

void CacheRespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void CacheRespPacket::visit (SimModule& module) is not implemented yet!";
}

void CacheRespPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<NocSim*>((SimBase*)(&simulator))) {
		dynamic_cast<NocSim*>((SimBase*)(&simulator))->handleCacheRespond(this);
	} else {
		CLASS_ERROR << "Invalid simulator type!";
	}
}

void CacheReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void CacheReqPacket::visit (SimModule& module) is not implemented yet!";
}

// When CacheSim visit this packet, it will create cacheRespPkt packed in nocRespPkt
// Then executing callback to return nocRespPkt
void CacheReqPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<CacheSim*>((SimBase*)(&simulator))) {
		dynamic_cast<CacheSim*>((SimBase*)(&simulator))->handleNOCRequest(this, when);
	} else {
		CLASS_ERROR << "Invalid simulator type!";
	}
}
