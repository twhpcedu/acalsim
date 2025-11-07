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
 * @file PE.cc
 * @brief Processing Element (PE) simulator implementation
 *
 * This file implements a simplified Processing Element simulator that demonstrates
 * packet handling through the visitor pattern. The PE acts as a computational unit
 * that can receive different types of packets (DataReqPacket, PEReqPacket) and
 * process them accordingly.
 *
 * **PE Role in Communication:**
 *
 * The PE sits at the end of the communication chain:
 * ```
 * TrafficGenerator ──[EventPacket(PEReqEvent)]──► PE
 *                                                  │
 *                                                  ▼
 *                                            Process Request
 *                                                  │
 *                                                  ▼
 *                                        Invoke Callback to TG
 * ```
 *
 * **Packet Reception Flow:**
 *
 * When a packet arrives at the PE:
 * ```
 * 1. EventPacket received via SlaveChannelPort "USTrafficGenerator"
 * 2. Framework extracts embedded SimEvent (e.g., PEReqEvent)
 * 3. Framework schedules event for target tick
 * 4. At target tick: event.process() invoked
 * 5. Event may call packet.visit(simulator) if needed
 * 6. visit() method calls appropriate packet handler
 * 7. Handler processes packet and generates response
 * ```
 *
 * **Visitor Pattern Implementation:**
 *
 * The PE uses the visitor pattern for packet handling:
 * - Packets implement visit(Tick, SimBase&) method
 * - visit() downcasts SimBase to PE
 * - visit() calls appropriate handler (dataPacketHandler, peReqPacketHandler)
 * - Handlers extract packet data and process it
 *
 * **PE Configuration Parameters:**
 * - peReqDelay: Base latency for processing requests (1 tick)
 * - peRespDelay: Base latency for generating responses (1 tick)
 * - getRespDelay(): Calculates response latency based on data size
 *
 * **Design Notes:**
 *
 * 1. **Why Visitor Pattern?**
 *    - Type-safe packet handling without explicit type checks
 *    - Each packet type knows how to visit each simulator type
 *    - Extensible: new packet types don't require changes to PE
 *
 * 2. **Packet Handler Separation:**
 *    - dataPacketHandler: For simple data transfers
 *    - peReqPacketHandler: For computational requests
 *    - Allows different processing logic per packet type
 *
 * 3. **Event-Driven Processing:**
 *    - PE doesn't actively poll for packets
 *    - Framework delivers events to PE's event queue
 *    - PE processes events when scheduled
 *
 * **Extension Points:**
 *
 * This PE implementation can be extended to:
 * - Add internal pipeline stages
 * - Implement resource contention
 * - Add local memory/cache hierarchy
 * - Support multiple outstanding requests
 * - Implement inter-PE communication (PE mesh)
 *
 * @see PE.hh for class definition and interface
 * @see PEReq.cc for PEReqPacket visitor implementation
 * @see DataReq.cc for DataReqPacket visitor implementation
 * @see PEEvent.cc for event processing logic
 */

#include "PE.hh"

#include "DataReq.hh"
#include "PEReq.hh"

/**
 * @brief Handle DataReqPacket received by PE
 *
 * This handler demonstrates a simple data reception pattern where the PE
 * receives raw data from an external source (e.g., TrafficGenerator).
 *
 * **Invocation Path:**
 * ```
 * 1. DataReqPacket.visit(when, simulator) called
 * 2. visit() downcasts simulator to PE*
 * 3. visit() calls PE::dataPacketHandler(when, packet)
 * 4. Handler extracts and processes data
 * ```
 *
 * **Packet Structure:**
 * - DataReqPacket wraps a void* data pointer
 * - getData() returns the raw pointer
 * - Handler casts to expected type (int* in this example)
 * - Data ownership managed by packet
 *
 * **Processing Logic:**
 * ```cpp
 * int* data = (int*)dataPkt->getData();  // Extract integer data
 * CLASS_INFO << "Received Data: " << *data;  // Log reception
 * // Could process data, update PE state, etc.
 * ```
 *
 * **Error Handling:**
 * - Dynamic cast checks if packet is actually DataReqPacket
 * - Logs error if unexpected packet type received
 * - Framework ensures type safety through visitor pattern
 *
 * **Usage Example:**
 *
 * From TrafficGenerator:
 * ```cpp
 * int* myData = new int(42);
 * DataReqPacket* pkt = new DataReqPacket(tick + 10, myData);
 * // Send to PE via channel...
 * ```
 *
 * @param when Simulation tick when packet was received
 * @param pkt Pointer to received SimPacket (should be DataReqPacket)
 *
 * @note This handler is currently unused in the main testCommunication flow
 *       but demonstrates how to handle alternative packet types.
 *
 * @warning The handler doesn't free the data pointer - ensure proper
 *          memory management in the packet destructor or caller.
 */
void PE::dataPacketHandler(Tick when, SimPacket* pkt) {
	auto dataPkt = dynamic_cast<DataReqPacket*>(pkt);
	if (dataPkt) {
		int* data = (int*)dataPkt->getData();
		CLASS_INFO << "Received Data from Traffic Generator : " + std::to_string(*data);
	} else {
		CLASS_ERROR << "Invalid Packet Type";
	}
}

/**
 * @brief Handle PEReqPacket received by PE
 *
 * This handler demonstrates packet reception through the visitor pattern.
 * However, in the current testCommunication example, actual PE request
 * processing happens in PEReqEvent::process(), not through this handler.
 *
 * **Invocation Path (Visitor Pattern):**
 * ```
 * 1. PEReqPacket.visit(when, simulator) called
 * 2. visit() downcasts simulator to PE*
 * 3. visit() calls PE::peReqPacketHandler(when, packet)
 * 4. Handler acknowledges packet reception
 * ```
 *
 * **Current Usage:**
 *
 * In testCommunication, PEReqPackets are wrapped in PEReqEvents, and
 * processing happens in PEReqEvent::process():
 * ```cpp
 * // TrafficEvent creates:
 * PEReqPacket* peReqPkt = new PEReqPacket(TEST, a, b, c, respPkt);
 * PEReqEvent* event = new PEReqEvent(tid, pe, callback, peReqPkt);
 * EventPacket* eventPkt = new EventPacket(event, targetTick);
 *
 * // PE receives EventPacket, framework schedules PEReqEvent
 * // At target tick: PEReqEvent::process() handles computation
 * ```
 *
 * **Alternative Usage (Direct Packet Handling):**
 *
 * This handler could be used for direct packet processing:
 * ```cpp
 * // Direct packet send (without event wrapper):
 * PEReqPacket* pkt = new PEReqPacket(TEST, 100, 2, 50, respPkt);
 * pkt->visit(currentTick, *peSimulator);
 * // Would invoke peReqPacketHandler
 * ```
 *
 * **Design Choice:**
 *
 * The example uses PEReqEvent::process() instead of this handler because:
 * - Events allow deferred execution (schedule for future tick)
 * - Events can carry callbacks for response handling
 * - Events integrate with framework's event queue
 * - More flexible for complex request-response protocols
 *
 * **Packet Type Validation:**
 * - dynamic_cast verifies packet is PEReqPacket
 * - Logs error if wrong type received
 * - Prevents type-related crashes
 *
 * **Processing Responsibilities:**
 *
 * If this handler were used for computation:
 * ```cpp
 * auto peReqPkt = dynamic_cast<PEReqPacket*>(pkt);
 * if (peReqPkt) {
 *     int result = peReqPkt->getA() * peReqPkt->getB() + peReqPkt->getC();
 *     peReqPkt->getPERespPkt()->setResult(result);
 *     // Invoke callback...
 * }
 * ```
 *
 * @param when Simulation tick when packet was received
 * @param pkt Pointer to received SimPacket (should be PEReqPacket)
 *
 * @note Currently only logs reception; actual computation happens in
 *       PEReqEvent::process() for this example.
 *
 * @see PEReqEvent::process() for actual request processing logic
 * @see PEReq.cc for PEReqPacket::visit() implementation
 */
void PE::peReqPacketHandler(Tick when, SimPacket* pkt) {
	auto peReqPkt = dynamic_cast<PEReqPacket*>(pkt);
	if (peReqPkt) {
		CLASS_INFO << "Received PE Request Packet";
	} else {
		CLASS_ERROR << "Invalid Packet Type";
	}
}
