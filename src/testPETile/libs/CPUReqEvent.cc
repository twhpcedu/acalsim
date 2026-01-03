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
 * @file CPUReqEvent.cc
 * @brief CPU memory request event with callback creation and trace generation
 *
 * This file implements the CPUReqEvent class, which represents a scheduled memory request
 * from the CPUTraffic module to the AXI Bus. It demonstrates event-driven simulation,
 * callback wrapping for response routing, and trace generation for performance analysis.
 *
 * **CPUReqEvent Role in Request Pipeline:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                       CPUReqEvent                                      │
 * │              (Memory Request Transmission Event)                       │
 * │                                                                        │
 * │  Scheduled By:                                                         │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ CPUTraffic::injectMemRequests()                                  │ │
 * │  │ ├─ Creates CPUReqEvent(tid, bus, callback, memReqPkt)            │ │
 * │  │ └─ Schedules at tick: 1, 11, 21, 31, 41                          │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Process Execution (process()):                                        │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Log event processing with transaction ID                      │ │
 * │  │                                                                  │ │
 * │  │ 2. Create new callback lambda:                                   │ │
 * │  │    └─ λ(id, resp) { cpuReqCallback(tid, resp); }                │ │
 * │  │       └─ Wraps original CPUTraffic::MemRespHandler callback     │ │
 * │  │                                                                  │ │
 * │  │ 3. Attach callback to memory request packet:                     │ │
 * │  │    └─ memReqPkt->setCallback(callback)                          │ │
 * │  │                                                                  │ │
 * │  │ 4. Send packet to AXI Bus via accept():                          │ │
 * │  │    └─ AXIBus::accept() → visit() → memReqPktHandler()           │ │
 * │  │                                                                  │ │
 * │  │ 5. Generate trace record:                                        │ │
 * │  │    ├─ Create CpuTrafficTraceRecord                              │ │
 * │  │    ├─ Include tick, req_type, tid, addr, size                   │ │
 * │  │    └─ Add to "CPUReq" category for chrome://tracing             │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Response Callback (cpuReqCallback()):                                 │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ Invoked when response received from bus:                         │ │
 * │  │ ├─ Log callback invocation with transaction ID                   │ │
 * │  │ └─ Invoke original callerCallback (CPUTraffic::MemRespHandler)   │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/CPUReqEvent.hh
 * - CPU Traffic: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUTraffic.cc
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - Mem Packets: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/MemReq.cc
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "CPUReqEvent.hh"

#include "AXIBus.hh"

/**
 * @brief Processes the CPU memory request event by sending packet to AXI Bus and generating trace
 *
 * This method is invoked when the event is dequeued from the event queue at its scheduled tick.
 * It performs three critical operations: (1) wraps the caller's callback for response routing,
 * (2) sends the memory request packet to the AXI Bus, and (3) generates a trace record for
 * performance analysis and debugging.
 *
 * @note This method is called automatically by the event queue
 * @note Callback is invoked later when response propagates back from SRAM
 * @note Trace record enables correlation of requests with responses
 *
 * @see CPUReqEvent::cpuReqCallback() Callback method invoked on response
 * @see AXIBus::memReqPktHandler() Handler that receives this packet
 * @see CpuTrafficTraceRecord::toJson() Trace serialization to JSON
 */
void CPUReqEvent::process() {
	CLASS_INFO << "Process CPUReqEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();
	auto callback = [this](int id, MemRespPacket* memRespPkt) {
		this->cpuReqCallback(this->tid, this->memReqPkt->getMemRespPkt());
	};
	this->memReqPkt->setCallback(callback);
	((AXIBus*)callee)->accept(top->getGlobalTick(), (SimPacket&)*memReqPkt);

	top->addTraceRecord(/* trace */ std::make_shared<CpuTrafficTraceRecord>(
	                        /* tick */ top->getGlobalTick(),
	                        /* req_type */ this->memReqPkt->reqType,
	                        /* transaction_id */ this->tid,
	                        /* addr */ this->memReqPkt->getAddr(),
	                        /* size */ this->memReqPkt->getSize()),
	                    /* category */ "CPUReq");
}
