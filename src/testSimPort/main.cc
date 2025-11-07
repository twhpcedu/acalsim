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
 * @file main.cc
 * @brief Port-based communication example - CPU-Bus-Memory architecture
 *
 * This example demonstrates **fundamental port-based communication** in ACALSim through
 * a canonical CPU-Bus-Memory system. It showcases request-response protocols, backpressure
 * handling, multi-master arbitration, and outstanding request tracking.
 *
 * **System Architecture:**
 * ```
 * ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
 * │    CPUCore       │            │    CrossBar      │            │     Memory       │
 * │                  │            │      (Bus)       │            │                  │
 * │  Request         │            │                  │            │                  │
 * │  Generator       │            │  Multi-Master    │            │  Fixed-Latency   │
 * │                  │            │  Arbiter         │            │  Response        │
 * │  Outstanding     │            │                  │            │  Generator       │
 * │  Request         │            │  Request         │            │                  │
 * │  Tracker         │            │  Forwarder       │            │                  │
 * └──────────────────┘            └──────────────────┘            └──────────────────┘
 *         ▲ │                            ▲ │                            ▲ │
 *  req ───┘ └─── rsp               req ─┘ └─── rsp               req ─┘ └─── rsp
 *  (M-port)  (S-port)              (M/S-ports)                    (S-port)  (M-port)
 *
 * Port Connections:
 *   CPU.bus-m ────► Bus.cpu-s  (CPU sends requests to Bus)
 *   Bus.cpu-m ────► CPU.bus-s  (Bus sends responses to CPU)
 *   Bus.mem-m ────► Mem.bus-s  (Bus sends requests to Memory)
 *   Mem.bus-m ────► Bus.mem-s  (Memory sends responses to Bus)
 * ```
 *
 * **Communication Flow (Request Path):**
 * ```
 * Tick 0-5:
 *   CPUCore:
 *     1. Generate request packet (ReqID=0)
 *     2. Push to internal queue (kInteralLatency delay)
 *     3. Schedule event to send at Tick 5
 *
 * Tick 5:
 *   CPUCore (Phase 1):
 *     4. Pop from internal queue
 *     5. Push to bus-m MasterPort
 *     6. Track in outstanding_req_queue
 *
 *   Framework (Phase 2):
 *     7. SimPortManager syncs ports
 *     8. CrossBar.cpu-s arbitrates (single master, immediate grant)
 *     9. Packet transferred to CrossBar's SlavePort queue
 *
 * Tick 6:
 *   CrossBar (Phase 1):
 *     10. Pop from cpu-s SlavePort
 *     11. Add bus latency (kBusLatency)
 *     12. Schedule forward event
 *
 * Tick 6+kBusLatency:
 *   CrossBar (Phase 1):
 *     13. Pop from internal queue
 *     14. Push to mem-m MasterPort
 *
 *   Framework (Phase 2):
 *     15. Memory.bus-s receives packet
 *
 * Tick 7+kBusLatency:
 *   Memory (Phase 1):
 *     16. Pop from bus-s SlavePort
 *     17. Process request (kMemLatency delay)
 *     18. Generate response packet
 *     19. Schedule response event
 * ```
 *
 * **Communication Flow (Response Path):**
 * ```
 * Tick 7+kBusLatency+kMemLatency:
 *   Memory (Phase 1):
 *     20. Create response packet
 *     21. Push to bus-m MasterPort
 *
 *   Framework (Phase 2):
 *     22. CrossBar.mem-s receives response
 *
 * Tick 8+kBusLatency+kMemLatency:
 *   CrossBar (Phase 1):
 *     23. Pop from mem-s SlavePort
 *     24. Add bus latency
 *     25. Push to cpu-m MasterPort
 *
 *   Framework (Phase 2):
 *     26. CPUCore.bus-s receives response
 *
 * Tick 9+kBusLatency+kMemLatency:
 *   CPUCore (Phase 1):
 *     27. Pop from bus-s SlavePort
 *     28. Match with outstanding request (ReqID)
 *     29. Mark request complete
 *     30. Generate next request if total not reached
 * ```
 *
 * **Key Features Demonstrated:**
 *
 * 1. **Port-Based Communication:**
 *    - MasterPort for sending packets
 *    - SlavePort for receiving packets
 *    - Automatic synchronization in Phase 2
 *
 * 2. **Backpressure Handling:**
 *    - MasterPort::push() returns false when full
 *    - Retry callback mechanism (masterPortRetry)
 *    - Outstanding request limit (max in-flight requests)
 *
 * 3. **Multi-Master Arbitration:**
 *    - CrossBar receives from CPU and Memory
 *    - Round-robin arbiter (default)
 *    - Fair bandwidth allocation
 *
 * 4. **Event-Driven Delays:**
 *    - CPU internal latency (kInteralLatency)
 *    - Bus latency (kBusLatency)
 *    - Memory latency (kMemLatency)
 *    - CallBackEvent for timed actions
 *
 * 5. **Request-Response Tracking:**
 *    - Unique request IDs (std::atomic counter)
 *    - Outstanding request queue
 *    - Response matching by ReqID
 *
 * 6. **RecycleContainer Memory Management:**
 *    - Packet allocation via acquire<>()
 *    - Automatic recycling after use
 *    - No manual new/delete
 *
 * **Command-Line Options:**
 * ```bash
 * # Run with default parameters
 * ./testSimPort
 *
 * # Customize CPU behavior
 * ./testSimPort --cpu-outstanding-requests 8 --cpu-total-requests 1000
 *
 * # Adjust latencies
 * ./testSimPort --cpu-latency 5 --bus-latency 3 --mem-latency 100
 *
 * # Change queue sizes
 * ./testSimPort --queue-size 16
 *
 * # Combine with framework options
 * ./testSimPort --max-tick 10000 --threadmanager V3 --verbose
 * ```
 *
 * **Expected Output:**
 * ```
 * [cpu] [0] Initialization
 * [cpu] ReqId-0: [1] Create a Request Packet and push to ReqOutQueue.
 * [cpu] ReqId-0: [2] Push the Request to MasterPort-bus-m
 * [bus] [3] Receive a Request from CPU
 * [bus] [4] Forward Request to Memory
 * [mem] [5] Receive Request from Bus
 * [mem] [6] Process Request and generate Response
 * [bus] [7] Receive Response from Memory
 * [bus] [8] Forward Response to CPU
 * [cpu] ReqId-0: [9] Receive Response from Bus
 * [cpu] All Requests have been issued to Memory!
 * [cpu] [9] All responses have been handled by the CPU!
 * ```
 *
 * **Learning Outcomes:**
 * - Understand MasterPort/SlavePort communication patterns
 * - Implement request-response protocols with IDs
 * - Handle backpressure and retry mechanisms
 * - Use events for modeling latencies
 * - Manage outstanding requests with flow control
 * - Apply RecycleContainer for efficient memory management
 *
 * **Performance Characteristics:**
 * - Total latency per request: kCPULatency + 2×kBusLatency + kMemLatency
 * - Maximum throughput: Limited by min(CPU rate, Bus bandwidth, Memory bandwidth)
 * - Concurrency: Up to cpu_outstanding_requests in-flight requests
 *
 * **Extending This Example:**
 * 1. Add cache hierarchy between CPU and bus
 * 2. Implement multiple CPUs with arbiter contention
 * 3. Add different packet types (read, write, prefetch)
 * 4. Implement priority-based arbitration
 * 5. Add memory bank interleaving
 *
 * @see TestSimPortTop For system configuration and CLI options
 * @see CPUCore For request generator implementation
 * @see CrossBar For multi-master bus arbiter
 * @see Memory For fixed-latency memory model
 * @see BasePacket For packet structure
 */

#include "ACALSim.hh"
#include "TestSimPortTop.hh"

int main(int argc, char** argv) {
	acalsim::top = std::make_shared<test_port::TestSimPortTop>();
	acalsim::top->init(argc, argv);
	acalsim::top->run();
	acalsim::top->finish();
	return 0;
}
