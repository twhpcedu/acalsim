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
 * @file Memory.cc
 * @brief Fixed-latency memory model with event-driven response generation
 *
 * This file implements the Memory simulator, which demonstrates a **simple memory controller**
 * with **fixed access latency**, **request queuing**, and **automatic response generation**.
 * It serves as the endpoint in the CPU-Bus-Memory system, completing the request-response cycle.
 *
 * **Component Overview:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                           Memory                                        │
 * │                                                                         │
 * │  ┌────────────┐       ┌─────────────────┐       ┌────────────┐       │
 * │  │ bus-s      │──────►│  process_queue  │──────►│  bus-m     │       │
 * │  │ SlavePort  │       │  (LimitedObj    │       │ MasterPort │       │
 * │  │            │       │   Container)    │       │            │       │
 * │  │ Receives   │       │                 │       │ Sends      │       │
 * │  │ requests   │       │ + mem_latency   │       │ responses  │       │
 * │  │ from Bus   │       │                 │       │ to Bus     │       │
 * │  └────────────┘       └─────────────────┘       └────────────┘       │
 * │       ▲                       │                        │              │
 * │       │                       │                        │              │
 * │       │                       ▼                        ▼              │
 * │  BaseReqPacket         Process Request          BaseRspPacket        │
 * │  (from Bus)            (Fixed latency)          (to Bus)             │
 * │                        Generate Response                             │
 * │                        Recycle Request                               │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Memory Model Characteristics:**
 * ```
 * Type: Fixed-Latency Memory
 *   - All requests take exactly kInteralLatency ticks to complete
 *   - No address dependencies or bank conflicts
 *   - Infinite capacity (no address validation)
 *   - Single-queue processing (FIFO order)
 *
 * Comparison with Real Memory:
 *   Simplified Model (this implementation):
 *     - Fixed latency for all accesses
 *     - No read/write differentiation
 *     - No address translation
 *     - No bank interleaving
 *
 *   Realistic Model (possible extensions):
 *     - Variable latency (bank conflicts, row buffer hits/misses)
 *     - Different latencies for read vs. write
 *     - Address decoding to banks
 *     - Memory controller scheduling policies
 * ```
 *
 * **Port Configuration:**
 * ```
 * SlavePort "bus-s":
 *   - Receives requests from Bus
 *   - Queue size: 1
 *   - Arbiter: Round-robin (default)
 *   - Connected to: CrossBar.mem-m
 *
 * MasterPort "bus-m":
 *   - Sends responses to Bus
 *   - No internal queue (master ports don't queue)
 *   - Connected to: CrossBar.mem-s
 * ```
 *
 * **Request Processing Flow:**
 * ```
 * Step 1: Accept Request from Bus
 *   acceptPacketFromBUS()
 *     └─► if (s_port->isPopValid())
 *          └─► accept(tick, packet)  // Calls handler(BaseReqPacket*)
 *
 * Step 2: Process Request Handler
 *   handler(BaseReqPacket* packet)
 *     ├─1─► Check process_queue.isPushReady()
 *     ├─2─► Pop from s_port SlavePort
 *     ├─3─► pushToProcessQueue(packet, tick + kInteralLatency)
 *     └─4─► If queue was empty → tryProcessRequest()
 *
 * Step 3: Schedule Request Processing
 *   tryProcessRequest()
 *     └─► If current_tick >= process_queue.getMinTick()
 *          └─► processRequest() immediately
 *          Else
 *          └─► Schedule processRequest() at min_tick
 *
 * Step 4: Process and Generate Response
 *   processRequest()
 *     ├─1─► Pop request packet from process_queue
 *     ├─2─► Extract ReqID from request
 *     ├─3─► Create BaseRspPacket with same ReqID
 *     ├─4─► Recycle request packet
 *     ├─5─► issueResponse(rsp_packet)
 *     └─6─► If queue not empty → tryProcessRequest() at tick+1
 *
 * Step 5: Send Response
 *   issueResponse(BaseRspPacket* packet)
 *     └─► Push to m_port MasterPort (to Bus)
 * ```
 *
 * **Latency Modeling:**
 * ```
 * Request arrives at tick T:
 *   T+0: Request received at bus-s SlavePort
 *   T+0: handler() called via accept()
 *   T+0: Request pushed to process_queue with ready_time = T + kInteralLatency
 *   T+kInteralLatency: processRequest() scheduled
 *   T+kInteralLatency: Response generated and sent to bus-m MasterPort
 *
 * Total Memory Latency: kInteralLatency ticks
 *
 * Example with kInteralLatency=5:
 *   Tick 10: Request arrives
 *   Tick 10: Added to process_queue (ready at tick 15)
 *   Tick 15: Response sent back to bus
 *   Latency: 5 ticks
 * ```
 *
 * **Request-Response Transformation:**
 * ```
 * Input: BaseReqPacket
 *   - ReqID: 42
 *   - (Other fields unused in this simple model)
 *
 * Processing:
 *   1. Extract ReqID from request
 *   2. Simulate memory access (wait kInteralLatency ticks)
 *   3. Create response with same ReqID
 *
 * Output: BaseRspPacket
 *   - ReqID: 42 (matches request)
 *   - (Data fields could be added for realistic model)
 *
 * Request Lifecycle:
 *   Created by:   CPUCore
 *   Forwarded by: CrossBar
 *   Received by:  Memory
 *   Recycled by:  Memory (after response created)
 *
 * Response Lifecycle:
 *   Created by:   Memory
 *   Forwarded by: CrossBar
 *   Received by:  CPUCore
 *   Recycled by:  CPUCore (after matching with outstanding request)
 * ```
 *
 * **Memory Queue (process_queue):**
 * ```
 * Type: LimitedObjectContainer<BaseReqPacket*>
 *   - Capacity: queue_size (from constructor)
 *   - Time-ordered: Earliest ready request processed first
 *   - Purpose: Model memory controller request buffer
 *
 * Queue Operations:
 *   push(packet, when):
 *     - Add request with completion time
 *     - when = current_tick + kInteralLatency
 *
 *   pop():
 *     - Remove earliest ready request
 *     - Only when current_tick >= ready_time
 *
 *   step():
 *     - Reset per-cycle push flag
 *     - Called at end of each simulation cycle
 * ```
 *
 * **Backpressure Handling:**
 * ```
 * Scenario 1: Bus Port Full (bus-m backpressure)
 *   issueResponse(packet) {
 *       if (!m_port->push(packet)) {
 *           // Push failed, retry via masterPortRetry()
 *           // Response stays in local state (not queued)
 *       }
 *   }
 *
 *   masterPortRetry(MasterPort* port) {
 *       if (port == m_port) {
 *           tryProcessRequest();  // Retry sending response
 *       }
 *   }
 *
 * Scenario 2: Process Queue Full
 *   handler(BaseReqPacket* packet) {
 *       if (!process_queue.isPushReady()) {
 *           // Queue full, cannot accept request
 *           // SlavePort will backpressure Bus
 *           // Bus will backpressure CPU
 *           return;
 *       }
 *   }
 * ```
 *
 * **RecycleContainer Integration:**
 * ```
 * Memory uses RecycleContainer for packet management:
 *
 * Request Recycling:
 *   processRequest() {
 *       auto req_packet = process_queue.pop();
 *       auto req_id = req_packet->getReqId();
 *
 *       // Create response
 *       auto rsp_packet = rc->acquire<BaseRspPacket>(
 *           &BaseRspPacket::renew, req_id
 *       );
 *
 *       // Recycle request (no longer needed)
 *       rc->recycle(req_packet);
 *
 *       issueResponse(rsp_packet);
 *   }
 *
 * Why Recycle:
 *   - Avoids repeated new/delete calls
 *   - Reduces memory fragmentation
 *   - Improves simulation performance
 *   - Pools memory for reuse
 * ```
 *
 * **Performance Characteristics:**
 *
 * | Configuration       | Effect                                        | Trade-off                   |
 * |---------------------|-----------------------------------------------|-----------------------------|
 * | High mem_latency    | Realistic DRAM timing (50-200 cycles)         | Longer request latency      |
 * | Low mem_latency     | Stress-test system, ideal memory              | Unrealistic performance     |
 * | Large queue_size    | Absorbs request bursts, reduces backpressure  | More memory usage           |
 * | Small queue_size    | Forces bus backpressure, tests flow control   | Frequent stalls             |
 *
 * **Usage Example:**
 * ```cpp
 * // In TestSimPortTop::registerSimulators()
 * auto mem = new Memory(
 *     "mem",          // Simulator name
 *     5,              // internal_resp_latency (memory access time)
 *     2               // queue_size (request buffer capacity)
 * );
 * this->addSimulator(mem);
 *
 * // Connect Bus → Memory (request path)
 * SimPortManager::ConnectPort(bus, mem, "mem-m", "bus-s");
 *
 * // Connect Memory → Bus (response path)
 * SimPortManager::ConnectPort(mem, bus, "bus-m", "mem-s");
 * ```
 *
 * **Extending to Realistic Memory:**
 *
 * 1. **Add Address-Based Timing:**
 *    ```cpp
 *    uint64_t calculateLatency(uint64_t addr) {
 *        uint32_t bank = (addr >> 6) & 0xF;
 *        if (last_bank[bank] == (addr >> 12)) {
 *            return row_buffer_hit_latency;  // Same row
 *        } else {
 *            return row_buffer_miss_latency; // Different row
 *        }
 *    }
 *    ```
 *
 * 2. **Add Bank-Level Parallelism:**
 *    ```cpp
 *    std::array<LimitedObjectContainer<BaseReqPacket*>, 16> bank_queues;
 *    // Dispatch requests to different banks
 *    // Banks can process in parallel
 *    ```
 *
 * 3. **Add Memory Controller Scheduling:**
 *    ```cpp
 *    class MemoryScheduler {
 *        BaseReqPacket* selectNextRequest() {
 *            // FR-FCFS: Prioritize row buffer hits
 *            // Or other policies
 *        }
 *    };
 *    ```
 *
 * 4. **Add Read/Write Differentiation:**
 *    ```cpp
 *    if (req_packet->type == READ) {
 *        latency = read_latency;
 *    } else {
 *        latency = write_latency;
 *    }
 *    ```
 *
 * **Design Patterns:**
 *
 * 1. **Request-Response Transform:**
 *    - Receive request, generate response
 *    - Preserve request ID for matching
 *
 * 2. **Fixed-Latency Delay:**
 *    - All operations take constant time
 *    - Simplified but predictable
 *
 * 3. **Event-Driven Processing:**
 *    - Requests scheduled for future processing
 *    - No busy-waiting or polling
 *
 * 4. **Packet Recycling:**
 *    - Request recycled after response created
 *    - Efficient memory management
 *
 * **Common Memory Models:**
 *
 * | Model Type         | Characteristics                              | Use Case                    |
 * |--------------------|----------------------------------------------|-----------------------------|
 * | Fixed Latency      | Constant access time (this implementation)   | Fast simulation, testing    |
 * | Bank-Level         | Multiple banks, interleaving                 | Parallelism study           |
 * | DRAM Timing        | Row buffer, tRCD, tRAS, tCAS                | Realistic performance       |
 * | Trace-Based        | Replay real memory traces                    | Validation                  |
 *
 * @see CPUCore For request generator
 * @see CrossBar For bus that forwards requests to memory
 * @see BasePacket For packet structure and request-response relationship
 * @see LimitedObjectContainer For time-ordered queue implementation
 */

#include "hw/Memory.hh"

#include "BasePacket.hh"
#include "CallBackEvent.hh"

namespace test_port {

Memory::Memory(const std::string& name, size_t internal_resp_latency, size_t queue_size)
    : acalsim::CPPSimBase(name),
      kInteralLatency(internal_resp_latency),
      process_queue(LimitedObjectContainer<BaseReqPacket*>(queue_size)) {
	this->m_port = this->addMasterPort("bus-m");
	this->s_port = this->addSlavePort("bus-s", 1);
}

void Memory::init() {}

void Memory::step() {
	this->acceptPacketFromBUS();

	// Flag Cleanup(make sure each cycle can only push 1 packet to queue)
	this->process_queue.step();
}

void Memory::cleanup() {}

void Memory::masterPortRetry(acalsim::MasterPort* port) {
	if (this->m_port == port) { this->tryProcessRequest(); }
}

void Memory::acceptPacketFromBUS() {
	if (this->s_port->isPopValid()) {
		auto packet = this->s_port->front();
		this->accept(acalsim::top->getGlobalTick(), *packet);
	}
}

void Memory::handler(BaseReqPacket* packet) {
	const auto rc = acalsim::top->getRecycleContainer();

	// 1. Ensure the ProcessQueue is ready to accept a new ReqPacket
	if (!this->process_queue.isPushReady()) { return; }
	LABELED_ASSERT_MSG(this->s_port->isPopValid(), name, "SlavePort is not ready to provide a valid RspPacket.");
	this->s_port->pop();

	// 1.1 Add Log & trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [5] Pop out the ReqPacket from CPU2BusPort and push to ReqOutQueue.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "BUS-MEM(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 2. Push the ReqPacket to the ProcessQueue with a delay based on the internal latency.
	this->pushToProcessQueue(packet, acalsim::top->getGlobalTick() + this->kInteralLatency);
}

void Memory::pushToProcessQueue(BaseReqPacket* packet, acalsim::Tick when) {
	// Check if the ReqOutQueue was previously empty
	const bool was_empty = this->process_queue.empty();

	// Attempt to push the packet into the ReqOutQueue and ensure success
	LABELED_ASSERT_MSG(this->process_queue.push(packet, when), name,
	                   "Failed to push to the ProcessQueue: the queue is full.");

	// Add Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "MEM(ProPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// This ensures that requests will be when the queue transitions from empty to non-empty.
	if (was_empty) { this->tryProcessRequest(); }
}

void Memory::tryProcessRequest() {
	if (this->process_queue.empty()) { return; }
	const auto rc       = acalsim::top->getRecycleContainer();
	const auto min_tick = this->process_queue.getMinTick();
	auto       callback = [this]() { this->processRequest(); };

	// Issue a request immediately if the earliest tick is less than or equal to CurrTick
	if (acalsim::top->getGlobalTick() >= min_tick) {
		callback();
	} else {
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, min_tick);
	}
}

void Memory::processRequest() {
	// 1. Make sure that ProcessQueue can pop the packet.
	if (!this->process_queue.isPopValid()) { return; }

	// 2. Ensure the output port is ready to accept a new packet.
	if (!this->m_port->isPushReady()) { return; }

	// 3. Check if the earliest request in the queue is ready to be issued
	if (this->process_queue.getMinTick() > acalsim::top->getGlobalTick()) { return; }

	// 4. Process: Issue Outbound Request the master port.
	auto req_packet = this->popFromProcessQueue();
	LABELED_ASSERT_MSG(req_packet, name, "Failed to pop the ReqPacket from the ProcessQueue.");
	std::string req_name = "ReqId-" + std::to_string(req_packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "MEM(ProPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	auto rsp_packet =
	    acalsim::top->getRecycleContainer()->acquire<BaseRspPacket>(&BaseRspPacket::renew, req_packet->getReqId());
	acalsim::top->getRecycleContainer()->recycle(req_packet);

	this->issueResponse(rsp_packet);
}

BaseReqPacket* Memory::popFromProcessQueue() {
	auto packet = this->process_queue.pop();
	if (!this->process_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryProcessRequest(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
	return packet;
}

void Memory::issueResponse(BaseRspPacket* packet) {
	LABELED_ASSERT_MSG(this->m_port->push(packet), name, "Failed to push the RespPacket to the Mem2BusPort.");
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [6] Pop out the RspPacket from RspQueue and push to Mem2BusPort.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "MEM-BUS(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));
}

}  // namespace test_port
