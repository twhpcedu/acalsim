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
 * @file CrossBar.cc
 * @brief Multi-master bus arbiter with latency modeling and packet forwarding
 *
 * This file implements the CrossBar simulator, which serves as a **shared interconnect**
 * between CPU and Memory. It demonstrates **multi-master arbitration**, **bus latency modeling**,
 * and **bidirectional packet forwarding** in discrete-event simulation.
 *
 * **Component Overview:**
 * ```
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │                            CrossBar (Bus)                                 │
 * │                                                                           │
 * │  Request Path (CPU → Memory):                                            │
 * │  ┌────────────┐       ┌─────────────────┐       ┌────────────┐         │
 * │  │ cpu-s      │──────►│   req_queue     │──────►│  mem-m     │         │
 * │  │ SlavePort  │       │  (LimitedObj    │       │ MasterPort │         │
 * │  │            │       │   Container)    │       │            │         │
 * │  │ Arbitrates │       │                 │       │ To Memory  │         │
 * │  │ from CPU   │       │ + bus_latency   │       │            │         │
 * │  └────────────┘       └─────────────────┘       └────────────┘         │
 * │       ▲                                                                   │
 * │       │                                                                   │
 * │       │ From CPU (BaseReqPacket)                                         │
 * │       │                                                                   │
 * │                                                                           │
 * │  Response Path (Memory → CPU):                                           │
 * │  ┌────────────┐       ┌─────────────────┐       ┌────────────┐         │
 * │  │ mem-s      │──────►│   rsp_queue     │──────►│  cpu-m     │         │
 * │  │ SlavePort  │       │  (LimitedObj    │       │ MasterPort │         │
 * │  │            │       │   Container)    │       │            │         │
 * │  │ Arbitrates │       │                 │       │ To CPU     │         │
 * │  │ from Mem   │       │ + bus_latency   │       │            │         │
 * │  └────────────┘       └─────────────────┘       └────────────┘         │
 * │       ▲                                                                   │
 * │       │                                                                   │
 * │       │ From Memory (BaseRspPacket)                                      │
 * │       │                                                                   │
 * └──────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Port Configuration:**
 * ```
 * SlavePort "cpu-s":
 *   - Receives requests from CPU
 *   - Queue size: 1
 *   - Arbiter: Round-robin (default)
 *   - Connected to: CPUCore.bus-m
 *
 * MasterPort "cpu-m":
 *   - Sends responses to CPU
 *   - No internal queue (master ports don't queue)
 *   - Connected to: CPUCore.bus-s
 *
 * MasterPort "mem-m":
 *   - Sends requests to Memory
 *   - No internal queue
 *   - Connected to: Memory.bus-s
 *
 * SlavePort "mem-s":
 *   - Receives responses from Memory
 *   - Queue size: 1
 *   - Arbiter: Round-robin (default)
 *   - Connected to: Memory.bus-m
 * ```
 *
 * **Request Path Flow (CPU → Memory):**
 * ```
 * Step 1: Accept from CPU
 *   acceptPacketFromCPU()
 *     └─► if (s_port_from_cpu->isPopValid())
 *          └─► accept(tick, packet)  // Calls handler(BaseReqPacket*)
 *
 * Step 2: Process Request Handler
 *   handler(BaseReqPacket* packet)
 *     ├─1─► Check req_queue.isPushReady()
 *     ├─2─► Pop from s_port_from_cpu
 *     ├─3─► pushToReqQueue(packet, tick + kInteralLatency)
 *     └─4─► If queue was empty → tryIssueRequest()
 *
 * Step 3: Schedule Request Forward
 *   tryIssueRequest()
 *     └─► If current_tick >= req_queue.getMinTick()
 *          └─► issueRequest() immediately
 *          Else
 *          └─► Schedule issueRequest() at min_tick
 *
 * Step 4: Forward to Memory
 *   issueRequest()
 *     ├─1─► Pop from req_queue
 *     ├─2─► Push to m_port_to_mem (MasterPort)
 *     └─3─► If queue not empty → tryIssueRequest() at tick+1
 * ```
 *
 * **Response Path Flow (Memory → CPU):**
 * ```
 * Step 1: Accept from Memory
 *   acceptPacketFromMEM()
 *     └─► if (s_port_from_mem->isPopValid())
 *          └─► accept(tick, packet)  // Calls handler(BaseRspPacket*)
 *
 * Step 2: Process Response Handler
 *   handler(BaseRspPacket* packet)
 *     ├─1─► Check rsp_queue.isPushReady()
 *     ├─2─► Pop from s_port_from_mem
 *     ├─3─► pushToRspQueue(packet, tick + kInteralLatency)
 *     └─4─► If queue was empty → tryIssueResponse()
 *
 * Step 3: Schedule Response Forward
 *   tryIssueResponse()
 *     └─► If current_tick >= rsp_queue.getMinTick()
 *          └─► issueResponse() immediately
 *          Else
 *          └─► Schedule issueResponse() at min_tick
 *
 * Step 4: Forward to CPU
 *   issueResponse()
 *     ├─1─► Pop from rsp_queue
 *     ├─2─► Push to m_port_to_cpu (MasterPort)
 *     └─3─► If queue not empty → tryIssueResponse() at tick+1
 * ```
 *
 * **Bus Latency Modeling:**
 * ```
 * Each packet incurs bus traversal delay:
 *
 * Request Path:
 *   CPU sends at tick T
 *     └─► Bus receives at tick T (Phase 2 port sync)
 *          └─► Bus adds kInteralLatency (bus_latency)
 *               └─► Memory receives at tick T + kInteralLatency
 *
 * Response Path:
 *   Memory sends at tick T
 *     └─► Bus receives at tick T
 *          └─► Bus adds kInteralLatency (bus_latency)
 *               └─► CPU receives at tick T + kInteralLatency
 *
 * Total Bus Overhead per Request:
 *   Request path:  +bus_latency
 *   Response path: +bus_latency
 *   Total:         2 × bus_latency
 * ```
 *
 * **Multi-Master Arbitration:**
 * ```
 * CrossBar has 2 SlavePort endpoints:
 *
 * 1. cpu-s (receives from CPU)
 *    - Single master (CPUCore)
 *    - No contention
 *    - Round-robin arbiter (unused, only 1 master)
 *
 * 2. mem-s (receives from Memory)
 *    - Single master (Memory)
 *    - No contention
 *    - Round-robin arbiter (unused, only 1 master)
 *
 * For Multi-CPU Extension:
 *   If multiple CPUs connect to cpu-s:
 *     ├─► SlavePort queue = 1 (only 1 packet per cycle)
 *     ├─► Round-robin arbiter grants access fairly
 *     └─► Other CPUs must retry via masterPortRetry()
 * ```
 *
 * **Backpressure Handling:**
 * ```
 * Scenario 1: Memory Port Full (mem-m backpressure)
 *   issueRequest() {
 *       if (!m_port_to_mem->push(packet)) {
 *           // Push failed, packet stays in req_queue
 *           // Retry triggered via masterPortRetry(m_port_to_mem)
 *       }
 *   }
 *
 *   masterPortRetry(MasterPort* port) {
 *       if (port == m_port_to_mem) {
 *           tryIssueRequest();  // Retry forwarding to memory
 *       }
 *   }
 *
 * Scenario 2: CPU Port Full (cpu-m backpressure)
 *   issueResponse() {
 *       if (!m_port_to_cpu->push(packet)) {
 *           // Push failed, packet stays in rsp_queue
 *           // Retry triggered via masterPortRetry(m_port_to_cpu)
 *       }
 *   }
 *
 *   masterPortRetry(MasterPort* port) {
 *       if (port == m_port_to_cpu) {
 *           tryIssueResponse();  // Retry forwarding to CPU
 *       }
 *   }
 * ```
 *
 * **Internal Queue Management:**
 * ```
 * req_queue (Request Queue):
 *   - Type: LimitedObjectContainer<BaseReqPacket*>
 *   - Capacity: queue_size (from constructor)
 *   - Purpose: Buffer requests during bus latency delay
 *   - Time-ordered: Earliest packet processed first
 *
 * rsp_queue (Response Queue):
 *   - Type: LimitedObjectContainer<BaseRspPacket*>
 *   - Capacity: queue_size (from constructor)
 *   - Purpose: Buffer responses during bus latency delay
 *   - Time-ordered: Earliest packet processed first
 * ```
 *
 * **Packet Visitor Pattern:**
 * ```
 * CrossBar uses overloaded handler() for packet routing:
 *
 * handler(BaseReqPacket* packet):
 *   - Called via accept() from cpu-s SlavePort
 *   - Packet came from CPU
 *   - Forward to Memory via mem-m MasterPort
 *
 * handler(BaseRspPacket* packet):
 *   - Called via accept() from mem-s SlavePort
 *   - Packet came from Memory
 *   - Forward to CPU via cpu-m MasterPort
 * ```
 *
 * **Performance Characteristics:**
 *
 * | Configuration     | Effect                                          | Trade-off                    |
 * |-------------------|-------------------------------------------------|------------------------------|
 * | High bus_latency  | More realistic interconnect delay               | Higher end-to-end latency    |
 * | Low bus_latency   | Faster packet transfer, stress-test logic       | May be unrealistic           |
 * | Large queue_size  | Absorbs traffic bursts, reduces backpressure    | More memory usage            |
 * | Small queue_size  | Forces flow control, tests backpressure logic   | Frequent stalls              |
 *
 * **Usage Example:**
 * ```cpp
 * // In TestSimPortTop::registerSimulators()
 * auto bus = new CrossBar(
 *     "bus",          // Simulator name
 *     2,              // internal_latency (bus traversal delay)
 *     2               // queue_size (internal buffer capacity)
 * );
 * this->addSimulator(bus);
 *
 * // Connect CPU → Bus → Memory (request path)
 * SimPortManager::ConnectPort(cpu, bus, "bus-m", "cpu-s");
 * SimPortManager::ConnectPort(bus, mem, "mem-m", "bus-s");
 *
 * // Connect Memory → Bus → CPU (response path)
 * SimPortManager::ConnectPort(mem, bus, "bus-m", "mem-s");
 * SimPortManager::ConnectPort(bus, cpu, "cpu-m", "bus-s");
 * ```
 *
 * **Extending to Multi-Master:**
 * ```cpp
 * // Multiple CPUs sharing the bus
 * for (int i = 0; i < num_cpus; i++) {
 *     auto cpu = new CPUCore("cpu" + std::to_string(i), ...);
 *     SimPortManager::ConnectPort(cpu, bus, "bus-m", "cpu-s");
 *     SimPortManager::ConnectPort(bus, cpu, "cpu-m" + std::to_string(i), "bus-s");
 * }
 *
 * // Bus arbitration:
 * // - cpu-s SlavePort arbitrates among multiple CPU masters
 * // - Round-robin grants one CPU access per cycle
 * // - Other CPUs retry via masterPortRetry() callback
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Packet Forwarder:**
 *    - Receives from one port, forwards to another
 *    - Adds latency in between
 *
 * 2. **Symmetric Request/Response Paths:**
 *    - Same logic for both directions
 *    - Separate queues and handlers
 *
 * 3. **Time-Delayed Forwarding:**
 *    - Packets scheduled for future delivery
 *    - Models physical bus propagation delay
 *
 * 4. **Overloaded Handlers:**
 *    - handler(BaseReqPacket*) for requests
 *    - handler(BaseRspPacket*) for responses
 *    - Visitor pattern dispatches to correct handler
 *
 * @see CPUCore For request generator that sends to bus
 * @see Memory For memory controller that receives from bus
 * @see BasePacket For packet types and visitor pattern
 * @see LimitedObjectContainer For time-ordered queue implementation
 */

#include "hw/CrossBar.hh"

#include "BasePacket.hh"
#include "CallBackEvent.hh"

namespace test_port {

CrossBar::CrossBar(const std::string& name, size_t internal_latency, size_t queue_size)
    : acalsim::CPPSimBase(name),
      kInteralLatency(internal_latency),
      req_queue(LimitedObjectContainer<BaseReqPacket*>(queue_size)),
      rsp_queue(LimitedObjectContainer<BaseRspPacket*>(queue_size)) {
	this->m_port_to_cpu   = this->addMasterPort("cpu-m");
	this->m_port_to_mem   = this->addMasterPort("mem-m");
	this->s_port_from_cpu = this->addSlavePort("cpu-s", 1);
	this->s_port_from_mem = this->addSlavePort("mem-s", 1);
}

void CrossBar::init() {}

void CrossBar::step() {
	this->acceptPacketFromMEM();
	this->acceptPacketFromCPU();

	// Flag Cleanup(make sure each cycle can only push 1 packet to queue)
	this->req_queue.step();
	this->rsp_queue.step();
}

void CrossBar::cleanup() {}

void CrossBar::masterPortRetry(acalsim::MasterPort* port) {
	if (this->m_port_to_cpu == port) { this->tryIssueResponse(); }
	if (this->m_port_to_mem == port) { this->tryIssueRequest(); }
}

/***
 * Request Path
 */
void CrossBar::acceptPacketFromCPU() {
	if (this->s_port_from_cpu->isPopValid()) {
		auto packet = this->s_port_from_cpu->front();
		this->accept(acalsim::top->getGlobalTick(), *packet);
	}
}

void CrossBar::handler(BaseReqPacket* packet) {  //  Packet from CPU
	const auto rc = acalsim::top->getRecycleContainer();

	// 1. Ensure the ReqQueue is ready to accept a new ReqPacket and pop out
	if (!this->req_queue.isPushReady()) { return; }
	LABELED_ASSERT_MSG(this->s_port_from_cpu->isPopValid(), name,
	                   "SlavePort is not ready to provide a valid RspPacket.");
	this->s_port_from_cpu->pop();

	// 1.1 Add Log & trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [3] Pop out the ReqPacket from CPU2BusPort and push to ReqOutQueue.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "CPU-BUS(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 2. Push the ReqPacket to the ReqQueue with a delay based on the internal latency.
	this->pushToReqQueue(packet, acalsim::top->getGlobalTick() + this->kInteralLatency);
}

void CrossBar::pushToReqQueue(BaseReqPacket* packet, acalsim::Tick when) {
	// Check if the ReqOutQueue was previously empty
	const bool was_empty = this->req_queue.empty();

	// Attempt to push the packet into the ReqOutQueue and ensure success
	LABELED_ASSERT_MSG(this->req_queue.push(packet, when), name, "Failed to push to the ReqQueue: the queue is full.");

	// Add Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "BUS(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// This ensures that requests will be when the queue transitions from empty to non-empty.
	if (was_empty) { this->tryIssueRequest(); }
}

void CrossBar::tryIssueRequest() {
	if (this->req_queue.empty()) { return; }

	const auto rc       = acalsim::top->getRecycleContainer();
	const auto min_tick = this->req_queue.getMinTick();
	auto       callback = [this]() { this->issueRequest(); };

	// Issue a request immediately if the earliest tick is less than or equal to CurrTick
	if (acalsim::top->getGlobalTick() >= min_tick) {
		callback();
	} else {
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, min_tick);
	}
}

void CrossBar::issueRequest() {
	// 1. Make sure that ReqQueue can pop the packet.
	if (!this->req_queue.isPopValid()) { return; }

	// 2. Ensure the output port is ready to accept a new packet.
	if (!this->m_port_to_mem->isPushReady()) { return; }

	// 3. Check if the earliest request in the queue is ready to be issued
	if (this->req_queue.getMinTick() > acalsim::top->getGlobalTick()) { return; }

	// 4. Process: Issue Outbound Request the master port.
	auto packet = this->popFromReqQueue();
	LABELED_ASSERT_MSG(packet, name, "Failed to pop the ReqPacket from the ReqQueue.");
	LABELED_ASSERT_MSG(this->m_port_to_mem->push(packet), name, "Failed to push the ReqPacket to the Bus2MemPort.");

	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [4] Pop out the ReqPacket from ReqOutQueue and push to Bus2MemPort.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "BUS(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "BUS-MEM(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));
}

BaseReqPacket* CrossBar::popFromReqQueue() {
	auto packet = this->req_queue.pop();
	if (!this->req_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryIssueRequest(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
	return packet;
}

/**
 * Response Path
 */
void CrossBar::acceptPacketFromMEM() {
	if (this->s_port_from_mem->isPopValid()) {
		auto packet = this->s_port_from_mem->front();
		this->accept(acalsim::top->getGlobalTick(), *packet);
	}
}

void CrossBar::handler(BaseRspPacket* packet) {  //  Packet from Memory
	const auto rc = acalsim::top->getRecycleContainer();

	// 1. Ensure the RspQueue is ready to accept a new RspPacket
	if (!this->rsp_queue.isPushReady()) { return; }
	LABELED_ASSERT_MSG(this->s_port_from_mem->isPopValid(), name,
	                   "SlavePort is not ready to provide a valid RspPacket.");
	this->s_port_from_mem->pop();

	// 1.1 Add Log & Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [7] Pop out the RspPacket from Mem2BusPort and push to RspQueue.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "MEM-BUS(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 2. Push the ReqPacket to the ReqInQueue with a delay based on the internal latency.
	this->pushToRspQueue(packet, acalsim::top->getGlobalTick() + this->kInteralLatency);
}

void CrossBar::pushToRspQueue(BaseRspPacket* packet, acalsim::Tick when) {
	// Check if the ReqOutQueue was previously empty
	const bool was_empty = this->rsp_queue.empty();

	// Attempt to push the packet into the ReqOutQueue and ensure success
	LABELED_ASSERT_MSG(this->rsp_queue.push(packet, when), name, "Failed to push to the RspQueue: the queue is full.");

	// Add Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "BUS(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// This ensures that requests will be when the queue transitions from empty to non-empty.
	if (was_empty) { this->tryIssueResponse(); }
}

void CrossBar::tryIssueResponse() {
	if (this->rsp_queue.empty()) { return; }

	const auto rc       = acalsim::top->getRecycleContainer();
	const auto min_tick = this->rsp_queue.getMinTick();
	auto       callback = [this]() { this->issueResponse(); };

	// Issue a request immediately if the earliest tick is less than or equal to CurrTick
	if (acalsim::top->getGlobalTick() >= min_tick) {
		callback();
	} else {
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, min_tick);
	}
}

void CrossBar::issueResponse() {
	// 1. Make sure that RspQueue can pop the packet.
	if (!this->rsp_queue.isPopValid()) { return; }

	// 2. Ensure the output port is ready to accept a new packet.
	if (!this->m_port_to_cpu->isPushReady()) { return; }

	// 3. Check if the earliest request in the queue is ready to be issued
	if (this->rsp_queue.getMinTick() > acalsim::top->getGlobalTick()) { return; }

	// 4-1. Process: Pop the RspPacket from RspQueue.
	auto packet = this->popFromRspQueue();
	LABELED_ASSERT_MSG(packet, name, "Failed to pop the RspPacket from the RspQueue.");
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [8] Pop out the RspPacket from RspQueue and push to Bus2CpuPort.";

	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "BUS(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 4-2. Process: Issue Outbound Request the master port.
	LABELED_ASSERT_MSG(this->m_port_to_cpu->push(packet), name, "Failed to push the RspPacket to the Bus2CpuPort.");
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "BUS-CPU(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));
}

BaseRspPacket* CrossBar::popFromRspQueue() {
	auto packet = this->rsp_queue.pop();
	if (!this->rsp_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryIssueResponse(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
	return packet;
}
}  // namespace test_port
