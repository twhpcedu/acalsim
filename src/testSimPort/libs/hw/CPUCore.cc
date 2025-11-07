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
 * @file CPUCore.cc
 * @brief Request generator with flow control and outstanding request tracking
 *
 * This file implements the CPUCore simulator, which demonstrates **advanced request generation**
 * with **flow control mechanisms**, **backpressure handling**, and **outstanding request tracking**.
 * It serves as a canonical example of a traffic generator in discrete-event simulation.
 *
 * **Component Overview:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                            CPUCore                                       │
 * │                                                                          │
 * │  ┌────────────────┐       ┌──────────────────┐      ┌────────────────┐ │
 * │  │   Request      │       │  Request Output  │      │  Outstanding   │ │
 * │  │  Generator     │──────►│     Queue        │─────►│  Request       │ │
 * │  │                │       │  (req_out_queue) │      │  Tracker       │ │
 * │  │  - Unique IDs  │       │                  │      │  (req_out_     │ │
 * │  │  - Flow ctrl   │       │  LimitedObject   │      │   queue)       │ │
 * │  │  - Back2back   │       │  Container       │      │                │ │
 * │  └────────────────┘       └──────────────────┘      └────────────────┘ │
 * │         ▲                          │                        ▲           │
 * │         │                          │                        │           │
 * │         │                          ▼                        │           │
 * │         │                    bus-m (MasterPort)             │           │
 * │         │                          │                        │           │
 * │         │                          │ Requests               │           │
 * │         │                          └───────────────────────►│           │
 * │         │                                                   │           │
 * │         │                    bus-s (SlavePort)              │           │
 * │         │                          │                        │           │
 * │         │                          │ Responses              │           │
 * │         │                          ▼                        │           │
 * │         │                  ┌──────────────────┐            │           │
 * │         │                  │  Response Input  │            │           │
 * │         └──────────────────│     Queue        │────────────┘           │
 * │           (Next request)   │  (rsp_in_queue)  │   (Match & release)    │
 * │                            │                  │                        │
 * │                            │  LimitedObject   │                        │
 * │                            │  Container       │                        │
 * │                            └──────────────────┘                        │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Key Data Structures:**
 *
 * 1. **req_out_queue (Request Output Queue):**
 *    ```cpp
 *    LimitedObjectContainer<BaseReqPacket*> req_out_queue
 *    - Type: Time-ordered queue with capacity limit
 *    - Purpose: Buffer generated requests before sending to bus
 *    - Capacity: Set by constructor parameter (default: cpu_outstanding_requests)
 *    - Scheduling: Each request has an associated "ready tick"
 *    ```
 *
 * 2. **rsp_in_queue (Response Input Queue):**
 *    ```cpp
 *    LimitedObjectContainer<BaseRspPacket*> rsp_in_queue
 *    - Type: Time-ordered queue with capacity limit
 *    - Purpose: Buffer responses received from bus before processing
 *    - Capacity: Set by constructor parameter (default: resp_queue_size)
 *    - Scheduling: Each response processed after internal latency
 *    ```
 *
 * 3. **outstanding_req_queue (Outstanding Request Tracker):**
 *    ```cpp
 *    OutStandingReqQueue outstanding_req_queue
 *    - Type: Map-based tracker (ReqID → bool)
 *    - Purpose: Track in-flight requests awaiting responses
 *    - Capacity: kMaxOutstandingRequests
 *    - Flow Control: Pauses generation when full
 *    ```
 *
 * **Request Generation Flow:**
 * ```
 * init()
 *   └─► generateRequest()  // Initial request at tick 0
 *        │
 *        ├─1─► Create unique BaseReqPacket (ReqID++)
 *        │
 *        ├─2─► pushToReqQueue(packet, tick + kInteralLatency)
 *        │      └─► req_out_queue.push(packet, when)
 *        │           └─► If queue was empty → tryIssueRequest()
 *        │
 *        └─3─► If conditions allow:
 *               - issued_requests < kTotalRequests
 *               - outstanding_req_queue.isPushReady()
 *               - req_out_queue not empty
 *               └─► Schedule back-to-back generateRequest() at tick+1
 *
 * tryIssueRequest()
 *   └─► If current_tick >= req_out_queue.getMinTick()
 *        └─► issueRequest() immediately
 *        Else
 *        └─► Schedule issueRequest() at min_tick
 *
 * issueRequest()
 *   ├─1─► Pop packet from req_out_queue
 *   ├─2─► Track in outstanding_req_queue
 *   ├─3─► Push to MasterPort "bus-m"
 *   └─4─► If queue not empty → tryIssueRequest() at tick+1
 * ```
 *
 * **Response Processing Flow:**
 * ```
 * step()
 *   └─► acceptPacketFromBus()
 *        └─► If SlavePort "bus-s" has packet
 *             └─► accept(tick, packet)  // Calls handler()
 *
 * handler(BaseRspPacket* packet)
 *   ├─1─► Pop from SlavePort
 *   ├─2─► pushToRspQueue(packet, tick + kInteralLatency)
 *   │      └─► rsp_in_queue.push(packet, when)
 *   │           └─► If queue was empty → tryProcessResponse()
 *   │
 *   └─3─► tryProcessResponse()
 *          └─► If current_tick >= rsp_in_queue.getMinTick()
 *               └─► processResponse() immediately
 *               Else
 *               └─► Schedule processResponse() at min_tick
 *
 * processResponse()
 *   ├─1─► Pop packet from rsp_in_queue
 *   ├─2─► Match ReqID in outstanding_req_queue
 *   ├─3─► outstanding_req_queue.remove(ReqID)
 *   ├─4─► Recycle packet to RecycleContainer
 *   ├─5─► Schedule generateRequest() at tick+1 (generate next)
 *   └─6─► If req_out_queue not empty → tryIssueRequest() at tick+1
 * ```
 *
 * **Outstanding Request Tracking:**
 * ```
 * Flow Control Mechanism:
 *
 *   generateRequest() checks:
 *     if (outstanding_req_queue.isPushReady()) {
 *         // OK to generate new request
 *     } else {
 *         // Max outstanding reached, pause generation
 *         return;
 *     }
 *
 *   issueRequest() adds to tracker:
 *     outstanding_req_queue.add(req_id, true);
 *     // Now tracked as in-flight
 *
 *   processResponse() releases slot:
 *     outstanding_req_queue.remove(req_id);
 *     // Slot freed, can generate next request
 * ```
 *
 * **Backpressure Handling:**
 * ```
 * Scenario 1: MasterPort Full (Bus queue saturated)
 *   issueRequest() {
 *       if (!m_port->push(packet)) {
 *           // Push failed, packet stays in req_out_queue
 *           // Retry triggered via masterPortRetry() callback
 *       }
 *   }
 *
 *   masterPortRetry(MasterPort* port) {
 *       // Called by framework when port has space
 *       if (port == m_port) {
 *           tryIssueRequest();  // Retry sending
 *       }
 *   }
 *
 * Scenario 2: Outstanding Limit Reached
 *   generateRequest() {
 *       if (!outstanding_req_queue.isPushReady()) {
 *           // Max in-flight requests reached
 *           // Pause generation until response arrives
 *           return;
 *       }
 *   }
 *
 *   processResponse() {
 *       outstanding_req_queue.remove(req_id);
 *       // Space freed, trigger generateRequest()
 *       scheduleEvent(..., [this]() { this->generateRequest(); });
 *   }
 * ```
 *
 * **Back-to-Back Request Generation:**
 * ```
 * When to Generate Multiple Requests:
 *
 *   generateRequest() {
 *       // Generate one request
 *       pushToReqQueue(packet, ...);
 *
 *       // Check if we can generate another immediately
 *       if (issued_requests < kTotalRequests &&
 *           outstanding_req_queue.isPushReady() &&
 *           !req_out_queue.empty()) {
 *
 *           // Schedule next generation at tick+1
 *           auto callback = [this]() { this->generateRequest(); };
 *           scheduleEvent(callback_event, tick + 1);
 *       }
 *   }
 *
 * Effect:
 *   - Maximizes request injection rate
 *   - Fills pipeline quickly
 *   - Saturates bus bandwidth (up to outstanding limit)
 * ```
 *
 * **Event-Driven Scheduling:**
 * ```
 * CPUCore uses CallBackEvent for all timed actions:
 *
 * 1. Request Output Delay:
 *    pushToReqQueue(packet, current_tick + kInteralLatency)
 *      └─► Request available after internal latency
 *
 * 2. Request Retry:
 *    tryIssueRequest() schedules issueRequest() at min_tick
 *      └─► Retry sending when earliest request is ready
 *
 * 3. Response Processing Delay:
 *    pushToRspQueue(packet, current_tick + kInteralLatency)
 *      └─► Response processed after internal latency
 *
 * 4. Next Request Generation:
 *    processResponse() schedules generateRequest() at tick+1
 *      └─► Trigger next request after handling current response
 * ```
 *
 * **Unique Request ID Generation:**
 * ```cpp
 * static std::atomic<uint32_t> uniqueReqID = 0;
 *
 * generateRequest() {
 *     auto packet = rc->acquire<BaseReqPacket>(
 *         &BaseReqPacket::renew,
 *         this->uniqueReqID  // Pass current ID
 *     );
 *     this->uniqueReqID++;   // Increment for next request
 * }
 *
 * Why atomic:
 *   - Thread-safe ID generation (if multiple CPUs in future)
 *   - Guarantees unique IDs across all CPUCore instances
 * ```
 *
 * **Port Configuration:**
 * ```
 * MasterPort "bus-m":
 *   - Created in constructor
 *   - Used for sending requests to bus
 *   - No queue size (master ports don't have internal queues)
 *
 * SlavePort "bus-s":
 *   - Created in constructor with queue_size=1
 *   - Used for receiving responses from bus
 *   - Single entry queue (responses processed immediately)
 * ```
 *
 * **LimitedObjectContainer Usage:**
 * ```cpp
 * Key Methods:
 *
 * 1. isPushReady() - Check if queue has space
 *    if (req_out_queue.isPushReady()) {
 *        // OK to push
 *    }
 *
 * 2. push(object, when) - Add to queue with timestamp
 *    req_out_queue.push(packet, tick + latency);
 *
 * 3. isPopValid() - Check if queue can pop now
 *    if (req_out_queue.isPopValid()) {
 *        auto packet = req_out_queue.pop();
 *    }
 *
 * 4. getMinTick() - Get earliest ready time
 *    auto when = req_out_queue.getMinTick();
 *    if (current_tick >= when) {
 *        // Earliest element is ready
 *    }
 *
 * 5. step() - Cleanup flags (call every iteration)
 *    req_out_queue.step();  // Reset per-cycle push flag
 * ```
 *
 * **Performance Considerations:**
 *
 * | Configuration           | Effect                                      | Trade-off                     |
 * |-------------------------|---------------------------------------------|-------------------------------|
 * | High outstanding_req    | More throughput, pipeline stays full        | More memory usage             |
 * | Low outstanding_req     | Less memory, simpler tracking               | Pipeline bubbles, low throughput|
 * | High kInteralLatency    | More realistic CPU delay                    | Lower request rate            |
 * | Low kInteralLatency     | Faster generation, stress-tests backpressure| May not be realistic          |
 * | Large queue_size        | Absorbs bursts, reduces backpressure        | More memory per queue         |
 * | Small queue_size        | Forces flow control, tests retry logic      | Frequent backpressure stalls  |
 *
 * **Usage Example:**
 * ```cpp
 * // In TestSimPortTop::registerSimulators()
 * auto cpu = new CPUCore(
 *     "cpu",              // Simulator name
 *     5,                  // max_outstanding_requests (in-flight limit)
 *     100,                // total_requests (stop after 100)
 *     1,                  // internal_resp_latency (CPU processing delay)
 *     2                   // resp_queue_size (response buffer size)
 * );
 * this->addSimulator(cpu);
 *
 * // Connect ports
 * SimPortManager::ConnectPort(cpu, bus, "bus-m", "cpu-s");
 * SimPortManager::ConnectPort(bus, cpu, "cpu-m", "bus-s");
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Producer-Consumer:**
 *    - CPUCore produces requests → Bus consumes
 *    - Memory produces responses → CPUCore consumes
 *
 * 2. **Callback-Based Retry:**
 *    - masterPortRetry() called on backpressure release
 *    - Avoids polling, efficient event-driven retry
 *
 * 3. **Time-Ordered Queue:**
 *    - LimitedObjectContainer maintains min-heap by timestamp
 *    - Efficient O(log N) insert, O(1) getMin
 *
 * 4. **Outstanding Tracker:**
 *    - Map-based tracking for fast lookup by ReqID
 *    - Bounded capacity for flow control
 *
 * @see CrossBar For bus arbiter that receives CPU requests
 * @see Memory For memory that processes requests and generates responses
 * @see BasePacket For packet structure and visitor pattern
 * @see LimitedObjectContainer For time-ordered queue implementation
 * @see OutStandingReqQueue For request tracking data structure
 */

#include "hw/CPUCore.hh"

#include "BasePacket.hh"
#include "CallBackEvent.hh"

namespace test_port {

// Static atomic counter for generating unique request IDs across all CPUCore instances
std::atomic<uint32_t> CPUCore::uniqueReqID = 0;

/**
 * @brief Constructor for CPUCore simulator.
 *
 * Initializes the CPU request generator with configurable parameters for flow control,
 * latency modeling, and queue sizing.
 *
 * @param name Unique simulator name (e.g., "cpu", "cpu0")
 * @param max_outstanding_requests Maximum in-flight requests (flow control limit)
 * @param total_requests Total number of requests to generate before stopping
 * @param internal_resp_latency CPU internal processing delay in ticks
 * @param resp_queue_size Size of response input queue
 *
 * **Initialization Steps:**
 * 1. Initialize CPPSimBase with simulator name
 * 2. Store configuration constants (kMaxOutstandingRequests, kTotalRequests, etc.)
 * 3. Initialize req_out_queue with capacity=max_outstanding_requests
 * 4. Initialize rsp_in_queue with capacity=resp_queue_size
 * 5. Initialize outstanding_req_queue with capacity=max_outstanding_requests
 * 6. Create and register MasterPort "bus-m"
 * 7. Create and register SlavePort "bus-s" with queue_size=1
 */
CPUCore::CPUCore(const std::string& name, size_t max_outstanding_requests, size_t total_requests,
                 size_t internal_resp_latency, size_t resp_queue_size)
    : acalsim::CPPSimBase(name),
      kMaxOutstandingRequests(max_outstanding_requests),
      kTotalRequests(total_requests),
      kInteralLatency(internal_resp_latency),
      req_out_queue(LimitedObjectContainer<BaseReqPacket*>(max_outstanding_requests)),
      rsp_in_queue(LimitedObjectContainer<BaseRspPacket*>(resp_queue_size)),
      outstanding_req_queue(OutStandingReqQueue(max_outstanding_requests)) {
	this->m_port = this->addMasterPort("bus-m");
	this->s_port = this->addSlavePort("bus-s", 1);
}

/**
 * @brief Initialization called before simulation loop starts.
 *
 * Triggers the initial request generation to begin the request-response cycle.
 *
 * **Actions:**
 * - Log initialization message
 * - Call generateRequest() to create first request
 * - First request will be scheduled for tick 0 + kInteralLatency
 */
void CPUCore::init() {
	LABELED_INFO(name) << "[0] Initialization";
	this->generateRequest();
}

/**
 * @brief Core execution method called every simulation tick (if active).
 *
 * Performs two main functions:
 * 1. Accept incoming responses from bus
 * 2. Cleanup queue flags for next iteration
 *
 * **Execution Order:**
 * - First: acceptPacketFromBus() checks for new responses
 * - Second: req_out_queue.step() resets per-cycle push flag
 * - Third: rsp_in_queue.step() resets per-cycle push flag
 *
 * **Why step() Cleanup:**
 * LimitedObjectContainer enforces "one push per cycle" limit.
 * Calling step() at end of each iteration resets this flag.
 */
void CPUCore::step() {
	this->acceptPacketFromBus();

	// Flag Cleanup(make sure each cycle can only push 1 packet to queue)
	this->req_out_queue.step();
	this->rsp_in_queue.step();
}

/**
 * @brief Cleanup called after simulation loop ends.
 *
 * Validates that all generated requests received responses and logs final status.
 *
 * **Validation:**
 * - finished_requests == kTotalRequests → Success
 * - finished_requests < kTotalRequests → Warning (some requests didn't complete)
 */
void CPUCore::cleanup() {
	if (finished_requests == kTotalRequests) {
		LABELED_INFO(name) << "[9] All responses have been handled by the CPU!";
	} else {
		LABELED_WARNING(name) << "There are unhandled responses!";
	}
}

/**
 * @brief Backpressure retry callback invoked when MasterPort has space available.
 *
 * Called by framework when:
 * - issueRequest() previously failed due to full MasterPort
 * - MasterPort now has space (slave consumed packets)
 * - Framework automatically triggers retry via this callback
 *
 * @param port The MasterPort that now has space
 *
 * **Retry Logic:**
 * - Check if it's our "bus-m" MasterPort
 * - Call tryIssueRequest() to retry sending pending requests
 *
 * **Why This Works:**
 * - Avoids polling (checking port availability every cycle)
 * - Event-driven notification when backpressure releases
 * - Efficient: Only retries when actually needed
 */
void CPUCore::masterPortRetry(acalsim::MasterPort* port) {
	if (this->m_port == port) { this->tryIssueRequest(); }
}

/***
 * Request Path
 */
void CPUCore::generateRequest() {
	const auto rc = acalsim::top->getRecycleContainer();

	// 1. Check if all requests have already been issued.
	if (this->issued_requests == this->kTotalRequests) {
		LABELED_INFO(this->name) << "All Requests have been issued to Memory!";
		return;
	}

	// 2. Check if the ReqOutQueue is ready to accept a new request.
	if (this->req_out_queue.isPushReady()) {
		auto packet = rc->acquire<BaseReqPacket>(&BaseReqPacket::renew, this->uniqueReqID);
		// 2.1 Add Log
		LABELED_INFO(this->name) << "ReqId-" << packet->getReqId()
		                         << ": [1] Create a Request Packet and push to ReqOutQueue.";
		this->uniqueReqID++;
		this->issued_requests++;

		// 2.2 Push the ReqPacket to the ReqQueue with a delay based on the internal latency.
		this->pushToReqQueue(packet, acalsim::top->getGlobalTick() + this->kInteralLatency);

		// 2.3 Issue back-to-back requests if conditions allow.
		if (this->issued_requests == this->kTotalRequests) { return; }
		if (!this->outstanding_req_queue.isPushReady()) { return; }
		if (this->req_out_queue.empty()) { return; }

		// Schedule a back2back request after a delay of 1 tick.
		auto callback = [this]() {
			LABELED_WARNING(this->getName()) << "[Req trigger] Generate Next Request callback ";
			this->generateRequest();
		};
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
}

void CPUCore::pushToReqQueue(BaseReqPacket* packet, acalsim::Tick when) {
	// Check if the ReqOutQueue was previously empty
	const bool was_empty = this->req_out_queue.empty();

	// Attempt to push the packet into the ReqOutQueue and ensure success
	LABELED_ASSERT_MSG(this->req_out_queue.push(packet, when), name,
	                   "Failed to push to the ReqOutQueue: the queue is full.");

	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "CPU(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// This ensures that requests will be when the queue transitions from empty to non-empty.
	if (was_empty) { this->tryIssueRequest(); }
}

void CPUCore::tryIssueRequest() {
	if (this->req_out_queue.empty()) { return; }

	const auto rc       = acalsim::top->getRecycleContainer();
	const auto min_tick = this->req_out_queue.getMinTick();
	auto       callback = [this]() { this->issueRequest(); };

	// Issue a request immediately if the earliest tick is less than or equal to CurrTick
	if (acalsim::top->getGlobalTick() >= min_tick) {
		callback();
	} else {
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, min_tick);
	}
}

void CPUCore::issueRequest() {
	// 1. Make sure that ReqOutQueue can pop the packet.
	if (!this->req_out_queue.isPopValid()) { return; }

	// 2. Verify that the number of outstanding requests does not exceed the hardware constraints.
	if (!this->outstanding_req_queue.isPushReady()) { return; }

	// 3. Ensure the output port is ready to accept a new packet.
	if (!this->m_port->isPushReady()) { return; }

	// 4. Check if the earliest request in the queue is ready to be issued
	if (this->req_out_queue.getMinTick() > acalsim::top->getGlobalTick()) { return; }

	// 5-1. Process: Pop the RspPacket from ReqOutQueue and update OutstandingReqQueue
	auto packet = this->popFromReqQueue();
	LABELED_ASSERT_MSG(packet, name, "Failed to pop the ReqPacket from the ReqOutQueue.");
	LABELED_ASSERT_MSG(this->outstanding_req_queue.add(packet->getReqId(), true), name,
	                   "Failed to push the ReqPacket to the OutstandingReqQueue.");
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [2] Pop out the ReqPacket from ReqOutQueue and push to OutPort.";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "CPU(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 5-2. Process: Issue Outbound Request the master port.
	LABELED_ASSERT_MSG(this->m_port->push(packet), name, "Failed to push the ReqPacket to the OutPort.");
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "CPU-BUS(ReqPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));
}

BaseReqPacket* CPUCore::popFromReqQueue() {
	auto packet = this->req_out_queue.pop();
	if (!this->req_out_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryIssueRequest(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
	return packet;
}

/**
 * Response Path
 */
void CPUCore::acceptPacketFromBus() {
	if (this->s_port->isPopValid() && this->rsp_in_queue.isPushReady()) {
		auto packet = this->s_port->front();
		this->accept(acalsim::top->getGlobalTick(), *packet);
	}
}

void CPUCore::handler(BaseRspPacket* packet) {
	const auto rc = acalsim::top->getRecycleContainer();

	// 1. Ensure the RspInQueue is ready to accept a new RspPacket
	if (!this->rsp_in_queue.isPushReady()) { return; }
	LABELED_ASSERT_MSG(this->s_port->isPopValid(), name, "SlavePort is not ready to provide a valid RspPacket.");
	this->s_port->pop();

	// 1.1 Add Log & Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	LABELED_INFO(this->name) << req_name << ": [9] Pop out the RspPacket from InPort and push to RspInQueue.";

	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "BUS-CPU(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// 2. Push the RspPacket to the RspInQueue with a delay based on the internal latency.
	this->pushToRspQueue(packet, acalsim::top->getGlobalTick() + this->kInteralLatency);
}

void CPUCore::pushToRspQueue(BaseRspPacket* packet, acalsim::Tick when) {
	// Check if the RspInQueue was previously empty
	const bool was_empty = this->rsp_in_queue.empty();

	// Attempt to push the packet into the RspInQueue and ensure success
	LABELED_ASSERT_MSG(this->rsp_in_queue.push(packet, when), name,
	                   "Failed to push to the RspInQueue: the queue is full.");

	// Add Trace
	std::string req_name = "ReqId-" + std::to_string(packet->getReqId());
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "TestSimPort", "CPU(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// This ensures that response will be processed when the queue transitions from empty to non-empty.
	if (was_empty) { this->tryProcessResponse(); }
}

void CPUCore::tryProcessResponse() {
	if (this->rsp_in_queue.empty()) { return; }
	const auto rc       = acalsim::top->getRecycleContainer();
	const auto min_tick = this->rsp_in_queue.getMinTick();
	auto       callback = [this]() { this->processResponse(); };

	// Issue a request immediately if the earliest tick is less than or equal to CurrTick
	if (acalsim::top->getGlobalTick() >= min_tick) {
		callback();
	} else {
		auto event = rc->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
		this->scheduleEvent(event, min_tick);
	}
}

void CPUCore::processResponse() {
	// 1. Ensure the RspInQueue has a valid packet to pop.
	if (!this->rsp_in_queue.isPopValid()) { return; }

	// 2. Retrieve the packet from the front of the RspInQueue without removing it.
	auto packet = this->popFromRspQueue();
	auto req_id = packet->getReqId();

	// 3. Assert that the request ID exists in the Outstanding ReqOutQueue
	// This ensures that the response corresponds to a valid and expected Outstanding request.
	LABELED_ASSERT_MSG(this->outstanding_req_queue.contains(req_id), name,
	                   "Unexpected response: Request ID not found in the outstanding request queue.");
	this->outstanding_req_queue.remove(req_id);
	acalsim::top->getRecycleContainer()->recycle(packet);

	// Add Log
	std::string req_name = "ReqId-" + std::to_string(req_id);
	LABELED_INFO(this->name) << req_name
	                         << ": [10] Pop out the RspPacket from RspInQueue and release Outstanding ReqQueue";
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "TestSimPort", "CPU(RespPath) " + req_name, acalsim::top->getGlobalTick(), "", req_name));

	// Schedule a callback to generate the next request after a delay of 1 tick.
	auto callback = [this]() {
		LABELED_WARNING(this->getName()) << "[Rsp trigger] Generate Next Request callback";
		this->generateRequest();
	};
	auto event =
	    acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(&CallBackEvent<void()>::renew, callback);
	this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);

	if (!this->req_out_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryIssueRequest(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
}

BaseRspPacket* CPUCore::popFromRspQueue() {
	auto packet = this->rsp_in_queue.pop();
	if (!this->rsp_in_queue.empty()) {
		auto event = acalsim::top->getRecycleContainer()->acquire<CallBackEvent<void()>>(
		    &CallBackEvent<void()>::renew, [this]() { this->tryProcessResponse(); });
		this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
	}
	return packet;
}

}  // namespace test_port
