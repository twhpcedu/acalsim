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
 * @file TBSim.cc
 * @brief Test bench simulator implementations for CrossBar validation
 *
 * @details
 * This file implements the test bench (TB) simulators used to validate CrossBar
 * functionality. It contains three key classes:
 * 1. **TBPacket**: Test packet for request/response transactions
 * 2. **MasterTBSim**: Master device simulator that generates requests
 * 3. **SlaveTBSim**: Slave device simulator that responds to requests
 *
 * ## Purpose
 *
 * These test bench components provide:
 * - **Traffic Generation**: Masters create configurable request streams
 * - **Response Handling**: Slaves echo requests back with transaction IDs
 * - **Validation Logic**: Verifies transaction ID integrity and completion
 * - **Backpressure Testing**: Exercises retry mechanisms under congestion
 * - **Performance Tracing**: Generates Chrome trace events for analysis
 *
 * ## Test Bench Architecture
 *
 * @code
 *   MasterTBSim                                        SlaveTBSim
 *   ┌─────────────────┐                          ┌─────────────────┐
 *   │ issueRequest()  │──[TBPacket]──────────►   │ handle()        │
 *   │  - create pkt   │                          │  - recv request │
 *   │  - push to bus  │      [CrossBar]          │  - issue resp   │
 *   │  - track tid    │                          │                 │
 *   │                 │   ◄──[TBPacket]──────────│ issueResponse() │
 *   │ handle()        │                          │  - create pkt   │
 *   │  - recv resp    │                          │  - push to bus  │
 *   │  - validate tid │                          └─────────────────┘
 *   │  - update count │
 *   └─────────────────┘
 * @endcode
 *
 * ## Transaction Flow
 *
 * 1. **Request Generation** (MasterTBSim):
 *    - Master creates TBPacket with destination slave ID
 *    - Assigns unique transaction ID from atomic counter
 *    - Pushes packet to CrossBar via PipeRegister
 *    - Stores transaction ID in local map for validation
 *    - Generates Chrome trace event marking request start
 *
 * 2. **Request Processing** (SlaveTBSim):
 *    - Slave receives packet via SlavePort pop
 *    - TBPacket::visit() dispatches to SlaveTBSim::handle()
 *    - Slave immediately generates response packet
 *    - Response packet carries same transaction ID
 *    - Pushes response to CrossBar response channel
 *
 * 3. **Response Validation** (MasterTBSim):
 *    - Master receives response via SlavePort pop
 *    - Validates transaction ID exists in pending map
 *    - Removes transaction ID from map (one-time use)
 *    - Increments finished_requests counter
 *    - Generates Chrome trace event marking completion
 *
 * ## Backpressure Handling
 *
 * When PipeRegister::push() returns false:
 * - **Master**: Retries in next step() cycle or on masterPortRetry() callback
 * - **Slave**: Retries in tryAcceptResponse() when backpressure releases
 *
 * The retry mechanism ensures lossless communication:
 * @code{.cpp}
 * if (!m_reg->push(packet)) {
 *     rc->recycle(packet);  // Don't leak memory
 *     // Will retry on next step() or masterPortRetry()
 * }
 * @endcode
 *
 * ## Transaction ID Management
 *
 * Transaction IDs ensure request-response pairing:
 * - **Allocation**: Atomic counter prevents collisions across masters
 * - **Storage**: std::map tracks outstanding requests per master
 * - **Validation**: Response handler checks ID exists before accepting
 * - **Cleanup**: IDs removed after successful response reception
 *
 * Example validation:
 * @code{.cpp}
 * auto iter = this->transaction_id_map.find(tid);
 * if (iter == this->transaction_id_map.end()) {
 *     LABELED_ERROR(getName()) << "Transaction ID not found!";
 * }
 * @endcode
 *
 * ## Chrome Tracing Integration
 *
 * Each transaction generates paired trace events:
 * - **FullPath**: Complete request → response latency
 * - **ReqPath**: Request issuance → slave reception
 * - **RespPath**: Response issuance → master reception
 *
 * Events use "B" (begin) and "E" (end) markers for duration visualization.
 *
 * ## Traffic Pattern
 *
 * Current implementation uses round-robin slave targeting:
 * @code{.cpp}
 * dst_idx = (src_idx + issued_requests) % num_slaves
 * @endcode
 *
 * This distributes load evenly and tests all master-slave pairs.
 *
 * ## Test Success Criteria
 *
 * **Master Success**:
 * - finished_requests == num_requests (all responses received)
 * - transaction_id_map is empty (all IDs validated)
 *
 * **Slave Success**:
 * - All received requests successfully echoed as responses
 * - No packets dropped or corrupted
 *
 * ## Code Example: Master Request Issuance
 *
 * @code{.cpp}
 * // Master creates and sends request
 * auto packet = rc->acquire<TBPacket>(&TBPacket::renew, my_id, target_slave);
 * packet->setTransactionId(transaction_id++);
 * transaction_id_map[tid] = true;
 *
 * if (m_reg->push(packet)) {
 *     issued_requests++;
 *     // Success: packet in flight to CrossBar
 * } else {
 *     rc->recycle(packet);
 *     // Backpressure: retry later
 * }
 * @endcode
 *
 * ## Best Practices for Test Bench Design
 *
 * 1. **Memory Management**: Always recycle packets on failure paths
 * 2. **Transaction Tracking**: Use unique IDs for request-response pairing
 * 3. **Backpressure**: Implement robust retry mechanisms
 * 4. **Validation**: Check transaction ID integrity on every response
 * 5. **Tracing**: Generate events for performance analysis
 * 6. **Cleanup**: Verify completion counters in cleanup() method
 *
 * ## Performance Considerations
 *
 * - **Atomic Operations**: transaction_id uses std::atomic for thread safety
 * - **Object Pooling**: RecycleContainer reuses packet objects
 * - **Event Scheduling**: forceStepInNextIteration() minimizes latency
 * - **Lazy Evaluation**: isPopValid() checks before pop() to avoid stalls
 *
 * @see testcrossbar::TBPacket Test packet implementation
 * @see testcrossbar::MasterTBSim Master simulator
 * @see testcrossbar::SlaveTBSim Slave simulator
 * @see acalsim::crossbar::CrossBar CrossBar under test
 * @see acalsim::CPPSimBase Simulation base class
 * @see acalsim::RecycleContainer Object pooling mechanism
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "TBSim.hh"

#include <random>

namespace testcrossbar {

/**
 * @brief Visitor pattern implementation for SimModule (unused)
 *
 * @details
 * This overload handles TBPacket visits to SimModule objects. Since test bench
 * packets only interact with SimBase-derived simulators (masters and slaves),
 * this function is intentionally empty.
 *
 * @param[in] when Simulation tick when packet is being processed
 * @param[in] module Target SimModule (unused in test bench)
 *
 * @see TBPacket::visit(acalsim::Tick, acalsim::SimBase&)
 */
void TBPacket::visit(acalsim::Tick when, acalsim::SimModule& module) {}

/**
 * @brief Visitor pattern implementation for packet dispatching
 *
 * @details
 * Implements the visitor pattern to dispatch packets to appropriate handler
 * methods based on the simulator type. This enables type-safe packet handling
 * without requiring explicit type casting at call sites.
 *
 * **Dispatch Logic**:
 * - If target is MasterTBSim → calls MasterTBSim::handle() (response processing)
 * - If target is SlaveTBSim → calls SlaveTBSim::handle() (request processing)
 * - Otherwise → logs error (invalid target type)
 *
 * This pattern is commonly used in hardware simulation frameworks to model
 * packet routing through heterogeneous components.
 *
 * @param[in] when Simulation tick when packet arrives at target
 * @param[in] simulator Target simulator receiving the packet
 *
 * @throws LABELED_ERROR if simulator is neither master nor slave
 *
 * @see MasterTBSim::handle()
 * @see SlaveTBSim::handle()
 */
void TBPacket::visit(acalsim::Tick when, acalsim::SimBase& simulator) {
	if (auto master = dynamic_cast<MasterTBSim*>(&simulator)) {
		master->handle(this);
	} else if (auto slave = dynamic_cast<SlaveTBSim*>(&simulator)) {
		slave->handle(this);
	} else {
		LABELED_ERROR("TBPacket") << "Invalid simulator: " << simulator.getName();
	}
}

/**
 * @brief Global transaction ID counter shared across all masters
 *
 * @details
 * Static atomic counter ensures unique transaction IDs across all MasterTBSim
 * instances. Atomic operations guarantee thread-safe ID allocation without locks.
 *
 * **Uniqueness Guarantee**: Each master increments this counter when creating
 * a new request, ensuring no two requests share the same transaction ID.
 */
std::atomic<size_t> MasterTBSim::transaction_id = 0;

/**
 * @brief Initializes master test bench simulator
 *
 * @details
 * Performs master-specific initialization including:
 * 1. Calls base class TBSim::init() to setup ports and pipe registers
 * 2. Retrieves test configuration parameters (n_requests, n_slave)
 * 3. Schedules initial request issuance event at tick+1
 *
 * The lambda event pattern enables deferred execution without explicit
 * event handler classes:
 *
 * @code{.cpp}
 * auto event = rc->acquire<LambdaEvent>(&LambdaEvent::renew,
 *     [this]() { this->issueRequest(); });
 * scheduleEvent(event, tick+1);
 * @endcode
 *
 * @see TBSim::init()
 * @see issueRequest()
 * @see acalsim::LambdaEvent
 */
void MasterTBSim::init() {
	this->TBSim::init();

	using LambdaEvent = acalsim::LambdaEvent<void()>;
	const auto& rc    = acalsim::top->getRecycleContainer();
	auto        event = rc->acquire<LambdaEvent>(&LambdaEvent::renew, [this]() { this->issueRequest(); });

	this->num_requests      = acalsim::top->getParameter<int>("crossbar_test", "n_requests");
	this->num_slave_device_ = acalsim::top->getParameter<int>("crossbar_test", "n_slave");
	this->scheduleEvent(event, acalsim::top->getGlobalTick() + 1);
}

/**
 * @brief Issues a single request packet to the CrossBar
 *
 * @details
 * Attempts to send one request packet to a target slave device. This function
 * implements the core request generation logic with proper backpressure handling.
 *
 * ## Operation Flow
 *
 * 1. **Completion Check**: Returns immediately if all requests already issued
 * 2. **Target Selection**: Computes destination slave using round-robin pattern
 * 3. **Packet Creation**: Acquires TBPacket from object pool
 * 4. **Transaction ID**: Assigns unique ID and tracks in local map
 * 5. **Transmission**: Attempts push to CrossBar pipe register
 * 6. **Tracing**: Generates Chrome trace events on success
 * 7. **Retry**: Recycles packet and waits for retry on backpressure
 *
 * ## Round-Robin Targeting
 *
 * @code{.cpp}
 * dst_idx = (src_idx + issued_requests) % num_slaves
 * @endcode
 *
 * Ensures uniform distribution across slaves and exercises all communication paths.
 *
 * ## Backpressure Handling
 *
 * If isStalled() is true or push() returns false:
 * - Packet is recycled to prevent memory leak
 * - issued_requests counter NOT incremented
 * - Will retry on next step() or masterPortRetry() callback
 *
 * ## Chrome Tracing
 *
 * Two trace events generated on successful send:
 * - **FullPath**: Tracks complete request→response round-trip
 * - **ReqPath**: Tracks request issuance→slave reception
 *
 * @note forceStepInNextIteration() schedules immediate re-execution to maximize throughput
 *
 * @see step()
 * @see masterPortRetry()
 * @see handle()
 */
void MasterTBSim::issueRequest() {
	if (this->issued_requests == this->num_requests) { return; }

	const auto& rc      = acalsim::top->getRecycleContainer();
	auto        src_idx = this->device_idx_;
	auto        dst_idx = (this->device_idx_ + this->issued_requests) % num_slave_device_;

	auto packet = rc->acquire<TBPacket>(&TBPacket::renew, src_idx, dst_idx);
	auto tid    = this->transaction_id++;
	packet->setTransactionId(tid);
	this->transaction_id_map[tid] = true;

	bool is_stalled = this->m_reg->isStalled();

	if (!is_stalled && this->m_reg->push(packet)) {
		this->issued_requests++;
		acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
		    "B", this->getName(), "SDevice:" + std::to_string(packet->getDstIdx()) + "-FullPath",
		    acalsim::top->getGlobalTick(), "", "packet:" + std::to_string(packet->getTransactionId())));

		acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
		    "B", this->getName(), "SDevice:" + std::to_string(packet->getDstIdx()) + "-ReqPath",
		    acalsim::top->getGlobalTick(), "", "packet:" + std::to_string(packet->getTransactionId())));

		LABELED_INFO(this->getName()) << " issue request at Master:" << this->device_idx_
		                              << " to Slave:" << packet->getDstIdx() << " tid=" << packet->getTransactionId()
		                              << " issued_request=" << this->issued_requests;

		this->forceStepInNextIteration();
	} else {
		rc->recycle(packet);
		LABELED_INFO(this->getName()) << this->getName() << " Cannot push the packet to Bus";
	}
}

/**
 * @brief Simulation step function for master device
 *
 * @details
 * Executed on each simulation cycle to process incoming responses and generate
 * new requests. This two-phase operation ensures progress on both channels.
 *
 * **Phase 1**: Response Processing
 * - Checks if response packet available (isPopValid)
 * - Pops and accepts response via accept() → calls handle()
 *
 * **Phase 2**: Request Generation
 * - Attempts to issue next request via issueRequest()
 * - Subject to backpressure and request quota
 *
 * The accept() method triggers packet visiting, which dispatches to handle().
 *
 * @see issueRequest()
 * @see handle()
 * @see acalsim::CPPSimBase::step()
 */
void MasterTBSim::step() {
	if (s_port->isPopValid()) { this->accept(acalsim::top->getGlobalTick(), *s_port->pop()); }
	issueRequest();
}

/**
 * @brief Cleanup and validation at simulation end
 *
 * @details
 * Validates test completion by checking if all issued requests received responses.
 * This is a critical validation step for GoogleTest success/failure determination.
 *
 * **Success Condition**:
 * @code{.cpp}
 * finished_requests == num_requests
 * @endcode
 *
 * **Failure Indicators**:
 * - finished_requests < issued_requests: Lost responses
 * - issued_requests < num_requests: Deadlock prevented issuance
 * - Remaining packets in SlavePort: Protocol violation
 *
 * Logs detailed diagnostic information on failure to aid debugging.
 *
 * @note LABELED_ERROR causes GoogleTest assertion failure
 *
 * @see init()
 * @see acalsim::CPPSimBase::cleanup()
 */
void MasterTBSim::cleanup() {
	if (this->s_port->isPopValid()) LABELED_INFO(getName()) << "cleanup : " << this->s_port->front()->getName();

	if (this->finished_requests == this->num_requests) {
		LABELED_INFO(this->getName()) << "Success";
	} else {
		LABELED_WARNING(this->getName()) << "Failed - finished_requests: " << finished_requests
		                                 << " issued_requests: " << issued_requests
		                                 << " num_requests: " << num_requests;
		LABELED_ERROR(this->getName()) << "Failed";
	}
}

/**
 * @brief Callback invoked when backpressure is released
 *
 * @details
 * The CrossBar (via PipeRegister) calls this method when a previously stalled
 * master port becomes available. This enables immediate retry without waiting
 * for the next scheduled step() cycle.
 *
 * **Backpressure Lifecycle**:
 * 1. Master attempts push() → returns false (stalled)
 * 2. Master waits (packet recycled)
 * 3. CrossBar frees buffer space
 * 4. CrossBar invokes masterPortRetry() → immediate retry
 *
 * This callback-based approach minimizes latency during congestion recovery.
 *
 * @param[in] port Master port that was released from backpressure
 *
 * @see issueRequest()
 * @see acalsim::MasterPort::retry()
 */
void MasterTBSim::masterPortRetry(acalsim::MasterPort* port) {
	LABELED_INFO(this->getName()) << " ----- " << this->getName() << " port " << port->getName()
	                              << " is released from pressure!";
	this->issueRequest();
}

/**
 * @brief Handles incoming response packets from slaves
 *
 * @details
 * Processes response packets by validating transaction IDs and updating completion
 * counters. This is the critical validation point for request-response integrity.
 *
 * ## Validation Sequence
 *
 * 1. **Counter Update**: Increment finished_requests
 * 2. **ID Lookup**: Search transaction_id_map for response's transaction ID
 * 3. **Validation**: Error if ID not found (protocol violation)
 * 4. **Cleanup**: Remove ID from map (single-use validation)
 * 5. **Tracing**: Generate Chrome trace end events
 * 6. **Recycling**: Return packet to object pool
 *
 * ## Transaction ID Validation
 *
 * @code{.cpp}
 * auto iter = transaction_id_map.find(tid);
 * if (iter == transaction_id_map.end()) {
 *     // ERROR: Response for unknown/duplicate transaction
 * } else {
 *     transaction_id_map.erase(iter);  // One-time use
 * }
 * @endcode
 *
 * ## Chrome Trace Events
 *
 * Two "E" (end) events generated:
 * - **FullPath**: Completes request→response duration
 * - **RespPath**: Completes response issuance→reception duration
 *
 * @param[in] packet Response packet from slave device
 *
 * @note Called via visitor pattern from TBPacket::visit()
 *
 * @see issueRequest()
 * @see TBPacket::visit()
 */
void MasterTBSim::handle(TBPacket* packet) {
	this->finished_requests++;

	LABELED_INFO(this->getName()) << " Master:" << this->device_idx_
	                              << " receive a response from Slave:" << packet->getSrcIdx()
	                              << " tid=" << packet->getTransactionId()
	                              << " finished_requests=" << this->finished_requests;

	auto tid = packet->getTransactionId();

	auto iter = this->transaction_id_map.find(tid);

	if (iter == this->transaction_id_map.end()) {
		LABELED_ERROR(this->getName()) << "MasterPort got wrong TransactionID";
	} else {
		this->transaction_id_map.erase(iter);
	}

	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", this->getName(), "SDevice:" + std::to_string(packet->getSrcIdx()) + "-FullPath",
	    acalsim::top->getGlobalTick(), "", "packet:" + std::to_string(packet->getTransactionId())));

	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", this->getName(), "SDevice:" + std::to_string(packet->getSrcIdx()) + "-RespPath",
	    acalsim::top->getGlobalTick(), "", "packet:" + std::to_string(packet->getTransactionId())));

	acalsim::top->getRecycleContainer()->recycle(packet);
}

/**
 * @brief Initializes slave test bench simulator
 *
 * @details
 * Calls base class TBSim::init() to setup ports and pipe registers. Unlike masters,
 * slaves don't need to schedule initial events since they operate reactively.
 *
 * Slaves respond to incoming requests on-demand without generating autonomous traffic.
 *
 * @see TBSim::init()
 */
void SlaveTBSim::init() { this->TBSim::init(); }

/**
 * @brief Simulation step function for slave device
 *
 * @details
 * Executed on each simulation cycle to process incoming requests. Slaves operate
 * reactively, only processing requests when they arrive via the SlavePort.
 *
 * Delegates to tryAcceptResponse() which checks for available requests and
 * processes them if the response channel is not stalled.
 *
 * @see tryAcceptResponse()
 * @see acalsim::CPPSimBase::step()
 */
void SlaveTBSim::step() { this->tryAcceptResponse(); }

/**
 * @brief Cleanup verification at simulation end
 *
 * @details
 * Logs any remaining packets in the SlavePort as diagnostic information. Unlike
 * masters, slaves don't track request counts since they respond passively.
 *
 * Success is implicit: if masters complete successfully, slaves operated correctly.
 *
 * @see acalsim::CPPSimBase::cleanup()
 */
void SlaveTBSim::cleanup() {
	if (this->s_port->isPopValid()) LABELED_INFO(getName()) << "cleanup : " << this->s_port->front()->getName();
}

/**
 * @brief Callback invoked when response channel backpressure is released
 *
 * @details
 * When a slave's response transmission stalls (due to CrossBar congestion),
 * this callback is invoked when buffer space becomes available. The slave
 * immediately retries processing pending requests.
 *
 * **Use Case**: Slave received request but couldn't send response due to
 * backpressure. When CrossBar frees space, this enables immediate retry.
 *
 * @param[in] port Master port that was released from backpressure
 *
 * @see tryAcceptResponse()
 * @see acalsim::MasterPort::retry()
 */
void SlaveTBSim::masterPortRetry(acalsim::MasterPort* port) { this->tryAcceptResponse(); }

/**
 * @brief Attempts to accept and process incoming request packets
 *
 * @details
 * Conditionally processes requests based on both request availability and
 * response channel readiness. This double-gating prevents deadlock scenarios.
 *
 * **Preconditions for Processing**:
 * 1. Response channel NOT stalled (!m_reg->isStalled())
 * 2. Request available in SlavePort (s_port->isPopValid())
 *
 * If both conditions met:
 * - Pops request packet from SlavePort
 * - Accepts packet (triggers visitor pattern)
 * - Visitor dispatches to handle() for processing
 *
 * ## Why Check Stall State?
 *
 * Prevents accepting requests when unable to send responses:
 * @code{.cpp}
 * if (!m_reg->isStalled() && s_port->isPopValid()) {
 *     // Safe to process: can both receive AND respond
 *     accept(...);
 * }
 * @endcode
 *
 * This maintains request-response atomicity and prevents buffer exhaustion.
 *
 * @see handle()
 * @see step()
 * @see masterPortRetry()
 */
void SlaveTBSim::tryAcceptResponse() {
	if (!this->m_reg->isStalled() && this->s_port->isPopValid()) {
		auto packet = dynamic_cast<TBPacket*>(this->s_port->front());
		this->accept(acalsim::top->getGlobalTick(), *this->s_port->pop());
	}
}

/**
 * @brief Handles incoming request packets from masters
 *
 * @details
 * Processes request packets by immediately generating response packets with
 * matching transaction IDs. This implements the echo/loopback pattern for testing.
 *
 * ## Processing Sequence
 *
 * 1. **Trace Event**: Generate "E" (end) event for request path
 * 2. **Logging**: Log request reception with master ID and transaction ID
 * 3. **Response**: Call issueResponse() to send response back to master
 * 4. **Cleanup**: Recycle request packet to object pool
 *
 * ## Echo Pattern
 *
 * Slaves don't model actual processing; they immediately echo requests as responses:
 * @code{.cpp}
 * handle(request) {
 *     // Immediately respond without modeling service time
 *     issueResponse(request.src, request.tid);
 * }
 * @endcode
 *
 * For more realistic testing, could add latency:
 * @code{.cpp}
 * scheduleEvent([this, src, tid]() {
 *     issueResponse(src, tid);
 * }, current_tick + service_latency);
 * @endcode
 *
 * @param[in] packet Request packet from master device
 *
 * @note Called via visitor pattern from TBPacket::visit()
 *
 * @see issueResponse()
 * @see TBPacket::visit()
 */
void SlaveTBSim::handle(TBPacket* packet) {
	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "E", "MDevice:" + std::to_string(packet->getSrcIdx()),
	    "SDevice:" + std::to_string(this->device_idx_) + "-ReqPath", acalsim::top->getGlobalTick(), "",
	    "packet:" + std::to_string(packet->getTransactionId())));

	LABELED_INFO(this->getName()) << " Slave:" << this->device_idx_
	                              << " handle a request from Master:" << packet->getSrcIdx()
	                              << " tid=" << packet->getTransactionId();

	this->issueResponse(packet->getSrcIdx(), packet->getTransactionId());
	acalsim::top->getRecycleContainer()->recycle(packet);
}

/**
 * @brief Issues response packet back to requesting master
 *
 * @details
 * Creates and transmits a response packet with the same transaction ID as the
 * original request. This enables the master to correlate responses with requests.
 *
 * ## Operation Flow
 *
 * 1. **Packet Creation**: Acquires TBPacket from object pool
 * 2. **ID Preservation**: Copies transaction ID from original request
 * 3. **Source/Dest Swap**: Sets src=slave_id, dst=master_id
 * 4. **Tracing**: Generates "B" (begin) event for response path
 * 5. **Transmission**: Pushes to CrossBar response channel
 * 6. **Error Handling**: Recycles packet and logs error on failure
 *
 * ## Transaction ID Preservation
 *
 * Critical for correctness:
 * @code{.cpp}
 * // Request:  src=master_id, dst=slave_id,  tid=X
 * // Response: src=slave_id,  dst=master_id, tid=X (same!)
 * @endcode
 *
 * Master uses tid to match response to original request.
 *
 * ## Backpressure Handling
 *
 * If push() fails (response channel stalled):
 * - Packet recycled to prevent leak
 * - ERROR logged (should not happen due to tryAcceptResponse() gating)
 * - No retry mechanism (tryAcceptResponse() prevents this scenario)
 *
 * @param[in] master_device_id ID of master that sent the request
 * @param[in] _transaction_id Transaction ID from original request
 *
 * @note ERROR condition indicates logic bug in tryAcceptResponse() gating
 *
 * @see handle()
 * @see tryAcceptResponse()
 */
void SlaveTBSim::issueResponse(size_t master_device_id, size_t _transaction_id) {
	const auto& rc = acalsim::top->getRecycleContainer();

	auto packet = rc->acquire<TBPacket>(&TBPacket::renew, this->device_idx_, master_device_id);
	packet->setTransactionId(_transaction_id);

	acalsim::top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createDurationEvent(
	    "B", "MDevice:" + std::to_string(master_device_id),
	    "SDevice:" + std::to_string(this->device_idx_) + "-RespPath", acalsim::top->getGlobalTick(), "",
	    "packet:" + std::to_string(packet->getTransactionId())));

	if (!this->m_reg->push(packet)) {
		rc->recycle(packet);
		LABELED_ERROR(this->getName()) << "Cannot push the packet to Bus";
	} else {
		LABELED_INFO(this->getName()) << " Slave :" << this->device_idx_
		                              << " issue response to src_idx:" << packet->getSrcIdx()
		                              << " for tid=" << packet->getTransactionId();
	}
}

}  // namespace testcrossbar
