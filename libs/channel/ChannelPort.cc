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
 * @file ChannelPort.cc
 * @brief ChannelPort implementation - lock-free inter-simulator communication with dual-queue ping-pong buffers
 *
 * This file implements the ChannelPort family (ChannelPort, MasterChannelPort, SlaveChannelPort),
 * which provides lock-free peer-to-peer communication between ChannelPortManager instances using
 * dual-queue ping-pong buffers for asynchronous message passing in ACALSim simulations.
 *
 * **ChannelPort vs Port Architecture Comparison:**
 * ```
 * Port-based Communication (SimPacket*):
 *   - Synchronous, Phase 2 arbitration
 *   - Multi-master to single-slave
 *   - AXI-like protocol with backpressure/retry
 *   - Single-tick latency
 *   - Used for intra-simulator component communication
 *
 * Channel-based Communication (std::shared_ptr<SimRequest>):
 *   - Asynchronous, lock-free queues
 *   - Peer-to-peer (one-to-one)
 *   - No arbitration needed
 *   - Multi-tick latency (configurable delay)
 *   - Used for inter-simulator communication
 * ```
 *
 * **Dual-Queue Ping-Pong Buffer Architecture:**
 * ```
 * MasterChannelPort                        TSimChannel                      SlaveChannelPort
 *   (Simulator A)                     (shared_ptr ownership)                 (Simulator B)
 *   │                                                                              │
 *   │  push(request)                                                               │
 *   └────────────────────►  Write to Push Queue                                    │
 *                            [req0][req1][req2]                                    │
 *                                 │                                                │
 *                                 │  swap() at configured delay                   │
 *                                 │  (e.g., 10 ticks network latency)             │
 *                                 ▼                                                │
 *                            Pop Queue ◄───────────────────────────────────────────┘
 *                            [req0][req1][req2]                           pop()
 *
 * Lock-Free Mechanism:
 *   - MasterChannelPort: Always writes to push_queue
 *   - SlaveChannelPort: Always reads from pop_queue
 *   - TSimChannel::swap(): Atomic exchange at configured intervals
 *   - No mutex needed: Single reader, single writer
 * ```
 *
 * **Master-Slave Asymmetry:**
 * | Role                | Class                | Operations       | Notification              | Use Case |
 * |---------------------|----------------------|------------------|---------------------------|---------------------------|
 * | MasterChannelPort   | Initiator (sender)   | push(request)    | Notifies mate on push     | Accelerator → Host CPU
 * | | SlaveChannelPort    | Receiver (consumer)  | pop() / empty()  | Receives inbound flag     | Host CPU ←
 * Accelerator    |
 *
 * **ChannelPort Communication Flow:**
 * ```
 * Phase 1 (Simulator A - MasterChannelPort owner):
 *   │
 *   ├─ SimBase::step()
 *   │    └─ Generate request: auto req = std::make_shared<SimRequest>(...)
 *   │
 *   ├─ MasterChannelPort::push(req)  ◄── THIS FILE (lines 51-54)
 *   │    │
 *   │    ├─ channel->operator<<(req)
 *   │    │    └─ TSimChannel::push(req)
 *   │    │         └─ Add to push_queue (no lock needed)
 *   │    │
 *   │    └─ channel_mate->handleInboundNotification()
 *   │         └─ Set inbound_request_flag in Simulator B's ChannelPortManager
 *   │              └─ Triggers Simulator B to stay active
 *   │
 *   └─ Notification propagates to Simulator B
 *
 * Phase 2 (TSimChannel - managed by ChannelPortManager):
 *   │
 *   ├─ ChannelPortManager::syncChannelPort()
 *   │    └─ channel->swap()
 *   │         └─ Atomically exchange push_queue ↔ pop_queue
 *   │              └─ Requests now visible to SlaveChannelPort
 *   │
 *   └─ Delay applied: Requests arrive after N ticks (configurable latency)
 *
 * Phase 1 (Next iteration - Simulator B - SlaveChannelPort owner):
 *   │
 *   ├─ SimBase::processInboundChannelRequests()
 *   │    │
 *   │    ├─ Check: !slave_channel_port->empty()  ◄── THIS FILE (line 78)
 *   │    │    └─ channel->nonEmptyForPop()
 *   │    │         └─ Check if pop_queue has items
 *   │    │
 *   │    ├─ auto req = slave_channel_port->pop()  ◄── THIS FILE (lines 72-76)
 *   │    │    └─ channel->operator>>(req)
 *   │    │         └─ Dequeue from pop_queue (no lock needed)
 *   │    │
 *   │    └─ Process request: handleRequest(req)
 *   │
 *   └─ Request consumed by Simulator B
 * ```
 *
 * **Stream Operator Overloads:**
 * Provide intuitive syntax for channel communication:
 *
 * ```cpp
 * // Push (send) syntax - operator<< (lines 56-59)
 * master_channel_port << request;  // Equivalent to: master_channel_port.push(request)
 *
 * // Pop (receive) syntax - operator>> (lines 80-83)
 * slave_channel_port >> request;   // Equivalent to: request = slave_channel_port.pop()
 *
 * // Example usage in user code:
 * class Accelerator : public SimBase {
 *     MasterChannelPort* to_host_port;
 *
 *     void step() {
 *         if (has_result()) {
 *             auto result = std::make_shared<SimRequest>(...);
 *             to_host_port << result;  // Stream-like push
 *         }
 *     }
 * };
 *
 * class HostCPU : public SimBase {
 *     SlaveChannelPort* from_accelerator_port;
 *
 *     void processInboundChannelRequests() {
 *         while (!from_accelerator_port->empty()) {
 *             std::shared_ptr<SimRequest> result;
 *             from_accelerator_port >> result;  // Stream-like pop
 *             handleResult(result);
 *         }
 *     }
 * };
 * ```
 *
 * **Shared Pointer Ownership:**
 * ```
 * TSimChannel Memory Management:
 *   │
 *   ├─ Created by: ChannelPortManager::ConnectChannelPort()
 *   │    └─ auto channel = std::make_shared<TSimChannel>(delay);
 *   │
 *   ├─ Shared ownership:
 *   │    ├─ MasterChannelPort::channel (shared_ptr)
 *   │    └─ SlaveChannelPort::channel (shared_ptr)
 *   │
 *   └─ Lifetime:
 *        └─ Destroyed when both ports go out of scope
 *             └─ Automatic cleanup (RAII)
 *
 * Why shared_ptr?
 *   - Both MasterChannelPort and SlaveChannelPort access the same queue
 *   - No clear ownership hierarchy (peer-to-peer)
 *   - Automatic memory management (no manual delete)
 *   - Thread-safe reference counting
 * ```
 *
 * **ChannelPort Initialization Flow:**
 * ```
 * 1. User registers ChannelPortManagers:
 *    class Accelerator : public SimBase, public ChannelPortManager {
 *        MasterChannelPort* to_host;
 *    };
 *
 *    class HostCPU : public SimBase, public ChannelPortManager {
 *        SlaveChannelPort* from_accel;
 *    };
 *
 * 2. ChannelPortManager::ConnectChannelPort(accel, host, "to_host", "from_accel", delay):
 *    │
 *    ├─ Create TSimChannel:
 *    │    auto channel = std::make_shared<TSimChannel>(delay);
 *    │
 *    ├─ Create MasterChannelPort:
 *    │    accel->to_host = new MasterChannelPort(host, channel);  ◄── THIS FILE (lines 46-49)
 *    │
 *    ├─ Create SlaveChannelPort:
 *    │    host->from_accel = new SlaveChannelPort(accel, channel); ◄── THIS FILE (lines 67-70)
 *    │
 *    └─ Register with ChannelPortManagers
 *
 * 3. Simulation execution:
 *    - Accelerator: to_host->push(request)
 *    - HostCPU: from_accel->pop() after delay
 * ```
 *
 * **TSimChannel (DualQueue) Configuration:**
 * ```
 * TSimChannel Properties:
 *   - Type: DualQueue<std::shared_ptr<SimRequest>>
 *   - Delay: Configurable (e.g., 10 ticks for network latency)
 *   - Swap policy: Every N ticks (delay parameter)
 *   - Queue size: Unbounded (std::queue backend)
 *   - Thread safety: Lock-free (single reader, single writer)
 *
 * Delay Example:
 *   ConnectChannelPort(accel, host, "to_host", "from_accel", 10);
 *     │
 *     └─ Requests sent at tick T arrive at tick T+10
 *
 * Network Latency Modeling:
 *   - PCIe link: 5-20 ticks
 *   - Memory bus: 100-300 ticks
 *   - Inter-chip: 500-1000 ticks
 * ```
 *
 * **Notification Mechanism:**
 * ```
 * handleInboundNotification() Flow:
 *   │
 *   ├─ Called by: MasterChannelPort::push() (line 53)
 *   │
 *   ├─ Effect on ChannelPortManager (receiver):
 *   │    └─ ChannelPortManager::handleInboundNotification()
 *   │         └─ Set inbound_request_flag = true
 *   │
 *   └─ Effect on SimBase (receiver):
 *        └─ SimBase::hasPendingActivityInChannelPort()
 *             └─ Returns true if inbound_request_flag set
 *                  └─ Keeps SimBase active in next iteration
 *                       └─ ThreadManager schedules for execution
 *
 * Why needed?
 *   - Receiver may be idle with no pending events
 *   - Inbound requests require immediate processing
 *   - Activity flag prevents premature simulation termination
 * ```
 *
 * **Implementation Functions:**
 *
 * 1. **ChannelPort::ChannelPort() (lines 31-34):**
 *    - Store channel_mate (peer ChannelPortManager)
 *    - Store shared_ptr to TSimChannel
 *    - Base class constructor for Master/Slave specializations
 *
 * 2. **ChannelPort::getChannelMate() (line 36):**
 *    - Return pointer to peer ChannelPortManager
 *    - Used for notification propagation
 *
 * 3. **ChannelPort::getChannel() (line 38):**
 *    - Return shared_ptr to TSimChannel
 *    - Used by push/pop operations
 *
 * 4. **MasterChannelPort::MasterChannelPort() (lines 46-49):**
 *    - Delegate to base constructor
 *    - Initialize sender side of channel
 *
 * 5. **MasterChannelPort::push() (lines 51-54):**
 *    - Write request to channel push_queue
 *    - Notify mate's ChannelPortManager
 *    - Non-blocking operation (queue unbounded)
 *
 * 6. **operator<<(MasterChannelPort) (lines 56-59):**
 *    - Stream-like push syntax
 *    - Delegates to push()
 *    - Returns port reference for chaining
 *
 * 7. **SlaveChannelPort::SlaveChannelPort() (lines 67-70):**
 *    - Delegate to base constructor
 *    - Initialize receiver side of channel
 *
 * 8. **SlaveChannelPort::pop() (lines 72-76):**
 *    - Read request from channel pop_queue
 *    - Non-blocking (user checks empty() first)
 *    - Return shared_ptr to SimRequest
 *
 * 9. **SlaveChannelPort::empty() (line 78):**
 *    - Check if pop_queue has available requests
 *    - Used before pop() to avoid undefined behavior
 *
 * 10. **operator>>(SlaveChannelPort) (lines 80-83):**
 *     - Stream-like pop syntax
 *     - Delegates to pop()
 *     - Returns port reference for chaining
 *
 * **Memory Management Strategy:**
 * ```
 * SimRequest Lifecycle:
 *   1. Creation (Sender - MasterChannelPort owner):
 *      auto req = std::make_shared<SimRequest>(...);
 *        └─ Heap allocation, ref_count = 1
 *
 *   2. Push to Channel:
 *      master_port << req;
 *        └─ Stored in push_queue, ref_count = 2 (local + queue)
 *
 *   3. Local variable out of scope:
 *      ref_count = 1 (only queue holds reference)
 *
 *   4. Swap operation:
 *      push_queue → pop_queue (move semantics)
 *        └─ ref_count = 1 (moved to pop_queue)
 *
 *   5. Pop from Channel (Receiver - SlaveChannelPort owner):
 *      slave_port >> req;
 *        └─ ref_count = 1 (transferred to local variable)
 *
 *   6. Processing complete:
 *      req goes out of scope
 *        └─ ref_count = 0 → automatic delete
 *
 * No manual delete needed: shared_ptr RAII guarantees cleanup
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * // Define custom request type
 * struct AcceleratorRequest : public SimRequest {
 *     uint64_t task_id;
 *     std::vector<uint8_t> data;
 * };
 *
 * // Accelerator (sender)
 * class Accelerator : public SimBase, public ChannelPortManager {
 *     MasterChannelPort* to_host;
 *
 * public:
 *     void step() {
 *         if (computation_done()) {
 *             auto result = std::make_shared<AcceleratorRequest>();
 *             result->task_id = task_counter++;
 *             result->data = get_result_data();
 *
 *             to_host << result;  // Send to host CPU
 *         }
 *     }
 * };
 *
 * // Host CPU (receiver)
 * class HostCPU : public SimBase, public ChannelPortManager {
 *     SlaveChannelPort* from_accel;
 *
 * public:
 *     void processInboundChannelRequests() {
 *         while (!from_accel->empty()) {
 *             std::shared_ptr<SimRequest> req;
 *             from_accel >> req;
 *
 *             // Downcast to specific type
 *             auto accel_req = std::static_pointer_cast<AcceleratorRequest>(req);
 *             handleResult(accel_req->task_id, accel_req->data);
 *         }
 *     }
 * };
 *
 * // Binding (in SimTop)
 * ChannelPortManager::ConnectChannelPort(
 *     accelerator, host_cpu,
 *     "to_host", "from_accel",
 *     10  // 10-tick network latency
 * );
 * ```
 *
 * @see ChannelPort.hh For interface documentation
 * @see ChannelPortManager.cc For port binding and synchronization
 * @see DualQueue.hh For dual-queue ping-pong buffer implementation
 * @see SimBase.cc For processInboundChannelRequests() mechanism
 */

#include "channel/ChannelPort.hh"

#include <memory>

#include "channel/ChannelPortManager.hh"

namespace acalsim {

/**********************************
 *                                *
 *          ChannelPort           *
 *                                *
 **********************************/

ChannelPort::ChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel)
    : channel_mate(_channel_mate), channel(_channel) {
	;
}

ChannelPortManager* ChannelPort::getChannelMate() const { return this->channel_mate; }

std::shared_ptr<ChannelPort::TSimChannel> ChannelPort::getChannel() const { return this->channel; }

/**********************************
 *                                *
 *      MasterChannelPort       *
 *                                *
 **********************************/

MasterChannelPort::MasterChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel)
    : ChannelPort(_channel_mate, _channel) {
	;
}

void MasterChannelPort::push(const TPayload& _item) {
	*(this->getChannel()) << _item;
	this->getChannelMate()->handleInboundNotification();
}

MasterChannelPort& operator<<(MasterChannelPort& _port, const ChannelPort::TPayload& _item) {
	_port.push(std::move(_item));
	return _port;
}

/**********************************
 *                                *
 *       SlaveChannelPort       *
 *                                *
 **********************************/

SlaveChannelPort::SlaveChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel)
    : ChannelPort(_channel_mate, _channel) {
	;
}

ChannelPort::TPayload SlaveChannelPort::pop() {
	TPayload item;
	*(this->getChannel()) >> item;
	return item;
}

bool SlaveChannelPort::empty() const { return !this->getChannel()->nonEmptyForPop(); }

SlaveChannelPort& operator>>(SlaveChannelPort& _port, ChannelPort::TPayload& _item) {
	_item = _port.pop();
	return _port;
}

}  // namespace acalsim
