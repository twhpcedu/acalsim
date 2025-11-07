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
 * @file SlavePort.cc
 * @brief SlavePort implementation - receiver port with FIFO queue and multi-master arbitration
 *
 * This file implements SlavePort, the receiver side of ACALSim's port-based communication
 * system. SlavePort provides a FIFO queue buffer with configurable depth and supports
 * multiple MasterPorts connecting to a single SlavePort with arbiter-based selection.
 *
 * **Multi-Master to Single-Slave Architecture:**
 * ```
 * MasterPort_0 ────┐
 *    entry_        │
 *                  │
 * MasterPort_1 ────┤
 *    entry_        │         ┌──────────────────────────────────┐
 *                  ├────────►│        SlavePort                 │
 * MasterPort_2 ────┤         │                                  │
 *    entry_        │         │  Arbiter (Round-Robin)           │
 *                  │         │    └─ selects from m_ports_[]    │
 * MasterPort_N ────┘         │                                  │
 *    entry_                  │  req_queue_: FifoQueue           │
 *                            │    [pkt0][pkt1][pkt2]...[pktN]   │
 *                            │    size: configurable (e.g., 8)  │
 *                            │                                  │
 *                            │  owner_: SimBase*                │
 *                            │    (receives packets via pop())  │
 *                            └──────────────────────────────────┘
 * ```
 *
 * **Arbitration and Synchronization Flow (Phase 2):**
 * ```
 * SlavePort::sync() - Called by SimPortManager in Phase 2
 *   │
 *   ├─ 1. Check if req_queue_ has space
 *   │     └─ isPushReady() → true if not full
 *   │
 *   ├─ 2. Arbitrate among MasterPorts
 *   │     │
 *   │     └─ arbitrate()
 *   │          │
 *   │          ├─ Start from last_idx (round-robin fairness)
 *   │          │
 *   │          ├─ For each MasterPort in m_ports_[]:
 *   │          │   │
 *   │          │   ├─ next_idx = arbiter_->select()
 *   │          │   │     └─ Round-robin, priority, or custom policy
 *   │          │   │
 *   │          │   ├─ Check: port->isPopValid()
 *   │          │   │     └─ Does MasterPort have packet? (entry_ != nullptr)
 *   │          │   │
 *   │          │   ├─ If YES: return port->pop()
 *   │          │   │     └─ Pop packet from MasterPort, clear entry_
 *   │          │   │
 *   │          │   └─ If NO: try next MasterPort
 *   │          │
 *   │          └─ Return: SimPacket* or nullptr (if all empty)
 *   │
 *   ├─ 3. Push arbitrated packet to queue
 *   │     │
 *   │     └─ push(packet)
 *   │          ├─ req_queue_->push(packet)
 *   │          │     └─ FifoQueue adds packet to tail
 *   │          │
 *   │          ├─ If success: owner_->setPendingActivityFlag()
 *   │          │     └─ Notify owner SimBase has data to process
 *   │          │
 *   │          └─ return success status
 *   │
 *   └─ Return (packet transferred from MasterPort → SlavePort queue)
 * ```
 *
 * **Example Multi-Master Scenario:**
 * ```
 * Memory Controller (SlavePort: "mem_slave", queue_size=4)
 *   │
 *   ├─ Connected MasterPorts:
 *   │    ├─ CPU_0::req_port (has packet A)
 *   │    ├─ CPU_1::req_port (empty)
 *   │    ├─ CPU_2::req_port (has packet B)
 *   │    └─ DMA::req_port (empty)
 *   │
 *   └─ Arbitration (Round-Robin, last winner: CPU_1):
 *        │
 *        ├─ Start: next_idx = 2 (after CPU_1)
 *        ├─ Check CPU_2: has packet B → Winner!
 *        ├─ Pop packet B from CPU_2::req_port
 *        ├─ Push packet B to mem_slave.req_queue_
 *        └─ Next arbitration starts from CPU_3 (DMA)
 * ```
 *
 * **FIFO Queue Buffer Management:**
 * ```
 * SlavePort Request Queue:
 *   Head                                         Tail
 *    │                                            │
 *    ▼                                            ▼
 *   [pkt0] [pkt1] [pkt2] [pkt3] [    ] [    ] [    ] [    ]
 *    oldest                newest   empty slots
 *
 * Operations:
 *   - push(pkt): Add to tail if not full
 *   - pop():     Remove from head (oldest first)
 *   - isPushReady(): Check if space available
 *   - empty():   Check if no packets
 *
 * Full Queue Behavior:
 *   - isPushReady() returns false
 *   - sync() skips arbitration
 *   - MasterPorts keep packets in entry_
 *   - Retry mechanism triggered on MasterPort side
 * ```
 *
 * **Port Binding and Initialization:**
 * ```
 * 1. Construction:
 *    SlavePort* mem_slave = new SlavePort("mem_slave", queue_size=8, arbiter);
 *
 * 2. Binding (SimPortManager::bind):
 *    mem_slave->addMasterPort(cpu0->req_port);
 *    mem_slave->addMasterPort(cpu1->req_port);
 *    mem_slave->addMasterPort(dma->req_port);
 *      └─ Builds m_ports_[] vector and port_map_ for fast lookup
 *
 * 3. Initialization (before simulation):
 *    mem_slave->init();
 *      └─ arbiter_->setComponentsNum(m_ports_.size())
 *           └─ Configure arbiter for 3 masters
 * ```
 *
 * **Arbiter Algorithms Supported:**
 * | Arbiter Type        | Selection Policy                         | Use Case                    |
 * |---------------------|------------------------------------------|-----------------------------|
 * | RoundRobinArbiter   | Fair rotation, no starvation             | General-purpose (default)   |
 * | PriorityArbiter     | Fixed priority order                     | QoS differentiation         |
 * | WeightedArbiter     | Weighted fair queuing                    | Bandwidth allocation        |
 * | Custom Arbiter      | User-defined policy                      | Domain-specific needs       |
 *
 * **Memory Management:**
 * ```
 * Destructor (~SlavePort):
 *   1. Drain all packets from req_queue_
 *      while (!empty()) {
 *          packet = pop();
 *          top->getRecycleContainer()->recycle(packet);
 *      }
 *      └─ Prevent memory leak from unprocessed packets
 *
 *   2. Delete req_queue_ (FifoQueue object)
 *   3. Delete arbiter_ (Arbiter object)
 *
 * Why needed:
 *   - Simulation may terminate with packets still in queue
 *   - RecycleContainer requires explicit recycling
 *   - Clean up dynamically allocated objects
 * ```
 *
 * **Owner Activity Notification:**
 * When packet is pushed to queue:
 * ```
 * push(packet)
 *   │
 *   ├─ req_queue_->push(packet) → true (success)
 *   │
 *   └─ owner_->setPendingActivityFlag()
 *        └─ Notify owner SimBase:
 *             "You have data to process!"
 *
 * Effect:
 *   - Owner's hasPendingActivityInSimPort() returns true
 *   - SimBase stays active in next iteration
 *   - ThreadManager schedules owner for Phase 1 execution
 * ```
 *
 * **Implementation Functions:**
 *
 * 1. **Constructor (lines 27-28):**
 *    - Create FifoQueue with specified size
 *    - Store arbiter reference
 *    - Initialize empty port connection list
 *
 * 2. **Destructor (lines 30-38):**
 *    - Drain and recycle all queued packets
 *    - Delete FifoQueue object
 *    - Delete Arbiter object
 *
 * 3. **addMasterPort() (lines 40-46):**
 *    - Add MasterPort to m_ports_ vector
 *    - Build port_map_ for name-based lookup
 *    - Prevent duplicate port names (ASSERT)
 *
 * 4. **arbitrate() (lines 48-58):**
 *    - Round-robin select from m_ports_[]
 *    - Check each port for valid packet
 *    - Pop from first non-empty port
 *    - Return nullptr if all ports empty
 *
 * 5. **init() (lines 60-63):**
 *    - Validate: At least one MasterPort connected
 *    - Configure arbiter with port count
 *    - Called before simulation starts
 *
 * 6. **sync() (lines 65-72):**
 *    - Check queue has space (isPushReady)
 *    - Arbitrate among MasterPorts
 *    - Push winner packet to queue
 *    - Called in Phase 2 by SimPortManager
 *
 * 7. **push() (lines 74-78):**
 *    - Attempt push to FifoQueue
 *    - If success: Notify owner of activity
 *    - Return success/fail status
 *
 * 8. **getEntryIndex() (lines 80-85):**
 *    - Lookup MasterPort index by name
 *    - Used for debugging, port identification
 *
 * **Usage Example:**
 * ```cpp
 * // Memory controller with 3 CPU requestors
 * class MemoryController : public SimBase {
 *     SlavePort* req_port;
 *
 * public:
 *     void init() {
 *         // Create SlavePort with 8-entry queue, round-robin arbiter
 *         req_port = new SlavePort("req", 8, new RoundRobinArbiter());
 *         addSlavePort(req_port);
 *     }
 *
 *     void step() {
 *         // Process requests from queue
 *         while (!req_port->empty()) {
 *             auto req = req_port->pop();
 *             processRequest(req);
 *         }
 *     }
 * };
 *
 * // CPUs bind their MasterPorts to MemoryController's SlavePort
 * // SimPortManager handles binding during initialization
 * ```
 *
 * @see SlavePort.hh For interface documentation
 * @see MasterPort.cc For initiator side implementation
 * @see SimPortManager.cc For port binding and synchronization
 * @see Arbiter.hh For arbiter policies documentation
 * @see FifoQueue.hh For queue implementation
 */

#include "port/SlavePort.hh"

#include "container/RecycleContainer/RecycleContainer.hh"
#include "packet/SimPacket.hh"
#include "port/MasterPort.hh"
#include "port/SimPortManager.hh"
#include "sim/SimTop.hh"

namespace acalsim {

SlavePort::SlavePort(const std::string& name, size_t req_queue_size, Arbiter* arbiter)
    : SimPort(name), arbiter_(arbiter), req_queue_(new FifoQueue<SimPacket*>(req_queue_size, name)) {}

SlavePort::~SlavePort() {
	while (!this->empty()) {
		auto packet = this->pop();
		if (packet) { top->getRecycleContainer()->recycle(packet); }
	}

	delete this->req_queue_;
	delete this->arbiter_;
}

void SlavePort::addMasterPort(MasterPort* port) {
	bool is_present = this->port_map_.contains(port->getName());
	CLASS_ASSERT_MSG(!is_present, "MasterPort and Entry `" + port->getName() + "`: has found in SlavePort");

	this->port_map_.insert(std::make_pair(port->getName(), this->m_ports_.size()));
	this->m_ports_.push_back(port);
}

SimPacket* SlavePort::arbitrate() {
	// Use Round-Robin Algorithm to get a SimPacket from an entry
	size_t last_idx = this->arbiter_->getCurIndex();
	size_t next_idx = last_idx;
	do {
		next_idx          = this->arbiter_->select();
		MasterPort*& port = this->m_ports_.at(next_idx);
		if (port->isPopValid()) { return port->pop(); }
	} while (next_idx != last_idx);
	return nullptr;
}

void SlavePort::init() {
	CLASS_ASSERT_MSG(this->m_ports_.size(), "SlavePort `" + this->name_ + "`  doesn't connect to any MasterPort");
	this->arbiter_->setComponentsNum(this->m_ports_.size());
}

void SlavePort::sync() {
	if (this->req_queue_->isPushReady()) {
		// 1. check all the request in the MasterEntry Array and do arbitration.
		// 2. pop the winner from the MasterPort::Entry and call the callback function.
		// 3. push it to the SlavePort::reqQueue
		if (auto packet = this->arbitrate()) { this->push(packet); }
	}
}

bool SlavePort::push(SimPacket* packet) {
	bool stat = this->req_queue_->push(packet);
	if (stat) { this->owner_->setPendingActivityFlag(); }
	return stat;
}

int SlavePort::getEntryIndex(const std::string& name) const {
	auto iter       = this->port_map_.find(name);
	bool is_present = (iter != this->port_map_.end());
	CLASS_ASSERT_MSG(is_present, "Port : `" + name + "` not found!");
	return iter->second;
}

}  // namespace acalsim
