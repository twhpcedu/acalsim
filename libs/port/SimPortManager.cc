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
 * @file SimPortManager.cc
 * @brief SimPortManager implementation - port lifecycle and synchronization orchestration
 *
 * This file implements SimPortManager, which manages the lifecycle of MasterPort and
 * SlavePort instances, handles port binding between components, and orchestrates Phase 2
 * synchronization for port-based communication in ACALSim's two-phase execution model.
 *
 * **Port Management Architecture:**
 * ```
 * SimPortManager (owned by SimBase/SimModule)
 *   │
 *   ├─ m_ports_: HashVector<MasterPort*>
 *   │    ├─ "req_port" → MasterPort instance
 *   │    ├─ "resp_port" → MasterPort instance
 *   │    └─ ... (multiple MasterPorts per manager)
 *   │
 *   ├─ s_ports_: HashVector<SlavePort*>
 *   │    ├─ "in_port" → SlavePort instance
 *   │    ├─ "cmd_port" → SlavePort instance
 *   │    └─ ... (multiple SlavePorts per manager)
 *   │
 *   ├─ pipeRegisters: HashVector<SimPipeRegister*>
 *   │    └─ Pipeline registers for staged communication
 *   │
 *   └─ hasPendingActivity_: bool
 *        └─ Tracks if any port has pending transactions
 * ```
 *
 * **Port Binding Flow (ConnectPort):**
 * ```
 * Scenario: CPU connects to Memory Controller
 *
 * CPU (SimPortManager):
 *   └─ MasterPort "req"
 *
 * MemCtrl (SimPortManager):
 *   └─ SlavePort "req_slave"
 *
 * Binding:
 *   SimPortManager::ConnectPort(
 *       cpu_manager, memctrl_manager,
 *       "req", "req_slave"
 *   )
 *   │
 *   ├─ 1. Lookup MasterPort "req" from cpu_manager
 *   │     └─ cpu_manager->getMasterPort("req")
 *   │
 *   ├─ 2. Lookup SlavePort "req_slave" from memctrl_manager
 *   │     └─ memctrl_manager->getSlavePort("req_slave")
 *   │
 *   ├─ 3. Bidirectional registration
 *   │     ├─ m_port->registerSlavePort(s_port)
 *   │     │     └─ MasterPort knows its destination
 *   │     │
 *   │     └─ s_port->addMasterPort(m_port)
 *   │           └─ SlavePort tracks all connected MasterPorts
 *   │
 *   └─ 4. Update connection statistics
 *        └─ connection_cnt_++ (global counter)
 *
 * Result: CPU can now send packets to Memory Controller via ports
 * ```
 *
 * **Phase 2 Synchronization (syncSimPort):**
 * Called by SimTop in Phase 2 to transfer packets from MasterPorts to SlavePorts:
 *
 * ```
 * SimTop::run() - Phase 2:
 *   │
 *   └─ SimPortManager::syncSimPort() for each SimBase
 *        │
 *        └─ For each SlavePort in s_ports_:
 *             │
 *             └─ s_port->sync()
 *                  │
 *                  ├─ Check: queue has space?
 *                  │   └─ isPushReady() → true
 *                  │
 *                  ├─ Arbitrate among connected MasterPorts
 *                  │   └─ arbitrate() → select winner packet
 *                  │
 *                  ├─ Pop packet from winning MasterPort
 *                  │   └─ m_port->pop(_retry=...)
 *                  │
 *                  └─ Push packet to SlavePort queue
 *                       └─ s_port->push(packet)
 *
 * Why Phase 2?
 *   - All MasterPort pushes complete in Phase 1 (parallel)
 *   - Phase 2 transfers packets atomically (no race conditions)
 *   - Ensures consistent global state before next iteration
 * ```
 *
 * **Retry Callback Mechanism (triggerRetryCallback):**
 * When SlavePort queue is full, MasterPort sets retry flag. In next iteration:
 *
 * ```
 * SimBase::stepWrapper() - Phase 1 (next iteration):
 *   │
 *   └─ SimPortManager::triggerRetryCallback(current_tick)
 *        │
 *        └─ For each MasterPort in m_ports_:
 *             │
 *             ├─ Check: port->isRetry()?
 *             │   └─ Retry flag set in previous Phase 2
 *             │
 *             ├─ Check: timestamp != current_tick?
 *             │   └─ Prevent duplicate retry in same iteration
 *             │
 *             ├─ If both true:
 *             │   ├─ masterPortRetry(port)
 *             │   │     └─ Execute user's retry_callback()
 *             │   │          └─ User re-attempts push(packet)
 *             │   │
 *             │   └─ port->finishRetry()
 *             │        └─ Clear retry flag, update timestamp
 *             │
 *             └─ Continue to next MasterPort
 *
 * Purpose:
 *   - Give user chance to re-send packet when space available
 *   - Prevent packet loss from backpressure
 *   - Maintain lossless communication semantics
 * ```
 *
 * **SimPipeRegister Integration:**
 * SimPipeRegister wraps MasterPort+SlavePort for pipeline stage communication:
 *
 * ```
 * Pipeline Stage Example (CPU with Fetch→Decode→Execute):
 *
 * Fetch Stage:
 *   └─ SimPipeRegister "fetch_decode"
 *        ├─ MasterPort (Fetch writes here)
 *        └─ SlavePort (Decode reads from here)
 *
 * addPRMasterPort("fetch_decode", fetch_decode_reg)
 *   └─ Adds MasterPort to m_ports_
 *   └─ Adds SimPipeRegister to pipeRegisters
 *
 * addPRSlavePort("fetch_decode", fetch_decode_reg)
 *   └─ Adds SlavePort to s_ports_
 *   └─ Adds SimPipeRegister to pipeRegisters
 *
 * Why separate methods?
 *   - Fetch stage uses MasterPort side
 *   - Decode stage uses SlavePort side
 *   - Same SimPipeRegister, different perspectives
 * ```
 *
 * **Port Factory Methods:**
 * SimPortManager provides factory methods for convenient port creation:
 *
 * ```cpp
 * // Manual creation + registration:
 * auto m_port = new MasterPort("req");
 * manager->addMasterPort("req", m_port);
 *
 * // Factory method (preferred):
 * auto m_port = manager->addMasterPort("req");
 *   └─ Creates MasterPort internally
 *   └─ Registers with manager
 *   └─ Returns pointer for user
 *
 * // SlavePort factory with configuration:
 * auto s_port = manager->addSlavePort("req_slave", queue_size=8, arbiter);
 *   └─ Creates SlavePort with specified queue size and arbiter
 * ```
 *
 * **Activity Tracking:**
 * ```
 * hasPendingActivityInSimPort() integration:
 *   │
 *   ├─ Purpose: Track if any port has pending transactions
 *   │
 *   ├─ Set by: SlavePort::push() when packet added
 *   │   └─ owner_->setPendingActivityFlag()
 *   │        └─ hasPendingActivity_ = true
 *   │
 *   ├─ Checked by: SimBase::interIterationUpdate()
 *   │   └─ Determine if SimBase should stay active
 *   │
 *   └─ Cleared by: clearHasPendingActivityInSimPortFlag()
 *        └─ Called each iteration after activity check
 *
 * Effect on simulation:
 *   - Prevents premature termination when ports have data
 *   - Ensures retry callbacks get triggered
 *   - Maintains correctness of activity-based scheduling
 * ```
 *
 * **Memory Management:**
 * ```
 * Destructor (~SimPortManager):
 *   1. Delete all MasterPorts in m_ports_
 *      for (auto& port : m_ports_) delete port;
 *
 *   2. Delete all SlavePorts in s_ports_
 *      for (auto& port : s_ports_) delete port;
 *
 * Note: SimPipeRegisters NOT deleted here
 *   - pipeRegisters stores raw pointers
 *   - Actual SimPipeRegister objects owned elsewhere
 *   - Prevents double-delete
 * ```
 *
 * **Statistics Collection (ACALSIM_STATISTICS):**
 * ```
 * #ifdef ACALSIM_STATISTICS
 *   connection_cnt_++;  // Global counter
 * #endif
 *
 * Tracks:
 *   - Total number of port connections
 *   - Used for performance analysis
 *   - Reported in SimTop::finish()
 * ```
 *
 * **Implementation Functions:**
 *
 * 1. **addMasterPort() (lines 30-41):**
 *    - Factory method or registration
 *    - Prevent duplicate names
 *    - Set owner relationship
 *
 * 2. **addSlavePort() (lines 69-80):**
 *    - Factory method with queue size
 *    - Create with arbiter
 *    - Set owner relationship
 *
 * 3. **addPRMasterPort/addPRSlavePort() (lines 43-55):**
 *    - Register SimPipeRegister ports
 *    - Track both port and pipe register
 *    - Enable pipeline stage lookup
 *
 * 4. **ConnectPort() (lines 115-132):**
 *    - Static method for binding
 *    - Bidirectional registration
 *    - Update connection statistics
 *
 * 5. **syncSimPort() (lines 93-95):**
 *    - Phase 2 synchronization
 *    - Call sync() on all SlavePorts
 *    - Transfer packets from Masters to Slaves
 *
 * 6. **triggerRetryCallback() (lines 97-104):**
 *    - Phase 1 retry handling
 *    - Execute user retry callbacks
 *    - Timestamp-based deduplication
 *
 * 7. **initSimPort() (lines 88-91):**
 *    - Pre-simulation initialization
 *    - Call init() on all ports
 *    - Configure arbiters
 *
 * **Usage Example:**
 * ```cpp
 * class CPU : public SimBase {
 *     MasterPort* req_port;
 *
 * public:
 *     void init() {
 *         // Create port via manager
 *         req_port = this->addMasterPort("req");
 *
 *         // Set retry callback
 *         req_port->setRetryCallback([this](Tick when) {
 *             // Re-send packet on retry
 *             if (pending_req) {
 *                 req_port->push(pending_req);
 *             }
 *         });
 *     }
 * };
 *
 * class Memory : public SimBase {
 *     SlavePort* req_slave;
 *
 * public:
 *     void init() {
 *         // Create SlavePort with 8-entry queue
 *         req_slave = this->addSlavePort("req_slave", 8, new RoundRobinArbiter());
 *     }
 * };
 *
 * // In SimTop::registerSimulators():
 * SimPortManager::ConnectPort(cpu, memory, "req", "req_slave");
 * ```
 *
 * @see SimPortManager.hh For interface documentation
 * @see MasterPort.cc For initiator port implementation
 * @see SlavePort.cc For receiver port implementation
 * @see SimBase.cc For port usage in simulation lifecycle
 */

#include "port/SimPortManager.hh"

#include <string>

#include "sim/SimTop.hh"

namespace acalsim {

SimPortManager::~SimPortManager() {
	for (auto& port : this->m_ports_) { delete port; }
	for (auto& port : this->s_ports_) { delete port; }
}

MasterPort* SimPortManager::addMasterPort(const std::string& name) {
	auto port = new MasterPort(name);
	this->addMasterPort(name, port);
	return port;
}

void SimPortManager::addMasterPort(const std::string& name, MasterPort* port) {
	bool is_present = this->m_ports_.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!is_present, "MasterPort : `" + name + "` is present in `" + this->name_ + "` MasterPorts!");
	port->setOwner(this);
	this->m_ports_.insert(std::make_pair(name, port));
}

void SimPortManager::addPRMasterPort(const std::string& name, SimPipeRegister* reg) {
	this->addMasterPort(name, reg->getMasterPort());
	bool is_present = this->pipeRegisters.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!is_present, "PipeRegister : `" + name + "` is present in `" + this->name_ + "`pipeRegisters!");
	this->pipeRegisters.insert(std::make_pair(name, reg));
}

void SimPortManager::addPRSlavePort(const std::string& name, SimPipeRegister* reg) {
	this->addSlavePort(name, reg->getSlavePort());
	bool is_present = this->pipeRegisters.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!is_present, "PipeRegister : `" + name + "` is present in `" + this->name_ + "`pipeRegisters!");
	this->pipeRegisters.insert(std::make_pair(name, reg));
}

SimPipeRegister* SimPortManager::getPipeRegister(const std::string& name) const {
	auto iter = this->pipeRegisters.getUMapRef().find(name);
	CLASS_ASSERT_MSG(iter != this->pipeRegisters.getUMapRef().end(), "PipeRegister :`" + name + "` Not Found !");
	return iter->second.get();
}

MasterPort* SimPortManager::getMasterPort(const std::string& name) const {
	auto iter = this->m_ports_.getUMapRef().find(name);
	CLASS_ASSERT_MSG(iter != this->m_ports_.getUMapRef().end(), "MasterPort :`" + name + "` Not Found !");
	return iter->second.get();
}

SlavePort* SimPortManager::addSlavePort(const std::string& name, size_t req_queue_size, Arbiter* arbiter) {
	auto port = new SlavePort(name, req_queue_size, arbiter);
	this->addSlavePort(name, port);
	return port;
}

void SimPortManager::addSlavePort(const std::string& name, SlavePort* port) {
	bool is_present = this->s_ports_.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!is_present, "SlavePort : `" + name + "` is present in `" + this->name_ + "` SlavePorts!");
	port->setOwner(this);
	this->s_ports_.insert(std::make_pair(name, port));
}

SlavePort* SimPortManager::getSlavePort(const std::string& name) const {
	auto iter = this->s_ports_.getUMapRef().find(name);
	CLASS_ASSERT_MSG(iter != this->s_ports_.getUMapRef().end(), "SlavePort :`" + name + "` Not Found!");
	return iter->second.get();
}

void SimPortManager::initSimPort() {
	for (auto& port : this->m_ports_) port->init();
	for (auto& port : this->s_ports_) port->init();
}

void SimPortManager::syncSimPort() {
	for (auto& port : this->s_ports_) port->sync();
}

void SimPortManager::triggerRetryCallback(Tick t) {
	for (auto& port : this->m_ports_) {
		if (port->isRetry() && port->getRetryTimestamp() != t) {
			this->masterPortRetry(port);
			port->finishRetry();
		}
	}
}

bool SimPortManager::pushToMasterPort(const std::string& name, SimPacket* packet) {
	bool ret = this->getMasterPort(name)->push(packet);
	return ret;
}

bool SimPortManager::hasPendingActivityInSimPort(bool pipeRegisterDump) const { return this->hasPendingActivity_; }

void SimPortManager::clearHasPendingActivityInSimPortFlag() { this->hasPendingActivity_ = false; }

void SimPortManager::ConnectPort(SimPortManager* m, SimPortManager* s, const std::string& m_port_name,
                                 const std::string& s_port_name) {
	ASSERT_MSG(m, "Sender not found");
	ASSERT_MSG(s, "Receiver not found");

	MasterPort* m_port = m->getMasterPort(m_port_name);
	SlavePort*  s_port = s->getSlavePort(s_port_name);

	ASSERT_MSG(m_port, "MasterPort : `" + m_port_name + "` not found");
	ASSERT_MSG(s_port, "SlavePort  : `" + s_port_name + "` not found");

	m_port->registerSlavePort(s_port);
	s_port->addMasterPort(m_port);

#ifdef ACALSIM_STATISTICS
	SimPortManager::connection_cnt_++;
#endif  // ACALSIM_STATISTICS
}

}  // end of namespace acalsim
