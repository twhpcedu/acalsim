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
 * @file SimPortManager.hh
 * @brief Port lifecycle and connection management for simulators and modules
 *
 * SimPortManager coordinates all MasterPorts and SlavePorts within a simulator
 * or module, handling creation, connection, initialization, and synchronization.
 *
 * **Port Management Architecture:**
 * ```
 * SimPortManager (e.g., CPUSimulator)
 *   │
 *   ├─── MasterPorts (Initiators)
 *   │     ├─ "mem_req"  ────────┐
 *   │     ├─ "io_req"   ─────┐  │
 *   │     └─ "dma_req"  ───┐ │  │
 *   │                       │ │  │
 *   └─── SlavePorts (Targets)  │ │
 *         ├─ "inst_fetch" <─────┘ │
 *         ├─ "data_access" <───────┘
 *         └─ "interrupt"
 *
 * ConnectPort() establishes connections between managers:
 *   CPUSimulator.m_ports_["mem_req"] ──► CacheSimulator.s_ports_["cpu_interface"]
 * ```
 *
 * **Initialization and Sync Flow:**
 * ```
 * Lifecycle Phase:
 *
 * 1. Port Creation (init phase):
 *    simPortMgr->addMasterPort("mem_req")
 *    simPortMgr->addSlavePort("inst_fetch", queue_size)
 *
 * 2. Port Connection (init phase):
 *    SimPortManager::ConnectPort(cpu, cache, "mem_req", "cpu_interface")
 *
 * 3. Port Initialization (initSimPort):
 *    - Validate all connections
 *    - Initialize arbiters
 *    - Setup retry callbacks
 *
 * 4. Port Synchronization (syncSimPort, per iteration):
 *    - Arbitrate all slave ports
 *    - Transfer winning packets to queues
 *    - Trigger retry callbacks for losers
 * ```
 *
 * **Key Features:**
 * - **Centralized Port Registry**: Single manager per simulator/module
 * - **Connection Validation**: Ensures all ports properly connected
 * - **Retry Coordination**: Manages backpressure via retry callbacks
 * - **Pipe Register Support**: Integrates with pipeline register management
 * - **Pending Activity Tracking**: Monitors port state for simulation progress
 *
 * @see MasterPort, SlavePort, SimPort, SimBase, SimModule
 */

#pragma once

/**
 * @defgroup Management
 * @brief Functions for managing ports.
 */

/**
 * @defgroup PacketTransmission
 * @brief Functions for transmission between masterport and slaveport
 */

/**
 * @defgroup PendingActivity
 * @brief Function for ACALSim to monitor pending activities and maintain simulation integrity.
 */

/**
 * @defgroup SimulationFlow
 * @brief Function for initializing, executing, and synchronizing ports in ACALSim to maintain simulations.
 */

#include <string>

#include "common/Arbiter.hh"
#include "common/HashVector.hh"
#include "hw/SimPipeRegister.hh"
#include "packet/SimPacket.hh"
#include "port/MasterPort.hh"
#include "port/SlavePort.hh"

#ifdef ACALSIM_STATISTICS
#include <atomic>
#endif  // ACALSIM_STATISTICS

namespace acalsim {

/**
 * @class SimPortManager
 * @brief Manager for all ports within a simulator or module
 *
 * Coordinates port creation, connection, initialization, and synchronization.
 * Each simulator (SimBase) and module (SimModule) inherits from SimPortManager
 * to manage their communication interfaces.
 *
 * **Usage Pattern:**
 * ```cpp
 * class CPUSimulator : public SimBase {  // SimBase → SimPortManager
 *     void init() override {
 *         // Create ports
 *         auto* memPort = addMasterPort("mem_req");
 *         auto* instPort = addSlavePort("inst_fetch", 4);
 *
 *         // Connect to cache (done externally or in SimTop)
 *         SimPortManager::ConnectPort(this, cache, "mem_req", "cpu_port");
 *     }
 * };
 * ```
 *
 * @note Uses virtual inheritance to work with multiple inheritance hierarchies
 */
class SimPortManager : virtual public HashableType {
public:
	/**
	 * @brief Construct port manager with name
	 * @param name Manager identifier (typically simulator or module name)
	 */
	SimPortManager(const std::string& name) : name_(name) {}

	/**
	 * @brief Destructor - cleans up all managed ports
	 */
	virtual ~SimPortManager();

	/**
	 * @ingroup Management
	 * @brief Add a master port with the specified name.
	 *
	 * @note
	 * This function adds a master port with the given name to the object.
	 * Master ports typically represent endpoints that initiate communication.
	 *
	 * @param name The name of the master port to add.
	 * @return A pointer to the newly added MasterPort object.
	 */
	MasterPort* addMasterPort(const std::string& name);

	/**
	 * @ingroup Management
	 * @brief Add a master port with the specified name.
	 *
	 * @note
	 * This function adds a master port with the given name to the object.
	 * Master ports typically represent endpoints that initiate communication.
	 *
	 * @param name The name of the master port.
	 * @param port The pointer of the master port.
	 */
	void addMasterPort(const std::string& name, MasterPort* port);
	/**
	 * @ingroup Management
	 * @brief Method to get a specific MasterPort pointer by name.
	 *
	 * @param _name The name of the SlavePort to retrieve.
	 *
	 * @return Pointer to the requested MasterPort, or nullptr if not found.
	 */
	MasterPort* getMasterPort(const std::string& name) const;

	/**
	 * @ingroup Management
	 * @brief Adds a slave port to the simulator.
	 *
	 * @note
	 * This function adds a slave port with the specified name and required queue size to the simulator.
	 *
	 * @param name The name of the slave port.
	 * @param req_queue_size The size of the required queue for the slave port.
	 * @param arbiter [optional] A pointer to an Arbiter object. Default is RoundRobin Arbiter.
	 *
	 * @return A pointer to the newly added SlavePort object.
	 * @throws std::runtime_error If a port with the same name already exists.
	 */
	SlavePort* addSlavePort(const std::string& name, size_t req_queue_size, Arbiter* arbiter = new RoundRobin());

	/**
	 * @ingroup Management
	 * @brief Adds a slave port to the simulator.
	 *
	 * @param name The name of the slave port.
	 * @param port The pointer of the slave port.
	 *
	 * @throws std::runtime_error if a port with the same name already exists.
	 */
	void addSlavePort(const std::string& name, SlavePort* port);

	void             addPRMasterPort(const std::string& name, SimPipeRegister* reg);
	void             addPRSlavePort(const std::string& name, SimPipeRegister* reg);
	SimPipeRegister* getPipeRegister(const std::string& name) const;
	/**
	 * @ingroup Management
	 * @brief Method to get a specific SlavePort pointer by name.
	 *
	 * @param _name The name of the SlavePort to retrieve.
	 *
	 * @return Pointer to the requested SlavePort, or nullptr if not found.
	 */
	SlavePort* getSlavePort(const std::string& name) const;

	/**
	 * @ingroup PacketTransmission
	 * @brief Push a SimPacket to the appropriate MasterPort.
	 *
	 * @param name The port name of the MasterPort.
	 * @param packet The SimPacket pointer to be pushed to the MasterPort.
	 *
	 * @return true if the SimPacket has been pushed to MasterPort::Entry, otherwise false.
	 */
	bool pushToMasterPort(const std::string& name, SimPacket* packet);

	/**
	 * @ingroup SimulationFlow
	 * @brief Initialize all slave ports and master ports
	 *
	 * @see MasterPort::init()
	 * @see SlavePort::init()
	 */
	virtual void initSimPort();

	/**
	 * @ingroup SimulationFlow
	 * @brief Synchronizes all slave ports with master ports and syncSimPort in modules.
	 *
	 * @note
	 * This function synchronizes all slave ports belonging to the SimPortManager instance with the
	 * MasterPort. It also calls the syncSimPort method on all modules.
	 *
	 * @see SimPort::sync()
	 */
	virtual void syncSimPort();

	/**
	 * @ingroup SimulationFlow
	 * @brief Trigger the master port owner to try pushing a packet to the master port again.
	 */
	virtual void masterPortRetry(MasterPort* port) = 0;

	virtual void triggerRetryCallback(Tick t);

	void setPendingActivityFlag() {
		if (!this->hasPendingActivity_) { this->hasPendingActivity_ = true; }
	}

	/**
	 * @ingroup PendingActivity
	 * @brief Check if any SlavePort in simulator has a packet in its entry.
	 *
	 * @note
	 * This method iterates through the slavePorts and modules to check if any SlavePort
	 * has a packet in its entry or if any SimPortManager has a SlavePort with a packet in its entry.
	 *
	 * @return true if any SlavePort or SimPortManager has a packet in its entry, otherwise false.
	 */
	virtual bool hasPendingActivityInSimPort(bool pipeRegisterDump) const;

	virtual void clearHasPendingActivityInSimPortFlag();

	/**
	 * @brief Connect master port to slave port across managers
	 *
	 * Static method establishing bidirectional connection between a MasterPort
	 * in one manager and a SlavePort in another. Configures both ports to
	 * recognize each other.
	 *
	 * @param master Manager containing MasterPort (initiator side)
	 * @param slave Manager containing SlavePort (target side)
	 * @param m_port_name Name of MasterPort in master manager
	 * @param s_port_name Name of SlavePort in slave manager
	 *
	 * **Connection Example:**
	 * ```cpp
	 * // Connect CPU memory request port to L1 cache slave port
	 * SimPortManager::ConnectPort(
	 *     cpu,    // CPUSimulator (has MasterPort "mem_req")
	 *     l1cache, // CacheSimulator (has SlavePort "cpu_interface")
	 *     "mem_req",
	 *     "cpu_interface"
	 * );
	 * ```
	 *
	 * @note Must be called after ports created but before initSimPort()
	 * @note Validates ports exist in respective managers
	 */
	static void ConnectPort(SimPortManager* master, SimPortManager* slave, const std::string& m_port_name,
	                        const std::string& s_port_name);

protected:
	/// @brief A unordered_map of slave ports
	HashVector<std::string, SlavePort*> s_ports_;

	/// @brief A unordered_map of Master ports
	HashVector<std::string, MasterPort*> m_ports_;

	/// @brief A unordered_map of Master ports
	HashVector<std::string, SimPipeRegister*> pipeRegisters;

private:
	std::string name_;

	bool hasPendingActivity_ = false;

#ifdef ACALSIM_STATISTICS
public:
	inline static size_t getConnectionCnt() { return SimPortManager::connection_cnt_; }

private:
	inline static std::atomic<size_t> connection_cnt_ = 0;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim
