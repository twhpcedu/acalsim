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
 * @file ChannelPortManager.hh
 * @brief Manager for thread-safe channel-based communication between simulators
 *
 * ChannelPortManager coordinates channel ports for inter-simulator communication
 * using SimChannel's dual-queue ping-pong buffers. Each simulator inherits from
 * this manager to create and manage its channel-based communication interfaces.
 *
 * **ChannelPortManager Architecture:**
 * ```
 * Simulator A                                          Simulator B
 * (ChannelPortManager)                                 (ChannelPortManager)
 *   │                                                     │
 *   ├─ MasterChannelPorts (Outbound)                    ├─ SlaveChannelPorts (Inbound)
 *   │   ├─ "to_cache"  ────┐                            │   ├─ "from_cpu"  ◄───┐
 *   │   ├─ "to_mem"    ──┐ │                            │   ├─ "from_noc"  ◄─┐ │
 *   │   └─ "to_noc"    ┐ │ │                            │   └─ "req_port"  ◄┐│ │
 *   │                  │ │ │                            │                   ││ │
 *   │                  │ │ └──► SimChannel<T*> ────────┼───────────────────┘│ │
 *   │                  │ └────► SimChannel<T*> ────────┼────────────────────┘ │
 *   │                  └──────► SimChannel<T*> ────────┼──────────────────────┘
 *   │
 *   └─ SlaveChannelPorts (Inbound)
 *       ├─ "resp_from_cache"  ◄───────── SimChannel<T*> ◄── (Other Simulator)
 *       └─ "ack_from_mem"     ◄───────── SimChannel<T*> ◄── (Other Simulator)
 * ```
 *
 * **Channel Port vs. Simulation Port:**
 * ```
 * ┌──────────────────────┬────────────────────────┬──────────────────────┐
 * │ Feature              │ ChannelPort            │ SimPort              │
 * ├──────────────────────┼────────────────────────┼──────────────────────┤
 * │ Thread Safety        │ Thread-safe            │ Thread-local         │
 * │ Use Case             │ Inter-simulator comm   │ Module connectivity  │
 * │ Underlying Mechanism │ SimChannel (dual-queue)│ Direct connection    │
 * │ Arbitration          │ Lock-free ping-pong    │ Arbiter-based        │
 * │ Notification         │ Callback-based         │ Direct push/pop      │
 * │ Typical Examples     │ CPU → Cache, NoC links │ Port → Queue → Module│
 * └──────────────────────┴────────────────────────┴──────────────────────┘
 * ```
 *
 * **Communication Flow:**
 * ```
 * Sender Simulator (Iteration N, Phase 1):
 *   1. pushToMasterChannelPort("to_cache", packet)
 *   2. MasterChannelPort → push to PING queue
 *   3. Notification sent to receiver (if configured)
 *
 * [Synchronization Barrier - Phase 1 Complete]
 *
 * Global Toggle:
 *   SimChannelGlobal::toggleChannelDualQueueStatus()
 *   (PING ↔ PONG roles swap)
 *
 * [Phase 2 Begins]
 *
 * Receiver Simulator (Iteration N, Phase 2):
 *   1. handleInboundNotification() called
 *   2. packet = popFromSlaveChannelPort("from_cpu")
 *   3. SlaveChannelPort → pop from PONG queue (was PING in Phase 1)
 *   4. Process packet
 * ```
 *
 * **Connection Establishment:**
 * ```
 * ConnectPort() creates bidirectional channel:
 *
 * CPUSimulator (sender)                    CacheSimulator (receiver)
 *      │                                            │
 *      │  ConnectPort(cpu, cache,                  │
 *      │             "mem_req",                     │
 *      │             "cpu_interface")               │
 *      │                                            │
 *      ├─ Creates: MasterChannelPort "mem_req"     │
 *      │            ↓                               │
 *      │        SimChannel<SimPacket*>             │
 *      │            ↓                               │
 *      └────────────┼───────────────────────────────┤
 *                   │                               │
 *                   └──► SlaveChannelPort "cpu_interface"
 * ```
 *
 * **Key Features:**
 * - **Thread-Safe Communication**: Uses SimChannel's lock-free dual-queue
 * - **Named Ports**: Human-readable identifiers for debugging
 * - **Bidirectional Setup**: ConnectPort() creates master/slave pair with shared channel
 * - **Notification System**: Optional callbacks when data arrives
 * - **Type-Safe**: Uses ChannelPort::TPayload (typically SimPacket*)
 *
 * **Usage Example:**
 * ```cpp
 * class CPUSimulator : public SimBase, public ChannelPortManager {
 * public:
 *     void init() {
 *         // Create channel ports
 *         auto masterPort = std::make_shared<MasterChannelPort>();
 *         addMasterChannelPort("mem_req", masterPort);
 *     }
 *
 *     void execute() {
 *         // Send memory request
 *         auto* req = new MemoryRequest(addr, data);
 *         pushToMasterChannelPort("mem_req", req);
 *     }
 *
 *     void handleInboundNotification() override {
 *         // Process incoming responses
 *         auto* resp = popFromSlaveChannelPort("mem_resp");
 *         if (resp) {
 *             processResponse(resp);
 *         }
 *     }
 * };
 *
 * class CacheSimulator : public SimBase, public ChannelPortManager {
 * public:
 *     void handleInboundNotification() override {
 *         // Process incoming requests
 *         auto* req = popFromSlaveChannelPort("cpu_interface");
 *         if (req) {
 *             handleCacheRequest(req);
 *
 *             // Send response back
 *             auto* resp = new MemoryResponse(req->addr, data);
 *             pushToMasterChannelPort("cpu_resp", resp);
 *         }
 *     }
 * };
 *
 * // In SimTop initialization:
 * ChannelPortManager::ConnectPort(cpu, cache, "mem_req", "cpu_interface");
 * ChannelPortManager::ConnectPort(cache, cpu, "cpu_resp", "mem_resp");
 * ```
 *
 * @see SimChannel For underlying thread-safe dual-queue implementation
 * @see ChannelPort For master/slave channel port abstractions
 * @see SimPortManager For module-level port management (thread-local)
 * @see SimBase For simulator base class that uses ChannelPortManager
 */

#pragma once

#include <string>

#include "channel/ChannelPort.hh"
#include "common/HashVector.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

// Forward declaration
class SimPacket;

/**
 * @class ChannelPortManager
 * @brief Manager for channel-based inter-simulator communication ports
 *
 * ChannelPortManager provides a registry and coordination layer for MasterChannelPorts
 * (outbound) and SlaveChannelPorts (inbound). Each simulator typically inherits from
 * ChannelPortManager to enable thread-safe communication with other simulators via
 * SimChannel's dual-queue mechanism.
 *
 * **Design Pattern:**
 * - Uses shared_ptr for port ownership to enable shared channel references
 * - Named ports for easy identification and debugging
 * - Static ConnectPort() method for establishing simulator-to-simulator links
 * - Virtual handleInboundNotification() for custom receive logic
 *
 * **Port Registry:**
 * ```
 * ChannelPortManager
 *   ├─ masterChannelPorts (HashVector<string, SharedPtr>)
 *   │   └─ "mem_req" → MasterChannelPort → SimChannel → Receiver
 *   │
 *   └─ slaveChannelPorts (HashVector<string, SharedPtr>)
 *       └─ "mem_resp" ← SimChannel ← MasterChannelPort ← Sender
 * ```
 *
 * @note Thread-safety provided by underlying SimChannel, not by this manager
 * @see SimChannel, ChannelPort, SimBase
 */
class ChannelPortManager {
	/// @brief Type alias for channel payload (typically SimPacket*)
	using TPayload = ChannelPort::TPayload;

public:
	/**
	 * @brief Construct channel port manager
	 *
	 * Initializes empty port registries for master and slave channel ports.
	 */
	ChannelPortManager();

	/**
	 * @brief Register slave channel port (inbound) with name
	 *
	 * Adds a SlaveChannelPort to the manager's registry. Slave ports receive
	 * data from other simulators via connected SimChannels.
	 *
	 * @param _name Port identifier (e.g., "cpu_interface", "mem_resp")
	 * @param _in_port Shared pointer to SlaveChannelPort instance
	 *
	 * @note Typically called by ConnectPort() during initialization
	 * @see ConnectPort(), SlaveChannelPort
	 */
	void addSlaveChannelPort(std::string _name, SlaveChannelPort::SharedPtr _in_port);

	/**
	 * @brief Retrieve slave channel port by name
	 *
	 * @param _name Port identifier
	 * @return SlaveChannelPort::SharedPtr Shared pointer to port, nullptr if not found
	 *
	 * @note Returns shared_ptr to allow multiple references
	 */
	SlaveChannelPort::SharedPtr getSlaveChannelPort(std::string _name) const;

	/**
	 * @brief Pop packet from slave channel port
	 *
	 * Retrieves and removes one packet from the named slave port's underlying
	 * SimChannel. Uses the channel's pop-side queue (determined by ping-pong status).
	 *
	 * @param _name Slave port identifier
	 * @return TPayload Packet pointer (typically SimPacket*), nullptr if empty
	 *
	 * **Usage Example:**
	 * ```cpp
	 * void handleInboundNotification() override {
	 *     auto* req = popFromSlaveChannelPort("cpu_interface");
	 *     if (req) {
	 *         processRequest(req);
	 *     }
	 * }
	 * ```
	 *
	 * @note Should be called during Phase 2 (after channel toggle)
	 * @see SlaveChannelPort::pop(), SimChannel
	 */
	TPayload popFromSlaveChannelPort(std::string _name);

	/**
	 * @brief Register master channel port (outbound) with name
	 *
	 * Adds a MasterChannelPort to the manager's registry. Master ports send
	 * data to other simulators via connected SimChannels.
	 *
	 * @param _name Port identifier (e.g., "mem_req", "cpu_resp")
	 * @param _out_port Shared pointer to MasterChannelPort instance
	 *
	 * @note Typically called by ConnectPort() during initialization
	 * @see ConnectPort(), MasterChannelPort
	 */
	void addMasterChannelPort(std::string _name, MasterChannelPort::SharedPtr _out_port);

	/**
	 * @brief Retrieve master channel port by name
	 *
	 * @param _name Port identifier
	 * @return MasterChannelPort::SharedPtr Shared pointer to port, nullptr if not found
	 *
	 * @note Returns shared_ptr to allow multiple references
	 */
	MasterChannelPort::SharedPtr getMasterChannelPort(std::string _name) const;

	/**
	 * @brief Push packet to master channel port
	 *
	 * Sends a packet to the named master port's underlying SimChannel.
	 * Uses the channel's push-side queue (determined by ping-pong status).
	 * May trigger notification callback in connected receiver.
	 *
	 * @param _name Master port identifier
	 * @param _item Packet to send (typically SimPacket*)
	 *
	 * **Usage Example:**
	 * ```cpp
	 * void sendMemoryRequest(Addr addr, uint8_t* data) {
	 *     auto* req = new MemoryRequest(addr, data);
	 *     pushToMasterChannelPort("mem_req", req);
	 * }
	 * ```
	 *
	 * @note Should be called during Phase 1 (parallel execution phase)
	 * @note Receiver will pop from opposite queue after toggle
	 * @see MasterChannelPort::push(), SimChannel
	 */
	void pushToMasterChannelPort(std::string _name, TPayload const& _item);

	/**
	 * @brief Handle inbound data arrival notification
	 *
	 * Virtual callback invoked when data arrives on any slave channel port.
	 * Called by MasterChannelPort::insert() on the receiving simulator when
	 * the sender pushes data to the channel.
	 *
	 * **Override Pattern:**
	 * Subclasses should override this method to:
	 * 1. Check which slave ports have data (via popFromSlaveChannelPort)
	 * 2. Process incoming packets
	 * 3. Generate responses if needed
	 *
	 * **Usage Example:**
	 * ```cpp
	 * class CacheSimulator : public SimBase {
	 *     void handleInboundNotification() override {
	 *         // Check CPU request port
	 *         while (auto* req = popFromSlaveChannelPort("cpu_req")) {
	 *             handleCacheRequest(req);
	 *
	 *             // Send response back to CPU
	 *             auto* resp = createResponse(req);
	 *             pushToMasterChannelPort("cpu_resp", resp);
	 *         }
	 *
	 *         // Check memory response port
	 *         while (auto* resp = popFromSlaveChannelPort("mem_resp")) {
	 *             handleMemoryResponse(resp);
	 *         }
	 *     }
	 * };
	 * ```
	 *
	 * @note Default implementation does nothing - subclasses must override
	 * @note Called asynchronously when sender pushes data (notification-based)
	 * @note Should pop from ALL relevant slave ports to drain queues
	 * @see popFromSlaveChannelPort(), pushToMasterChannelPort()
	 */
	virtual void handleInboundNotification();

	/**
	 * @brief Connect channel ports between sender and receiver simulators
	 *
	 * Static method establishing unidirectional communication channel between two
	 * simulators. Creates a shared SimChannel and registers MasterChannelPort in
	 * sender and SlaveChannelPort in receiver.
	 *
	 * **What ConnectPort() Creates:**
	 * ```
	 * Before:
	 *   Sender (empty)        Receiver (empty)
	 *
	 * After ConnectPort(sender, receiver, "out", "in"):
	 *   Sender                                      Receiver
	 *     │                                            │
	 *     └─ MasterChannelPort "out"                  │
	 *            │                                     │
	 *            └──► SimChannel<TPayload> ──────────►└─ SlaveChannelPort "in"
	 * ```
	 *
	 * **Bidirectional Communication:**
	 * For request-response patterns, call ConnectPort twice:
	 * ```cpp
	 * // CPU → Cache (request channel)
	 * ChannelPortManager::ConnectPort(cpu, cache, "mem_req", "cpu_req");
	 *
	 * // Cache → CPU (response channel)
	 * ChannelPortManager::ConnectPort(cache, cpu, "cpu_resp", "mem_resp");
	 * ```
	 *
	 * **Complete Example:**
	 * ```cpp
	 * // In SimTop initialization:
	 * auto* cpu = new CPUSimulator();
	 * auto* l1cache = new CacheSimulator();
	 * auto* memory = new MemoryController();
	 *
	 * // Connect CPU ↔ L1 Cache
	 * ChannelPortManager::ConnectPort(cpu, l1cache, "mem_req", "cpu_interface");
	 * ChannelPortManager::ConnectPort(l1cache, cpu, "cpu_resp", "mem_resp");
	 *
	 * // Connect L1 Cache ↔ Memory
	 * ChannelPortManager::ConnectPort(l1cache, memory, "mem_access", "cache_req");
	 * ChannelPortManager::ConnectPort(memory, l1cache, "cache_resp", "mem_data");
	 * ```
	 *
	 * @param _sender Sender simulator (will get MasterChannelPort)
	 * @param _receiver Receiver simulator (will get SlaveChannelPort)
	 * @param _sender_port_name Name for MasterChannelPort in sender
	 * @param _receiver_port_name Name for SlaveChannelPort in receiver
	 *
	 * @note Creates shared SimChannel owned by both port references
	 * @note Must be called during initialization, before simulation starts
	 * @note Creates unidirectional link - use twice for bidirectional communication
	 * @see SimChannel, MasterChannelPort, SlaveChannelPort
	 */
	static inline void ConnectPort(ChannelPortManager* _sender, ChannelPortManager* _receiver,
	                               std::string _sender_port_name, std::string _receiver_port_name);

protected:
	/// @brief Registry of master (outbound) channel ports indexed by name
	HashVector<std::string, MasterChannelPort::SharedPtr> masterChannelPorts;

	/// @brief Registry of slave (inbound) channel ports indexed by name
	HashVector<std::string, SlaveChannelPort::SharedPtr> slaveChannelPorts;

#ifdef ACALSIM_STATISTICS
public:
	/**
	 * @brief Get accumulated communication cost statistics
	 * @return double Total accumulated cost metric
	 *
	 * @note Only available when compiled with ACALSIM_STATISTICS
	 */
	inline static double getCostStat() { return ChannelPortManager::cost_stat.sum(); }

private:
	/// @brief Global statistics accumulator for channel communication costs
	inline static Statistics<double, acalsim::StatisticsMode::Accumulator, true> cost_stat;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "channel/ChannelPortManager.inl"
