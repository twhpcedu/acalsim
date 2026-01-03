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
 * @file MasterPort.hh
 * @brief Initiator port for request-response communication patterns
 *
 * MasterPort implements the initiator side of port-based communication,
 * managing request transmission with single-entry buffering and retry-based
 * backpressure handling.
 *
 * **MasterPort Request Flow:**
 * ```
 * Module (Initiator)          MasterPort               SlavePort          Module (Target)
 *      |                           |                         |                   |
 *      |-- push(packet) ---------->|                         |                   |
 *      |                           |-- send(packet) -------->|                   |
 *      |                           |                         |-- accept() ------>|
 *      |                           |                         |                   |
 *      |                    [If slave busy]                  |                   |
 *      |<-- retry callback --------|<----- retry ------------|                   |
 *      |                           |                         |                   |
 *      |-- push(packet) [retry] -->|                         |                   |
 *      |                           |-- send(packet) -------->|-- accept() ------>|
 * ```
 *
 * **Single-Entry Buffer Model:**
 * ```
 * MasterPort State Machine:
 *
 *   [Empty]
 *      │
 *      │ push(pkt)
 *      ▼
 *   [Full: pkt]
 *      │
 *      │ pop()
 *      ▼
 *   [Empty]
 *
 * - Only one packet can be buffered at a time
 * - push() blocks if entry is full
 * - Retry mechanism handles backpressure
 * ```
 *
 * **Retry Mechanism:**
 * ```
 * Time: T0      T1      T2      T3      T4
 *       |       |       |       |       |
 * push()────X   │       │       │       │  (blocked: slave busy)
 *       │   │   │       │       │       │
 * retry────────>│       │       │       │  (callback triggered)
 *       │       │       │       │       │
 * push()────────────>OK │       │       │  (retry successful)
 * ```
 *
 * **Key Features:**
 * - **Single-Entry Buffering**: Lightweight with minimal state
 * - **Retry-Based Backpressure**: Callback mechanism for flow control
 * - **SlavePort Binding**: One-to-one connection with target
 * - **Pending Activity Tracking**: Notifies owner when retry needed
 *
 * @see SlavePort For target port implementation
 * @see SimPort For base class
 * @see SimPortManager For port connection management
 */

#pragma once

#include <functional>
#include <string>

#include "external/gem5/Event.hh"
#include "port/SimPort.hh"
#include "utils/Logging.hh"

/**
 * @defgroup Management
 * @brief Functions for managing MasterPort.
 */

/**
 * @defgroup Operation
 * @brief Functions for operating MasterPort.
 */

/**
 * @defgroup Status
 * @brief Functions for getting the status of MasterPort.
 */

namespace acalsim {

// Forward declarations
class SimPacket;
class SlavePort;

/**
 * @class MasterPort
 * @brief Initiator port with single-entry buffering and retry mechanism
 *
 * MasterPort sends packets to connected SlavePort. Uses single-entry buffer
 * for simplicity and retry callbacks for flow control when slave is busy.
 *
 * **Design Pattern:**
 * - Single-entry buffer (not a queue) - only one outstanding request
 * - Retry callbacks for backpressure (not blocking)
 * - Friend relationship with SlavePort for direct access
 *
 * **Usage Example:**
 * ```cpp
 * auto* masterPort = new MasterPort("cpu_req");
 * auto* slavePort = new SlavePort("cache_interface");
 * masterPort->registerSlavePort(slavePort);
 * masterPort->init();
 *
 * // Send request
 * auto* memReq = new MemoryRequest(addr, data);
 * if (masterPort->isPushReady()) {
 *     masterPort->push(memReq);
 * }
 * ```
 *
 * @note Thread-local - not thread-safe (accessed from single simulator thread)
 * @see SlavePort, SimPacket, SimPortManager
 */
class MasterPort : public SimPort {
	friend class SlavePort;       ///< Allow SlavePort to access private members
	friend class SimPortManager;  ///< Allow manager to orchestrate connections

public:
	/**
	 * @brief Construct master port with name
	 * @param name Port identifier (e.g., "cpu_mem_req", "l1_miss")
	 */
	MasterPort(const std::string& name);

	/**
	 * @brief Destructor
	 */
	~MasterPort();

	/**
	 * @brief Push packet to slave port
	 *
	 * Attempts to send packet to connected SlavePort. If slave accepts,
	 * packet is transferred and entry becomes empty. If slave is busy,
	 * packet remains in entry and retry mechanism activates.
	 *
	 * @param packet Packet to send (ownership transferred if accepted)
	 * @return bool True if packet accepted, false if slave busy (retry later)
	 *
	 * @warning Entry must be empty (isPushReady() == true) before calling
	 * @note If false returned, packet remains in entry until retry succeeds
	 */
	bool push(SimPacket* packet);

	/**
	 * @brief Check if entry is empty and ready for new packet
	 *
	 * @return bool True if entry empty (can push), false if full
	 *
	 * @note Check this before push() to avoid overwriting buffered packet
	 */
	bool isPushReady() const { return this->entry_ == nullptr; }

	/**
	 * @brief Remove and return packet from entry
	 *
	 * Used internally for retry mechanism. Removes packet from entry buffer
	 * after successful transmission to slave.
	 *
	 * @param _retry Enable retry mechanism if true (default)
	 * @return SimPacket* Pointer to packet removed from entry, nullptr if empty
	 *
	 * @note Typically called by internal retry logic, not by users directly
	 */
	SimPacket* pop(bool _retry = true);

	/**
	 * @brief Get current packet value without removing from entry
	 *
	 * @return SimPacket* Pointer to buffered packet, nullptr if empty
	 *
	 * @note Non-destructive read - packet remains in entry
	 */
	SimPacket* value();

	/**
	 * @brief Check if entry contains a packet
	 *
	 * @return bool True if entry has packet (can pop), false if empty
	 */
	bool isPopValid() const { return this->entry_ != nullptr; }

	/**
	 * @brief Register target slave port
	 *
	 * Establishes one-to-one connection between this master and a slave.
	 * Required before init().
	 *
	 * @param port Pointer to connected SlavePort
	 *
	 * @note Called automatically by SimPortManager::connectPorts()
	 * @see init()
	 */
	void registerSlavePort(SlavePort* port) { this->slave_port_ = port; }

	/**
	 * @brief Initialize port and verify connection
	 *
	 * Validates that port is connected to a SlavePort. Must be called
	 * after registerSlavePort() and before use.
	 *
	 * @throws Assertion failure if not connected to SlavePort
	 *
	 * @note Called automatically during simulator initialization
	 */
	void init() {
		CLASS_ASSERT_MSG(this->slave_port_, "MasterPort `" + this->name_ + "` doesn't connect to any SlavePort");
	}

	/**
	 * @brief Check if port is in retry state
	 * @return bool True if waiting for retry callback, false otherwise
	 */
	bool isRetry() { return this->is_retry_; }

	/**
	 * @brief Mark port as needing retry
	 *
	 * Sets retry flag and notifies owner (SimPortManager) that port has
	 * pending activity requiring retry callback.
	 *
	 * @note Called internally when SlavePort rejects packet
	 */
	void setRetryFlag();

	/**
	 * @brief Clear retry flag after successful retry
	 *
	 * Called after retry callback successfully sends buffered packet.
	 */
	void finishRetry() { this->is_retry_ = false; }

	/**
	 * @brief Record timestamp when retry flag was set
	 * @param t Simulation tick when retry was triggered
	 */
	void setRetryTimestamp(Tick t) { retryTimestamp = t; }

	/**
	 * @brief Get timestamp of retry flag setting
	 * @return Tick Time when retry was triggered
	 */
	Tick getRetryTimestamp() { return retryTimestamp; }

	/**
	 * @brief Notify owner that port has pending activity
	 *
	 * Informs SimPortManager that this port requires processing
	 * in next iteration (due to retry or other pending operation).
	 */
	void setPendingActivityFlag();

private:
	/// @brief Single-entry buffer holding one packet (nullptr if empty)
	SimPacket* entry_ = nullptr;

	/// @brief Connected slave port (set by registerSlavePort)
	SlavePort* slave_port_ = nullptr;

	/// @brief Retry flag - true if waiting for retry callback
	bool is_retry_ = false;

	/// @brief Timestamp when retry was triggered (for debugging/profiling)
	Tick retryTimestamp;
};

}  // namespace acalsim
