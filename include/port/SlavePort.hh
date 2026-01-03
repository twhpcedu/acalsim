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
 * @file SlavePort.hh
 * @brief Target port with FIFO queue and arbitration for multiple masters
 *
 * SlavePort implements the target side of port communication, handling requests
 * from multiple MasterPorts with arbitration and queueing.
 *
 * **SlavePort Architecture (Many-to-One):**
 * ```
 * MasterPort 0 ──┐
 * MasterPort 1 ──┤
 * MasterPort 2 ──┼──► Arbiter ──► FIFO Queue ──► SlavePort ──► Target Module
 * MasterPort 3 ──┤                (req_queue)
 * MasterPort N ──┘
 * ```
 *
 * **Arbitration and Sync Flow:**
 * ```
 * sync() called per iteration:
 *   1. Check if req_queue has space (isPushReady())
 *   2. Arbitrate among MasterPort entries (round-robin by default)
 *   3. Winner transfers packet to req_queue
 *   4. Losers may set retry flag
 *
 * Time: T0        T1        T2        T3
 *       |         |         |         |
 * M0 ──►req0      │         │         │
 * M1 ──►req1      │         │         │
 * M2 ──────────►req2       │         │
 *       │         │         │         │
 * sync()│         │         │         │
 *   ├─ Arbiter selects M0 (winner)    │
 *   ├─ M0 packet → req_queue          │
 *   ├─ M1, M2 set retry               │
 *       │         │         │         │
 * sync()          │         │         │
 *   ├─ Arbiter selects M1 (retry)     │
 *   ├─ M1 packet → req_queue          │
 * ```
 *
 * @see MasterPort, Arbiter, FifoQueue, SimPort
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/Arbiter.hh"
#include "common/FifoQueue.hh"
#include "packet/SimPacket.hh"
#include "port/SimPort.hh"

namespace acalsim {

// Forward declaration
class MasterPort;

/**
 * @class SlavePort
 * @brief Target port with arbitrated queue for multiple initiators
 *
 * Sl

avePort accepts requests from multiple connected MasterPorts, arbitrates
 * among them, and queues accepted packets in FIFO order.
 *
 * **Key Features:**
 * - Many-to-one connectivity (multiple masters → one slave)
 * - Arbitration policy (default: round-robin)
 * - FIFO request queue with configurable depth
 * - Automatic retry signaling for blocked masters
 *
 * @note Thread-local - not thread-safe
 */
class SlavePort : public SimPort {
	friend class SimPortManager;

public:
	/**
	 * @brief Construct slave port with queue and arbiter
	 * @param name Port identifier
	 * @param req_queue_size Depth of request FIFO
	 * @param arbiter Arbiter for master port selection
	 */
	SlavePort(const std::string& name, size_t req_queue_size, Arbiter* arbiter);

	~SlavePort();

	/** @brief Pop packet from request queue */
	SimPacket* pop() { return this->req_queue_->pop(); }

	/** @brief Check if queue has packets */
	bool isPopValid() const { return this->req_queue_->isPopValid(); }

	/** @brief Peek front packet without removing */
	SimPacket* front() const { return this->req_queue_->front(); }

	/** @brief Check if request queue empty */
	bool empty() const { return this->req_queue_->empty(); }

	/**
	 * @brief Register master port connection
	 * @param port Master port to connect
	 */
	void addMasterPort(MasterPort* port);

	/**
	 * @brief Arbitrate among master ports and transfer winner
	 * @return Winning packet (transferred to queue)
	 */
	SimPacket* arbitrate();

	/**
	 * @brief Initialize port and arbiter
	 */
	void init();

	/**
	 * @brief Reset queue state
	 */
	void reset() { this->req_queue_->reset(); }

	/**
	 * @brief Synchronize port - arbitrate and queue winner
	 */
	void sync();

	/**
	 * @brief Update queue state
	 */
	void update() { this->req_queue_->update(); }

	/**
	 * @brief Push packet directly to queue (bypass arbitration)
	 * @param packet Packet to enqueue
	 * @return true if queued, false if full
	 */
	bool push(SimPacket* packet);

	/**
	 * @brief Check if queue has space
	 * @return true if can accept packet
	 */
	bool isPushReady() const { return this->req_queue_->isPushReady(); }

	/**
	 * @brief Get current queue occupancy
	 * @return Number of packets in queue
	 */
	size_t getReqQueueSize() const { return this->req_queue_->size(); }

protected:
	/**
	 * @brief Get master port index by name
	 * @param name Master port name
	 * @return Index in m_ports_ vector
	 */
	int getEntryIndex(const std::string& name) const;

private:
	/// @brief Vector of connected master ports
	std::vector<MasterPort*> m_ports_;

	/// @brief Map of port names to indices
	std::unordered_map<std::string, int> port_map_;

	/// @brief FIFO queue for incoming requests
	FifoQueue<SimPacket*>* req_queue_ = nullptr;

	/// @brief Arbiter for selecting among masters
	Arbiter* arbiter_ = nullptr;
};

}  // namespace acalsim
