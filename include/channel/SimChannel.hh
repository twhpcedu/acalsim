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
 * @file SimChannel.hh
 * @brief Thread-safe dual-queue channel for inter-simulator communication
 *
 * SimChannel implements a ping-pong buffer pattern using two queues that
 * alternate roles between iterations, enabling lock-free communication between
 * simulator threads running in different phases.
 *
 * **Dual-Queue Ping-Pong Architecture:**
 * ```
 * Iteration N:                    Iteration N+1:
 *
 * Phase 1 (Parallel Execution):   Phase 1 (Parallel Execution):
 *   Producer → PING Queue             Producer → PONG Queue
 *   Consumer ← PONG Queue             Consumer ← PING Queue
 *            (push)    (pop)                  (push)    (pop)
 *
 * Phase 2 (Sync Barrier):         Phase 2 (Sync Barrier):
 *   Toggle queue roles                Toggle queue roles
 *   PING ↔ PONG                       PONG ↔ PING
 * ```
 *
 * **Status Toggle Pattern:**
 * ```
 * Iteration:  1         2         3         4
 * Status:     PING→PONG PONG→PING PING→PONG PONG→PING
 *             ↓         ↓         ↓         ↓
 * Push to:    Ping      Pong      Ping      Pong
 * Pop from:   Pong      Ping      Pong      Ping
 * ```
 *
 * **Key Features:**
 * - **Lock-Free**: No mutexes - uses dual buffering
 * - **Phase Separation**: Producers and consumers use different queues
 * - **Global Coordination**: SimChannelGlobal toggles all channels simultaneously
 * - **Template-Based**: Works with any packet/message type
 *
 * **Usage Example:**
 * ```cpp
 * // In simulator A (sender):
 * SimChannel<MemoryRequest*> reqChannel;
 * auto* req = new MemoryRequest(addr, data);
 * reqChannel << req;  // Push to current ping/pong queue
 *
 * // After Phase 1 barrier:
 * SimChannelGlobal::toggleChannelDualQueueStatus();
 *
 * // In simulator B (receiver - Phase 2):
 * MemoryRequest* req;
 * if (reqChannel.nonEmptyForPop()) {
 *     reqChannel >> req;  // Pop from swapped queue
 *     processRequest(req);
 * }
 * ```
 *
 * @see ChannelPort, ChannelPortManager, SimBase
 */

#pragma once

#include <msd/channel.hpp>

namespace acalsim {

/**
 * @enum SimChannelStatus
 * @brief Indicates which queue is for push and which for pop
 *
 * Global state controlling all SimChannels' buffer roles.
 * Toggled at iteration boundaries to swap producer/consumer queues.
 */
enum class SimChannelStatus {
	PING_PUSH_PONG_POP,  ///< Push to ping queue, pop from pong queue
	PONG_PUSH_PING_POP   ///< Push to pong queue, pop from ping queue
};

/**
 * @class SimChannelGlobal
 * @brief Global coordinator for dual-queue status across all channels
 *
 * Manages the ping-pong toggle state shared by all SimChannel instances.
 * Toggled once per iteration by ThreadManager or SimTop after Phase 1.
 */
class SimChannelGlobal {
private:
	/// @brief Global ping-pong state (shared by all SimChannels)
	static inline SimChannelStatus channelDualQueueStatus = SimChannelStatus::PING_PUSH_PONG_POP;

public:
	/**
	 * @brief Get current dual-queue status
	 * @return SimChannelStatus Current push/pop queue assignment
	 */
	static inline SimChannelStatus getChannelDualQueueStatus();

	/**
	 * @brief Toggle ping-pong status (swap producer/consumer queues)
	 *
	 * Called once per iteration at Phase 1/Phase 2 boundary.
	 * Affects all SimChannel instances globally.
	 *
	 * @note Must be called by single control thread (SimTop)
	 */
	static inline void toggleChannelDualQueueStatus();
};

/**
 * @class SimChannel
 * @brief Template-based dual-queue channel for thread-safe message passing
 *
 * Provides lock-free inter-simulator communication using ping-pong buffers.
 * Producer and consumer use separate queues that swap roles each iteration.
 *
 * **Thread Safety:**
 * - Producer thread: Pushes to one queue (ping or pong)
 * - Consumer thread: Pops from other queue (pong or ping)
 * - No contention between push/pop operations
 * - Global toggle occurs at synchronization barrier
 *
 * @tparam T Message type (typically pointers: SimPacket*, MemoryRequest*, etc.)
 *
 * @note Uses msd::channel (lock-free SPSC queue) internally
 */
template <typename T>
class SimChannel {
private:
	msd::channel<T>* pingQueue;  ///< First buffer in ping-pong pair
	msd::channel<T>* pongQueue;  ///< Second buffer in ping-pong pair

	/**
	 * @brief Get queue for push based on current global status
	 * @return msd::channel<T>* Queue to push to (ping or pong)
	 */
	msd::channel<T>* getQueueForPush();

	/**
	 * @brief Get queue for pop based on current global status
	 * @return msd::channel<T>* Queue to pop from (pong or ping)
	 */
	msd::channel<T>* getQueueForPop();

public:
	/**
	 * @brief Construct dual-queue channel
	 */
	SimChannel();

	/**
	 * @brief Destructor - cleans up both queues
	 */
	~SimChannel();

	/**
	 * @brief Push element to channel (stream operator)
	 * @param ch Channel to push to
	 * @param in Element to push
	 * @return Reference to channel (for chaining)
	 */
	template <typename TT>
	friend SimChannel<typename std::decay<TT>::type>& operator<<(SimChannel<typename std::decay<TT>::type>& ch,
	                                                             const TT&                                  in);

	/**
	 * @brief Pop element from channel (stream operator)
	 * @param ch Channel to pop from
	 * @param out Output variable to receive element
	 * @return Reference to channel (for chaining)
	 */
	template <typename TT>
	friend SimChannel<TT>& operator>>(SimChannel<TT>& ch, TT& out);

	/**
	 * @brief Check if channel has elements available for pop
	 * @return bool True if pop queue non-empty, false otherwise
	 */
	bool nonEmptyForPop();
};

}  // end of namespace acalsim

#include "channel/SimChannel.inl"
