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
 * @file ChannelPort.hh
 * @brief Thread-safe channel port abstraction for inter-simulator communication
 *
 * ChannelPort provides the connection interface between simulators via SimChannel's
 * dual-queue ping-pong buffers. Defines MasterChannelPort (sender) and SlaveChannelPort
 * (receiver) roles for type-safe, thread-safe message passing.
 *
 * **ChannelPort Architecture:**
 * ```
 * Sender Simulator                                       Receiver Simulator
 * (ChannelPortManager)                                   (ChannelPortManager)
 *         │                                                      │
 *         │                                                      │
 *         ├─► MasterChannelPort                                 │
 *         │      │                                               │
 *         │      ├─ channel_mate ─────────────────────────────►├─ (Points to receiver)
 *         │      │                                               │
 *         │      └─ channel (shared_ptr)                        │
 *         │            │                                         │
 *         │            ▼                                         │
 *         │      ┌──────────────────┐                           │
 *         │      │  SimChannel<T*>  │                           │
 *         │      │  ┌────┬────┐     │                           │
 *         │      │  │PING│PONG│     │                           │
 *         │      │  └────┴────┘     │                           │
 *         │      └──────────────────┘                           │
 *         │            ▲                                         │
 *         │            │                                         │
 *         │      channel (shared_ptr)                           │
 *         │      │                                               │
 *         │      └─ channel_mate ◄──────────────────────────────┤
 *         │      │                                               │
 *         ◄──────┤ SlaveChannelPort                             │
 *                                                                │
 *                                                  (Points to sender)
 * ```
 *
 * **Master vs. Slave Roles:**
 * ```
 * ┌────────────────────┬──────────────────────┬──────────────────────┐
 * │ Feature            │ MasterChannelPort    │ SlaveChannelPort     │
 * ├────────────────────┼──────────────────────┼──────────────────────┤
 * │ Role               │ Sender (Producer)    │ Receiver (Consumer)  │
 * │ Primary Operation  │ push() / <<          │ pop() / >>           │
 * │ SimChannel Access  │ Push-side queue      │ Pop-side queue       │
 * │ Created By         │ ConnectPort(sender)  │ ConnectPort(receiver)│
 * │ Triggers           │ Receiver notification│ N/A                  │
 * └────────────────────┴──────────────────────┴──────────────────────┘
 * ```
 *
 * **Communication Flow with Ping-Pong:**
 * ```
 * Iteration 1 (PING_PUSH_PONG_POP):
 *   MasterChannelPort::push(pkt)
 *       ↓
 *   SimChannel (PING queue) ← pkt
 *       ↓
 *   [Phase 1 Complete - Barrier]
 *       ↓
 *   toggleChannelDualQueueStatus()  (PING ↔ PONG)
 *       ↓
 *   [Phase 2 Begins]
 *       ↓
 *   SlaveChannelPort::pop()
 *       ↓
 *   SimChannel (PONG queue) → pkt (was PING)
 *
 * Iteration 2 (PONG_PUSH_PING_POP):
 *   MasterChannelPort::push(pkt)
 *       ↓
 *   SimChannel (PONG queue) ← pkt
 *       ↓
 *   [Toggle again]
 *       ↓
 *   SlaveChannelPort::pop()
 *       ↓
 *   SimChannel (PING queue) → pkt (was PONG)
 * ```
 *
 * **Key Features:**
 * - **Thread-Safe**: Uses SimChannel's lock-free dual-queue mechanism
 * - **Type-Safe**: TPayload = SimPacket* by default
 * - **Shared Ownership**: Both master and slave share SimChannel via shared_ptr
 * - **Channel Mate**: Each port knows its connected ChannelPortManager
 * - **Stream Operators**: << for push, >> for pop (optional convenience)
 *
 * **Usage Example:**
 * ```cpp
 * // Typically created by ConnectPort(), but can be used manually:
 *
 * class SenderSim : public SimBase, public ChannelPortManager {
 * public:
 *     void sendData() {
 *         auto* port = getMasterChannelPort("output");
 *         auto* packet = new MemoryRequest(addr, data);
 *
 *         // Method 1: Direct push
 *         port->push(packet);
 *
 *         // Method 2: Stream operator
 *         *port << packet;
 *     }
 * };
 *
 * class ReceiverSim : public SimBase, public ChannelPortManager {
 * public:
 *     void handleInboundNotification() override {
 *         auto* port = getSlaveChannelPort("input");
 *
 *         // Check if data available
 *         if (!port->empty()) {
 *             // Method 1: Direct pop
 *             auto* packet = port->pop();
 *
 *             // Method 2: Stream operator
 *             SimPacket* pkt;
 *             *port >> pkt;
 *
 *             processPacket(packet);
 *         }
 *     }
 * };
 *
 * // Connection (in SimTop):
 * ChannelPortManager::ConnectPort(sender, receiver, "output", "input");
 * ```
 *
 * @see SimChannel For underlying dual-queue implementation
 * @see ChannelPortManager For port lifecycle and connection management
 * @see SimPacket For default payload type
 */

#pragma once

#include <memory>

#include "channel/SimChannel.hh"
#include "packet/SimPacket.hh"

namespace acalsim {

// Forward declaration
class ChannelPortManager;

/**
 * @class ChannelPort
 * @brief Base class for channel-based communication ports
 *
 * ChannelPort provides the common infrastructure for MasterChannelPort and
 * SlaveChannelPort, holding shared references to the SimChannel and the
 * connected ChannelPortManager (channel mate).
 *
 * **Design Pattern:**
 * - Protected inheritance prevents direct use as ChannelPort
 * - Forces type-safe usage via MasterChannelPort or SlaveChannelPort
 * - Shared ownership of SimChannel via shared_ptr
 * - Bidirectional reference between connected simulators
 *
 * **Ownership Model:**
 * ```
 * MasterChannelPort                           SlaveChannelPort
 *   │                                              │
 *   ├─ channel_mate ──────────►ChannelPortManager◄┴─ channel_mate
 *   │                          (opposite sim)
 *   │
 *   ├─ channel (shared_ptr)
 *   │      │
 *   │      └──────► SimChannel ◄────────┬─ channel (shared_ptr)
 *   │                                    │
 *   └────────────────────────────────────┘
 *   (Both ports share same SimChannel)
 * ```
 *
 * @note Not meant to be used directly - use MasterChannelPort or SlaveChannelPort
 * @see MasterChannelPort, SlaveChannelPort, SimChannel, ChannelPortManager
 */
class ChannelPort {
public:
	/// @brief Type alias for channel payload (default: SimPacket*)
	using TPayload = SimPacket*;

	/// @brief Type alias for the underlying channel type
	using TSimChannel = SimChannel<TPayload>;

public:
	/**
	 * @brief Construct channel port with mate and shared channel
	 *
	 * Initializes the port with a reference to the connected ChannelPortManager
	 * (the opposite simulator) and a shared pointer to the SimChannel that both
	 * ends use for communication.
	 *
	 * @param _channel_mate Pointer to connected ChannelPortManager (opposite simulator)
	 * @param _channel Shared pointer to SimChannel for bidirectional reference
	 *
	 * @note Typically called by ChannelPortManager::ConnectPort(), not by users
	 * @note Both master and slave ports share the same SimChannel instance
	 */
	ChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel);

	/**
	 * @brief Get connected channel port manager (opposite simulator)
	 *
	 * @return ChannelPortManager* Pointer to the simulator on the other end
	 *
	 * @note Useful for identifying communication partner
	 */
	ChannelPortManager* getChannelMate() const;

	/**
	 * @brief Get shared pointer to underlying SimChannel
	 *
	 * @return std::shared_ptr<TSimChannel> Shared channel used for communication
	 *
	 * @note Both master and slave ports return the same shared_ptr
	 */
	std::shared_ptr<TSimChannel> getChannel() const;

private:
	/// @brief Pointer to connected simulator's ChannelPortManager
	ChannelPortManager* channel_mate;

	/// @brief Shared pointer to SimChannel (dual-queue ping-pong buffer)
	std::shared_ptr<TSimChannel> channel;
};

/**
 * @class MasterChannelPort
 * @brief Sender-side channel port for pushing packets to connected simulator
 *
 * MasterChannelPort implements the producer/sender role in channel-based
 * communication. Uses SimChannel's push-side queue (determined by current
 * ping-pong status) to send packets to the connected receiver.
 *
 * **Role in Communication:**
 * ```
 * Sender Simulator                       Receiver Simulator
 *   MasterChannelPort                     SlaveChannelPort
 *         │                                     │
 *         │  push(packet)                       │
 *         │────────────►SimChannel──────────────►│  pop()
 *         │             (PING/PONG)              │
 *         │                                      │
 *         │  [May trigger notification]          │
 *         │──────────────────────────────────────►│  handleInboundNotification()
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * class CPUSimulator : public SimBase, public ChannelPortManager {
 * public:
 *     void sendMemoryRequest(Addr addr, uint8_t* data, size_t size) {
 *         auto* port = getMasterChannelPort("mem_req");
 *         auto* req = new MemoryRequest(addr, data, size);
 *
 *         // Method 1: Direct push
 *         port->push(req);
 *
 *         // Method 2: Stream operator (equivalent)
 *         *port << req;
 *     }
 * };
 * ```
 *
 * @note Inherits from ChannelPort via protected inheritance
 * @note Thread-safe due to SimChannel's lock-free dual-queue
 * @see SlaveChannelPort, SimChannel, ChannelPortManager
 */
class MasterChannelPort : protected ChannelPort {
public:
	/// @brief Shared pointer type for port ownership
	using SharedPtr = std::shared_ptr<MasterChannelPort>;

public:
	/**
	 * @brief Construct master channel port
	 *
	 * @param _channel_mate Pointer to receiver's ChannelPortManager
	 * @param _channel Shared pointer to SimChannel for communication
	 *
	 * @note Typically created by ChannelPortManager::ConnectPort()
	 */
	MasterChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel);

	/**
	 * @brief Push packet to channel
	 *
	 * Sends packet to SimChannel's push-side queue. Uses current ping-pong
	 * status to determine which queue (PING or PONG). May trigger notification
	 * callback in receiver's handleInboundNotification().
	 *
	 * @param _item Packet to send (typically SimPacket*)
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* port = getMasterChannelPort("output");
	 * auto* pkt = new MemoryRequest(addr, data);
	 * port->push(pkt);
	 * ```
	 *
	 * @note Should be called during Phase 1 (parallel execution)
	 * @note Receiver will pop from opposite queue after toggle
	 * @see SimChannel::operator<<(), SlaveChannelPort::pop()
	 */
	void push(const TPayload& _item);

	/**
	 * @brief Stream operator for pushing packets
	 *
	 * Convenience operator equivalent to push(). Provides stream-like syntax.
	 *
	 * @param _port MasterChannelPort to push to
	 * @param _item Packet to send
	 * @return MasterChannelPort& Reference to port (for chaining)
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* port = getMasterChannelPort("output");
	 * *port << packet1 << packet2 << packet3;
	 * ```
	 */
	friend MasterChannelPort& operator<<(MasterChannelPort& _port, const ChannelPort::TPayload& _item);
};

/**
 * @class SlaveChannelPort
 * @brief Receiver-side channel port for popping packets from connected simulator
 *
 * SlaveChannelPort implements the consumer/receiver role in channel-based
 * communication. Uses SimChannel's pop-side queue (determined by current
 * ping-pong status) to receive packets from the connected sender.
 *
 * **Role in Communication:**
 * ```
 * Sender Simulator                       Receiver Simulator
 *   MasterChannelPort                     SlaveChannelPort
 *         │                                     │
 *         │  push(packet)                       │
 *         │────────────►SimChannel──────────────►│
 *         │             (PING/PONG)              │
 *         │                                      │
 *         │  [Notification triggered]            │
 *         │──────────────────────────────────────►│  handleInboundNotification()
 *         │                                      │      └──► pop()
 *         │                                      │      └──► processPacket(pkt)
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * class CacheSimulator : public SimBase, public ChannelPortManager {
 * public:
 *     void handleInboundNotification() override {
 *         auto* port = getSlaveChannelPort("cpu_req");
 *
 *         // Process all pending requests
 *         while (!port->empty()) {
 *             // Method 1: Direct pop
 *             auto* req = port->pop();
 *
 *             // Method 2: Stream operator (equivalent)
 *             SimPacket* req2;
 *             *port >> req2;
 *
 *             handleCacheRequest(req);
 *         }
 *     }
 * };
 * ```
 *
 * @note Inherits from ChannelPort via protected inheritance
 * @note Thread-safe due to SimChannel's lock-free dual-queue
 * @see MasterChannelPort, SimChannel, ChannelPortManager
 */
class SlaveChannelPort : protected ChannelPort {
public:
	/// @brief Shared pointer type for port ownership
	using SharedPtr = std::shared_ptr<SlaveChannelPort>;

public:
	/**
	 * @brief Construct slave channel port
	 *
	 * @param _channel_mate Pointer to sender's ChannelPortManager
	 * @param _channel Shared pointer to SimChannel for communication
	 *
	 * @note Typically created by ChannelPortManager::ConnectPort()
	 */
	SlaveChannelPort(ChannelPortManager* _channel_mate, std::shared_ptr<TSimChannel> _channel);

	/**
	 * @brief Pop packet from channel
	 *
	 * Retrieves and removes one packet from SimChannel's pop-side queue.
	 * Uses current ping-pong status to determine which queue (opposite of
	 * the push-side queue).
	 *
	 * @return TPayload Packet pointer (typically SimPacket*), nullptr if empty
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* port = getSlaveChannelPort("input");
	 * if (!port->empty()) {
	 *     auto* pkt = port->pop();
	 *     processPacket(pkt);
	 * }
	 * ```
	 *
	 * @note Should be called during Phase 2 (after channel toggle)
	 * @note Returns nullptr if no packets available
	 * @see SimChannel::operator>>(), MasterChannelPort::push(), empty()
	 */
	TPayload pop();

	/**
	 * @brief Check if channel has packets available
	 *
	 * Queries SimChannel's pop-side queue to determine if any packets are
	 * waiting to be consumed.
	 *
	 * @return bool True if queue empty (no packets), false if packets available
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* port = getSlaveChannelPort("input");
	 * while (!port->empty()) {
	 *     auto* pkt = port->pop();
	 *     handlePacket(pkt);
	 * }
	 * ```
	 *
	 * @note Check before pop() to avoid nullptr handling
	 */
	bool empty() const;

	/**
	 * @brief Stream operator for popping packets
	 *
	 * Convenience operator equivalent to pop(). Provides stream-like syntax.
	 *
	 * @param _port SlaveChannelPort to pop from
	 * @param _item Reference to receive popped packet
	 * @return SlaveChannelPort& Reference to port (for chaining)
	 *
	 * **Usage:**
	 * ```cpp
	 * auto* port = getSlaveChannelPort("input");
	 * SimPacket *pkt1, *pkt2, *pkt3;
	 * *port >> pkt1 >> pkt2 >> pkt3;
	 * ```
	 *
	 * @note Sets _item to nullptr if queue empty
	 */
	friend SlaveChannelPort& operator>>(SlaveChannelPort& _port, ChannelPort::TPayload& _item);
};

}  // namespace acalsim
