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

#pragma once

#include <functional>

#include "event/SimEvent.hh"

namespace acalsim {

/**
 * @file CallbackEvent.hh
 * @brief Template event class with callback function and transaction tracking
 *
 * @details
 * CallbackEvent extends SimEvent to support callback-based event processing
 * with transaction ID tracking and caller context. This is essential for
 * modeling asynchronous operations where the response must be routed back
 * to the original requestor.
 *
 * **Callback Model:**
 * ```
 * Caller                          Callee
 *   |                               |
 *   |--- Issue Request (tid=42) --->|
 *   |                               |
 *   |                           [Processing]
 *   |                               |
 *   |<-- Callback (tid=42) ---------|
 *   |                               |
 * [Match tid=42, process response]
 * ```
 *
 * **Key Features:**
 *
 * - **Transaction ID**: Unique identifier for request/response matching
 * - **Caller Context**: Void pointer to caller object for context
 * - **Callback Function**: std::function for flexible callback handling
 * - **Template-based**: Generic callback signature via template parameter
 * - **Exit Events**: Special flag for simulation termination events
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Example |
 * |----------|-------------|---------|
 * | **Memory Read** | Async read with callback | CPU → Memory → CPU callback |
 * | **NoC Transfer** | Packet delivery notification | Router → Destination → Ack |
 * | **DMA Transfer** | DMA completion callback | DMA start → Transfer → Done callback |
 * | **RPC Calls** | Remote procedure call response | Client → Server → Response |
 * | **Cache Miss** | Miss handling with refill callback | L1 miss → L2 → Refill callback |
 *
 * **Transaction ID Tracking:**
 * ```
 * Outstanding Requests Map:
 * tid=100 → {addr=0x1000, callback=func1, context=cpu0}
 * tid=101 → {addr=0x2000, callback=func2, context=cpu1}
 * tid=102 → {addr=0x3000, callback=func3, context=cpu0}
 *
 * On response (tid=101):
 *   → Lookup tid=101
 *   → Execute callback func2
 *   → Remove from map
 * ```
 *
 * **Memory Management:**
 *
 * | Pattern | Managed Flag | Lifecycle |
 * |---------|-------------|-----------|
 * | **Pooled (no callback)** | Managed=true | Auto-recycle after process() |
 * | **With callback** | Managed=false | Callback must call release() |
 * | **Exit event** | Managed=false | Never recycled (termination) |
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | Constructor | O(1) | Clears Managed flag if callback set |
 * | renew() | O(1) | Reset state for recycling |
 * | setExitFlag() | O(1) | Mark as exit event |
 * | process() | Varies | User-defined callback logic |
 *
 * **Memory:** sizeof(CallbackEvent<T>) ≈ sizeof(SimEvent) + 24 bytes (tid + callee + function)
 *
 * **Thread Safety:**
 * - **Callback Execution**: Not thread-safe - caller must ensure safety
 * - **Transaction ID**: Managed by user - ensure uniqueness
 * - **Flag Manipulation**: Not atomic - single-threaded access required
 *
 * @tparam T Callback function signature (typically void() or void(SomeType*))
 *
 * @code{.cpp}
 * // Example: Memory read with callback
 * class MemoryReadEvent : public CallbackEvent<void()> {
 * public:
 *     MemoryReadEvent(uint64_t tid, void* cpu, std::function<void()> cb, uint64_t addr)
 *         : CallbackEvent<void()>(tid, cpu, cb), address(addr) {}
 *
 *     void renew(uint64_t tid, void* cpu, std::function<void()> cb, uint64_t addr) {
 *         CallbackEvent<void()>::renew(tid, cpu, cb);
 *         this->address = addr;
 *     }
 *
 *     void process() override {
 *         // Read memory at address
 *         uint64_t data = memory->read(address);
 *
 *         // Invoke callback to deliver data
 *         if (callerCallback) {
 *             callerCallback();  // CPU processes the read data
 *         }
 *     }
 *
 * private:
 *     uint64_t address;
 * };
 *
 * // Usage in CPU
 * class CPU {
 * public:
 *     void issueRead(uint64_t addr) {
 *         uint64_t tid = nextTID++;
 *
 *         // Create callback to handle response
 *         auto callback = [this, tid, addr]() {
 *             this->handleReadResponse(tid, addr);
 *         };
 *
 *         // Create and schedule event
 *         auto* evt = new MemoryReadEvent(tid, this, callback, addr);
 *         scheduler.schedule(evt, currentTick() + memoryLatency);
 *
 *         // Track outstanding request
 *         outstandingReads[tid] = {addr, currentTick()};
 *     }
 *
 *     void handleReadResponse(uint64_t tid, uint64_t addr) {
 *         // Process the read response
 *         outstandingReads.erase(tid);
 *         LOG_INFO << "Read tid=" << tid << " completed";
 *     }
 *
 * private:
 *     uint64_t nextTID = 0;
 *     std::map<uint64_t, ReadInfo> outstandingReads;
 * };
 *
 * // Example: Exit event for simulation termination
 * class ExitSimEvent : public CallbackEvent<void()> {
 * public:
 *     ExitSimEvent() : CallbackEvent<void()>() {
 *         setExitFlag();  // Mark as exit event
 *     }
 *
 *     void process() override {
 *         LOG_INFO << "Simulation terminating...";
 *         // Cleanup and exit
 *     }
 * };
 * @endcode
 *
 * @note Transaction IDs must be unique across concurrent requests
 * @note Callee pointer is not owned - caller manages lifetime
 * @note Callbacks with captures must manage their own lifetime
 *
 * @warning Setting callback clears Managed flag - caller must call release()
 * @warning Exit events are never recycled - only use for termination
 *
 * @see SimEvent for base event class
 * @see LambdaEvent for simpler lambda-based events
 * @since ACALSim 0.1.0
 */
template <typename T>
class CallbackEvent : public SimEvent {
protected:
	/** @brief Transaction ID for request/response matching */
	uint64_t tid;

	/** @brief Pointer to the callee object (not owned) */
	void* callee = nullptr;

	/** @brief Callback function to invoke on event processing */
	std::function<T> callerCallback = nullptr;

public:
	/**
	 * @brief Construct a callback event with transaction ID, callee, and callback
	 *
	 * @param _tid Transaction ID (default: 0)
	 * @param _callee Pointer to callee object (default: nullptr)
	 * @param _callback Callback function (default: nullptr)
	 *
	 * @note If callback is provided, clears Managed flag (manual release required)
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * auto cb = [this]() { this->handleResponse(); };
	 * auto* evt = new CallbackEvent<void()>(42, this, cb);
	 * @endcode
	 */
	CallbackEvent(uint64_t _tid = 0, void* _callee = nullptr, std::function<T> _callback = nullptr)
	    : SimEvent(), tid(_tid), callee(_callee), callerCallback(_callback) {
		if (_callback) { this->clearFlags(this->Managed); }
	}

	/**
	 * @brief Virtual destructor
	 */
	virtual ~CallbackEvent() {}

	/**
	 * @brief Reset event state for recycling
	 *
	 * @param _tid New transaction ID (default: 0)
	 * @param _callee New callee pointer (default: nullptr)
	 * @param _callback New callback function (default: nullptr)
	 *
	 * @note Called by RecycleContainer before reuse
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * // Recycled event reused with new parameters
	 * CallbackEvent<void()>* evt = pool.get<CallbackEvent<void()>>();
	 * evt->renew(newTid, newCallee, newCallback);
	 * @endcode
	 */
	void renew(uint64_t _tid = 0, void* _callee = nullptr, std::function<T> _callback = nullptr) {
		this->tid            = _tid;
		this->callee         = _callee;
		this->callerCallback = _callback;
	}

	/**
	 * @brief Mark this event as an exit event (simulation termination)
	 *
	 * @note Sets IsExitEvent flag
	 * @note Clears Managed flag (never recycled)
	 * @note Used for simulation termination events
	 *
	 * @code{.cpp}
	 * CallbackEvent<void()>* exitEvt = new CallbackEvent<void()>();
	 * exitEvt->setExitFlag();
	 * scheduler.schedule(exitEvt, exitTick);
	 * @endcode
	 */
	void setExitFlag() {
		this->setFlags(IsExitEvent);
		this->clearFlags(Managed);
	}

	/**
	 * @brief Process the event (pure virtual - must be implemented)
	 *
	 * @note Called by scheduler at scheduled tick
	 * @note Typically invokes callerCallback with response data
	 * @note Transaction ID used to match request/response
	 *
	 * @code{.cpp}
	 * class ResponseEvent : public CallbackEvent<void()> {
	 *     void process() override {
	 *         // Do work
	 *         if (callerCallback) {
	 *             callerCallback();  // Notify caller
	 *         }
	 *     }
	 * };
	 * @endcode
	 */
	virtual void process() = 0;
};

}  // end of namespace acalsim
