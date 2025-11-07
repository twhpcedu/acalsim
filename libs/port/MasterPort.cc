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
 * @file MasterPort.cc
 * @brief MasterPort implementation - initiator port with single-entry buffer and retry mechanism
 *
 * This file implements MasterPort, the initiator side of ACALSim's port-based communication
 * system. MasterPort provides a single-entry buffer with backpressure handling via retry
 * callbacks, enabling AXI-like master-slave protocol semantics.
 *
 * **Single-Entry Buffer Model:**
 * ```
 * MasterPort (Initiator)
 *   │
 *   ├─ entry_: SimPacket* (single slot)
 *   │    │
 *   │    ├─ State: Empty (nullptr)
 *   │    │   └─ Can accept new push()
 *   │    │
 *   │    └─ State: Full (valid packet)
 *   │        └─ push() returns false (backpressure)
 *   │
 *   └─ slave_port_: SlavePort* (bound destination)
 *
 * Push Flow:
 *   push(pkt)
 *     │
 *     ├─ entry_ == nullptr?
 *     │   ├─ YES: entry_ = pkt, return true
 *     │   └─ NO:  return false (full, backpressure)
 *     │
 *     └─ (In Phase 2: SimPortManager syncs entry_ → SlavePort)
 *
 * Pop Flow (by framework in Phase 2):
 *   pop(_retry=false)
 *     │
 *     ├─ ret = entry_
 *     ├─ entry_ = nullptr (clear buffer)
 *     ├─ if (_retry) setRetryFlag()
 *     └─ return ret
 * ```
 *
 * **Backpressure and Retry Mechanism:**
 * When SlavePort cannot accept a packet, MasterPort uses a retry mechanism:
 *
 * ```
 * Scenario: SlavePort is full, cannot accept packet
 *
 * Phase 1 (Current Iteration):
 *   CPU:    push(req_pkt) → returns true (stored in entry_)
 *     │
 *     └─ Packet in MasterPort.entry_
 *
 * Phase 2 (Current Iteration):
 *   SimPortManager::syncSimPort()
 *     │
 *     ├─ Attempt: SlavePort.push(entry_)
 *     │   └─ Returns false (SlavePort full)
 *     │
 *     ├─ MasterPort.pop(_retry=true)
 *     │   ├─ Clear entry_ (nullptr)
 *     │   └─ setRetryFlag()
 *     │        ├─ is_retry_ = true
 *     │        └─ owner_->setPendingActivityFlag()
 *     │             └─ Marks owner SimBase as active
 *     │
 *     └─ Packet returned to CPU (still in CPU's ownership)
 *
 * Phase 1 (Next Iteration):
 *   SimBase::triggerRetryCallback()
 *     │
 *     ├─ SimPortManager::triggerRetryCallback()
 *     │   └─ If is_retry_ == true:
 *     │        └─ Execute user's retry_callback()
 *     │
 *     └─ User re-sends packet: push(req_pkt)
 * ```
 *
 * **Retry Callback Pattern:**
 * ```cpp
 * // User code example:
 * class CPU : public SimBase {
 *     MasterPort* req_port;
 *     SimPacket* pending_request;
 *
 * public:
 *     void init() {
 *         req_port = new MasterPort("req");
 *
 *         // Register retry callback
 *         req_port->setRetryCallback([this](Tick when) {
 *             // Re-attempt send when SlavePort has space
 *             if (!req_port->push(pending_request)) {
 *                 // Still full, retry will be called again
 *             } else {
 *                 pending_request = nullptr;  // Sent successfully
 *             }
 *         });
 *     }
 *
 *     void step() {
 *         if (has_new_request()) {
 *             pending_request = createRequest();
 *             if (!req_port->push(pending_request)) {
 *                 // Port full, will retry via callback
 *             } else {
 *                 pending_request = nullptr;
 *             }
 *         }
 *     }
 * };
 * ```
 *
 * **Owner Activity Flag Propagation:**
 * The setPendingActivityFlag() mechanism ensures SimBase stays active:
 *
 * ```
 * SimBase Activity Tracking:
 *   │
 *   ├─ has_pending_events → Active
 *   ├─ has_inbound_channel_requests → Active
 *   └─ hasPendingActivityInSimPort() → Active
 *        │
 *        └─ MasterPort::setPendingActivityFlag()
 *             └─ owner_->setPendingActivityFlag()
 *                  └─ SimBase marked active for next iteration
 *
 * Purpose:
 *   - Ensure retry callbacks get triggered
 *   - Keep SimBase scheduled in ThreadManager
 *   - Prevent premature simulation termination
 * ```
 *
 * **Memory Management:**
 * ```
 * Destructor (~MasterPort):
 *   if (entry_ != nullptr) {
 *       top->getRecycleContainer()->recycle(entry_);
 *   }
 *
 * Why needed:
 *   - entry_ may contain unsent packet at simulation end
 *   - RecycleContainer requires explicit cleanup
 *   - Prevents memory leak from object pool
 *
 * Typical lifecycle:
 *   1. push(pkt) → entry_ = pkt
 *   2. Phase 2 sync → entry_ moved to SlavePort
 *   3. entry_ = nullptr (ownership transferred)
 *   4. If unsent at end: destructor recycles
 * ```
 *
 * **AXI Protocol Analogy:**
 * MasterPort behavior mirrors AXI handshake protocol:
 *
 * | AXI Signal    | MasterPort Equivalent              | Description            |
 * |---------------|------------------------------------|------------------------|
 * | VALID         | entry_ != nullptr                  | Master has data        |
 * | READY         | SlavePort::push() returns true     | Slave can accept       |
 * | Handshake     | VALID && READY → transfer          | Successful push        |
 * | Backpressure  | VALID && !READY → retry            | Slave full, retry      |
 *
 * **Implementation Functions:**
 *
 * 1. **Constructor (line 25):**
 *    - Initialize entry_ = nullptr (empty buffer)
 *    - Initialize slave_port_ = nullptr (unbound)
 *    - SimPortManager::bind() will set slave_port_ later
 *
 * 2. **Destructor (lines 27-29):**
 *    - Recycle any unsent packet in entry_
 *    - Prevent memory leak from RecycleContainer
 *    - Called at simulation teardown
 *
 * 3. **push(pkt) (lines 31-38):**
 *    - If entry_ empty: Store packet, return true
 *    - If entry_ full: Return false (backpressure)
 *    - User must check return value and handle retry
 *
 * 4. **pop(_retry) (lines 40-45):**
 *    - Pop entry_, clear to nullptr
 *    - If _retry=true: Call setRetryFlag()
 *    - Called by SimPortManager::syncSimPort() in Phase 2
 *
 * 5. **value() (line 47):**
 *    - Peek at entry_ without popping
 *    - Used for debugging, state inspection
 *
 * 6. **setRetryFlag() (lines 49-52):**
 *    - Set is_retry_ = true (callback will trigger next iteration)
 *    - Notify owner to stay active
 *    - Ensures SimBase scheduled in Phase 1
 *
 * 7. **setPendingActivityFlag() (line 54):**
 *    - Propagate activity to owner SimBase
 *    - Called when port has pending transactions
 *    - Part of global activity tracking system
 *
 * **Usage in Two-Phase Execution:**
 * ```
 * Phase 1 (Parallel):
 *   - User calls push(pkt) in SimBase::step()
 *   - Packet stored in entry_ (if space available)
 *   - User-defined retry callback may be triggered
 *
 * Phase 2 (Synchronization):
 *   - SimPortManager::syncSimPort()
 *     ├─ For each MasterPort:
 *     │   ├─ If entry_ != nullptr:
 *     │   │   ├─ success = SlavePort::push(entry_)
 *     │   │   ├─ If success: pop(_retry=false)
 *     │   │   └─ If fail: pop(_retry=true)
 *     │   └─ Transfer packet ownership
 *     └─ Advance to next iteration
 * ```
 *
 * @see MasterPort.hh For interface documentation
 * @see SlavePort.cc For receiver side implementation
 * @see SimPortManager.cc For port synchronization orchestration
 * @see SimBase.cc For triggerRetryCallback() mechanism
 */

#include "port/MasterPort.hh"

#include "container/RecycleContainer/RecycleContainer.hh"
#include "port/SimPortManager.hh"
#include "sim/SimTop.hh"

namespace acalsim {

MasterPort::MasterPort(const std::string& name) : SimPort(name), entry_(nullptr), slave_port_(nullptr) {}

MasterPort::~MasterPort() {
	if (this->entry_) { top->getRecycleContainer()->recycle(this->entry_); }
}

bool MasterPort::push(SimPacket* pkt) {
	if (this->entry_ == nullptr) {
		entry_ = pkt;
		return true;
	}

	return false;
}

SimPacket* MasterPort::pop(bool _retry) {
	auto ret = entry_;
	entry_   = nullptr;
	if (_retry) { this->setRetryFlag(); }
	return ret;
}

SimPacket* MasterPort::value() { return entry_; }

void MasterPort::setRetryFlag() {
	this->is_retry_ = true;
	this->owner_->setPendingActivityFlag();
}

void MasterPort::setPendingActivityFlag() { this->owner_->setPendingActivityFlag(); }

}  // namespace acalsim
