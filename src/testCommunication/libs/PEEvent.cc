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
 * @file PEEvent.cc
 * @brief PE request event implementation for computation processing
 *
 * This file implements the PEReqEvent class, which processes computation requests
 * at the Processing Element. It demonstrates event-based computation, callback
 * invocation, and the completion of the request-response cycle.
 *
 * **PEReqEvent Role:**
 *
 * PEReqEvent is the "computational work processor" in the system:
 * ```
 * TrafficEvent                     PEReqEvent                    TrafficGenerator
 * (Traffic Injection)             (PE Processing)                (Response Handler)
 *       │                               │                              │
 *   Create Event                        │                              │
 *       │                               │                              │
 *   Send via Channel ──────────►  Framework schedules                  │
 *       │                         event for tick T                     │
 *       │                               │                              │
 *       │                         Tick T arrives                       │
 *       │                               │                              │
 *       │                         process() called                     │
 *       │                               │                              │
 *       │                         Extract a, b, c                      │
 *       │                               │                              │
 *       │                         Compute d = a*b + c                  │
 *       │                               │                              │
 *       │                         Update PERespPacket                  │
 *       │                               │                              │
 *       │                         Invoke callback ─────────────────────►│
 *       │                               │                        PERespHandler
 *       │                               │                        Extract result
 *       │                               │                        Log completion
 * ```
 *
 * **Event Lifecycle:**
 *
 * From creation to completion:
 * ```
 * 1. TrafficEvent::process()
 *    - Creates PEReqPacket(TEST, 200, 2, 400, respPkt)
 *    - Creates PEReqEvent(tid, pe, callback, reqPkt)
 *    - Wraps in EventPacket(event, tick + 5)
 *    - Pushes to channel "DSPE"
 *
 * 2. Framework (Phase 2 - Channel Transfer)
 *    - Transfers EventPacket from TG to PE
 *    - Extracts PEReqEvent from EventPacket
 *    - Schedules event in PE's event queue
 *
 * 3. Framework (Target Tick)
 *    - Invokes PEReqEvent::process()
 *
 * 4. PEReqEvent::process()
 *    - Extracts parameters from PEReqPacket
 *    - Performs computation
 *    - Updates PERespPacket with result
 *    - Invokes caller's callback immediately
 *
 * 5. Callback Execution (same tick)
 *    - TrafficGenerator::PERespHandler() runs
 *    - Extracts and logs result
 *    - Frees response packet
 *
 * 6. Event Cleanup
 *    - Framework deletes PEReqEvent
 *    - PEReqPacket deleted (if not managed)
 * ```
 *
 * **PEReqEvent Inheritance:**
 *
 * PEReqEvent extends CallbackEvent:
 * ```
 * SimEvent (base)
 *   └─ CallbackEvent<void(int, PERespPacket*)>
 *       └─ PEReqEvent
 * ```
 *
 * CallbackEvent provides:
 * - `tid`: Transaction ID
 * - `callee`: Destination simulator (PE)
 * - `callerCallback`: Response callback function
 * - Template parameter: Callback signature
 *
 * **Request Processing Details:**
 *
 * The computation is simple but demonstrates key concepts:
 * ```cpp
 * int a = this->peReqPkt->getA();  // Extract operand A (200)
 * int b = this->peReqPkt->getB();  // Extract operand B (2)
 * int c = this->peReqPkt->getC();  // Extract operand C (400)
 *
 * int d = a * b + c;  // Compute: 200 * 2 + 400 = 800
 *
 * this->peReqPkt->getPERespPkt()->setResult(d);  // Store result
 *
 * callerCallback(this->tid, this->peReqPkt->getPERespPkt());  // Notify
 * ```
 *
 * **Callback Invocation:**
 *
 * The callback mechanism completes the request-response cycle:
 * ```
 * callerCallback(tid, respPkt)
 *   │
 *   └─► Executes lambda defined in TrafficEvent:
 *       [this](int id, PERespPacket* pkt) {
 *           dynamic_cast<TrafficGenerator*>(this->sim)
 *               ->PERespHandler(id, pkt);
 *       }
 *       │
 *       └─► TrafficGenerator::PERespHandler(tid, respPkt)
 *           │
 *           ├─► Logs transaction completion
 *           ├─► Extracts result value
 *           └─► Frees response packet
 * ```
 *
 * **Timing Behavior:**
 *
 * All operations happen in same tick:
 * - No additional latency for computation (could be added)
 * - Callback executes immediately (same tick as computation)
 * - No response delay modeling (callback not via channel)
 *
 * Adding computation latency:
 * ```cpp
 * void PEReqEvent::process() {
 *     // Compute result
 *     int d = a * b + c;
 *     peReqPkt->getPERespPkt()->setResult(d);
 *
 *     // Schedule callback for future tick (model computation delay)
 *     Tick responseTick = top->getGlobalTick() + computeLatency;
 *     auto* respEvent = new ResponseEvent(tid, callback, respPkt);
 *     scheduleEvent(respEvent, responseTick);
 * }
 * ```
 *
 * **Why CallbackEvent?**
 *
 * CallbackEvent simplifies asynchronous operations:
 * - Template parameter defines callback signature
 * - Automatic transaction ID management
 * - Callee tracking for routing
 * - Standard pattern for request-response
 *
 * Alternative (without CallbackEvent):
 * ```cpp
 * class PEReqEvent : public SimEvent {
 * public:
 *     PEReqEvent(int tid, SimBase* pe, std::function<void(...)> cb, ...)
 *         : tid(tid), callee(pe), callback(cb), ... {}
 *
 *     void process() override {
 *         // ... computation ...
 *         callback(tid, respPkt);  // Manual callback invocation
 *     }
 *
 * private:
 *     int tid;
 *     SimBase* callee;
 *     std::function<void(int, PERespPacket*)> callback;
 *     PEReqPacket* reqPkt;
 * };
 * // CallbackEvent provides this boilerplate automatically
 * ```
 *
 * **Error Handling:**
 *
 * Current implementation assumes success. Could add:
 * ```cpp
 * void PEReqEvent::process() {
 *     if (!peReqPkt) {
 *         CLASS_ERROR << "Null request packet";
 *         return;
 *     }
 *
 *     if (peReqPkt->getReqType() != PEReqTypeEnum::TEST) {
 *         CLASS_ERROR << "Unsupported request type";
 *         respPkt->setError(ERR_UNSUPPORTED);
 *         callerCallback(tid, respPkt);
 *         return;
 *     }
 *
 *     // Normal processing...
 * }
 * ```
 *
 * **Resource Modeling:**
 *
 * Could extend with resource contention:
 * ```cpp
 * void PEReqEvent::process() {
 *     if (!pe->hasAvailableResources()) {
 *         // Reschedule for later
 *         Tick retryTick = top->getGlobalTick() + 1;
 *         scheduleEvent(this, retryTick);
 *         return;
 *     }
 *
 *     pe->allocateResources();
 *
 *     // Compute result...
 *     int d = a * b + c;
 *
 *     // Schedule resource release
 *     pe->scheduleResourceRelease(top->getGlobalTick() + processTime);
 *
 *     // Invoke callback
 *     callerCallback(tid, respPkt);
 * }
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Event-Driven Processing**
 *    - Computation triggered by event
 *    - No active polling or blocking
 *    - Integrates with simulation framework
 *
 * 2. **Callback Pattern**
 *    - Asynchronous response notification
 *    - Decouples caller and callee
 *    - Context preserved via closure
 *
 * 3. **Request-Response Protocol**
 *    - Request packet carries parameters
 *    - Response packet carries results
 *    - Transaction ID links request to response
 *
 * **Memory Management:**
 *
 * Ownership and lifecycle:
 * - PEReqEvent: Created by TrafficEvent, deleted by framework after process()
 * - peReqPkt: Created by TrafficEvent, owned by PEReqEvent
 * - PERespPacket: Created by TrafficEvent, owned by caller (freed in callback)
 * - Result storage (int*): Created by TrafficEvent, freed with PERespPacket
 *
 * **Extension Examples:**
 *
 * Multiple operation types:
 * ```cpp
 * void PEReqEvent::process() {
 *     int result;
 *     switch (peReqPkt->getReqType()) {
 *         case PEReqTypeEnum::TEST:
 *             result = a * b + c;
 *             break;
 *         case PEReqTypeEnum::ADD:
 *             result = a + b;
 *             break;
 *         case PEReqTypeEnum::MUL:
 *             result = a * b;
 *             break;
 *         default:
 *             CLASS_ERROR << "Unknown operation";
 *             return;
 *     }
 *     peReqPkt->getPERespPkt()->setResult(result);
 *     callerCallback(tid, peReqPkt->getPERespPkt());
 * }
 * ```
 *
 * Pipeline modeling:
 * ```cpp
 * void PEReqEvent::process() {
 *     CLASS_INFO << "Stage 1: Fetch operands";
 *     int a = peReqPkt->getA();
 *     int b = peReqPkt->getB();
 *     int c = peReqPkt->getC();
 *
 *     CLASS_INFO << "Stage 2: Execute multiply";
 *     int temp = a * b;
 *
 *     CLASS_INFO << "Stage 3: Execute add";
 *     int d = temp + c;
 *
 *     CLASS_INFO << "Stage 4: Write back result";
 *     peReqPkt->getPERespPkt()->setResult(d);
 *
 *     callerCallback(tid, peReqPkt->getPERespPkt());
 * }
 * ```
 *
 * @see PEEvent.hh for class definition
 * @see TrafficEvent.cc for request creation
 * @see TrafficGenerator.cc for response handling
 * @see PEReq.cc for packet definitions
 */

#include "PEEvent.hh"

/**
 * @brief Process PE computation request
 *
 * This method is invoked by the framework when the PEReqEvent is scheduled
 * to execute. It performs the requested computation, updates the response
 * packet, and invokes the caller's callback to notify completion.
 *
 * **Execution Flow:**
 *
 * 1. **Log Event Execution**
 *    ```cpp
 *    CLASS_INFO << "Process PEReqEvent with transaction id: " << tid
 *               << " at Tick=" << top->getGlobalTick();
 *    ```
 *    - Logs event processing for debugging
 *    - Includes transaction ID for tracing
 *    - Shows execution tick for timing analysis
 *
 * 2. **Extract Request Parameters**
 *    ```cpp
 *    int a = this->peReqPkt->getA();  // Operand A (200)
 *    int b = this->peReqPkt->getB();  // Operand B (2)
 *    int c = this->peReqPkt->getC();  // Operand C (400)
 *    ```
 *    - Retrieves computation parameters from request packet
 *    - Type-safe accessors ensure correct data extraction
 *    - Values set by TrafficEvent during request creation
 *
 * 3. **Perform Computation**
 *    ```cpp
 *    int d = a * b + c;  // 200 * 2 + 400 = 800
 *    ```
 *    - Executes requested operation: multiply-accumulate
 *    - Simple integer arithmetic for demonstration
 *    - Could be replaced with complex operations:
 *      - Vector operations
 *      - Matrix multiplications
 *      - Memory accesses
 *      - Custom accelerator functions
 *
 * 4. **Update Response Packet**
 *    ```cpp
 *    this->peReqPkt->getPERespPkt()->setResult(d);
 *    ```
 *    - Writes result to pre-allocated response packet
 *    - Response packet created by caller (TrafficEvent)
 *    - Caller retrieves result via callback parameter
 *
 * 5. **Invoke Callback**
 *    ```cpp
 *    callerCallback(this->tid, this->peReqPkt->getPERespPkt());
 *    ```
 *    - Calls back to TrafficGenerator with result
 *    - Passes transaction ID for matching
 *    - Passes response packet with computed result
 *    - Executes immediately (same tick)
 *
 * **Data Flow:**
 *
 * Complete request-response data flow:
 * ```
 * TrafficEvent:
 *   a = 200, b = 2, c = 400
 *   ├─► PEReqPacket(TEST, 200, 2, 400, respPkt)
 *   └─► PERespPacket(TEST, resultPtr)
 *
 * Channel Transfer:
 *   EventPacket(PEReqEvent(tid=1, callback, reqPkt), tick=6)
 *
 * PEReqEvent::process():
 *   ├─► Extract: a=200, b=2, c=400
 *   ├─► Compute: d = 200 * 2 + 400 = 800
 *   ├─► Store: respPkt->setResult(800)
 *   └─► Callback: callerCallback(1, respPkt)
 *
 * TrafficGenerator::PERespHandler():
 *   ├─► Receive: tid=1, respPkt
 *   ├─► Extract: result = 800
 *   ├─► Log: "Received result: 800"
 *   └─► Free: free(respPkt)
 * ```
 *
 * **Timing Example:**
 *
 * Detailed tick-by-tick execution:
 * ```
 * Tick 1: TrafficEvent #1 executes
 *   - Creates PEReqEvent for tick 6
 *   - Sends via channel
 *
 * Tick 1 (Phase 2): Channel transfer
 *   - EventPacket moved to PE
 *   - PEReqEvent scheduled for tick 6
 *
 * Tick 6: PEReqEvent::process() executes
 *   - Extracts a=200, b=2, c=400
 *   - Computes d=800
 *   - Updates response packet
 *   - Invokes callback
 *   - TrafficGenerator::PERespHandler() runs immediately
 *   - Logs: "Receive PE computation result : 800"
 * ```
 *
 * **CallbackEvent Template:**
 *
 * PEReqEvent inherits from CallbackEvent<void(int, PERespPacket*)>:
 * - Template specifies callback signature
 * - Provides `callerCallback` member
 * - Automatically manages transaction ID (`tid`)
 * - Tracks callee simulator
 *
 * Using CallbackEvent:
 * ```cpp
 * // In PEReqEvent.hh:
 * class PEReqEvent : public CallbackEvent<void(int, PERespPacket*)> {
 * public:
 *     PEReqEvent(int tid, SimBase* callee,
 *                std::function<void(int, PERespPacket*)> callback,
 *                PEReqPacket* reqPkt)
 *         : CallbackEvent(tid, callee, callback), peReqPkt(reqPkt) {}
 * };
 *
 * // In process():
 * callerCallback(tid, respPkt);  // Provided by CallbackEvent
 * ```
 *
 * **Callback Execution Context:**
 *
 * The callback runs in PEReqEvent's context:
 * - Same simulation tick
 * - PE simulator's event processing phase
 * - Before event deletion
 * - Synchronous execution (not scheduled)
 *
 * **Error Conditions:**
 *
 * Potential errors (not currently handled):
 * - peReqPkt is nullptr
 * - Response packet is nullptr
 * - Invalid operation type
 * - Arithmetic overflow
 * - Callback throws exception
 *
 * Robust implementation:
 * ```cpp
 * void PEReqEvent::process() {
 *     try {
 *         if (!peReqPkt || !peReqPkt->getPERespPkt()) {
 *             CLASS_ERROR << "Invalid packet state";
 *             return;
 *         }
 *
 *         int a = peReqPkt->getA();
 *         int b = peReqPkt->getB();
 *         int c = peReqPkt->getC();
 *
 *         // Check for overflow
 *         if (willOverflow(a, b, c)) {
 *             peReqPkt->getPERespPkt()->setError(ERR_OVERFLOW);
 *         } else {
 *             int d = a * b + c;
 *             peReqPkt->getPERespPkt()->setResult(d);
 *         }
 *
 *         callerCallback(tid, peReqPkt->getPERespPkt());
 *     } catch (const std::exception& e) {
 *         CLASS_ERROR << "Exception in PEReqEvent: " << e.what();
 *     }
 * }
 * ```
 *
 * **Performance Modeling:**
 *
 * Adding realistic latency:
 * ```cpp
 * void PEReqEvent::process() {
 *     CLASS_INFO << "Process PEReqEvent...";
 *
 *     int a = peReqPkt->getA();
 *     int b = peReqPkt->getB();
 *     int c = peReqPkt->getC();
 *
 *     // Model multi-cycle operation
 *     int multiplyLatency = 3;  // Multiply takes 3 cycles
 *     int addLatency = 1;       // Add takes 1 cycle
 *     int totalLatency = multiplyLatency + addLatency;
 *
 *     int d = a * b + c;
 *     peReqPkt->getPERespPkt()->setResult(d);
 *
 *     // Schedule callback for future tick (model computation delay)
 *     Tick callbackTick = top->getGlobalTick() + totalLatency;
 *     auto* respEvent = new ResponseCallbackEvent(
 *         tid, callerCallback, peReqPkt->getPERespPkt()
 *     );
 *     scheduleEvent(respEvent, callbackTick);
 * }
 * ```
 *
 * **Extension: Pipelined Processing:**
 *
 * ```cpp
 * void PEReqEvent::process() {
 *     // Stage 1: Issue (current tick)
 *     CLASS_INFO << "Issue computation at tick " << top->getGlobalTick();
 *
 *     // Extract operands
 *     int a = peReqPkt->getA();
 *     int b = peReqPkt->getB();
 *     int c = peReqPkt->getC();
 *
 *     // Check resource availability
 *     if (!pe->canIssueNewOp()) {
 *         // Stall - reschedule for next tick
 *         scheduleEvent(this, top->getGlobalTick() + 1);
 *         return;
 *     }
 *
 *     // Reserve pipeline resources
 *     pe->reservePipeline(MULTIPLY_UNIT, 3);
 *     pe->reservePipeline(ADD_UNIT, 1);
 *
 *     // Schedule completion event
 *     int d = a * b + c;
 *     peReqPkt->getPERespPkt()->setResult(d);
 *
 *     Tick completionTick = top->getGlobalTick() + 4;  // Pipeline depth
 *     auto* completeEvent = new PECompleteEvent(
 *         tid, callerCallback, peReqPkt->getPERespPkt()
 *     );
 *     scheduleEvent(completeEvent, completionTick);
 * }
 * ```
 *
 * @note The method executes in the callee's (PE's) context, but the callback
 *       executes code in the caller's (TrafficGenerator's) context.
 *
 * @see CallbackEvent for base class functionality
 * @see TrafficEvent::process() for request creation
 * @see TrafficGenerator::PERespHandler() for callback implementation
 */
void PEReqEvent::process() {
	CLASS_INFO << "Process PEReqEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();

	int a = this->peReqPkt->getA();
	int b = this->peReqPkt->getB();
	int c = this->peReqPkt->getC();

	int d = a * b + c;

	this->peReqPkt->getPERespPkt()->setResult(d);

	callerCallback(this->tid, this->peReqPkt->getPERespPkt());
}
