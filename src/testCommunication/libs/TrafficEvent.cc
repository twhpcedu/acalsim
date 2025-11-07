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
 * @file TrafficEvent.cc
 * @brief Traffic generation event implementation for PE communication
 *
 * This file implements the TrafficEvent class, which orchestrates the injection
 * of computation requests from TrafficGenerator to PE. It demonstrates event-driven
 * traffic generation, callback creation, packet composition, and channel communication.
 *
 * **TrafficEvent Role:**
 *
 * TrafficEvent is the "traffic injection trigger" in the system:
 * ```
 * TrafficGenerator::init()
 *   │
 *   ├─► Schedule TrafficEvent #1 for tick 1
 *   └─► Schedule TrafficEvent #2 for tick 2
 *
 * Tick 1:
 *   TrafficEvent #1::process()
 *     │
 *     ├─► Create PEReqPacket with parameters
 *     ├─► Create PERespPacket for result
 *     ├─► Create callback lambda
 *     ├─► Create PEReqEvent with callback
 *     ├─► Wrap in EventPacket for PE delivery
 *     └─► Push to channel "DSPE"
 * ```
 *
 * **Event Processing Flow:**
 *
 * When TrafficEvent::process() executes:
 * ```
 * 1. Retrieve downstream PE via getDownStream("DSPE")
 * 2. Allocate result storage (int* _d)
 * 3. Create PERespPacket with result storage
 * 4. Define callback lambda for response handling
 * 5. Create PEReqPacket with computation parameters + callback
 * 6. Create PEReqEvent wrapping PEReqPacket
 * 7. Create EventPacket with target tick (current + 5)
 * 8. Push EventPacket to MasterChannelPort "DSPE"
 * 9. Log traffic injection
 * ```
 *
 * **Detailed Step-by-Step Execution:**
 *
 * ```cpp
 * // Step 1: Get destination PE
 * SimBase* pe = this->sim->getDownStream("DSPE");
 * // Returns pointer to PE simulator registered as downstream
 *
 * // Step 2: Allocate result storage
 * int* _d = new int;
 * // Pre-allocate storage for PE to write result
 *
 * // Step 3: Calculate target execution tick
 * Tick t = top->getGlobalTick() + 5;
 * // Request will be processed 5 ticks in the future
 *
 * // Step 4: Create response packet
 * PERespPacket* peRespPkt = new PERespPacket(PEReqTypeEnum::TEST, _d);
 * // Pre-allocated packet for PE to fill with result
 *
 * // Step 5: Define callback lambda
 * std::function<void(int, PERespPacket*)> callback =
 *     [this](int id, PERespPacket* pkt) {
 *         dynamic_cast<TrafficGenerator*>(this->sim)->PERespHandler(id, pkt);
 *     };
 * // Captures 'this' to access TrafficGenerator context
 * // Will be invoked by PE when computation completes
 *
 * // Step 6: Create request packet
 * PEReqPacket* peReqPkt = new PEReqPacket(
 *     PEReqTypeEnum::TEST,  // Request type
 *     200,                  // a = 200
 *     2,                    // b = 2
 *     400,                  // c = 400
 *     peRespPkt             // Response packet to update
 * );
 * // Request: compute d = 200 * 2 + 400 = 800
 *
 * // Step 7: Create PE request event
 * PEReqEvent* peReqEvent = new PEReqEvent(
 *     this->getTID(),  // Transaction ID
 *     pe,              // Callee (PE simulator)
 *     callback,        // Callback function
 *     peReqPkt         // Request packet
 * );
 * // PEReqEvent will process the computation
 *
 * // Step 8: Wrap in EventPacket for channel delivery
 * EventPacket* eventPkt = new EventPacket(peReqEvent, t);
 * // Specifies when event should execute (tick = current + 5)
 *
 * // Step 9: Send via channel
 * this->sim->pushToMasterChannelPort("DSPE", eventPkt);
 * // Packet placed in channel's master queue
 * // Framework will transfer during Phase 2
 * ```
 *
 * **Packet Nesting Structure:**
 *
 * The final packet sent through the channel has nested structure:
 * ```
 * EventPacket (outermost)
 *   │
 *   ├─ targetTick: globalTick + 5
 *   └─ event: PEReqEvent
 *       │
 *       ├─ transactionID: getTID()
 *       ├─ callee: PE simulator pointer
 *       ├─ callback: lambda function
 *       └─ peReqPacket: PEReqPacket
 *           │
 *           ├─ reqType: TEST
 *           ├─ a: 200
 *           ├─ b: 2
 *           ├─ c: 400
 *           └─ respPacket: PERespPacket
 *               │
 *               ├─ reqType: TEST
 *               └─ resultPtr: int* _d
 * ```
 *
 * **Callback Lambda Explained:**
 *
 * The callback lambda captures context for response handling:
 * ```cpp
 * [this, peRespPkt](int id, PERespPacket* pkt) {
 *     dynamic_cast<TrafficGenerator*>(this->sim)->PERespHandler(id, pkt);
 * };
 * ```
 *
 * Breakdown:
 * - `[this, peRespPkt]`: Capture clause
 *   - `this`: Captures TrafficEvent's 'this' pointer
 *   - `peRespPkt`: Captures response packet pointer (unused in current code)
 * - `(int id, PERespPacket* pkt)`: Parameters
 *   - `id`: Transaction ID passed by PE
 *   - `pkt`: Response packet with result
 * - `this->sim`: Pointer to TrafficGenerator
 * - `dynamic_cast<TrafficGenerator*>`: Safe downcast to access PERespHandler
 * - `PERespHandler(id, pkt)`: Calls response handling method
 *
 * **Transaction ID Management:**
 *
 * Each TrafficEvent has unique transaction ID:
 * ```cpp
 * // In TrafficGenerator::init():
 * int _tID = 1;
 * TrafficEvent* event1 = new TrafficEvent(this, "TG2PE_1", _tID);
 * // Event 1 gets TID = 1
 *
 * _tID = 2;
 * TrafficEvent* event2 = new TrafficEvent(this, "TG2PE_2", _tID);
 * // Event 2 gets TID = 2
 * ```
 *
 * TID used for:
 * - Matching responses to requests
 * - Logging and debugging
 * - Performance tracking (latency per transaction)
 * - Identifying transactions in callbacks
 *
 * **Timing and Scheduling:**
 *
 * Two levels of scheduling:
 * 1. **TrafficEvent scheduling** (by TrafficGenerator)
 *    - Event #1: scheduled for tick 1
 *    - Event #2: scheduled for tick 2
 *
 * 2. **PEReqEvent scheduling** (by TrafficEvent)
 *    - Scheduled for current tick + 5
 *    - If TrafficEvent runs at tick 1, PEReqEvent runs at tick 6
 *    - If TrafficEvent runs at tick 2, PEReqEvent runs at tick 7
 *
 * Timeline example:
 * ```
 * Tick 0: Initialization
 * Tick 1: TrafficEvent #1 → Creates PEReqEvent for tick 6
 * Tick 2: TrafficEvent #2 → Creates PEReqEvent for tick 7
 * Tick 6: PEReqEvent #1 processes, callback invoked
 * Tick 7: PEReqEvent #2 processes, callback invoked
 * ```
 *
 * **Channel Communication Details:**
 *
 * pushToMasterChannelPort() operation:
 * ```
 * 1. EventPacket placed in MasterChannelPort "DSPE" queue
 * 2. During Phase 2 (channel transfer phase):
 *    a. ChannelPortManager reads from master queues
 *    b. Transfers packets to connected slave queues
 *    c. PE's SlaveChannelPort "USTrafficGenerator" receives packet
 * 3. Framework extracts PEReqEvent from EventPacket
 * 4. Schedules PEReqEvent in PE's event queue for target tick
 * ```
 *
 * **Memory Management:**
 *
 * Allocation responsibilities:
 * - `int* _d`: Allocated by TrafficEvent, freed by PERespHandler
 * - `PERespPacket`: Allocated by TrafficEvent, freed by PERespHandler
 * - `PEReqPacket`: Allocated by TrafficEvent, ownership transferred to PEReqEvent
 * - `PEReqEvent`: Allocated by TrafficEvent, auto-deleted after process()
 * - `EventPacket`: Allocated by TrafficEvent, framework handles deletion
 *
 * **Design Patterns:**
 *
 * 1. **Event-Driven Generation**
 *    - Traffic triggered by scheduled events
 *    - Decouples generation logic from simulation loop
 *    - Flexible timing control
 *
 * 2. **Builder Pattern**
 *    - Constructs complex nested packet structure
 *    - Step-by-step object creation
 *    - Clear dependencies between components
 *
 * 3. **Callback Pattern**
 *    - Asynchronous response handling
 *    - Context preservation via closure
 *    - Decouples request and response timing
 *
 * **Extension Examples:**
 *
 * Dynamic traffic generation:
 * ```cpp
 * void TrafficEvent::process() {
 *     // Generate traffic
 *     // ... existing code ...
 *
 *     // Schedule next traffic event (periodic generation)
 *     int nextTID = this->getTID() + 10;
 *     TrafficEvent* nextEvent = new TrafficEvent(
 *         this->sim, "Periodic", nextTID
 *     );
 *     this->sim->scheduleEvent(nextEvent, top->getGlobalTick() + 100);
 * }
 * ```
 *
 * Multiple PE destinations:
 * ```cpp
 * void TrafficEvent::process() {
 *     // Select PE based on load balancing, address, etc.
 *     int peID = selectPE(address);
 *     std::string peName = "DSPE" + std::to_string(peID);
 *     SimBase* pe = this->sim->getDownStream(peName);
 *     // ... create and send request ...
 * }
 * ```
 *
 * @see TrafficEvent.hh for class definition
 * @see TrafficGenerator.cc for event scheduling and response handling
 * @see PEEvent.cc for request processing
 * @see PEReq.cc for packet definitions
 */

#include "TrafficEvent.hh"

#include "PEEvent.hh"
#include "PEReq.hh"

/**
 * @brief Process traffic event and inject request into PE
 *
 * This is the core traffic injection method, invoked by the framework when
 * the scheduled event fires. It creates all necessary packets, events, and
 * callbacks to send a computation request to the PE.
 *
 * **Processing Steps:**
 *
 * 1. **Destination Lookup**
 *    ```cpp
 *    SimBase* pe = this->sim->getDownStream("DSPE");
 *    ```
 *    - Retrieves PE simulator registered as downstream
 *    - "DSPE" is the downstream name used in registerSimulators()
 *    - Returns nullptr if not found (error condition)
 *
 * 2. **Result Storage Allocation**
 *    ```cpp
 *    int* _d = new int;
 *    ```
 *    - Pre-allocates memory for PE's computation result
 *    - PE will write result to this location
 *    - Caller (PERespHandler) responsible for freeing
 *
 * 3. **Target Tick Calculation**
 *    ```cpp
 *    Tick t = top->getGlobalTick() + 5;
 *    ```
 *    - Adds 5-tick delay for request processing
 *    - Models communication + queuing latency
 *    - Adjustable for different latency scenarios
 *
 * 4. **Response Packet Creation**
 *    ```cpp
 *    PERespPacket* peRespPkt = new PERespPacket(PEReqTypeEnum::TEST, _d);
 *    ```
 *    - Pre-allocates response packet
 *    - Contains pointer to result storage
 *    - PE will update this packet with result
 *
 * 5. **Callback Definition**
 *    ```cpp
 *    std::function<void(int, PERespPacket*)> callback =
 *        [this](int id, PERespPacket* pkt) {
 *            dynamic_cast<TrafficGenerator*>(this->sim)->PERespHandler(id, pkt);
 *        };
 *    ```
 *    - Lambda captures TrafficEvent context
 *    - Calls TrafficGenerator::PERespHandler when invoked
 *    - Parameters: transaction ID and response packet
 *    - Note: peRespPkt captured but not used (passes pkt parameter)
 *
 * 6. **Request Packet Creation**
 *    ```cpp
 *    PEReqPacket* peReqPkt = new PEReqPacket(
 *        PEReqTypeEnum::TEST, 200, 2, 400, peRespPkt
 *    );
 *    ```
 *    - Operation: TEST (d = a*b + c)
 *    - Parameters: a=200, b=2, c=400
 *    - Expected result: 200 * 2 + 400 = 800
 *    - Includes response packet pointer
 *
 * 7. **PE Request Event Creation**
 *    ```cpp
 *    PEReqEvent* peReqEvent = new PEReqEvent(
 *        this->getTID(), pe, callback, peReqPkt
 *    );
 *    ```
 *    - Wraps request in CallbackEvent
 *    - Associates transaction ID
 *    - Carries callback and request packet
 *    - Will execute at target tick
 *
 * 8. **Event Packet Creation**
 *    ```cpp
 *    EventPacket* eventPkt = new EventPacket(peReqEvent, t);
 *    ```
 *    - Wraps PEReqEvent for channel transmission
 *    - Specifies target execution tick
 *    - Framework will schedule event at this tick
 *
 * 9. **Logging**
 *    ```cpp
 *    CLASS_INFO << "Issue traffic with transaction id: " << getTID()
 *               << " at Tick=" << top->getGlobalTick();
 *    ```
 *    - Logs traffic injection for debugging
 *    - Includes transaction ID and current tick
 *    - Helps trace request flow
 *
 * 10. **Channel Transmission**
 *     ```cpp
 *     this->sim->pushToMasterChannelPort("DSPE", eventPkt);
 *     ```
 *     - Pushes packet to TrafficGenerator's MasterChannelPort
 *     - Port name "DSPE" matches downstream connection
 *     - Packet queued for Phase 2 transfer
 *     - Will arrive at PE's SlaveChannelPort "USTrafficGenerator"
 *
 * **Execution Context:**
 *
 * This method runs in TrafficGenerator's context:
 * - `this`: Pointer to TrafficEvent
 * - `this->sim`: Pointer to TrafficGenerator (owner)
 * - `top->getGlobalTick()`: Current simulation tick
 *
 * **Error Handling:**
 *
 * Current implementation assumes success. Could add:
 * ```cpp
 * SimBase* pe = this->sim->getDownStream("DSPE");
 * if (!pe) {
 *     CLASS_ERROR << "PE destination not found";
 *     return;
 * }
 * ```
 *
 * **Performance Considerations:**
 *
 * - Multiple allocations per request (response, request, event packets)
 * - Could use object pools for high-frequency traffic
 * - Lambda capture minimal overhead (two pointers)
 * - Channel push is lock-free queue operation
 *
 * **Timing Analysis:**
 *
 * Example for TrafficEvent scheduled at tick 1:
 * ```
 * Tick 1: TrafficEvent::process() executes
 *   - Creates PEReqEvent for tick 6 (current + 5)
 *   - Pushes to channel
 *
 * Tick 1 (Phase 2): Framework transfers channel packets
 *   - EventPacket moved to PE's slave port
 *   - PEReqEvent extracted and scheduled
 *
 * Tick 6: PEReqEvent::process() executes
 *   - Computes result
 *   - Invokes callback
 *   - TrafficGenerator::PERespHandler() runs immediately
 * ```
 *
 * **Modification Examples:**
 *
 * Variable delay:
 * ```cpp
 * Tick t = top->getGlobalTick() + uniformRandom(1, 10);
 * ```
 *
 * Different operation parameters:
 * ```cpp
 * int a = generateAddress();
 * int b = generateSize();
 * int c = generateOffset();
 * PEReqPacket* pkt = new PEReqPacket(PEReqTypeEnum::LOAD, a, b, c, resp);
 * ```
 *
 * @note The method assumes "DSPE" downstream exists and is a valid PE.
 *       Add null checks for production code.
 *
 * @see PEReqEvent::process() for request processing
 * @see TrafficGenerator::PERespHandler() for response handling
 * @see ChannelPortManager for channel transfer mechanism
 */
void TrafficEvent::process() {
	SimBase*                                pe        = this->sim->getDownStream("DSPE");
	int*                                    _d        = new int;
	Tick                                    t         = top->getGlobalTick() + 5;
	PERespPacket*                           peRespPkt = new PERespPacket(PEReqTypeEnum::TEST, _d);
	std::function<void(int, PERespPacket*)> callback  = [this, peRespPkt](int id, PERespPacket* pkt) {
        dynamic_cast<TrafficGenerator*>(this->sim)->PERespHandler(this->getTID(), peRespPkt);
	};
	PEReqPacket* peReqPkt   = new PEReqPacket(PEReqTypeEnum::TEST, 200, 2, 400, peRespPkt);
	PEReqEvent*  peReqEvent = new PEReqEvent(this->getTID(), pe, callback, peReqPkt);
	EventPacket* eventPkt   = new EventPacket(peReqEvent, t);

	CLASS_INFO << "Issue traffic with transaction id: " << this->getTID() << " at Tick=" << top->getGlobalTick();
	this->sim->pushToMasterChannelPort("DSPE", eventPkt);
}
