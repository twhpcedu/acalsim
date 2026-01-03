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
 * @file TrafficEvent.cc
 * @brief Traffic generation events initiating request transactions
 *
 * Implements TrafficEvent (request generation) and TrafficRespEvent (final response processing).
 * These events demonstrate callback-based asynchronous communication patterns with multi-tick
 * latency modeling.
 *
 * **Event Flow:**
 * ```
 * Tick 3: TrafficEvent::process()
 *   → Creates NocReqPacket with callback
 *   → Wraps in NocReqEvent
 *   → Sends via channel to NOC
 *
 * Tick 15: TrafficEvent::NocRespHandler() (via callback)
 *   → Receives NocRespPacket
 *   → Creates TrafficRespEvent
 *   → Sends to self for final processing
 *
 * Tick 16: TrafficRespEvent::process()
 *   → Logs transaction completion
 *   → Adds Chrome trace record
 * ```
 *
 * **Callback Pattern:**
 * ```cpp
 * auto callback = [this](Tick when, int id, NocRespPacket* pkt, SimBase* sim) {
 *     this->NocRespHandler(when, this->id, pkt, sim);
 * };
 * ```
 * Captures transaction context for asynchronous response handling.
 *
 * @see TrafficGenerator For event scheduling
 * @see NocReqEvent For request forwarding
 * @see TrafficRespEvent For final completion
 */

#include "TrafficEvent.hh"

#include "CacheEvent.hh"
#include "CachePacket.hh"
#include "NocEvent.hh"
#include "NocPacket.hh"
#include "SharedData.hh"
#include "TrafficGenerator.hh"

/**
 * @brief Process TrafficEvent - initiate request transaction
 *
 * Entry point for traffic generation. Calls sendNoCEvent() to create and send
 * a request to the NOC.
 *
 * @note Alternative path sendSharedData() is commented out
 */
void TrafficEvent::process() {
	CLASS_INFO << "TrafficEvent Processed.";

	// Test #1
	sendNoCEvent();

	// Test #2
	// sendSharedData();
}

/**
 * @brief Send shared data packet (alternative test path)
 *
 * Demonstrates zero-copy data sharing using SharedContainer and std::shared_ptr.
 * Creates multiple TestSharedData objects, initializes them, and sends via channel.
 *
 * **Steps:**
 * 1. Create shared container
 * 2. Add 3 TestSharedData objects
 * 3. Initialize and configure each object
 * 4. Wrap in SharedDataPacket
 * 5. Send via DSNOC channel
 *
 * @note Currently commented out in process()
 * @see SharedDataPacket For packet implementation
 */
void TrafficEvent::sendSharedData() {
	std::shared_ptr<acalsim::SharedContainer<TestSharedData>> ptr =
	    std::make_shared<acalsim::SharedContainer<TestSharedData>>();

	for (int i = 0; i < 3; i++) {
		ptr->add();

		// initialize the shared data object
		ptr->run(i, &TestSharedData::init);

		// modify each object for testing
		ptr->run(i, &TestSharedData::set, i, 2 << i + 1, i + 1);
		char msg[256];
		sprintf(msg, "-- SharedContainer initialize shared data #");
		CLASS_INFO << std::string(msg);
		// print each object
		ptr->run(i, &TestSharedData::print);
	}
	SharedDataPacket* pkt = new SharedDataPacket(top->getGlobalTick() + 1, ptr);

	sim->pushToMasterChannelPort("DSNOC", pkt);
}

/**
 * @brief Send NOC request event (main test path)
 *
 * Creates memory request packet with callback, wraps in event, and sends to NOC.
 *
 * **Request Creation:**
 * - Address: 0
 * - Size: 256 bytes
 * - Callback: Lambda capturing this->id
 *
 * **Event Wrapping:**
 * - NocReqPacket → NocReqEvent → EventPacket
 * - Target tick: current + 10
 *
 * **Channel Transmission:**
 * - Pushes to MasterChannelPort "DSNOC"
 * - Connects to NocSim.USTrafficGenerator
 *
 * @note Adds Chrome trace with 10-tick duration
 * @see NocRespHandler() Callback destination
 */
void TrafficEvent::sendNoCEvent() {
	CLASS_INFO << "Traffic Event Processed and sendNocEvent with transaction id: " << this->id
	           << " at Tick=" << top->getGlobalTick();
	// TODO: "DSNOC" can be replaced with a variable set in the SimTop::init() function
	// when the connection of simulators is made.
	int _size = 256;
	int _addr = 0;

	auto callback = [this](Tick when, int id, NocRespPacket* pkt, SimBase* sim) {
		this->NocRespHandler(when, this->id, pkt, sim);
	};
	SimBase* noc = sim->getDownStream("DSNOC");
	CLASS_ASSERT(noc);
	top->addChromeTraceRecord(
	    acalsim::ChromeTraceRecord::createCompleteEvent("TrafficGenerator", "NocReqEvent", top->getGlobalTick(), 10));
	NocReqPacket* nocReqPkt   = new NocReqPacket(NocPktTypeEnum::TEST, _addr, _size);
	NocReqEvent*  nocReqEvent = new NocReqEvent(this->id, "tg2Noc", callback, noc, nocReqPkt);
	EventPacket*  eventPkt    = new EventPacket(nocReqEvent, top->getGlobalTick() + 10);

	sim->pushToMasterChannelPort("DSNOC", eventPkt);
}

/**
 * @brief Handle NOC response (callback from NocSim)
 *
 * Invoked asynchronously when response arrives from NOC. Creates final TrafficRespEvent
 * for transaction completion.
 *
 * **Processing:**
 * 1. Log response reception
 * 2. Create TrafficRespEvent with response data
 * 3. Wrap in EventPacket (tick + 1)
 * 4. Send to self via USTrafficGenerator port
 *
 * **Self-Delivery Pattern:**
 * Sends event to own SlaveChannelPort for final processing at next tick.
 *
 * @param when Tick when response should be delivered
 * @param id Transaction ID
 * @param pkt Response packet from NOC
 * @param sim Pointer to TrafficGenerator (this simulator)
 *
 * @see TrafficRespEvent::process() Final processing
 */
void TrafficEvent::NocRespHandler(Tick when, int id, NocRespPacket* pkt, SimBase* sim) {
	CLASS_INFO << "Thread " << std::this_thread::get_id() << "executes NocRespHandler with transaction id: " << this->id
	           << " at Tick=" << top->getGlobalTick();
	top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createCompleteEvent(
	    "NocSim", "NocSim2TrafficGenerator", when, ((TrafficGenerator*)sim)->getRespDelay()));
	TrafficRespEvent* tgRespEvent = new TrafficRespEvent(id, "TEST", pkt);
	EventPacket*      eventPkt    = new EventPacket(tgRespEvent, when + ((TrafficGenerator*)sim)->getRespDelay());
	*(sim->getMasterChannelPort("USTrafficGenerator")) << eventPkt;
}
