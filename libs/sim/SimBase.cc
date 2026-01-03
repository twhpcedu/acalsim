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
 * @file SimBase.cc
 * @brief SimBase core implementation - simulator lifecycle and synchronization
 *
 * This file implements the core simulation execution logic for SimBase, including:
 * - Two-phase parallel execution model (Phase 1: computation, Phase 2: sync)
 * - Event queue processing with recycling optimization
 * - Channel-based inter-simulator communication
 * - Module management and port synchronization
 * - Activity tracking for thread pool coordination
 *
 * **Implementation Architecture:**
 * ```
 * SimBase Execution Flow:
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     stepWrapper()                           │
 * │  Called by thread pool worker in parallel phase            │
 * └────────────────┬────────────────────────────────────────────┘
 *                  │
 *                  ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ Phase 1: Inter-iteration Update & Hardware Modeling        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. Process inbound channel requests                         │
 * │    - Drain SlaveChannelPorts from other simulators          │
 * │    - Drain fromTopChannel from SimTop                       │
 * │    - Call accept() for each received SimPacket              │
 * │                                                             │
 * │ 2. Process event queue                                      │
 * │    - Drain all events scheduled for current tick            │
 * │    - Handle ExitEvents (simulation termination)             │
 * │    - Recycle events via RecycleContainer                    │
 * │                                                             │
 * │ 3. Trigger retry callbacks                                  │
 * │    - Handle backpressure retry for MasterPorts              │
 * │                                                             │
 * │ 4. Execute user-defined step()                              │
 * │    - Hardware modeling logic runs here                      │
 * │    - Can generate events, packets, port transactions        │
 * └─────────────────────────────────────────────────────────────┘
 *                  │
 *                  ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ Phase 2: Synchronization (interIterationUpdate())          │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. Check pending activity status                            │
 * │    - Inbound channel requests                               │
 * │    - Pending events in event queue                          │
 * │    - Activity in SimPorts                                   │
 * │                                                             │
 * │ 2. Update SimTop activity bitmask                           │
 * │    - Positive edge: setPendingEventBitMask()                │
 * │    - Negative edge: clearPendingEventBitMask()              │
 * │                                                             │
 * │ 3. Sync ports and prepare for next iteration               │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Memory Management Strategy:**
 * - All events and packets use RecycleContainer for object pooling
 * - ExitEvents have Managed flag cleared, require manual recycling
 * - Destructor explicitly cleans up unmanaged events before EventQueue deletion
 * - Module deletion handled in destructor (modules vector owns pointers)
 *
 * **Channel Communication Implementation:**
 * - sendPacketViaChannel() supports three delay models:
 *   1. No delay (remoteDelay=0, localDelay=0): Immediate push to channel
 *   2. Local delay (localDelay!=0): Schedule DelayEvent to push later
 *   3. Remote delay (remoteDelay!=0): Wrap in ChannelEventPacket for remote scheduling
 *
 * **Activity Tracking for Thread Pool:**
 * - interIterationUpdate() maintains bitmask in SimTop
 * - Edge detection prevents redundant bitmask updates
 * - Thread pool uses bitmask to schedule only active simulators
 *
 * @see SimBase.hh For detailed API documentation
 * @see SimTop.cc For global coordination and thread pool management
 * @see ThreadManager.cc For parallel execution implementation
 */

#include "sim/SimBase.hh"

#include <memory>
#include <vector>

#include "container/RecycleContainer/RecycleContainer.hh"
#include "packet/EventPacket.hh"
#include "port/SimPortManager.hh"
#include "sim/SimModule.hh"
#include "sim/SimTop.hh"
#include "utils/Logging.hh"

namespace acalsim {

void DelayEvent::process() {
	if (isChannelReq) {
		this->callee->pushToMasterChannelPort(dsChannelPortName, pkt);
	} else {
		this->callee->accept(acalsim::top->getGlobalTick(), (SimPacket&)*this->getPacket());
	}
}
void ChannelEventPacket::visit(Tick when, SimBase& simulator) {
	if (auto delayEvent = dynamic_cast<DelayEvent*>(this->getEvent())) { delayEvent->setCallee(&simulator); }
	simulator.handler(when, this);
}

SimBase::SimBase(std::string _name)
    : EventManager(new EventQueue("Private EventQueue")),
      LinkManager<SimBase*>(),
      SimPortManager(_name),
      name(_name),
      bPendingActivityLastIteration(false),
      bInboundChannelReq(false),
      bInboundChannelReqLastIteration(false),
      bForceStepInNextIteration(false),
      bStepInCurrIteration(false) {
	CLASS_ASSERT_MSG(!name.empty(), "Invalid arguments.");
}

SimBase::~SimBase() {
	for (auto& module : this->modules) { delete module; }

	// Clean up any remaining ExitEvents before deleting the event queue.
	// ExitEvents have the Managed flag cleared, so they won't be automatically
	// recycled when the EventQueue destructor calls deschedule(). We need to
	// manually recycle them to prevent memory leaks.
	if (this->eventq && !this->eventq->empty()) {
		auto recycleContainer = std::static_pointer_cast<RecycleContainer>(top->getRecycleContainer());
		std::vector<SimEvent*> exitEvents;

		// Collect all ExitEvents by repeatedly checking the head of the queue
		while (!this->eventq->empty()) {
			auto event = this->eventq->getHead();
			if (event && event->isExitEvent()) { exitEvents.push_back(static_cast<SimEvent*>(event)); }
			// Deschedule removes the event from the queue
			this->eventq->deschedule(event);
		}

		// Now manually recycle all ExitEvents (deschedule won't recycle unmanaged events)
		for (auto exitEvent : exitEvents) { recycleContainer->recycle(exitEvent); }
	}

	delete this->eventq;
}

void SimBase::stepWrapper() {
	MT_DEBUG_CLASS_INFO << "Phase 1 starts";

	/* ------------------------------------------------------------
	 * [Phase #1 Inter-iteration Framework Update phase START ]
	 * 1. Handle InBound Channel Requests
	 * ------------------------------------------------------------
	 */
	// The assumption is to drain all the requests from all the inbound channels
	this->processInBoundChannelRequest();

	/* ------------------------------------------------------------
	 * [Phase #1 Inter-iteration Framework Update phase CONTINUE ]
	 * 2. Handle EventQ
	 * ------------------------------------------------------------
	 */
	// process Events in SimBase::EventQueue. Drain events expired at current tick
	if (!this->eventQueueEmpty()) {
		// Process the events
		Tick curTick = top->getGlobalTick();
		do {
			if (auto event = this->drainEventQueue(curTick)) {
				MT_DEBUG_CLASS_INFO << "Reach ExitEvent. Terminate the simulation.";
				std::static_pointer_cast<RecycleContainer>(top->getRecycleContainer())->recycle(event);
				return;
			}
		} while (!this->eventQueueEmpty() && this->getEventNextTick() == curTick);

		MT_DEBUG_CLASS_INFO << "Finish processing events expired at current cycle.";
	}

	this->triggerRetryCallback();

	/* ------------------------------------------------------------
	 * [Phase #1 Inter-iteration Hardware Modeling ] :
	 * - Internal Simulator' step() function
	 * ------------------------------------------------------------
	 */

	// reset the flag to avoid inefficient steps
	this->bForceStepInNextIteration = false;
	// user can defined their own step function for each simulator
	this->step();
}

void SimBase::processInBoundChannelRequest() {
	// pop a packet pointer from other simulator through inChannel
	for (auto& inChannelPort : this->slaveChannelPorts) { this->drainInBoundChannelRequest(inChannelPort); }
	// pop a packet pointer from SimTop through fromTopChannel
	this->drainInBoundChannelRequest(this->fromTopChannelPort);
}

void SimBase::drainInBoundChannelRequest(SlaveChannelPort::SharedPtr channel_port) {
	while (!channel_port->empty()) {
		void* ptr = channel_port->pop();
		this->processInBoundPacket(ptr);
	}
}

void SimBase::processInBoundPacket(void* ptr) {
	// route SimPacket to the accept() function for the users
	// to write their own visit() function for the corresponding packet
	this->accept(top->getGlobalTick(), *reinterpret_cast<SimPacket*>(ptr));
}

void SimBase::addModule(SimModule* module) {
	module->setID(this->modules.size());
	module->setSimID(id);
	module->setSimulator(this);

	auto name     = module->getName();
	auto existing = this->modules.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!existing, "SimMoudle `" + name + "` already exists!");
	this->modules.insert(std::make_pair(name, module));
}

SimModule* SimBase::getModule(std::string name) const {
	auto iter = this->modules.getUMapRef().find(name);
	CLASS_ASSERT_MSG(iter != this->modules.getUMapRef().end(), "The module \'" + name + "\' does not exist.");
	return iter->second.get();
}

void SimBase::initSimPort() {
	this->SimPortManager::initSimPort();
	for (auto& mod : this->modules) mod->initSimPort();
}

void SimBase::syncSimPort() {
	this->SimPortManager::syncSimPort();
	for (auto& mod : this->modules) mod->syncSimPort();
}

void SimBase::triggerRetryCallback() {
	this->SimPortManager::triggerRetryCallback(top->getGlobalTick());
	for (auto& mod : this->modules) mod->triggerRetryCallback(top->getGlobalTick());
}

bool SimBase::hasPendingActivityInSimPort(bool pipeRegisterDump) const {
	if (this->SimPortManager::hasPendingActivityInSimPort(pipeRegisterDump)) return true;
	for (auto& mod : this->modules)
		if (mod->hasPendingActivityInSimPort(pipeRegisterDump)) return true;
	return false;
}

void SimBase::clearHasPendingActivityInSimPortFlag() {
	this->SimPortManager::clearHasPendingActivityInSimPortFlag();
	for (auto& mod : this->modules) mod->clearHasPendingActivityInSimPortFlag();
}

Tick SimBase::getSimNextTick() {
	Tick next_tick = -1;

	if (this->getStepInCurrIteration() || this->isForceStepInNextIteration()) {
		// has packets in the inbound channels or slave port.
		next_tick = top->isRunning() ? top->getGlobalTick() + 1 : 0;
	} else {
		next_tick = this->eventQueueEmpty() ? -1 : this->getEventNextTick();
	}

	VERBOSE_CLASS_INFO << "next_tick : " << next_tick;
	return next_tick;
}

void SimBase::issueExitEvent(Tick when) {
	auto exitEvent = top->getRecycleContainer()->acquire<ExitEvent>(&ExitEvent::renew, this->name);
	CLASS_ASSERT(exitEvent->isExitEvent());
	this->scheduleEvent(exitEvent, when);
}

void SimBase::sendPacketViaChannel(std::string dsChannelPortName, Tick localDelay, Tick remoteDelay, SimPacket* pkt) {
	// 1.remoteDelay == 0 && localDelay == 0
	//     - Push the packet to the downstream channel right away and
	//       pop it out for processing in the next cycle

	// 2. localDelay!=0  - create a DelayEvent. Stuff the packet into the event first.
	//          Schedule the event to the eventqueue. When Tick expires, push it to the channel
	//          Pop it out from the channel for processing in the next cycle in the subsequent cycle

	// 3. remoteDelay!=0 -
	// Push the packet to the downstream channel right away.
	// Pop out the packet at the receiver side and process the packet in a future time Tick later than 𝑇.
	// In this case, we need to create a DownstreamEvent. Stuff the packet into the event first.
	// Use an EventPacket to pack the packet and push it to the channel connected to the downstream simulator.
	// When the downstream simulator pop it out from the channel, for the EventPacket,
	// it will detect the EventPacket Type and schedule it to the downstream simulator’s eventQueue.

	const auto rc = top->getRecycleContainer();

	Tick       now        = top->getGlobalTick();
	SimPacket* channelPkt = pkt;

	if (remoteDelay != 0) {
		auto downstreamEvent =
		    rc->acquire<DelayEvent>(static_cast<void (DelayEvent::*)(SimPacket*)>(&DelayEvent::renew), pkt);
		channelPkt = rc->acquire<ChannelEventPacket>(&ChannelEventPacket::renew, downstreamEvent,
		                                             now + localDelay + 1 + remoteDelay);
	}

	if (localDelay == 0) {
		this->pushToMasterChannelPort(dsChannelPortName, channelPkt);
	} else {
		auto delayEvent = rc->acquire<DelayEvent>(
		    static_cast<void (DelayEvent::*)(SimBase*, std::string, SimPacket*)>(&DelayEvent::renew), this,
		    dsChannelPortName, channelPkt);
		this->scheduleEvent(delayEvent, now + localDelay);
	}
}

void SimBase::handler(int when, ChannelEventPacket* packet) {
	this->_scheduleEvent(packet->getEvent(), packet->getWhen());
	top->getRecycleContainer()->recycle(packet);
}

bool CPPSimBase::interIterationUpdate() {
	// inbound request in Inbound SimChannel and SlavePort.
	// hasInboundChannelReq() shows whether there is a pending request in the inbound channels
	// hasPendingActivityInSlavePort() shows whether there is a pending request in the slavePorts
	bool inboundRequestNotDone = this->hasInboundChannelReq() || this->hasPendingActivityInSimPort(true);

	// eventQueueEmpty() shows whether there is a pending event in the event queue
	bool notDone = (!this->eventQueueEmpty() || inboundRequestNotDone || this->isForceStepInNextIteration());

	if (notDone && !this->hasPendingActivityLastIteration()) {  // capture positive edge transition
		// Only set PendingEventBitMask when it changes on the positive edge transition
		top->setPendingEventBitMask(this->getID());
		VERBOSE_CLASS_INFO << name << ": SetPendingEventBitMask";
	}

	if (this->hasPendingActivityLastIteration() && !notDone) {  // capture negative edge transition
		// Only clear PendingEventBitMask when it changes on the negative edge transition
		top->clearPendingEventBitMask(this->getID());
		VERBOSE_CLASS_INFO << name << ": ClearPendingEventBitMask";
	}

	this->setInboundChannelReqLastIteration(this->hasInboundChannelReq());
	this->setPendingActivityLastIteration(notDone);
	this->setInboundChannelReq(false);
	this->setStepInCurrIteration(inboundRequestNotDone || this->isForceStepInNextIteration());

	this->clearHasPendingActivityInSimPortFlag();

	return inboundRequestNotDone;
}

}  // end of namespace acalsim
