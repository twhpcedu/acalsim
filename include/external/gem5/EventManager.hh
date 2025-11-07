/*
 * Copyright (c) 2000-2005 The Regents of The University of Michigan
 * Copyright (c) 2013 Advanced Micro Devices, Inc.
 * Copyright (c) 2013 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* @file
 * EventManager Interface
 */
#pragma once

#include "external/gem5/EventQueue.hh"

namespace acalsim {

class EventManager {
protected:
	/** A pointer to this object's event queue */
	EventQueue* eventq;

public:
	/**
	 * Event manger manages events in the event queue. Where
	 * you can schedule and deschedule different events.
	 *
	 * @ingroup api_eventq
	 * @{
	 */
	// EventManager(EventManager &em) : eventq(em.eventq) {}
	// EventManager(EventManager *em) : eventq(em->eventq) {}
	EventManager(EventQueue* eq) : eventq(eq) {}
	/** @}*/  // end of api_eventq group

	/**
	 * @ingroup api_eventq
	 */
	EventQueue* eventQueue() const { return eventq; }

	/**
	 * @ingroup api_eventq
	 */
	void schedule(Event& event, Tick when) { eventq->schedule(&event, when); }

	void schedule_async(Event& event, Tick when) { eventq->schedule_async(&event, when); }

	/**
	 * @ingroup api_eventq
	 */
	void deschedule(Event& event) { eventq->deschedule(&event); }

	/**
	 * @ingroup api_eventq
	 */
	void reschedule(Event& event, Tick when, bool always = false) { eventq->reschedule(&event, when, always); }

	/**
	 * @ingroup api_eventq
	 */
	void schedule(Event* event, Tick when) { eventq->schedule(event, when); }

	void schedule_async(Event* event, Tick when) { eventq->schedule_async(event, when); }

	/**
	 * @ingroup api_eventq
	 */
	void deschedule(Event* event) { eventq->deschedule(event); }

	/**
	 * @ingroup api_eventq
	 */
	void reschedule(Event* event, Tick when, bool always = false) { eventq->reschedule(event, when, always); }

	/**
	 * This function is not needed by the usual gem5 event loop
	 * but may be necessary in derived EventQueues which host gem5
	 * on other schedulers.
	 * @ingroup api_eventq
	 */
	void wakeupEventQueue(Tick when = (Tick)-1) { eventq->wakeup(when); }

	void serviceOne() { eventq->serviceOne(); }

	void handleAsyncInsertions() { eventq->handleAsyncInsertions(); }

	void setCurTick(Tick newVal) { eventq->setCurTick(newVal); }
};

}  // end of namespace acalsim
