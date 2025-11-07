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
 * EventQueue Interface
 */

#pragma once

#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "external/gem5/Event.hh"
#include "external/gem5/base/uncontended_mutex.hh"
#include "utils/Logging.hh"

namespace acalsim {

//! Current number of allocated main event queues.
extern uint32_t numMainEventQueues;

//! Array for main event queues.
extern std::vector<EventQueue*> mainEventQueue;

//! The current event queue for the running thread. Access to this queue
//! does not require any locking from the thread.

extern __thread EventQueue* _curEventQueue;

//! Current mode of execution: parallel / serial
extern bool inParallelMode;

inline EventQueue* curEventQueue() { return _curEventQueue; }
inline void        curEventQueue(EventQueue* q) { _curEventQueue = q; }

class EventQueue {
private:
	friend void curEventQueue(EventQueue*);

	std::string objName;
	Event*      head;
	Tick        _curTick;

	//! Mutex to protect async queue.
	UncontendedMutex async_queue_mutex;

	//! List of events added by other threads to this event queue.
	std::list<Event*> async_queue;

	/**
	 * Lock protecting event handling.
	 *
	 * This lock is always taken when servicing events. It is assumed
	 * that the thread scheduling new events (not asynchronous events
	 * though) have taken this lock. This is normally done by
	 * serviceOne() since new events are typically scheduled as a
	 * response to an earlier event.
	 *
	 * This lock is intended to be used to temporarily steal an event
	 * queue to support inter-thread communication when some
	 * deterministic timing can be sacrificed for speed. For example,
	 * the KVM CPU can use this support to access devices running in a
	 * different thread.
	 *
	 * @see EventQueue::ScopedMigration.
	 * @see EventQueue::ScopedRelease
	 * @see EventQueue::lock()
	 * @see EventQueue::unlock()
	 */
	UncontendedMutex service_mutex;

	//! Insert / remove event from the queue. Should only be called
	//! by thread operating this queue.
	void insert(Event* event);
	void remove(Event* event);

	//! Function for adding events to the async queue. The added events
	//! are added to main event queue later. Threads, other than the
	//! owning thread, should call this function instead of insert().
	void asyncInsert(Event* event);

	EventQueue(const EventQueue&);

public:
	class ScopedMigration {
	public:
		/**
		 * Temporarily migrate execution to a different event queue.
		 *
		 * An instance of this class temporarily migrates execution to
		 * different event queue by releasing the current queue, locking
		 * the new queue, and updating curEventQueue(). This can, for
		 * example, be useful when performing IO across thread event
		 * queues when timing is not crucial (e.g., during fast
		 * forwarding).
		 *
		 * ScopedMigration does nothing if both eqs are the same
		 *
		 * @ingroup api_eventq
		 */
		ScopedMigration(EventQueue* _new_eq, bool _doMigrate = true)
		    : new_eq(*_new_eq), old_eq(*curEventQueue()), doMigrate((&new_eq != &old_eq) && _doMigrate) {
			if (doMigrate) {
				old_eq.unlock();
				new_eq.lock();
				curEventQueue(&new_eq);
			}
		}

		~ScopedMigration() {
			if (doMigrate) {
				new_eq.unlock();
				old_eq.lock();
				curEventQueue(&old_eq);
			}
		}

	private:
		EventQueue& new_eq;
		EventQueue& old_eq;
		bool        doMigrate;
	};

	class ScopedRelease {
	public:
		/**
		 * Temporarily release the event queue service lock.
		 *
		 * There are cases where it is desirable to temporarily release
		 * the event queue lock to prevent deadlocks. For example, when
		 * waiting on the global barrier, we need to release the lock to
		 * prevent deadlocks from happening when another thread tries to
		 * temporarily take over the event queue waiting on the barrier.
		 *
		 * @group api_eventq
		 */
		ScopedRelease(EventQueue* _eq) : eq(*_eq) { eq.unlock(); }

		~ScopedRelease() { eq.lock(); }

	private:
		EventQueue& eq;
	};

	/**
	 * @ingroup api_eventq
	 */
	EventQueue(const std::string& n);

	/**
	 * @ingroup api_eventq
	 * @{
	 */
	virtual const std::string name() const { return objName; }
	void                      name(const std::string& st) { objName = st; }
	/** @}*/  // end of api_eventq group

	/**
	 * Schedule the given event on this queue. Safe to call from any thread.
	 *
	 * @ingroup api_eventq
	 */
	void schedule(Event* event, Tick when, bool global = false) {
#ifdef ACALSIM_VERBOSE
		INFO << std::string(__func__) + ": " + event->name() + ", tick = " << when;
#endif  // ACALSIM_VERBOSE

		assert(when >= getCurTick());
		assert(!event->scheduled());
		assert(event->initialized());
		event->setWhen(when, this);

		// The check below is to make sure of two things
		// a. A thread schedules local events on other queues through the
		//    asyncq.
		// b. A thread schedules global events on the asyncq, whether or not
		//    this event belongs to this eventq. This is required to maintain
		//    a total order amongst the global events. See global_event.{cc,hh}
		//    for more explanation.
		if (inParallelMode && (this != curEventQueue() || global)) {
			asyncInsert(event);
		} else {
			insert(event);
		}
		event->flags.set(Event::Scheduled);
		event->acquire();
	}

	void schedule_async(Event* event, Tick when) {
		std::cout << __func__ << ": " << event->name() << ", tick = " << when << std::endl;
		assert(when >= getCurTick());
		assert(!event->scheduled());
		assert(event->initialized());

		event->setWhen(when, this);
		asyncInsert(event);

		event->flags.set(Event::Scheduled);
		event->acquire();
	}

	/**
	 * Deschedule the specified event. Should be called only from the owning
	 * thread.
	 * @ingroup api_eventq
	 */
	void deschedule(Event* event) {
		std::cout << __func__ << ": " << event->name() << std::endl;
		assert(event->scheduled());
		assert(event->initialized());
		assert(!inParallelMode || this == curEventQueue());

		remove(event);

		event->flags.clear(Event::Squashed);
		event->flags.clear(Event::Scheduled);

		event->release();
	}

	/**
	 * Reschedule the specified event. Should be called only from the owning
	 * thread.
	 *
	 * @ingroup api_eventq
	 */
	void reschedule(Event* event, Tick when, bool always = false) {
		std::cout << __func__ << ": " << event->name() << ", tick = " << when << std::endl;
		assert(when >= getCurTick());
		assert(always || event->scheduled());
		assert(event->initialized());
		assert(!inParallelMode || this == curEventQueue());

		if (event->scheduled()) {
			remove(event);
		} else {
			event->acquire();
		}

		event->setWhen(when, this);
		insert(event);
		event->flags.clear(Event::Squashed);
		event->flags.set(Event::Scheduled);
	}

	Tick nextTick() const { return head->when(); }
	void setCurTick(Tick newVal) { _curTick = newVal; }

	/**
	 * While curTick() is useful for any object assigned to this event queue,
	 * if an object that is assigned to another event queue (or a non-event
	 * object) need to access the current tick of this event queue, this
	 * function is used.
	 *
	 * Tick is the unit of time used in gem5.
	 *
	 * @return Tick The current tick of this event queue.
	 * @ingroup api_eventq
	 */
	Tick   getCurTick() const { return _curTick; }
	Event* getHead() const { return head; }

	Event* serviceOne();

	Event* serviceOne(Tick when);

	/**
	 * process all events up to the given timestamp.  we inline a quick test
	 * to see if there are any events to process; if so, call the internal
	 * out-of-line version to process them all.
	 *
	 * Notes:
	 *  - This is only used for "instruction" event queues. Instead of counting
	 *    ticks, this is actually counting instructions.
	 *  - This updates the current tick value to the value of the entry at the
	 *    head of the queue.
	 *
	 * @ingroup api_eventq
	 */
	void serviceEvents(Tick when) {
		while (!empty()) {
			if (nextTick() > when) break;

			/**
			 * @todo this assert is a good bug catcher.  I need to
			 * make it true again.
			 */
			// assert(head->when() >= when && "event scheduled in the past");
			serviceOne();
		}

		setCurTick(when);
	}

	/**
	 * Returns true if no events are queued
	 *
	 * @ingroup api_eventq
	 */
	bool empty() const { return head == NULL; }

	/**
	 * This is a debugging function which will print everything on the event
	 * queue.
	 *
	 * @ingroup api_eventq
	 */
	void dump() const;

	bool debugVerify() const;

	/**
	 * Function for moving events from the async_queue to the main queue.
	 */
	void handleAsyncInsertions();

	/**
	 *  Function to signal that the event loop should be woken up because
	 *  an event has been scheduled by an agent outside the gem5 event
	 *  loop(s) whose event insertion may not have been noticed by gem5.
	 *  This function isn't needed by the usual gem5 event loop but may
	 *  be necessary in derived EventQueues which host gem5 onto other
	 *  schedulers.
	 *
	 *  @param when Time of a delayed wakeup (if known). This parameter
	 *  can be used by an implementation to schedule a wakeup in the
	 *  future if it is sure it will remain active until then.
	 *  Or it can be ignored and the event queue can be woken up now.
	 *
	 *  @ingroup api_eventq
	 */
	virtual void wakeup(Tick when = (Tick)-1) { (void)when; }

	/**
	 *  function for replacing the head of the event queue, so that a
	 *  different set of events can run without disturbing events that have
	 *  already been scheduled. Already scheduled events can be processed
	 *  by replacing the original head back.
	 *  USING THIS FUNCTION CAN BE DANGEROUS TO THE HEALTH OF THE SIMULATOR.
	 *  NOT RECOMMENDED FOR USE.
	 */
	Event* replaceHead(Event* s);

	/**@{*/
	/**
	 * Provide an interface for locking/unlocking the event queue.
	 *
	 * @warn Do NOT use these methods directly unless you really know
	 * what you are doing. Incorrect use can easily lead to simulator
	 * deadlocks.
	 *
	 * @see EventQueue::ScopedMigration.
	 * @see EventQueue::ScopedRelease
	 * @see EventQueue
	 */
	void lock() { service_mutex.lock(); }
	void unlock() { service_mutex.unlock(); }
	/**@}*/

	/**
	 * Reschedule an event after a checkpoint.
	 *
	 * Since events don't know which event queue they belong to,
	 * parent objects need to reschedule events themselves. This
	 * method conditionally schedules an event that has the Scheduled
	 * flag set. It should be called by parent objects after
	 * unserializing an object.
	 *
	 * @warn Only use this method after unserializing an Event.
	 */
	void checkpointReschedule(Event* event);

	virtual ~EventQueue() {
		while (!empty()) deschedule(getHead());
	}
};

}  // end of namespace acalsim
