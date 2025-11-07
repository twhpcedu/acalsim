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
 * Event classes
 */

#pragma once

#include <assert.h>

#include <iostream>
#include <string>

#include "external/gem5/base/flags.hh"

namespace acalsim {

#undef SCHAR_MIN
#undef SCHAR_MAX
#define SCHAR_MAX __SCHAR_MAX__
#define SCHAR_MIN (-__SCHAR_MAX__ - 1)

typedef uint64_t Tick;
class EventBase {
protected:
	typedef unsigned short            FlagsType;
	typedef acalsim::Flags<FlagsType> Flags;

	static const FlagsType PublicRead  = 0x003f;   // public readable flags
	static const FlagsType PublicWrite = 0x001d;   // public writable flags
	static const FlagsType Squashed    = 0x0001;   // has been squashed
	static const FlagsType Scheduled   = 0x0002;   // has been scheduled
	static const FlagsType Managed     = 0x0004;   // Use life cycle manager
	static const FlagsType AutoDelete  = Managed;  // delete after dispatch
	/**
	 * This used to be AutoSerialize. This value can't be reused
	 * without changing the checkpoint version since the flag field
	 * gets serialized.
	 */
	static const FlagsType Reserved0   = 0x0008;
	static const FlagsType IsExitEvent = 0x0010;  // special exit event
	static const FlagsType IsMainQueue = 0x0020;  // on main event queue
	static const FlagsType Initialized = 0x7a40;  // somewhat random bits
	static const FlagsType InitMask    = 0xffc0;  // mask for init bits

public:
	/**
	 * @ingroup api_eventq
	 */
	typedef int8_t Priority;

	/// Event priorities, to provide tie-breakers for events scheduled
	/// at the same cycle.  Most events are scheduled at the default
	/// priority; these values are used to control events that need to
	/// be ordered within a cycle.

	/**
	 * Minimum priority
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Minimum_Pri = SCHAR_MIN;

	/**
	 * If we enable tracing on a particular cycle, do that as the
	 * very first thing so we don't miss any of the events on
	 * that cycle (even if we enter the debugger).
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Debug_Enable_Pri = -101;

	/**
	 * Breakpoints should happen before anything else (except
	 * enabling trace output), so we don't miss any action when
	 * debugging.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Debug_Break_Pri = -100;

	/**
	 * CPU switches schedule the new CPU's tick event for the
	 * same cycle (after unscheduling the old CPU's tick event).
	 * The switch needs to come before any tick events to make
	 * sure we don't tick both CPUs in the same cycle.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority CPU_Switch_Pri = -31;

	/**
	 * For some reason "delayed" inter-cluster writebacks are
	 * scheduled before regular writebacks (which have default
	 * priority).  Steve?
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Delayed_Writeback_Pri = -1;

	/**
	 * Default is zero for historical reasons.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Default_Pri = 0;

	/**
	 * DVFS update event leads to stats dump therefore given a lower priority
	 * to ensure all relevant states have been updated
	 *
	 * @ingroup api_eventq
	 */
	static const Priority DVFS_Update_Pri = 31;

	/**
	 * Serailization needs to occur before tick events also, so
	 * that a serialize/unserialize is identical to an on-line
	 * CPU switch.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Serialize_Pri = 32;

	/**
	 * CPU ticks must come after other associated CPU events
	 * (such as writebacks).
	 *
	 * @ingroup api_eventq
	 */
	static const Priority CPU_Tick_Pri = 50;

	/**
	 * If we want to exit a thread in a CPU, it comes after CPU_Tick_Pri
	 *
	 * @ingroup api_eventq
	 */
	static const Priority CPU_Exit_Pri = 64;

	/**
	 * Statistics events (dump, reset, etc.) come after
	 * everything else, but before exit.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Stat_Event_Pri = 90;

	/**
	 * Progress events come at the end.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Progress_Event_Pri = 95;

	/**
	 * If we want to exit on this cycle, it's the very last thing
	 * we do.
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Sim_Exit_Pri = 100;

	/**
	 * Maximum priority
	 *
	 * @ingroup api_eventq
	 */
	static const Priority Maximum_Pri = SCHAR_MAX;
};

typedef int64_t Counter;

class EventQueue;

class Event : public EventBase {
	friend class EventQueue;

private:
	// The event queue is now a linked list of linked lists.  The
	// 'nextBin' pointer is to find the bin, where a bin is defined as
	// when+priority.  All events in the same bin will be stored in a
	// second linked list (a stack) maintained by the 'nextInBin'
	// pointer.  The list will be accessed in LIFO order.  The end
	// result is that the insert/removal in 'nextBin' is
	// linear/constant, and the lookup/removal in 'nextInBin' is
	// constant/constant.  Hopefully this is a significant improvement
	// over the current fully linear insertion.
	Event* nextBin;
	Event* nextInBin;

	static Event* insertBefore(Event* event, Event* curr);
	static Event* removeItem(Event* event, Event* last);

	Tick     _when;      //!< timestamp when event should be processed
	Priority _priority;  //!< event priority
	Flags    flags;

#ifndef NDEBUG
	/// Global counter to generate unique IDs for Event instances
	static Counter instanceCounter;

	/// This event's unique ID.  We can also use pointer values for
	/// this but they're not consistent across runs making debugging
	/// more difficult.  Thus we use a global counter value when
	/// debugging.
	Counter instance;

	/// queue to which this event belongs (though it may or may not be
	/// scheduled on this queue yet)
	EventQueue* queue;
#endif

#ifdef EVENTQ_DEBUG
	Tick whenCreated;    //!< time created
	Tick whenScheduled;  //!< time scheduled
#endif

	void setWhen(Tick when, EventQueue* q) {
		_when = when;
#ifndef NDEBUG
		queue = q;
#endif
#ifdef EVENTQ_DEBUG
		whenScheduled = curTick();
#endif
	}

	bool initialized() const { return (flags & InitMask) == Initialized; }

protected:
	Flags getFlags() const { return flags & PublicRead; }

	bool isFlagSet(Flags _flags) const {
		assert(_flags.noneSet(~PublicRead));
		return flags.isSet(_flags);
	}

	void setFlags(Flags _flags) {
		assert(_flags.noneSet(~PublicWrite));
		flags.set(_flags);
	}

	void setExitFlag() { setFlags(IsExitEvent); }

	void clearFlags(Flags _flags) {
		assert(_flags.noneSet(~PublicWrite));
		flags.clear(_flags);
	}

	void clearFlags() { flags.clear(PublicWrite); }

	/// Return the instance number as a string.
	const std::string instanceString() const;

protected: /* Memory management */
	/**
	 * @{
	 * Memory management hooks for events that have the Managed flag set
	 *
	 * Events can use automatic memory management by setting the
	 * Managed flag. The default implementation automatically deletes
	 * events once they have been removed from the event queue. This
	 * typically happens when events are descheduled or have been
	 * triggered and not rescheduled.
	 *
	 * The methods below may be overridden by events that need custom
	 * memory management. For example, events exported to Python need
	 * to impement reference counting to ensure that the Python
	 * implementation of the event is kept alive while it lives in the
	 * event queue.
	 *
	 * @note Memory managers are responsible for implementing
	 * reference counting (by overriding both acquireImpl() and
	 * releaseImpl()) or checking if an event is no longer scheduled
	 * in releaseImpl() before deallocating it.
	 */

	/**
	 * Managed event scheduled and being held in the event queue.
	 */
	void acquire() {
		if (flags.isSet(Event::Managed)) acquireImpl();
	}

	/**
	 * Managed event removed from the event queue.
	 */
	void release() {
		if (flags.isSet(Event::Managed)) releaseImpl();
	}

	virtual void acquireImpl() {}

	virtual void releaseImpl() {
		if (!scheduled()) delete this;
	}

	/** @} */

public:
	/*
	 * Event constructor
	 * @param queue that the event gets scheduled on
	 *
	 * @ingroup api_eventq
	 */
	Event(Priority p = Default_Pri, Flags f = 0)
	    : nextBin(nullptr), nextInBin(nullptr), _when(0), _priority(p), flags(Initialized | f) {
		assert(f.noneSet(~PublicWrite));

#ifndef NDEBUG
		instance = ++instanceCounter;
		queue    = NULL;
#endif
#ifdef EVENTQ_DEBUG
		whenCreated   = curTick();
		whenScheduled = 0;
#endif
	}
	/**
	 * @ingroup api_eventq
	 * @{
	 */
	virtual ~Event();
	virtual const std::string name() const;

	/// Return a C string describing the event.  This string should
	/// *not* be dynamically allocated; just a const char array
	/// describing the event class.
	virtual const char* description() const;

	/// Dump the current event data
	void dump() const;
	/** @}*/  // end of api group

public:
	/*
	 * This member function is invoked when a functional event is processed
	 * (occurs).  There is no default implementation; each subclass
	 * must provide its own implementation.  The event is not
	 * automatically deleted after it is processed (to allow for
	 * statically allocated event objects).
	 *
	 * If the AutoDestroy flag is set, the object is deleted once it
	 * is processed.
	 *
	 * @ingroup api_eventq
	 */
	virtual void process() = 0;

	/**
	 * Determine if the current event is scheduled
	 *
	 * @ingroup api_eventq
	 */
	bool scheduled() const { return flags.isSet(Scheduled); }

	/**
	 * Squash the current event
	 *
	 * @ingroup api_eventq
	 */
	void squash() { flags.set(Squashed); }

	/**
	 * Check whether the event is squashed
	 *
	 * @ingroup api_eventq
	 */
	bool squashed() const { return flags.isSet(Squashed); }

	/**
	 * See if this is a SimExitEvent (without resorting to RTTI)
	 *
	 * @ingroup api_eventq
	 */
	bool isExitEvent() const { return flags.isSet(IsExitEvent); }

	/**
	 * Check whether this event will auto-delete
	 *
	 * @ingroup api_eventq
	 */
	bool isManaged() const { return flags.isSet(Managed); }

	/**
	 * The function returns true if the object is automatically
	 * deleted after the event is processed.
	 *
	 * @ingroup api_eventq
	 */
	bool isAutoDelete() const { return isManaged(); }

	/**
	 * Get the time that the event is scheduled
	 *
	 * @ingroup api_eventq
	 */
	Tick when() const { return _when; }

	/**
	 * Get the event priority
	 *
	 * @ingroup api_eventq
	 */
	Priority priority() const { return _priority; }

	//! If this is part of a GlobalEvent, return the pointer to the
	//! Global Event.  By default, there is no GlobalEvent, so return
	//! NULL.  (Overridden in GlobalEvent::BarrierEvent.)
	virtual Event* globalEvent() { return NULL; }
};
/**
 * @ingroup api_eventq
 */
inline bool operator<(const Event& l, const Event& r) {
	return l.when() < r.when() || (l.when() == r.when() && l.priority() < r.priority());
}

/**
 * @ingroup api_eventq
 */
inline bool operator>(const Event& l, const Event& r) {
	return l.when() > r.when() || (l.when() == r.when() && l.priority() > r.priority());
}

/**
 * @ingroup api_eventq
 */
inline bool operator<=(const Event& l, const Event& r) {
	return l.when() < r.when() || (l.when() == r.when() && l.priority() <= r.priority());
}

/**
 * @ingroup api_eventq
 */
inline bool operator>=(const Event& l, const Event& r) {
	return l.when() > r.when() || (l.when() == r.when() && l.priority() >= r.priority());
}

/**
 * @ingroup api_eventq
 */
inline bool operator==(const Event& l, const Event& r) { return l.when() == r.when() && l.priority() == r.priority(); }

/**
 * @ingroup api_eventq
 */
inline bool operator!=(const Event& l, const Event& r) { return l.when() != r.when() || l.priority() != r.priority(); }

}  // end of namespace acalsim
