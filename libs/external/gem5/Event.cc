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
 * Event Class Implementation
 */

#include "external/gem5/Event.hh"

#include <string>

#include "container/RecycleContainer/RecycleContainer.hh"
#include "external/gem5/base/cprintf.hh"
#include "sim/SimTop.hh"
#include "utils/Logging.hh"

namespace acalsim {

#ifndef NDEBUG
Counter Event::instanceCounter = 0;
#endif

Event* Event::insertBefore(Event* event, Event* curr) {
	// Either way, event will be the top element in the 'in bin' list
	// which is the pointer we need in order to look into the list, so
	// we need to insert that into the bin list.
	if (!curr || *event < *curr) {
		// Insert the event before the current list since it is in the future.
		event->nextBin   = curr;
		event->nextInBin = NULL;
	} else {
		// Since we're on the correct list, we need to point to the next list
		event->nextBin = curr->nextBin;  // curr->nextBin can now become stale

		// Insert event at the top of the stack
		event->nextInBin = curr;
	}

	return event;
}

Event* Event::removeItem(Event* event, Event* top) {
	Event* curr = top;
	Event* next = top->nextInBin;

	// if we removed the top item, we need to handle things specially
	// and just remove the top item, fixing up the next bin pointer of
	// the new top item
	if (event == top) {
		if (!next) return top->nextBin;
		next->nextBin = top->nextBin;
		return next;
	}

	// Since we already checked the current element, we're going to
	// keep checking event against the next element.
	while (event != next) {
		if (!next) printf("event not found!");

		curr = next;
		next = next->nextInBin;
	}

	// remove next from the 'in bin' list since it's what we're looking for
	curr->nextInBin = next->nextInBin;
	return top;
}

const std::string Event::instanceString() const {
#ifndef NDEBUG
	return csprintf("%d", instance);
#else
	return csprintf("%#x", (uintptr_t)this);
#endif
}

Event::~Event() {
	assert(!scheduled());
	flags = 0;
}

const std::string Event::name() const { return csprintf("Event_%s", instanceString()); }

const char* Event::description() const { return "generic"; }

void Event::dump() const {
	cprintf("Event %s (%s)\n", name(), description());
	cprintf("Flags: %#x\n", flags);
#ifdef EVENTQ_DEBUG
	cprintf("Created: %d\n", whenCreated);
#endif
	if (scheduled()) {
#ifdef EVENTQ_DEBUG
		cprintf("Scheduled at  %d\n", whenScheduled);
#endif
		cprintf("Scheduled for %d, priority %d\n", when(), _priority);
	} else {
		cprintf("Not Scheduled\n");
	}
}

}  // end of namespace acalsim
