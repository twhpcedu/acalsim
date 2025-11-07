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
 * EventQueue classes
 */

#include "external/gem5/EventQueue.hh"

#include <iostream>
#include <unordered_map>
#include <vector>

#include "external/gem5/base/cprintf.hh"

namespace acalsim {
uint32_t numMainEventQueues = 0;

std::vector<EventQueue*> mainEventQueue;

__thread EventQueue* _curEventQueue = NULL;

bool inParallelMode = false;

EventQueue* getEventQueue(uint32_t index) {
	while (numMainEventQueues <= index) {
		numMainEventQueues++;
		mainEventQueue.push_back(new EventQueue(csprintf("MainEventQueue-%d", index)));
	}

	return mainEventQueue[index];
}

void EventQueue::insert(Event* event) {
	// Deal with the head case
	if (!head || *event <= *head) {
		head = Event::insertBefore(event, head);
		return;
	}

	// Figure out either which 'in bin' list we are on, or where a new list
	// needs to be inserted
	Event* prev = head;
	Event* curr = head->nextBin;
	while (curr && *curr < *event) {
		prev = curr;
		curr = curr->nextBin;
	}

	// Note: this operation may render all nextBin pointers on the
	// prev 'in bin' list stale (except for the top one)
	prev->nextBin = Event::insertBefore(event, curr);
}

void EventQueue::remove(Event* event) {
	if (head == NULL) printf("event not found!");

	assert(event->queue == this);

	// deal with an event on the head's 'in bin' list (event has the same
	// time as the head)
	if (*head == *event) {
		head = Event::removeItem(event, head);
		return;
	}

	// Find the 'in bin' list that this event belongs on
	Event* prev = head;
	Event* curr = head->nextBin;
	while (curr && *curr < *event) {
		prev = curr;
		curr = curr->nextBin;
	}

	if (!curr || *curr != *event) printf("event not found!");

	// curr points to the top item of the the correct 'in bin' list, when
	// we remove an item, it returns the new top item (which may be
	// unchanged)
	prev->nextBin = Event::removeItem(event, curr);
}

Event* EventQueue::serviceOne(Tick when) {
	std::lock_guard<EventQueue> lock(*this);
	Event*                      event = head;
	Event*                      next  = head->nextInBin;
	if (event->_when == when) {
		event->flags.clear(Event::Scheduled);

		if (next) {
			// update the next bin pointer since it could be stale
			next->nextBin = head->nextBin;

			// pop the stack
			head = next;
		} else {
			// this was the only element on the 'in bin' list, so get rid of
			// the 'in bin' list and point to the next bin list
			head = head->nextBin;
		}

		// handle action
		if (!event->squashed()) {
			event->process();
			if (event->isExitEvent()) {
				assert(!event->flags.isSet(Event::Managed) ||
				       !event->flags.isSet(Event::IsMainQueue));  // would be silly
				return event;
			}
		} else {
			event->flags.clear(Event::Squashed);
		}

		event->release();
	}

	return NULL;
}

Event* EventQueue::serviceOne() {
	std::lock_guard<EventQueue> lock(*this);
	Event*                      event = head;
	Event*                      next  = head->nextInBin;
	event->flags.clear(Event::Scheduled);

	if (next) {
		// update the next bin pointer since it could be stale
		next->nextBin = head->nextBin;

		// pop the stack
		head = next;
	} else {
		// this was the only element on the 'in bin' list, so get rid of
		// the 'in bin' list and point to the next bin list
		head = head->nextBin;
	}

	// handle action
	if (!event->squashed()) {
		// forward current cycle to the time when this event occurs.
		setCurTick(event->when());
		event->process();
		if (event->isExitEvent()) {
			assert(!event->flags.isSet(Event::Managed) || !event->flags.isSet(Event::IsMainQueue));  // would be silly
			return event;
		}
	} else {
		event->flags.clear(Event::Squashed);
	}

	event->release();

	return NULL;
}

void EventQueue::checkpointReschedule(Event* event) {
	// It's safe to call insert() directly here since this method
	// should only be called when restoring from a checkpoint (which
	// happens before thread creation).
	if (event->flags.isSet(Event::Scheduled)) insert(event);
}
void EventQueue::dump() const {
	// cprintf("============================================================\n");
	// cprintf("EventQueue Dump  (cycle %d)\n", *_curTickPtr);
	// cprintf("------------------------------------------------------------\n");

	if (empty())
		cprintf("<No Events>\n");
	else {
		Event* nextBin = head;
		while (nextBin) {
			Event* nextInBin = nextBin;
			while (nextInBin) {
				nextInBin->dump();
				nextInBin = nextInBin->nextInBin;
			}

			nextBin = nextBin->nextBin;
		}
	}

	cprintf("============================================================\n");
}

bool EventQueue::debugVerify() const {
	std::unordered_map<long, bool> map;

	Tick  time     = 0;
	short priority = 0;

	Event* nextBin = head;
	while (nextBin) {
		Event* nextInBin = nextBin;
		while (nextInBin) {
			if (nextInBin->when() < time) {
				cprintf("time goes backwards!");
				nextInBin->dump();
				return false;
			} else if (nextInBin->when() == time && nextInBin->priority() < priority) {
				cprintf("priority inverted!");
				nextInBin->dump();
				return false;
			}

			if (map[reinterpret_cast<long>(nextInBin)]) {
				cprintf("Node already seen");
				nextInBin->dump();
				return false;
			}
			map[reinterpret_cast<long>(nextInBin)] = true;

			time     = nextInBin->when();
			priority = nextInBin->priority();

			nextInBin = nextInBin->nextInBin;
		}

		nextBin = nextBin->nextBin;
	}

	return true;
}

Event* EventQueue::replaceHead(Event* s) {
	Event* t = head;
	head     = s;
	return t;
}

EventQueue::EventQueue(const std::string& n) : objName(n), head(NULL), _curTick(0) {}

void EventQueue::asyncInsert(Event* event) {
	std::cout << "eventq." << __func__ << std::endl;
	async_queue_mutex.lock();
	async_queue.push_back(event);
	async_queue_mutex.unlock();
}

void EventQueue::handleAsyncInsertions() {
	async_queue_mutex.lock();

	while (!async_queue.empty()) {
		insert(async_queue.front());
		async_queue.pop_front();
	}

	async_queue_mutex.unlock();
}

}  // end of namespace acalsim
