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

#pragma once

#include "channel/SimChannel.hh"

namespace acalsim {

/**********************************
 *                                *
 *        SimChannelGlobal        *
 *                                *
 **********************************/

SimChannelStatus SimChannelGlobal::getChannelDualQueueStatus() { return SimChannelGlobal::channelDualQueueStatus; }

void SimChannelGlobal::toggleChannelDualQueueStatus() {
	// A SimChannel oboject toggles status when the queue for popping is completely drained
	// Here are the assumptions for the SimChannel implementation:
	// At a given iteration,
	//   1) one of the queues is used for pushing new reuests
	//   2) the other queue is used for popping requests
	//   3) everything in the queue for popping will be drained completely in each iteration
	//   4) The status toggle will be done by the SimTop control thread

	SimChannelGlobal::channelDualQueueStatus =
	    ((SimChannelGlobal::channelDualQueueStatus == SimChannelStatus::PING_PUSH_PONG_POP)
	         ? SimChannelStatus::PONG_PUSH_PING_POP
	         : SimChannelStatus::PING_PUSH_PONG_POP);
}

/**********************************
 *                                *
 *           SimChannel           *
 *                                *
 **********************************/

template <typename T>
msd::channel<T>* SimChannel<T>::getQueueForPush() {
	return (SimChannelGlobal::getChannelDualQueueStatus() == SimChannelStatus::PING_PUSH_PONG_POP) ? this->pingQueue
	                                                                                               : this->pongQueue;
}

template <typename T>
msd::channel<T>* SimChannel<T>::getQueueForPop() {
	return (SimChannelGlobal::getChannelDualQueueStatus() == SimChannelStatus::PONG_PUSH_PING_POP) ? this->pingQueue
	                                                                                               : this->pongQueue;
}

template <typename T>
SimChannel<T>::SimChannel() {
	pingQueue = new msd::channel<T>;
	pongQueue = new msd::channel<T>;
}

template <typename T>
SimChannel<T>::~SimChannel() {
	pingQueue->close();
	pongQueue->close();

	delete this->pingQueue;
	delete this->pongQueue;
}

template <typename T>
SimChannel<typename std::decay<T>::type>& operator<<(SimChannel<typename std::decay<T>::type>& ch, const T& in) {
	*(ch.getQueueForPush()) << in;
	return ch;
}

template <typename T>
SimChannel<T>& operator>>(SimChannel<T>& ch, T& out) {
	*(ch.getQueueForPop()) >> out;
	return ch;
}

template <typename T>
bool SimChannel<T>::nonEmptyForPop() {
	return !this->getQueueForPop()->empty();
}

}  // namespace acalsim
