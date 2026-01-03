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

#include <string>

#include "common/FifoQueue.hh"
#include "utils/Logging.hh"

namespace acalsim {

template <typename T>
FifoQueue<T>::FifoQueue(size_t _qSize, std::string _name) : queueSize(_qSize), name(_name) {
	this->reset();
}

template <typename T>
FifoQueue<T>::~FifoQueue() {
	this->reset();
}

template <typename T>
T FifoQueue<T>::pop() {
	T element = T();
	if (this->popValid) {
		element = this->queue.front();
		this->queue.pop();
	} else {
		CLASS_ERROR << "The FifoQueue is out of range";
	}
	this->update();
	return element;
}

template <typename T>
bool FifoQueue<T>::push(T t) {
	bool status = false;
	if (this->pushReady) {
		this->queue.push(t);
		status = true;
	}
	this->update();
	return status;
}

template <typename T>
void FifoQueue<T>::update() {
	this->popValid  = !this->queue.empty();
	this->pushReady = this->queue.size() < this->queueSize;
}

template <typename T>
void FifoQueue<T>::reset() {
	while (!this->queue.empty()) this->queue.pop();
	this->update();
}

template <typename T>
T FifoQueue<T>::front() {
	return this->empty() ? nullptr : this->queue.front();
}

}  // namespace acalsim
