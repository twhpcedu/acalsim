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

#pragma once

#include <limits>

#include "common/PriorityQueue.hh"

namespace acalsim {

template <typename TPriority, typename TElem>
PriorityQueue<TPriority, TElem>::PriorityQueue() {
	;
}

template <typename TPriority, typename TElem>
PriorityQueue<TPriority, TElem>::~PriorityQueue() {
	for (auto& [priority, elem_set] : this->prioritySetMap) { delete elem_set; }
	for (auto& elem_set : this->elemSetReclcyeBin) { delete elem_set; }
}

template <typename TPriority, typename TElem>
void PriorityQueue<TPriority, TElem>::insert(TElem _elem, TPriority _priority) {
	if (this->prioritySetMap.find(_priority) == this->prioritySetMap.end())
		this->prioritySetMap[_priority] = this->getNewElemSet();
	this->prioritySetMap[_priority]->insert(_elem);
}

template <typename TPriority, typename TElem>
TElem PriorityQueue<TPriority, TElem>::getTopElem() const {
	return *(this->prioritySetMap.begin()->second->begin());
}

template <typename TPriority, typename TElem>
TElem PriorityQueue<TPriority, TElem>::popTopElem() {
	TElem elem = *(this->prioritySetMap.begin()->second->begin());
	this->remove(elem);
	return elem;
}

template <typename TPriority, typename TElem>
void PriorityQueue<TPriority, TElem>::getTopElements(std::function<void(const TElem&)> _func) {
	std::unordered_set<TElem>* elem_set = this->prioritySetMap.begin()->second;

	for (const auto& elem : *elem_set) { _func(elem); }

	elem_set->clear();
	this->prioritySetMap.erase(this->prioritySetMap.begin());
	this->elemSetReclcyeBin.push_back(elem_set);
}

template <typename TPriority, typename TElem>
void PriorityQueue<TPriority, TElem>::getTopElements(std::function<void(const std::unordered_set<TElem>&)> _func) {
	std::unordered_set<TElem>* elem_set = this->prioritySetMap.begin()->second;

	_func(*elem_set);

	elem_set->clear();
	this->prioritySetMap.erase(this->prioritySetMap.begin());
	this->elemSetReclcyeBin.push_back(elem_set);
}

template <typename TPriority, typename TElem>
void PriorityQueue<TPriority, TElem>::getTopElements(std::function<void(std::unordered_set<TElem>&)> _func) {
	std::unordered_set<TElem>* elem_set = this->prioritySetMap.begin()->second;

	_func(*elem_set);

	elem_set->clear();
	this->prioritySetMap.erase(this->prioritySetMap.begin());
	this->elemSetReclcyeBin.push_back(elem_set);
}

template <typename TPriority, typename TElem>
TPriority PriorityQueue<TPriority, TElem>::getTopPriority() const {
	return !this->empty() ? this->prioritySetMap.begin()->first : std::numeric_limits<TPriority>::max();
}

template <typename TPriority, typename TElem>
bool PriorityQueue<TPriority, TElem>::empty() const {
	return this->prioritySetMap.empty();
}

template <typename TPriority, typename TElem>
void PriorityQueue<TPriority, TElem>::remove(const TElem& _elem) {
	for (auto it = this->prioritySetMap.begin(); it != this->prioritySetMap.end(); ++it) {
		std::unordered_set<TElem>* elem_set = it->second;
		if (elem_set->erase(_elem) > 0) {
			// Element found and removed
			if (elem_set->empty()) {
				// Priority level is now empty, remove it and recycle the set
				this->prioritySetMap.erase(it);
				this->elemSetReclcyeBin.push_back(elem_set);
			}
			return;
		}
	}
}

template <typename TPriority, typename TElem>
std::unordered_set<TElem>* PriorityQueue<TPriority, TElem>::getNewElemSet() {
	if (!this->elemSetReclcyeBin.empty()) {
		std::unordered_set<TElem>* set = this->elemSetReclcyeBin.back();
		this->elemSetReclcyeBin.pop_back();
		return set;
	} else {
		return new std::unordered_set<TElem>();
	}
}

}  // namespace acalsim
