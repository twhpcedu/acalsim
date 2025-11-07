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
#include <vector>

#include "sim/experimental/ThreadManagerV7/PriorityQueueV7.hh"

namespace acalsim {

template <typename TPriority, typename TElem>
PriorityQueueV7<TPriority, TElem>::PriorityQueueV7() {
	;
}

template <typename TPriority, typename TElem>
PriorityQueueV7<TPriority, TElem>::~PriorityQueueV7() {
	for (auto& [priority, elem_vec] : this->priorityMap) { delete elem_vec; }
	for (auto& elem_vec : this->elemVecReclcyeBin) { delete elem_vec; }
}

template <typename TPriority, typename TElem>
void PriorityQueueV7<TPriority, TElem>::insert(const TElem& _elem, const TPriority& _priority) {
	if (!this->priorityMap.contains(_priority)) { this->priorityMap[_priority] = this->getNewElemVec(); }
	this->priorityMap[_priority]->emplace_back(_elem);
}

template <typename TPriority, typename TElem>
TElem& PriorityQueueV7<TPriority, TElem>::getTopElem() const {
	return *(this->priorityMap.begin()->second->back());
}

template <typename TPriority, typename TElem>
TElem PriorityQueueV7<TPriority, TElem>::popTopElem() {
	TElem elem = *(this->priorityMap.begin()->second->back());
	this->priorityMap.begin()->second->pop_back();
	return elem;
}

template <typename TPriority, typename TElem>
void PriorityQueueV7<TPriority, TElem>::getTopElements(std::function<void(std::vector<TElem>*)> _func) {
	std::vector<TElem>* elem_vec = this->priorityMap.begin()->second;

	_func(elem_vec);

	elem_vec->clear();
	this->priorityMap.erase(this->priorityMap.begin());
	this->elemVecReclcyeBin.push_back(elem_vec);
}

template <typename TPriority, typename TElem>
TPriority PriorityQueueV7<TPriority, TElem>::getTopPriority() const {
	return !this->empty() ? this->priorityMap.begin()->first : std::numeric_limits<TPriority>::max();
}

template <typename TPriority, typename TElem>
bool PriorityQueueV7<TPriority, TElem>::empty() const {
	return this->priorityMap.empty();
}

template <typename TPriority, typename TElem>
std::vector<TElem>* PriorityQueueV7<TPriority, TElem>::getNewElemVec() {
	if (!this->elemVecReclcyeBin.empty()) {
		std::vector<TElem>* vec = this->elemVecReclcyeBin.back();
		this->elemVecReclcyeBin.pop_back();
		return vec;
	} else {
		return new std::vector<TElem>();
	}
}

}  // namespace acalsim
