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

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <memory>

#include "container/RecycleContainer/LinkedList.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**
 * @class LinkedListArray
 * @brief A collection of linked lists managed as an array structure.
 *
 * This class manages multiple linked lists and provides an interface to insert nodes
 * and manipulate these lists collectively. The primary purpose of organizing objects
 * into multiple lists is to enable efficient movement of many objects (i.e., `maxListSize`
 * elements at once) across different `LinkedListArray` instances in O(1) time complexity.
 * Each list in the array can hold a maximum number of nodes, defined by `maxListSize`.
 */
class LinkedListArray : virtual public HashableType {
public:
	/**
	 * @brief Constructor for `LinkedListArray`.
	 *
	 * Initializes a new `LinkedListArray` with a given maximum list size.
	 * @param _maxListSize The maximum size each linked list can grow to before a new list is started.
	 *                     Defaults to 128.
	 */
	LinkedListArray(size_t _maxListSize = 128) : maxListSize(_maxListSize) { ; }

	/**
	 * @brief Destructor for `LinkedListArray`.
	 *
	 * Cleans up all linked lists contained within the array to free memory.
	 */
	virtual ~LinkedListArray() {}

	/**
	 * @brief Inserts a node into the linked list array.
	 *
	 * If the last list has reached its maximum size, a new list is created.
	 * @param _node The node to be inserted. The node should inherit from `LinkedList::Node`.
	 */
	void insert(LinkedList::Node* _node) {
		if (this->lists.empty() || this->lists.back()->size() == this->maxListSize) [[unlikely]] {
			this->lists.emplace_back(std::make_shared<LinkedList>());
		}

		this->lists.back()->insertBack(_node);
	}

	/**
	 * @brief Retrieves the last node of the last list in the array.
	 *
	 * @return The node at the last of the last linked list.
	 * @throws An assertion error if there are no available lists.
	 */
	LinkedList::Node* back() const {
		CLASS_ASSERT_MSG(!this->lists.empty(), "There is no available list.");
		return this->lists.back()->back();
	}

	/**
	 * @brief Removes and returns the front node from the last list in the array.
	 *
	 * If the last list becomes empty after removal, it is removed from the array.
	 *
	 * @tparam SIZE_CHECK If true, performs an empty check before popping.
	 * @return The node that was removed from the front of the last linked list.
	 * @throws An assertion error if there are no available lists.
	 */
	template <bool SIZE_CHECK = false>
	LinkedList::Node* pop() {
		if constexpr (SIZE_CHECK) {
			if (this->empty()) return nullptr;
		} else {
			CLASS_ASSERT_MSG(!this->lists.empty(), "There is no available list.");
		}

		LinkedList::Node* elem = this->lists.back()->popBack();

		if (this->lists.back()->empty()) [[unlikely]] { this->lists.pop_back(); }

		return elem;
	}

	/**
	 * @brief Checks if the `LinkedListArray` is empty.
	 *
	 * @return true if there are no linked lists, false otherwise.
	 */
	bool empty() const { return this->lists.empty(); }

	/**
	 * @brief Checks if there are any full lists in the array.
	 *
	 * @return true if the first list in the array is full, false otherwise.
	 */
	bool hasFullLists() const { return this->lists.front()->size() == this->maxListSize; }

	/**
	 * @brief Counts the number of full linked lists in the array.
	 *
	 * @return The number of linked lists that have reached `maxListSize`.
	 */
	size_t getNumFullLists() const {
		return std::count_if(
		    this->lists.begin(), this->lists.end(),
		    [this](const std::shared_ptr<const LinkedList>& _list) { return _list->size() == this->maxListSize; });
	}

	/**
	 * @brief Inserts a new linked list at the front of the array.
	 *
	 * It is more suitable for the input list to match the maxListSize to maintain
	 * optimal performance when transferring lists between LinkedListArray instances.
	 *
	 * @param _list The linked list to be inserted.
	 */
	void insertList(std::shared_ptr<LinkedList> _list) {
		CLASS_ASSERT_MSG(_list->size() == this->maxListSize,
		                 "The size of the input list does not match the maxListSize of this LinkedListArray.");

		this->lists.emplace_front(_list);
	}

	/**
	 * @brief Removes and returns the front linked list from the array.
	 *
	 * Enables the efficient transfer of lists between LinkedListArrays.
	 * It is more suitable for the list being popped to match the maxListSize
	 * to maintain optimal performance.
	 *
	 * @tparam SIZE_CHECK If true, performs an empty check before popping.
	 * @return The linked list that was removed from the front.
	 */
	template <bool SIZE_CHECK = false>
	std::shared_ptr<LinkedList> popList() {
		if constexpr (SIZE_CHECK) {
			if (this->empty()) return nullptr;
		}

		std::shared_ptr<LinkedList> list = this->lists.front();

		CLASS_ASSERT_MSG(list->size() == this->maxListSize,
		                 "The size of the popped list does not match the maxListSize of this LinkedListArray.");

		this->lists.pop_front();
		return list;
	}

private:
	std::deque<std::shared_ptr<LinkedList>> lists;

	size_t maxListSize = 128;
};

}  // namespace acalsim
