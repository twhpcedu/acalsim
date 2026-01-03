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

#include <cstdlib>

#include "utils/Logging.hh"

namespace acalsim {

/**
 * @class LinkedList
 * @brief A singly linked list implementation.
 *
 * This class provides basic functionalities to manipulate a singly linked list.
 * It allows insertion at the back, removal from the front, and splicing another list.
 * Objects stored in this list should inherit from the `LinkedList::Node` class.
 */
class LinkedList : virtual public HashableType {
public:
	/**
	 * @class Node
	 * @brief A node in the LinkedList.
	 *
	 * This nested class represents a node in the linked list, which holds
	 * a pointer to the previous node. Any object to be stored in the LinkedList
	 * must inherit from this Node class.
	 */
	class Node {
		friend class LinkedList;

	public:
		virtual ~Node() { ; }

	private:
		Node* prev = nullptr;
	};

	/**
	 * @brief Constructor for LinkedList.
	 *
	 * Initializes a new linked list.
	 */
	LinkedList() { ; }

	/**
	 * @brief Destructor for LinkedList.
	 *
	 * Deletes all nodes in the linked list to free memory.
	 */
	~LinkedList() {
		while (!this->empty()) { delete this->popBack(); }
	}

	/**
	 * @brief Inserts a node at the back of the list.
	 *
	 * @param _node The node to be inserted at the back. It must inherit from LinkedList::Node.
	 */
	void insertBack(Node* _node) {
		if (!this->empty()) {
			_node->prev = this->tail;
		} else {
			this->head = _node;
		}

		this->tail = _node;

		this->length += 1;
	}

	/**
	 * @brief Retrieves the front node of the list.
	 *
	 * @return The node at the front of the list.
	 * @throws An assertion error if the list is empty.
	 */
	Node* front() const {
		CLASS_ASSERT_MSG(!this->empty(), "The list is empty.");
		return this->head;
	}

	/**
	 * @brief Retrieves the last node of the list.
	 *
	 * @return The node at the tail of the list.
	 * @throws An assertion error if the list is empty.
	 */
	Node* back() const {
		CLASS_ASSERT_MSG(!this->empty(), "The list is empty.");
		return this->tail;
	}

	/**
	 * @brief Removes and returns the last node of the list.
	 *
	 * @return The node that was at the tail of the list.
	 * @throws An assertion error if the list is empty.
	 */
	Node* popBack() {
		CLASS_ASSERT_MSG(!this->empty(), "The list is empty.");

		Node* obj = this->tail;

		// Update connectivity
		this->tail = obj->prev;
		obj->prev  = nullptr;
		if (obj == this->head) [[unlikely]] { this->head = nullptr; }

		// Update list length
		this->length -= 1;

		return obj;
	}

	/**
	 * @brief Checks if the list is empty.
	 *
	 * @return true if the list is empty, false otherwise.
	 */
	bool empty() const { return this->head == nullptr; }

	/**
	 * @brief Returns the number of nodes in the list.
	 *
	 * @return The size of the list.
	 */
	size_t size() const { return this->length; }

	/**
	 * @brief Splices another linked list into this one after the current tail.
	 *
	 * This operation removes all nodes from the other list and appends them to the current list.
	 *
	 * @param _other The other linked list to splice into this one.
	 */
	void spliceAfter(LinkedList& _other) {
		if (!_other.empty()) {
			if (!this->empty()) {
				_other.head->prev = this->tail;
			} else {
				this->head = _other.head;
			}

			this->tail  = _other.tail;
			_other.head = nullptr;
			_other.tail = nullptr;
		}
	}

private:
	Node* head = nullptr;
	Node* tail = nullptr;

	size_t length = 0;
};

}  // namespace acalsim
