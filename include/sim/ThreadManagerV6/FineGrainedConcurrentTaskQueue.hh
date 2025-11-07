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

#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace acalsim {

template <typename T>
class FineGrainedConcurrentUpdateablePriorityQueue {
private:
	struct Node {
		T                     value;
		std::atomic<uint64_t> priority;
		std::atomic<bool>     needsUpdate;
		std::atomic<int>      index;

		Node(const T& val, uint64_t prio, bool update, int idx)
		    : value(val), priority(prio), needsUpdate(update), index(idx) {}
		Node(T&& val, uint64_t prio, bool update, int idx)
		    : value(std::move(val)), priority(prio), needsUpdate(update), index(idx) {}
	};

	std::vector<std::shared_ptr<Node>>           nodes;
	std::atomic<size_t>                          size_;
	std::shared_mutex                            vector_mutex;
	std::shared_mutex                            index_map_mutex;
	std::unordered_map<int, std::weak_ptr<Node>> index_map;

	bool compare(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
		return a->priority.load(std::memory_order_acquire) < b->priority.load(std::memory_order_acquire);
	}

	int parent(int index) const { return (index - 1) / 2; }
	int leftChild(int index) const { return 2 * index + 1; }
	int rightChild(int index) const { return 2 * index + 2; }

	void siftUp(int index) {
		while (index > 0) {
			int                   parent_idx = parent(index);
			std::shared_ptr<Node> current_node, parent_node;
			{
				std::shared_lock<std::shared_mutex> lock(vector_mutex);
				if (index >= nodes.size() || parent_idx >= nodes.size()) break;
				current_node = nodes[index];
				parent_node  = nodes[parent_idx];
				if (!current_node || !parent_node) break;
			}

			if (current_node->priority.load(std::memory_order_acquire) <
			    parent_node->priority.load(std::memory_order_acquire)) {
				{
					std::unique_lock<std::shared_mutex> lock(vector_mutex);
					if (index < nodes.size() && parent_idx < nodes.size() && nodes[index] == current_node &&
					    nodes[parent_idx] == parent_node) {
						std::swap(nodes[index], nodes[parent_idx]);
						current_node->index.store(parent_idx, std::memory_order_release);
						parent_node->index.store(index, std::memory_order_release);
						index = parent_idx;
					} else {
						break;
					}
				}
			} else {
				break;
			}
		}
	}

	void siftDown(int index) {
		size_t current_size = size_.load(std::memory_order_acquire);
		while (true) {
			int                   smallest = index;
			int                   left     = leftChild(index);
			int                   right    = rightChild(index);
			std::shared_ptr<Node> current_node, left_node, right_node;
			{
				std::shared_lock<std::shared_mutex> lock(vector_mutex);
				if (index >= nodes.size() || index >= current_size) break;
				current_node = nodes[index];
				if (left < nodes.size() && left < current_size) left_node = nodes[left];
				if (right < nodes.size() && right < current_size) right_node = nodes[right];
				if (!current_node) break;
			}

			if (left_node && left_node->priority.load(std::memory_order_acquire) <
			                     current_node->priority.load(std::memory_order_acquire)) {
				smallest = left;
			}
			if (right_node &&
			    right_node->priority.load(std::memory_order_acquire) <
			        (smallest == index
			             ? current_node->priority.load(std::memory_order_acquire)
			             : (smallest == left ? left_node->priority.load(std::memory_order_acquire)
			                                 : nodes[smallest]->priority.load(std::memory_order_acquire)))) {
				smallest = right;
			}

			if (smallest != index) {
				{
					std::unique_lock<std::shared_mutex> lock(vector_mutex);
					if (index < nodes.size() && smallest < nodes.size() && nodes[index] == current_node &&
					    nodes[smallest] == (smallest == left ? left_node : right_node)) {
						std::swap(nodes[index], nodes[smallest]);
						nodes[index]->index.store(index, std::memory_order_release);
						nodes[smallest]->index.store(smallest, std::memory_order_release);
						index = smallest;
					} else {
						break;
					}
				}
			} else {
				break;
			}
		}
	}

	void processUpdates() {
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) return;

		std::shared_ptr<Node> top_node;
		{
			std::shared_lock<std::shared_mutex> lock(vector_mutex);
			if (nodes.empty() || current_size == 0) return;
			top_node = nodes[0];
		}

		if (top_node && top_node->needsUpdate.load(std::memory_order_acquire)) {
			if (top_node->needsUpdate.exchange(false, std::memory_order_acq_rel)) { siftDown(0); }
		}
	}

public:
	FineGrainedConcurrentUpdateablePriorityQueue() : size_(0) { nodes.reserve(1024); }

	void push(const T& value, uint64_t priority) {
		auto   new_node = std::make_shared<Node>(value, priority, false, -1);
		size_t current_size;
		{
			std::unique_lock<std::shared_mutex> lock(vector_mutex);
			current_size = size_.fetch_add(1, std::memory_order_release);
			if (current_size >= nodes.capacity()) { nodes.reserve(nodes.capacity() * 2); }
			if (current_size >= nodes.size()) {
				nodes.push_back(new_node);
			} else {
				nodes[current_size] = new_node;
			}
			new_node->index.store(current_size, std::memory_order_release);
			{
				std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
				index_map[value.id] = new_node;
			}
		}
		siftUp(current_size);
	}

	void update(int simID, uint64_t newPriority) {
		std::shared_ptr<Node> target_node;
		{
			std::shared_lock<std::shared_mutex> lock_map(index_map_mutex);
			auto                                it = index_map.find(simID);
			if (it != index_map.end()) { target_node = it->second.lock(); }
		}

		if (target_node) {
			uint64_t old_priority = target_node->priority.exchange(newPriority, std::memory_order_acq_rel);
			target_node->value.next_execution_cycle = newPriority;
			target_node->needsUpdate.store(true, std::memory_order_release);
			int index = target_node->index.load(std::memory_order_acquire);
			if (newPriority < old_priority) {
				siftUp(index);
			} else if (newPriority > old_priority) {
				siftDown(index);
			}
		}
	}

	bool hasReadyTask(uint64_t priority) {
		processUpdates();
		std::shared_lock<std::shared_mutex> lock(vector_mutex);
		return !nodes.empty() && size_.load(std::memory_order_acquire) > 0 &&
		       nodes[0]->priority.load(std::memory_order_acquire) <= priority;
	}

	T top() {
		processUpdates();
		std::shared_lock<std::shared_mutex> lock(vector_mutex);
		if (nodes.empty() || size_.load(std::memory_order_acquire) == 0) {
			throw std::out_of_range("Priority queue is empty");
		}
		return nodes[0]->value;
	}

	void pop() {
		processUpdates();
		std::unique_lock<std::shared_mutex> lock(vector_mutex);
		size_t                              current_size = size_.load(std::memory_order_acquire);
		if (nodes.empty() || current_size == 0) { throw std::out_of_range("Priority queue is empty"); }
		std::shared_ptr<Node> last_node = nodes.back();
		nodes[0]                        = last_node;
		nodes.pop_back();
		size_.store(current_size - 1, std::memory_order_release);
		if (!nodes.empty()) {
			nodes[0]->index.store(0, std::memory_order_release);
			{
				std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
				index_map[nodes[0]->value.id] = nodes[0];
				for (auto it = index_map.begin(); it != index_map.end(); ++it) {
					if (it->second.expired()) continue;
					if (it->second.lock() == last_node) {
						index_map.erase(it);
						break;
					}
				}
			}
			siftDown(0);
		} else {
			std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
			index_map.clear();
		}
	}

	bool tryGetReadyTask(uint64_t currentTick, T& outTask) {
		processUpdates();
		std::unique_lock<std::shared_mutex> lock(vector_mutex);
		if (nodes.empty() || size_.load(std::memory_order_acquire) == 0 ||
		    nodes[0]->priority.load(std::memory_order_acquire) > currentTick) {
			return false;
		}
		outTask                         = nodes[0]->value;
		std::shared_ptr<Node> last_node = nodes.back();
		nodes[0]                        = last_node;
		nodes.pop_back();
		size_.store(size_.load(std::memory_order_acquire) - 1, std::memory_order_release);
		if (!nodes.empty()) {
			nodes[0]->index.store(0, std::memory_order_release);
			{
				std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
				index_map[nodes[0]->value.id] = nodes[0];
				for (auto it = index_map.begin(); it != index_map.end(); ++it) {
					if (it->second.expired()) continue;
					if (it->second.lock() == last_node) {
						index_map.erase(it);
						break;
					}
				}
			}
			siftDown(0);
		} else {
			std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
			index_map.clear();
		}
		return true;
	}

	bool tryGetAnyTask(T& outTask) {
		std::unique_lock<std::shared_mutex> lock(vector_mutex);
		if (nodes.empty()) return false;
		outTask                         = nodes[0]->value;
		std::shared_ptr<Node> last_node = nodes.back();
		nodes[0]                        = last_node;
		nodes.pop_back();
		size_.store(size_.load(std::memory_order_acquire) - 1, std::memory_order_release);
		if (!nodes.empty()) {
			nodes[0]->index.store(0, std::memory_order_release);
			{
				std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
				index_map[nodes[0]->value.id] = nodes[0];
				for (auto it = index_map.begin(); it != index_map.end(); ++it) {
					if (it->second.expired()) continue;
					if (it->second.lock() == last_node) {
						index_map.erase(it);
						break;
					}
				}
			}
			siftDown(0);
		} else {
			std::unique_lock<std::shared_mutex> lock_map(index_map_mutex);
			index_map.clear();
		}
		return true;
	}

	bool empty() const {
		std::shared_lock<std::shared_mutex> lock(const_cast<std::shared_mutex&>(vector_mutex));
		return size_.load(std::memory_order_acquire) == 0;
	}

	size_t size() const {
		std::shared_lock<std::shared_mutex> lock(const_cast<std::shared_mutex&>(vector_mutex));
		return size_.load(std::memory_order_acquire);
	}

	std::string dump() {
		processUpdates();
		std::shared_lock<std::shared_mutex> lock(vector_mutex);
		std::stringstream                   ss;
		ss << "Concurrent Priority Queue Content: size=" << size_.load(std::memory_order_acquire) << "\n";
		if (!nodes.empty() && size_.load(std::memory_order_acquire) > 0) {
			ss << "  - Top element data (ID: " << nodes[0]->value.id << "): " << nodes[0]->value << "\n";
		}
		return ss.str();
	}
};

}  // namespace acalsim
