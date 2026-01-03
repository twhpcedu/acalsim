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

#include <gtest/gtest.h>

#include <queue>
#include <vector>

#include "workloads/graph/DAG.hh"
#include "workloads/operator/MicroOp.hh"
#include "workloads/tensor/SimTensor.hh"

using namespace acalsim;

class DAGTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Setup will be called before each test
	}

	void TearDown() override {
		// Clean up called after each test
	}

	// Declarations of helper methods
	graph::DAG<MicroOp, SimTensor> createBasicGraph(const std::string& graphName, const std::string& prefix = "");
	graph::DAG<MicroOp, SimTensor> createDiamondGraph(const std::string& graphName, const std::string& prefix = "");
	graph::DAG<MicroOp, SimTensor> createTiledGEMMGraph(const std::string& graphName, const std::string& prefix = "");
};

// New test class for testing the DAGScheduler with round-robin scheduling
class DAGSchedulerTest : public DAGTest {  // Inherit from DAGTest

protected:
	class TestDAGScheduler : public acalsim::graph::DAGScheduler<acalsim::MicroOp, acalsim::SimTensor> {
	public:
		// Store execution order and ready nodes for testing
		std::vector<std::string> executionOrder;

		TestDAGScheduler(size_t numGraphs) {}
		// Override processExecuteQueue to capture the execution order
		void processExecuteQueue(
		    std::queue<acalsim::MicroOp*>&                                          queue,
		    std::vector<acalsim::graph::DAG<acalsim::MicroOp, acalsim::SimTensor>>& graphs) override {
			// Store the queue contents for testing
			std::queue<acalsim::MicroOp*> queueCopy = queue;
			while (!queueCopy.empty()) {
				std::cout << "queueCopy->name" << queueCopy.front()->name;
				executionOrder.push_back(queueCopy.front()->name);
				queueCopy.pop();
			}

			// Process the queue normally (simplified for testing)
			while (!queue.empty()) {
				acalsim::MicroOp* node = queue.front();
				queue.pop();

				// Simulate actual processing
				node->setStatus(acalsim::graph::NodeStatus::ACTIVE);
				node->setStatus(acalsim::graph::NodeStatus::DONE);

				// Update dependent operations
				for (auto consumer : node->consumers) {
					// Decrement dependency count
					consumer->dependency_count--;

					// If all dependencies satisfied, mark ready
					if (consumer->dependency_count == 0) {
						consumer->setStatus(acalsim::graph::NodeStatus::READY);
						// Find which graph this node belongs to
						for (size_t i = 0; i < graphs.size(); ++i) {
							if (graphs[i].NodeNameMap.find(consumer->name) != graphs[i].NodeNameMap.end()) {
								// Add to this graph's new ready nodes
								graphs[i].readyPool.push_back(consumer);
								break;
							}
						}
					}
				}
			}
		}
		virtual ~TestDAGScheduler() override = default;  // Defined destructor
	};
};

class PriorityTestScheduler : public DAGTest {  // Inherit from DAGTest

protected:
	class TestSchedulerPriority : public acalsim::graph::DAGScheduler<acalsim::MicroOp, acalsim::SimTensor> {
	public:
		// Store execution order and ready nodes for testing
		std::vector<std::string> executionOrder;

		TestSchedulerPriority(size_t numGraphs) {}

		void updateExecuteQueue() override {
			// Sort graphs by priority in descending order (higher priority first)
			std::vector<std::pair<int, acalsim::graph::DAG<acalsim::MicroOp, acalsim::SimTensor>*>> sortedGraphs;
			for (auto& graph : this->graphs) { sortedGraphs.emplace_back(graph.getPriority(), &graph); }
			std::sort(sortedGraphs.begin(), sortedGraphs.end(),
			          [](const auto& a, const auto& b) { return a.first > b.first; });

			// Clear the execution queue
			std::queue<acalsim::MicroOp*> newExecutionQueue;

			// Process nodes from higher-priority graphs first
			for (auto& [priority, graph] : sortedGraphs) {
				// schedule from the highest-priority graph first
				if (graph->readyPool.empty()) continue;
				while (!graph->readyPool.empty()) {
					acalsim::MicroOp* node = graph->readyPool.front();
					graph->readyPool.erase(graph->readyPool.begin());
					newExecutionQueue.push(node);
				}
				break;
			}

			// Swap executionQueue with the sorted one
			this->executionQueue = std::move(newExecutionQueue);
		}

		// Override processExecuteQueue to capture the execution order
		void processExecuteQueue(
		    std::queue<acalsim::MicroOp*>&                                          queue,
		    std::vector<acalsim::graph::DAG<acalsim::MicroOp, acalsim::SimTensor>>& graphs) override {
			// Store the queue contents for testing
			std::queue<acalsim::MicroOp*> queueCopy = queue;
			while (!queueCopy.empty()) {
				std::cout << "queueCopy->name" << queueCopy.front()->name;
				executionOrder.push_back(queueCopy.front()->name);
				queueCopy.pop();
			}

			// Process the queue normally (simplified for testing)
			while (!queue.empty()) {
				acalsim::MicroOp* node = queue.front();
				queue.pop();

				// Simulate actual processing
				node->setStatus(acalsim::graph::NodeStatus::ACTIVE);
				node->setStatus(acalsim::graph::NodeStatus::DONE);

				// Update dependent operations
				for (auto consumer : node->consumers) {
					// Decrement dependency count
					consumer->dependency_count--;

					// If all dependencies satisfied, mark ready
					if (consumer->dependency_count == 0) {
						consumer->setStatus(acalsim::graph::NodeStatus::READY);
						// Find which graph this node belongs to
						for (size_t i = 0; i < graphs.size(); ++i) {
							if (graphs[i].NodeNameMap.find(consumer->name) != graphs[i].NodeNameMap.end()) {
								// Add to this graph's new ready nodes
								graphs[i].readyPool.push_back(consumer);
								break;
							}
						}
					}
				}
			}
		}
		virtual ~TestSchedulerPriority() override = default;  // Defined destructor
	};
};
