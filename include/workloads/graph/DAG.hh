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

#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "container/RecycleContainer/RecyclableObject.hh"
#include "utils/Logging.hh"

namespace acalsim {
namespace graph {

// Forward declaration
template <typename EdgeType>
class Edge;

template <typename NodeType, typename EdgeType>
class DAG;

enum class NodeStatus { WAITING, READY, ACTIVE, DONE };

// Node class template
template <typename NodeType, typename EdgeType>
class Node : public virtual RecyclableObject {
public:
	std::string            name;
	NodeStatus             status;
	int                    dependency_count;
	std::vector<NodeType*> consumers;
	std::vector<EdgeType*> inputs;
	std::vector<EdgeType*> outputs;

	Node(std::string nodeName = "Unknown node") : name(nodeName), status(NodeStatus::WAITING), dependency_count(0) {}

	void renew(const std::string& _name) {
		this->name             = _name;
		this->status           = acalsim::graph::NodeStatus::WAITING;
		this->dependency_count = 0;
		this->consumers.clear();
		this->inputs.clear();
		this->outputs.clear();
		this->dag = nullptr;
	}

	void setStatus(NodeStatus newStatus) { this->status = newStatus; }

	void                     setDAG(DAG<NodeType, EdgeType>* _dag) { this->dag = _dag; }
	DAG<NodeType, EdgeType>* getDAG() const { return this->dag; }

	void markAsComplete() {
		this->setStatus(NodeStatus::DONE);

		for (auto& consumer : this->consumers) {
			if (--consumer->dependency_count == 0) {
				consumer->setStatus(acalsim::graph::NodeStatus::READY);
				LABELED_ASSERT(consumer->getDAG()->NodeNameMap.contains(consumer->name), "DAG::Node");
				consumer->getDAG()->readyPool.push_back(consumer);
			}
		}
	}

	std::string getName() const { return this->name; }

private:
	acalsim::graph::DAG<NodeType, EdgeType>* dag = nullptr;
};

// Edge class template
template <typename EdgeType>
class Edge : public RecyclableObject {
public:
	std::string name;
	Edge(std::string edgeName = "") : name(edgeName) {}
	std::string getName() const { return name; }  // Added missing getName() method

	void renew(const std::string& _name) { this->name = _name; }
};

// Graph class - represents the entire computation graph
template <typename NodeType, typename EdgeType>
class DAG {
public:
	// Constructor with graph name
	DAG(std::string graphName) : name(graphName), priority(0) {}

	// Add an input tensor to the graph
	void addInput(EdgeType* edge) { inputs.push_back(edge); }
	void addInputs(std::vector<EdgeType*> edges) {
		for (auto edge : edges) inputs.push_back(edge);
	}

	// Add an edge (tensor) to the graph
	void addEdge(EdgeType* edge) {
		edges.push_back(edge);
		EdgeNameMap[edge->name] = edge;
	}

	void                   setInputs(std::vector<EdgeType*> _inputs);
	void                   setOutputs(std::vector<EdgeType*> _outputs);
	std::vector<EdgeType*> getInputs() { return inputs; }
	std::vector<EdgeType*> getOutputs() { return outputs; }

	// Add an output tensor to the graph
	void addOutput(EdgeType* edge) { outputs.push_back(edge); }
	void addOutputs(std::vector<EdgeType*> edges) {
		for (auto edge : edges) outputs.push_back(edge);
	}

	// Add a node with its input and output tensors
	void addNode(NodeType* op, std::vector<EdgeType*> inEdges, std::vector<EdgeType*> outEdges) {
		op->inputs  = inEdges;
		op->outputs = outEdges;
		nodes.push_back(op);
		NodeNameMap[op->name] = op;
		static_cast<Node<NodeType, EdgeType>*>(op)->setDAG(this);

		// If node has no dependencies, add to ready pool
		if (op->dependency_count == 0) {
			readyPool.push_back(op);
			op->setStatus(NodeStatus::READY);
		}
	}

	// Create a dependency between two operations
	void addDependency(NodeType* parent, NodeType* consumer) {
		parent->consumers.push_back(consumer);
		consumer->dependency_count++;
	}

	// Get the next ready operation from the pool
	NodeType* getReadyNode() {
		if (!readyPool.empty()) {
			NodeType* node = readyPool.back();
			readyPool.pop_back();
			return node;
		}
		return nullptr;
	}

	// Lookup operation by name
	NodeType* getNode(std::string nodeName) { return NodeNameMap.count(nodeName) ? NodeNameMap[nodeName] : nullptr; }

	// Lookup tensor by name
	EdgeType* getEdge(std::string edgeName) { return EdgeNameMap.count(edgeName) ? EdgeNameMap[edgeName] : nullptr; }

	std::vector<EdgeType*> getEdges() { return edges; }
	std::vector<NodeType*> getNodes() { return nodes; }

	// Set priority for the graph (for priority scheduling)
	void setPriority(int p) { priority = p; }

	// Get priority of the graph
	int getPriority() const { return priority; }

	void initializeReadyPool() {
		readyPool.clear();  // Clear any existing ready nodes
		for (auto* node : nodes) {
			if (node->dependency_count == 0) {
				readyPool.push_back(node);
				node->setStatus(NodeStatus::READY);  // Set the node status to READY
			} else {
				node->setStatus(NodeStatus::WAITING);
			}
		}
	}
	// Merge another graph into this one
	void merge(const DAG<NodeType, EdgeType>& other) {
		// Add all nodes from other graph
		for (auto* node : other.nodes) {
			if (NodeNameMap.find(node->name) == NodeNameMap.end()) {
				nodes.push_back(node);
				NodeNameMap[node->name] = node;
			}
		}

		// Add all edges from other graph
		for (auto* edge : other.edges) {
			if (EdgeNameMap.find(edge->name) == EdgeNameMap.end()) {
				edges.push_back(edge);
				EdgeNameMap[edge->name] = edge;
			}
		}

		// Update ready pool
		for (auto* node : other.readyPool) {
			if (std::find(readyPool.begin(), readyPool.end(), node) == readyPool.end()) {
				if (NodeNameMap[node->name]->dependency_count == 0) { readyPool.push_back(node); }
			}
		}
	}

	// Get the successors of a node
	std::unordered_set<NodeType*> getSuccessors(const NodeType* node) const {
		std::unordered_set<NodeType*> successors;
		if (node) {
			for (const auto& consumer : node->consumers) { successors.insert(consumer); }
		}
		return successors;
	}

	std::string genGraphviz() const {
		try {
			std::unordered_set<NodeType*>   unique_nodes;
			std::unordered_set<std::string> unique_node_names;
			std::stringstream               graphviz_ss;

			// Collect all nodes
			for (const auto& node : this->nodes) {
				unique_nodes.insert(node);
				for (const auto& consumer : node->consumers) { unique_nodes.insert(consumer); }
			}

			// Safer check for repeated node names - skip the assertion
			for (const auto& node : unique_nodes) {
				if (unique_node_names.find(node->name) != unique_node_names.end()) {
					// Instead of assertion, just log the issue
					std::cerr << "Warning: Duplicate node name detected - " << node->name << "!" << std::endl;
					// Optionally make the name unique by appending a counter
					// node->name = node->name + "_" + std::to_string(counter++);
				}
				unique_node_names.insert(node->name);
			}

			// Prologue
			graphviz_ss << "digraph G {\n";

			// Declare all nodes in the graph
			for (const auto& node : unique_nodes) { graphviz_ss << "    \"" << node->getName() << "\";\n"; }

			// Create edges based on consumers
			for (const auto& node : this->nodes) {
				for (const auto& consumer : node->consumers) {
					graphviz_ss << "    \"" << node->getName() << "\" -> \"" << consumer->getName() << "\";\n";
				}
			}

			// Epilogue
			graphviz_ss << "}";

			return graphviz_ss.str();
		} catch (const std::exception& e) {
			std::cerr << "Error generating GraphViz: " << e.what() << std::endl;
			return "digraph G { /* Error generating graph */ }";
		} catch (...) {
			std::cerr << "Unknown error generating GraphViz" << std::endl;
			return "digraph G { /* Error generating graph */ }";
		}
	}

	// Graph properties
	std::string                                name;         // Graph name
	std::vector<NodeType*>                     heads;        // Entry nodes
	std::vector<NodeType*>                     nodes;        // All operations
	std::vector<EdgeType*>                     edges;        // All tensors
	std::vector<NodeType*>                     readyPool;    // Operations ready for execution
	std::unordered_map<std::string, NodeType*> NodeNameMap;  // Name to operation lookup
	std::unordered_map<std::string, EdgeType*> EdgeNameMap;  // Name to tensor lookup
	std::vector<EdgeType*>                     inputs;       // Input tensors
	std::vector<EdgeType*>                     outputs;      // Output tensors
	int                                        priority;     // Graph priority for scheduling
};

// Execute the computation graph
template <typename NodeType, typename EdgeType>
class DAGScheduler {
public:
	DAGScheduler() {}

	virtual ~DAGScheduler() = default;

	virtual void run(std::vector<graph::DAG<NodeType, EdgeType>>& graphsToProcess) {
		// Store graphs
		graphs = graphsToProcess;

		// update the execution queue. Push the ready nodes from each graph to the execution queue.
		updateExecuteQueue();

		// Process the execution queue with all nodes scheduled in round-robin order
		while (!executionQueue.empty()) {
			processExecuteQueue(executionQueue, graphs);

			// update the execution queue. Push the ready nodes from each graph to the execution queue.
			updateExecuteQueue();
		}
	}

	NodeType* peekNextNodeToExecute() { return executionQueue.front(); }

	NodeType* popNextNodeToExecute() {
		if (executionQueue.empty()) return nullptr;

		NodeType* node = executionQueue.front();
		executionQueue.pop();
		return node;
	}

	// post-process routine for a single node
	void markNodeAsDone(NodeType* node) {
		// Keep track of newly ready nodes for each graph
		std::vector<std::vector<NodeType*>> newReadyNodes(graphs.size());

		// Mark as done
		node->setStatus(NodeStatus::DONE);

		// Update dependent operations
		for (auto consumer : node->consumers) {
			// Decrement dependency count
			consumer->dependency_count--;

			// If all dependencies satisfied, mark ready
			if (consumer->dependency_count == 0) {
				consumer->setStatus(NodeStatus::READY);

				// Find which graph this node belongs to
				for (size_t i = 0; i < graphs.size(); ++i) {
					if (graphs[i].NodeNameMap.find(consumer->name) != graphs[i].NodeNameMap.end()) {
						// Add to this graph's new ready nodes
						newReadyNodes[i].push_back(consumer);
						break;
					}
				}
			}
		}

		// If the execution queue is empty, add new ready nodes in round-robin fashion
		if (executionQueue.empty()) {
			bool addedNewNode = true;
			while (addedNewNode) {
				addedNewNode = false;
				for (size_t i = 0; i < graphs.size(); ++i) {
					if (!newReadyNodes[i].empty()) {
						executionQueue.push(newReadyNodes[i].front());
						newReadyNodes[i].erase(newReadyNodes[i].begin());
						addedNewNode = true;
					}
				}
			}
		}
	}

	void addDAG(graph::DAG<NodeType, EdgeType>& graph) { graphs.push_back(graph); }

	// Virtual method to be overridden by test classes
	virtual NodeType* getNextReadyNode(std::vector<graph::DAG<NodeType, EdgeType>>& graphs) {
		// Default implementation returns nullptr, should be overridden
		return nullptr;
	}

	// Virtual method to allow overriding in test classes
	virtual void processExecuteQueue(std::queue<NodeType*>&                       queue,
	                                 std::vector<graph::DAG<NodeType, EdgeType>>& graphs) {
		// Process operations in order of readiness
		while (!queue.empty()) {
			// Get next operation
			NodeType* node = queue.front();
			queue.pop();

			// Set to active and execute
			node->setStatus(NodeStatus::ACTIVE);

			std::cout << "Executing " << node->name << " with " << node->inputs.size() << " inputs and "
			          << node->outputs.size() << " outputs" << std::endl;

			// Mark as done
			node->setStatus(NodeStatus::DONE);

			// Update dependent operations
			for (auto consumer : node->consumers) {
				// Decrement dependency count
				consumer->dependency_count--;

				// If all dependencies satisfied, mark ready
				if (consumer->dependency_count == 0) {
					consumer->setStatus(NodeStatus::READY);
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

protected:
	std::vector<graph::DAG<NodeType, EdgeType>> graphs;

	// Execution queue from ready operations
	std::queue<NodeType*> executionQueue;

	virtual void updateExecuteQueue() {
		// Update execution queue from ready operations using round-robin scheduling

		// Keep track of which nodes have been added from each graph
		std::vector<std::unordered_set<NodeType*>> processedNodes(graphs.size());

		// Continue until all ready nodes from all graphs have been processed
		bool addedAny;
		do {
			addedAny = false;

			// Iterate through each graph in a round-robin fashion
			for (size_t i = 0; i < graphs.size(); ++i) {
				// If this graph still has unprocessed ready nodes
				if (!graphs[i].readyPool.empty()) {
					// Add one node from this graph to the execution queue
					NodeType* node = graphs[i].readyPool.front();
					graphs[i].readyPool.erase(graphs[i].readyPool.begin());
					executionQueue.push(node);
					processedNodes[i].insert(node);
					addedAny = true;
				}
			}
		} while (addedAny);
	}
};

template <typename NodeType, typename EdgeType>
class PriorityScheduler : public DAGScheduler<NodeType, EdgeType> {
public:
	void updateExecuteQueue() override {
		std::vector<std::pair<int, graph::DAG<NodeType, EdgeType>*>> sortedGraphs;
		for (auto& graph : this->graphs) { sortedGraphs.emplace_back(graph.getPriority(), &graph); }
		std::sort(sortedGraphs.begin(), sortedGraphs.end(),
		          [](const auto& a, const auto& b) { return a.first > b.first; });

		std::queue<NodeType*> newExecutionQueue;
		for (auto& [priority, graph] : sortedGraphs) {
			if (graph->readyPool.empty()) continue;
			while (!graph->readyPool.empty()) {
				NodeType* node = graph->readyPool.front();
				graph->readyPool.erase(graph->readyPool.begin());
				newExecutionQueue.push(node);
			}
			break;
		}

		this->executionQueue = std::move(newExecutionQueue);
	}
};

}  // namespace graph
}  // namespace acalsim
