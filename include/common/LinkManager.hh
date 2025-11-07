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

#include <string>
#include <unordered_map>

#include "utils/HashableType.hh"

namespace acalsim {

/**
 * @file LinkManager.hh
 * @brief Manages named bidirectional connections between simulation components
 *
 * @details
 * LinkManager provides a registry for tracking upstream and downstream connections
 * in a simulation hierarchy. This is essential for modeling dataflow and control
 * dependencies in complex hardware systems.
 *
 * **Key Concepts:**
 *
 * - **Upstream**: Components that send data/signals TO this component
 * - **Downstream**: Components that receive data/signals FROM this component
 *
 * **Dataflow Direction:**
 * ```
 * [Upstream] ---> [This Component] ---> [Downstream]
 *   (Source)         (Processor)           (Sink)
 * ```
 *
 * **Use Cases:**
 *
 * | Scenario | Upstream | This Component | Downstream |
 * |----------|----------|----------------|------------|
 * | **Cache Hierarchy** | L1 Cache | L2 Cache | L3 Cache |
 * | **NoC Router** | Input Ports | Router Logic | Output Ports |
 * | **Pipeline Stage** | Decode | Execute | Memory |
 * | **Memory System** | CPU | Memory Controller | DRAM |
 * | **Interconnect** | Masters | Crossbar | Slaves |
 *
 * **Design Pattern:**
 * ```cpp
 * // Setup phase (during construction):
 * component->addUpStream(sourcePtr, "cpu0");
 * component->addDownStream(sinkPtr, "memory");
 *
 * // Runtime phase (during simulation):
 * auto* source = component->getUpStream("cpu0");
 * auto* sink = component->getDownStream("memory");
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | addUpStream() | O(1) avg | Hash table insert |
 * | addDownStream() | O(1) avg | Hash table insert |
 * | getUpStream() | O(1) avg | Hash table lookup |
 * | getDownStream() | O(1) avg | Hash table lookup |
 *
 * **Memory:** O(U + D) where U = upstreams, D = downstreams
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Typically initialized once during setup, read-only during simulation
 *
 * @tparam T Type of linked components (typically pointers: SimBase*, Module*, etc.)
 *
 * @code{.cpp}
 * // Example: L2 cache connected to L1 and L3
 * class L2Cache : public LinkManager<Cache*> {
 * public:
 *     L2Cache(L1Cache* l1, L3Cache* l3) {
 *         // L1 is upstream (sends requests to L2)
 *         addUpStream(l1, "L1");
 *
 *         // L3 is downstream (receives requests from L2)
 *         addDownStream(l3, "L3");
 *     }
 *
 *     void handleRequest() {
 *         Cache* l1 = getUpStream("L1");
 *         if (l1->hasPendingRequest()) {
 *             Request req = l1->getRequest();
 *             processRequest(req);
 *
 *             if (miss) {
 *                 Cache* l3 = getDownStream("L3");
 *                 l3->forwardRequest(req);
 *             }
 *         }
 *     }
 * };
 * @endcode
 *
 * @note Names must be unique within upstream/downstream collections separately
 * @note Duplicate names will overwrite previous entries silently
 * @note Type T is typically a pointer (no ownership semantics)
 *
 * @warning getUpStream/getDownStream may throw if name not found (implementation-dependent)
 * @warning No automatic cleanup - caller manages component lifetimes
 *
 * @see SimBase for simulation component base class
 * @see SimChannel for data communication between components
 * @since ACALSim 0.1.0
 */
template <typename T>
class LinkManager : virtual public HashableType {
public:
	/**
	 * @brief Default constructor - creates empty link manager
	 *
	 * @note No upstreams or downstreams initially
	 * @note Use add*Stream() methods to register connections
	 */
	LinkManager() {}

	/**
	 * @brief Add an upstream component (data source)
	 *
	 * @param _mate Component to add as upstream
	 * @param _upStreamName Unique name to identify this upstream
	 *
	 * @note If name already exists, silently overwrites previous entry
	 * @note Complexity: O(1) average case
	 * @note Name can be any string (e.g., "CPU0", "Port1", "Master")
	 *
	 * @code{.cpp}
	 * LinkManager<Port*> router;
	 * router.addUpStream(inputPort0, "north");
	 * router.addUpStream(inputPort1, "south");
	 * router.addUpStream(inputPort2, "east");
	 * @endcode
	 */
	void addUpStream(T _mate, std::string _upStreamName);

	/**
	 * @brief Get an upstream component by name
	 *
	 * @param _name Name of the upstream component
	 * @return Component associated with the name
	 *
	 * @note Complexity: O(1) average case
	 * @note Behavior if name not found is implementation-dependent (may throw or return default)
	 *
	 * @code{.cpp}
	 * Port* northInput = router.getUpStream("north");
	 * if (northInput->hasData()) {
	 *     processData(northInput->getData());
	 * }
	 * @endcode
	 */
	T getUpStream(std::string _name) const;

	/**
	 * @brief Add a downstream component (data sink)
	 *
	 * @param _mate Component to add as downstream
	 * @param _downStreamName Unique name to identify this downstream
	 *
	 * @note If name already exists, silently overwrites previous entry
	 * @note Complexity: O(1) average case
	 * @note Name can be any string (e.g., "Memory", "Port0", "Slave")
	 *
	 * @code{.cpp}
	 * LinkManager<Module*> cpu;
	 * cpu.addDownStream(l1Cache, "L1");
	 * cpu.addDownStream(tlb, "TLB");
	 * cpu.addDownStream(branchPredictor, "BP");
	 * @endcode
	 */
	void addDownStream(T _mate, std::string _downStreamName);

	/**
	 * @brief Get a downstream component by name
	 *
	 * @param _name Name of the downstream component
	 * @return Component associated with the name
	 *
	 * @note Complexity: O(1) average case
	 * @note Behavior if name not found is implementation-dependent (may throw or return default)
	 *
	 * @code{.cpp}
	 * Cache* l1 = cpu.getDownStream("L1");
	 * l1->issueRequest(addr, data);
	 * @endcode
	 */
	T getDownStream(std::string _name) const;

private:
	/**
	 * @brief Map of downstream components indexed by name
	 * @details Components that receive data FROM this component
	 */
	std::unordered_map<std::string, T> downStreams;

	/**
	 * @brief Map of upstream components indexed by name
	 * @details Components that send data TO this component
	 */
	std::unordered_map<std::string, T> upStreams;
};

}  // namespace acalsim

#include "LinkManager.inl"
