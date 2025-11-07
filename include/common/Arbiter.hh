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

#include <cstddef>

namespace acalsim {

/**
 * @file Arbiter.hh
 * @brief Base class for arbitration mechanisms in hardware modeling
 *
 * @details
 * Arbiter provides a common interface for implementing various arbitration policies
 * used in hardware simulation. Arbitration is needed when multiple components compete
 * for shared resources (e.g., ports, buses, memory).
 *
 * **Arbitration Scenarios:**
 *
 * | Use Case | Description | Typical Policy |
 * |----------|-------------|----------------|
 * | **NoC Crossbar** | Multiple inputs to single output | Round Robin |
 * | **Memory Bus** | Multiple requesters to memory | Priority / Fair |
 * | **Cache Replacement** | Which line to evict | LRU / Random |
 * | **Multi-Port Access** | Multiple ports to same buffer | Round Robin |
 * | **CPU Scheduler** | Multiple threads/cores | Priority / Fair |
 *
 * **Design Pattern:**
 * ```
 * 1. Create arbiter with N components
 * 2. Each cycle, call select() to get winner index
 * 3. Service the winner
 * 4. Repeat
 * ```
 *
 * **Implemented Policies:**
 *
 * - **RoundRobin**: Fair rotation through all components
 * - **Future**: Priority, Weighted Fair Queuing, LRU, etc.
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | setComponentsNum() | O(1) | |
 * | select() | O(1) - O(n) | Depends on policy |
 * | reset() | O(1) | |
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Each arbiter instance should be accessed by single thread
 *
 * @code{.cpp}
 * // Example: Arbitrate between 4 NoC input ports
 * RoundRobin arbiter;
 * arbiter.setComponentsNum(4);
 *
 * // Each cycle, select which port gets access
 * while (simulation_running) {
 *     size_t winner = arbiter.select();  // Returns 0, 1, 2, 3, 0, 1, ...
 *
 *     if (ports[winner]->hasData()) {
 *         processData(ports[winner]);
 *     }
 * }
 * @endcode
 *
 * @note Extend this base class to implement custom arbitration policies
 * @note Component indices are 0-based
 *
 * @see RoundRobin for fair round-robin arbitration
 * @see CrossBar for usage in NoC modeling
 * @since ACALSim 0.1.0
 */
class Arbiter {
public:
	/**
	 * @brief Default constructor - creates arbiter with 0 components
	 *
	 * @note Must call setComponentsNum() before using select()
	 * @note curIndex starts at 0
	 */
	Arbiter() : curIndex(0), componentNum(0) {}

	/**
	 * @brief Virtual destructor for safe polymorphic use
	 */
	virtual ~Arbiter() = default;

	/**
	 * @brief Set the number of components competing for arbitration
	 *
	 * @param _num Number of components (must be > 0 for select() to work)
	 *
	 * @note Should be called before first select()
	 * @note Does not reset curIndex - call reset() for that
	 *
	 * @code{.cpp}
	 * RoundRobin arbiter;
	 * arbiter.setComponentsNum(8);  // 8 competing ports
	 * @endcode
	 */
	void setComponentsNum(size_t _num) { this->componentNum = _num; }

	/**
	 * @brief Get the number of components
	 *
	 * @return Number of components in arbitration
	 *
	 * @code{.cpp}
	 * size_t numPorts = arbiter.getComponentsNum();
	 * @endcode
	 */
	size_t getComponentsNum() const { return this->componentNum; }

	/**
	 * @brief Get the index of the last selected component
	 *
	 * @return Index of current selection (0-based)
	 *
	 * @note Returns the result of the most recent select() call
	 * @note Returns 0 if select() has never been called
	 *
	 * @code{.cpp}
	 * size_t lastWinner = arbiter.getCurIndex();
	 * @endcode
	 */
	size_t getCurIndex() { return this->curIndex; }

	/**
	 * @brief Perform arbitration and select the next component
	 *
	 * @return Index of the selected component (0-based, < componentNum)
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note Updates curIndex internally
	 * @note Behavior depends on specific arbitration policy
	 *
	 * @warning Calling with componentNum == 0 may cause assertion/crash
	 *
	 * @code{.cpp}
	 * // RoundRobin example:
	 * size_t winner = arbiter.select();  // Returns next index in round-robin
	 * handleRequest(winner);
	 * @endcode
	 */
	virtual size_t select() = 0;

	/**
	 * @brief Reset arbitration state to initial conditions
	 *
	 * @note Resets curIndex to 0
	 * @note Does not change componentNum
	 * @note Can be overridden for more complex state reset
	 *
	 * @code{.cpp}
	 * arbiter.setComponentsNum(4);
	 * arbiter.select();  // Returns 0
	 * arbiter.select();  // Returns 1
	 *
	 * arbiter.reset();
	 * arbiter.select();  // Returns 0 again
	 * @endcode
	 */
	virtual void reset() { this->curIndex = 0; }

protected:
	/**
	 * @brief Number of components competing for arbitration
	 * @note Set via setComponentsNum()
	 */
	size_t componentNum;

	/**
	 * @brief Current/last selected component index
	 * @note Updated by select(), read by getCurIndex()
	 */
	size_t curIndex;
};

/**
 * @brief Round-robin arbitration policy
 *
 * @details
 * RoundRobin implements a fair arbitration scheme that cycles through all
 * components in order, giving each an equal opportunity for selection.
 *
 * **Algorithm:**
 * ```
 * curIndex = (curIndex + 1) % componentNum
 * return curIndex
 * ```
 *
 * **Fairness:**
 * - Every component gets exactly 1/N of the bandwidth
 * - No starvation - all components eventually serviced
 * - Simple and predictable behavior
 *
 * **Use Cases:**
 *
 * | Scenario | Why Round-Robin |
 * |----------|----------------|
 * | **Fair Resource Sharing** | Equal access for all requesters |
 * | **Load Balancing** | Distribute work evenly |
 * | **Preventing Starvation** | Guarantee service for all |
 * | **Simple Hardware** | Easy to implement in RTL |
 *
 * **Performance:**
 *
 * | Operation | Complexity |
 * |-----------|-----------|
 * | select() | O(1) |
 * | reset() | O(1) |
 *
 * **Example Sequence:**
 * ```
 * setComponentsNum(4);
 * select() -> 0
 * select() -> 1
 * select() -> 2
 * select() -> 3
 * select() -> 0  // Wraps around
 * select() -> 1
 * ...
 * ```
 *
 * @code{.cpp}
 * // Example: Fair arbitration among 4 CPU cores for memory access
 * RoundRobin memArbiter;
 * memArbiter.setComponentsNum(4);
 *
 * // Each memory cycle, select next core
 * void memoryControllerCycle() {
 *     size_t core = memArbiter.select();  // 0 -> 1 -> 2 -> 3 -> 0 ...
 *
 *     if (cores[core]->hasMemRequest()) {
 *         serviceMemRequest(cores[core]);
 *     }
 * }
 * @endcode
 *
 * @note First call to select() returns index 0
 * @note Asserts if componentNum == 0
 *
 * @warning Not priority-aware - all components treated equally
 *
 * @see Arbiter for base class interface
 * @since ACALSim 0.1.0
 */
class RoundRobin : public Arbiter {
public:
	/**
	 * @brief Default constructor
	 * @note Inherits componentNum = 0, curIndex = 0 from Arbiter
	 */
	RoundRobin() : Arbiter() {}

	/**
	 * @brief Select the next component in round-robin order
	 *
	 * @return Index of selected component (0 to componentNum-1)
	 *
	 * @note Increments curIndex and wraps around using modulo
	 * @note First call returns 0, second returns 1, etc.
	 * @note Complexity: O(1)
	 *
	 * @warning Asserts if componentNum == 0 (division by zero)
	 *
	 * @code{.cpp}
	 * RoundRobin rr;
	 * rr.setComponentsNum(3);
	 *
	 * size_t a = rr.select();  // Returns 0
	 * size_t b = rr.select();  // Returns 1
	 * size_t c = rr.select();  // Returns 2
	 * size_t d = rr.select();  // Returns 0 (wrapped)
	 * @endcode
	 */
	size_t select() override;
};

}  // namespace acalsim
