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

/**
 * @file SharedData.cc
 * @brief Shared data container packet for zero-copy data sharing
 *
 * This file implements SharedDataPacket, an alternative communication pattern that demonstrates
 * sharing complex data structures between simulators without copying. Uses std::shared_ptr
 * and SharedContainer for efficient multi-object data transfer.
 *
 * **SharedDataPacket vs Regular Packets:**
 * ```
 * Regular Packet (NocReqPacket, CacheReqPacket):
 * ═══════════════════════════════════════════════
 *   - Contains primitive data (int, addr, size)
 *   - Copied during transfer
 *   - Simple ownership (packet owns data)
 *   - Used in main testChannel flow
 *
 * SharedDataPacket:
 * ═════════════════
 *   - Contains std::shared_ptr to SharedContainer
 *   - Zero-copy transfer (reference counting)
 *   - Shared ownership (multiple simulators can access)
 *   - Used in alternative test path (currently disabled)
 *   - Enables complex data sharing (arrays, structs, etc.)
 * ```
 *
 * **SharedContainer Architecture:**
 * ```
 * std::shared_ptr<SharedContainer<TestSharedData>>
 *         ↓
 *   SharedContainer
 *   ├─ TestSharedData object[0]: { vInt[5], vLong[5], pInt[5] }
 *   ├─ TestSharedData object[1]: { vInt[5], vLong[5], pInt[5] }
 *   └─ TestSharedData object[2]: { vInt[5], vLong[5], pInt[5] }
 *
 * Benefits:
 *   - Single allocation for multiple objects
 *   - Thread-safe reference counting
 *   - Method invocation via SharedContainer::run()
 *   - Zero-copy sharing between simulators
 * ```
 *
 * **Usage Pattern:**
 * ```cpp
 * // Create shared container
 * auto ptr = std::make_shared<SharedContainer<TestSharedData>>();
 *
 * // Add objects
 * for (int i = 0; i < 3; i++) {
 *     ptr->add();                                 // Allocate object
 *     ptr->run(i, &TestSharedData::init);         // Initialize
 *     ptr->run(i, &TestSharedData::set, i, val);  // Configure
 * }
 *
 * // Create packet
 * SharedDataPacket* pkt = new SharedDataPacket(tick + 1, ptr);
 *
 * // Send via channel
 * sim->pushToMasterChannelPort("DSNOC", pkt);
 *
 * // Receiver accesses same data (no copy)
 * auto container = pkt->data;
 * container->run(0, &TestSharedData::print);
 * ```
 *
 * **Method Invocation via SharedContainer::run():**
 * ```cpp
 * // Run method with no arguments
 * container->run(index, &TestSharedData::init);
 *
 * // Run method with arguments
 * container->run(index, &TestSharedData::set, which, seed, pindex);
 *
 * // Iterate over all objects
 * for (int i = 0; i < container->size(); i++) {
 *     container->run(i, &TestSharedData::print);
 * }
 * ```
 *
 * @see TestSharedData For shared data structure definition
 * @see SharedContainer For container implementation
 * @see TrafficEvent::sendSharedData() For example usage
 */

#include "SharedData.hh"

/**
 * @brief Visit function for SimModule (not implemented)
 *
 * This visitor method would be invoked if SharedDataPacket is sent to a SimModule.
 * Currently not used in testChannel example.
 *
 * @param when Tick when packet should be processed
 * @param module Target SimModule object
 *
 * @note Logs error message indicating not implemented
 */
// visit function when a SharedDataPacket is sent to a SimModule object
void SharedDataPacket::visit(Tick when, SimModule& module) {
	CLASS_INFO << "void SharedDataPacket::visit(SimBase& module) is not implemented yet!";
}

/**
 * @brief Visit function for SimBase - demonstrates SharedContainer usage
 *
 * This visitor method is invoked when SharedDataPacket arrives at a simulator.
 * Demonstrates:
 * - Accessing shared data container
 * - Iterating over multiple objects
 * - Invoking methods via SharedContainer::run()
 * - Modifying shared data (visible to all holders)
 *
 * **Processing Steps:**
 * 1. Log packet reception
 * 2. Extract SharedContainer from packet
 * 3. Print all objects (before modification)
 * 4. Modify first object via run()
 * 5. Print all objects (after modification)
 *
 * **Modification Example:**
 * ```cpp
 * container->run(0, &TestSharedData::set, 0, 100000, 2);
 * // Calls: container[0].set(0, 100000, 2)
 * // Sets: vInt[0] = 100000, vLong[0] = 100000*10000, pInt[0] = vInt+2
 * ```
 *
 * @param when Tick when packet should be processed
 * @param simulator Target simulator object
 *
 * @note Demonstrates shared data access pattern
 * @note Modifications visible to all shared_ptr holders
 * @see TestSharedData::print() For object printing
 * @see TestSharedData::set() For object modification
 */
// visit function when a SharedDataPacket is sent to a simulator
void SharedDataPacket::visit(Tick when, SimBase& simulator) {
	char msg[256];
	sprintf(msg, "receive a ShardDataPAcket id:%d @when: %ld\n",
	        /*((&simulator)->getName()).c_str(),*/ (&simulator)->getID(), when);
	CLASS_INFO << std::string(msg);
	std::shared_ptr<SharedContainer<TestSharedData>> container = ((SharedDataPacket*)this)->data;
	for (int i = 0; i < container->size(); i++) {
		sprintf(msg, "-- SharedContainer shared data #%d", i);
		CLASS_INFO << std::string(msg);
		// print each object
		container->run(i, &TestSharedData::print);
	}

	// modify a object for testing
	container->run(0, &TestSharedData::set, 0, 100000, 2);
	for (int i = 0; i < container->size(); i++) {
		sprintf(msg, "-- SharedContainer shared data #%d", i);
		CLASS_INFO << std::string(msg);
		// print each object
		container->run(i, &TestSharedData::print);
	}
}
