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
 * @file ChannelPortManager.cc
 * @brief ChannelPortManager implementation - channel port registration, lookup, and statistics collection
 *
 * This file implements ChannelPortManager, a mixin interface that enables SimBase objects to
 * participate in lock-free inter-simulator communication via channel ports. It provides port
 * registration, name-based lookup, and optional performance statistics collection for channel
 * operations.
 *
 * **ChannelPortManager as Mixin Architecture:**
 * ```
 * SimBase (core simulation functionality)
 *   │
 *   └─ Multiple Inheritance Pattern:
 *        │
 *        ├─ class Accelerator : public SimBase, public ChannelPortManager
 *        │     │
 *        │     ├─ SimBase functionality: step(), event queue, clock
 *        │     └─ ChannelPortManager functionality: channel communication
 *        │
 *        └─ class HostCPU : public SimBase, public ChannelPortManager
 *              │
 *              ├─ SimBase functionality: step(), event queue, clock
 *              └─ ChannelPortManager functionality: channel communication
 *
 * Why mixin?
 *   - Optional capability: Not all SimBase objects need channels
 *   - Single inheritance from SimBase: Preserves class hierarchy
 *   - Composition alternative: Could use has-a, but is-a more natural
 *   - Interface segregation: Channel logic separate from core SimBase
 * ```
 *
 * **Port Registration and Storage:**
 * ```
 * ChannelPortManager Internal Storage:
 *   │
 *   ├─ masterChannelPorts: IterableUMap<string, SharedPtr<MasterChannelPort>>
 *   │    │
 *   │    ├─ "to_host" → MasterChannelPort (sends to HostCPU)
 *   │    ├─ "to_memory" → MasterChannelPort (sends to MemoryController)
 *   │    └─ "to_network" → MasterChannelPort (sends to NetworkInterface)
 *   │
 *   └─ slaveChannelPorts: IterableUMap<string, SharedPtr<SlaveChannelPort>>
 *        │
 *        ├─ "from_cpu" → SlaveChannelPort (receives from CPU)
 *        ├─ "from_dma" → SlaveChannelPort (receives from DMA)
 *        └─ "from_cache" → SlaveChannelPort (receives from Cache)
 *
 * IterableUMap Benefits:
 *   - Fast O(1) lookup by name
 *   - Iteration support for batch operations
 *   - Wrapper around std::unordered_map with safer API
 * ```
 *
 * **Channel Port Lifecycle:**
 * ```
 * 1. SimBase Construction:
 *    class Accelerator : public SimBase, public ChannelPortManager {
 *        MasterChannelPort* to_host;
 *        SlaveChannelPort* from_host;
 *    };
 *
 * 2. ChannelPortManager::ConnectChannelPort() (static method):
 *    ConnectChannelPort(accel, host, "to_host", "from_accel", 10);
 *      │
 *      ├─ Create shared TSimChannel:
 *      │    auto channel = std::make_shared<TSimChannel>(delay=10);
 *      │
 *      ├─ Create MasterChannelPort:
 *      │    auto master = std::make_shared<MasterChannelPort>(host, channel);
 *      │    accel->addMasterChannelPort("to_host", master);  ◄── THIS FILE (lines 55-60)
 *      │    accel->to_host = master.get();
 *      │
 *      └─ Create SlaveChannelPort:
 *           auto slave = std::make_shared<SlaveChannelPort>(accel, channel);
 *           host->addSlaveChannelPort("from_accel", slave);  ◄── THIS FILE (lines 29-33)
 *           host->from_accel = slave.get();
 *
 * 3. Simulation Execution:
 *    - Accelerator: pushToMasterChannelPort("to_host", request)  ◄── THIS FILE (lines 69-75)
 *    - HostCPU: popFromSlaveChannelPort("from_accel")  ◄── THIS FILE (lines 42-53)
 *
 * 4. Destruction (RAII):
 *    - SharedPtrs in IterableUMap automatically destroyed
 *    - TSimChannel destroyed when last SharedPtr released
 *    - No manual cleanup needed
 * ```
 *
 * **Name-Based Lookup API:**
 * ```
 * User Code (SimBase::step):
 *   // Push using string name
 *   pushToMasterChannelPort("to_host", request);
 *     │
 *     ├─ getMasterChannelPort("to_host")  ◄── THIS FILE (lines 62-67)
 *     │    └─ Lookup in masterChannelPorts map
 *     │         └─ ASSERT if not found
 *     │
 *     └─ port->push(request)
 *
 * Alternative Direct Access:
 *   // Use raw pointer stored in class
 *   to_host_port->push(request);
 *
 * Why both approaches?
 *   - Name-based: Flexible, dynamic lookup, used in generic code
 *   - Direct access: Faster, no map lookup overhead, used in hot paths
 *   - Trade-off: Convenience vs performance
 * ```
 *
 * **Statistics Collection Integration:**
 * ```
 * When ACALSIM_STATISTICS is defined:
 *
 * pushToMasterChannelPort() (lines 69-75):
 *   MEASURE_TIME_MICROSECONDS(
 *       push,  // Variable name: push_lat
 *       this->getMasterChannelPort(_name)->push(_item)
 *   );
 *   ChannelPortManager::cost_stat.push(push_lat);
 *     │
 *     └─ Accumulates push latency (wall-clock time, not simulation ticks)
 *
 * popFromSlaveChannelPort() (lines 42-53):
 *   MEASURE_TIME_MICROSECONDS(
 *       pop,  // Variable name: pop_lat
 *       TPayload payload = this->getSlaveChannelPort(_name)->pop();
 *   );
 *   ChannelPortManager::cost_stat.push(pop_lat);
 *     │
 *     └─ Accumulates pop latency
 *
 * MEASURE_TIME_MICROSECONDS Macro (from profiling/Utils.hh):
 *   - Records wall-clock time (chrono::high_resolution_clock)
 *   - Measures framework overhead, not simulated latency
 *   - Used for profiling/optimization, not simulation results
 *   - Minimal overhead when ACALSIM_STATISTICS disabled
 *
 * cost_stat Analysis:
 *   - Static member: Statistics::CostStat<double> cost_stat
 *   - Aggregates min/max/avg/std_dev of push/pop latencies
 *   - Reported in Statistics::report() at simulation end
 *   - Example output:
 *       ChannelPortManager push latency: avg=1.23μs, max=5.67μs
 *       ChannelPortManager pop latency: avg=0.89μs, max=3.21μs
 * ```
 *
 * **Inbound Notification Override Pattern:**
 * ```
 * handleInboundNotification() (line 77):
 *   - Default implementation: Empty (no-op)
 *   - Called by: MasterChannelPort::push() in remote simulator
 *   - Purpose: Notify receiver of incoming requests
 *
 * Typical Override in SimBase subclass:
 *   class HostCPU : public SimBase, public ChannelPortManager {
 *       void handleInboundNotification() override {
 *           this->inbound_request_flag = true;
 *           // Flag propagates to hasPendingActivityInChannelPort()
 *           // Keeps SimBase active for next iteration
 *       }
 *   };
 *
 * Call Chain (from sender to receiver):
 *   1. Accelerator: pushToMasterChannelPort("to_host", req)
 *   2. MasterChannelPort::push(req)
 *   3. channel_mate->handleInboundNotification()  // Call across simulators
 *   4. HostCPU::handleInboundNotification()       // Override executed
 *   5. inbound_request_flag = true
 *   6. SimBase::hasPendingActivityInChannelPort() returns true
 *   7. ThreadManager schedules HostCPU for next Phase 1
 *
 * Why virtual?
 *   - Each SimBase may handle notifications differently
 *   - Some may set flags, others may wake threads
 *   - Extensibility for custom notification logic
 * ```
 *
 * **IterableUMap Storage Details:**
 * ```
 * IterableUMap<string, SharedPtr<ChannelPort>>:
 *   - Wrapper: Custom container from container/IterableUMap.hh
 *   - Backend: std::unordered_map<string, Ownership<SharedPtr<ChannelPort>>>
 *   - Ownership: Smart pointer wrapper for RAII management
 *
 * Why not raw std::unordered_map?
 *   - Ownership<SharedPtr>: Explicit ownership semantics
 *   - getUMapRef(): Controlled access to underlying map
 *   - insert(): Wrapper with ownership transfer
 *   - Iteration: Range-based for loops supported
 *   - Type safety: Prevents accidental misuse
 *
 * Example iteration:
 *   for (auto& [name, port_owner] : slaveChannelPorts) {
 *       auto port = port_owner.get();
 *       if (!port->empty()) {
 *           processRequest(port->pop());
 *       }
 *   }
 * ```
 *
 * **Error Handling Strategy:**
 * | Error Type                  | Detection Method                  | Action                        | Line  |
 * |-----------------------------|-----------------------------------|-------------------------------|-------|
 * | Duplicate port name         | contains() check                  | ASSERT, halt execution        | 31, 57|
 * | Port not found (get)        | map::find() == end()              | ASSERT, halt execution        | 37, 64|
 * | Empty port pop              | User responsibility (check empty) | Undefined behavior            | -     |
 *
 * Why ASSERT instead of exceptions?
 *   - Configuration errors: Should be caught during initialization
 *   - Performance: Zero overhead in release builds (NDEBUG)
 *   - Determinism: No exception handling unpredictability
 *   - Debugging: Immediate stack trace at error location
 * ```
 *
 * **Integration with SimBase Lifecycle:**
 * ```
 * SimBase::stepWrapper() Flow:
 *   │
 *   ├─ Phase 1a: processInboundChannelRequests()
 *   │     │
 *   │     └─ If (hasPendingActivityInChannelPort()):
 *   │          └─ For each SlaveChannelPort:
 *   │               └─ while (!port->empty()):
 *   │                    ├─ req = popFromSlaveChannelPort(name)  ◄── THIS FILE (lines 42-53)
 *   │                    └─ handleRequest(req)
 *   │
 *   ├─ Phase 1b: processEventQueue()
 *   │
 *   ├─ Phase 1c: triggerRetryCallback()
 *   │
 *   └─ Phase 1d: step()
 *          │
 *          └─ User code may call:
 *               pushToMasterChannelPort(name, req)  ◄── THIS FILE (lines 69-75)
 *
 * Activity Tracking:
 *   - hasPendingActivityInChannelPort() checks inbound_request_flag
 *   - Set by handleInboundNotification() when remote push occurs
 *   - Cleared after processing all inbound requests
 *   - Part of global activity tracking (keeps SimBase alive)
 * ```
 *
 * **Implementation Functions:**
 *
 * 1. **ChannelPortManager::ChannelPortManager() (line 27):**
 *    - Default constructor (empty implementation)
 *    - IterableUMap members use default constructors
 *    - No manual initialization needed
 *
 * 2. **addSlaveChannelPort() (lines 29-33):**
 *    - Register incoming channel port by name
 *    - Check for duplicates (ASSERT if exists)
 *    - Store SharedPtr in slaveChannelPorts map
 *
 * 3. **getSlaveChannelPort() (lines 35-40):**
 *    - Lookup SlaveChannelPort by name
 *    - ASSERT if not found (configuration error)
 *    - Return raw pointer from SharedPtr
 *
 * 4. **popFromSlaveChannelPort() (lines 42-53):**
 *    - Wrapper for SlaveChannelPort::pop()
 *    - Conditional statistics collection (ACALSIM_STATISTICS)
 *    - Measure wall-clock time (MEASURE_TIME_MICROSECONDS)
 *    - Accumulate latency in cost_stat
 *
 * 5. **addMasterChannelPort() (lines 55-60):**
 *    - Register outgoing channel port by name
 *    - Check for duplicates (ASSERT if exists)
 *    - Store SharedPtr in masterChannelPorts map
 *
 * 6. **getMasterChannelPort() (lines 62-67):**
 *    - Lookup MasterChannelPort by name
 *    - ASSERT if not found (configuration error)
 *    - Return raw pointer from SharedPtr
 *
 * 7. **pushToMasterChannelPort() (lines 69-75):**
 *    - Wrapper for MasterChannelPort::push()
 *    - Always measure time (MEASURE_TIME_MICROSECONDS)
 *    - Conditional statistics collection (ACALSIM_STATISTICS)
 *    - Accumulate latency in cost_stat
 *
 * 8. **handleInboundNotification() (line 77):**
 *    - Virtual method for notification handling
 *    - Default implementation: Empty (no-op)
 *    - Override in SimBase subclass to set activity flags
 *    - Called by remote MasterChannelPort::push()
 *
 * **Usage Example:**
 * ```cpp
 * // Define simulators with channel capability
 * class Accelerator : public SimBase, public ChannelPortManager {
 *     MasterChannelPort* to_host;
 *     SlaveChannelPort* from_host;
 *
 * public:
 *     void init() {
 *         // Ports created and registered by ConnectChannelPort()
 *     }
 *
 *     void step() {
 *         // Send result to host
 *         if (computation_done()) {
 *             auto result = std::make_shared<SimRequest>(...);
 *             pushToMasterChannelPort("to_host", result);
 *             // Or: to_host->push(result);  // Direct access
 *         }
 *
 *         // Process commands from host
 *         // (Handled in processInboundChannelRequests)
 *     }
 *
 *     void handleInboundNotification() override {
 *         this->inbound_request_flag = true;
 *         // Keep active for next iteration
 *     }
 * };
 *
 * class HostCPU : public SimBase, public ChannelPortManager {
 *     MasterChannelPort* to_accel;
 *     SlaveChannelPort* from_accel;
 *
 * public:
 *     void init() {
 *         // Ports created and registered by ConnectChannelPort()
 *     }
 *
 *     void processInboundChannelRequests() override {
 *         // Process results from accelerator
 *         while (!getSlaveChannelPort("from_accel")->empty()) {
 *             auto result = popFromSlaveChannelPort("from_accel");
 *             handleResult(result);
 *         }
 *     }
 *
 *     void step() {
 *         // Send commands to accelerator
 *         if (has_work()) {
 *             auto cmd = std::make_shared<SimRequest>(...);
 *             pushToMasterChannelPort("to_accel", cmd);
 *         }
 *     }
 *
 *     void handleInboundNotification() override {
 *         this->inbound_request_flag = true;
 *     }
 * };
 *
 * // Binding in SimTop
 * auto accel = new Accelerator("accel");
 * auto host = new HostCPU("host");
 *
 * // Connect bidirectional channels
 * ChannelPortManager::ConnectChannelPort(
 *     accel, host, "to_host", "from_accel", 10);  // Accel → Host (10 tick delay)
 *
 * ChannelPortManager::ConnectChannelPort(
 *     host, accel, "to_accel", "from_host", 10);  // Host → Accel (10 tick delay)
 * ```
 *
 * **Performance Considerations:**
 * - Name-based lookup: O(1) hash map, ~10ns overhead
 * - Direct pointer access: O(1) dereference, ~1ns overhead
 * - Statistics collection: ~50-100ns per operation (disabled in release)
 * - SharedPtr reference counting: Thread-safe atomic operations
 * - Recommendation: Use direct pointers in hot paths (step()), names in setup
 *
 * @see ChannelPortManager.hh For interface documentation and ConnectChannelPort()
 * @see ChannelPort.cc For channel port implementation
 * @see SimBase.cc For processInboundChannelRequests() and activity tracking
 * @see IterableUMap.hh For storage container documentation
 * @see profiling/Utils.hh For MEASURE_TIME_MICROSECONDS macro
 * @see profiling/Statistics.hh For cost_stat and statistics reporting
 */

#include "channel/ChannelPortManager.hh"

#include "profiling/Utils.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

ChannelPortManager::ChannelPortManager() { ; }

void ChannelPortManager::addSlaveChannelPort(std::string _name, SlaveChannelPort::SharedPtr _in_port) {
	bool is_present = this->slaveChannelPorts.getUMapRef().contains(_name);
	ASSERT_MSG(!is_present, "SlaveChannelPort `" + _name + "` is present in `ChannelPortManager::slaveChannelPorts`.");
	this->slaveChannelPorts.insert(std::make_pair(_name, _in_port));
}

SlaveChannelPort::SharedPtr ChannelPortManager::getSlaveChannelPort(std::string _name) const {
	auto iter = this->slaveChannelPorts.getUMapRef().find(_name);
	ASSERT_MSG(iter != this->slaveChannelPorts.getUMapRef().end(),
	           "SlaveChannelPort `" + _name + "` is not present in `ChannelPortManager::slaveChannelPorts`.");
	return iter->second.get();
}

ChannelPortManager::TPayload ChannelPortManager::popFromSlaveChannelPort(std::string _name) {
#ifndef ACALSIM_STATISTICS
	return this->getSlaveChannelPort(_name)->pop();
#else
	MEASURE_TIME_MICROSECONDS(
	    /* var_name */ pop,
	    /* code_block */ ChannelPortManager::TPayload payload = this->getSlaveChannelPort(_name)->pop(););

	ChannelPortManager::cost_stat.push(pop_lat);
	return payload;
#endif  // ACALSIM_STATISTICS
}

void ChannelPortManager::addMasterChannelPort(std::string _name, MasterChannelPort::SharedPtr _out_port) {
	bool is_present = this->masterChannelPorts.getUMapRef().contains(_name);
	ASSERT_MSG(!is_present,
	           "MasterChannelPort `" + _name + "` is present in `ChannelPortManager::masterChannelPorts`.");
	this->masterChannelPorts.insert(std::make_pair(_name, _out_port));
}

MasterChannelPort::SharedPtr ChannelPortManager::getMasterChannelPort(std::string _name) const {
	auto iter = this->masterChannelPorts.getUMapRef().find(_name);
	ASSERT_MSG(iter != this->masterChannelPorts.getUMapRef().end(),
	           "MasterChannelPort `" + _name + "` is not present in `ChannelPortManager::masterChannelPorts`.");
	return iter->second.get();
}

void ChannelPortManager::pushToMasterChannelPort(std::string _name, ChannelPortManager::TPayload const& _item) {
	MEASURE_TIME_MICROSECONDS(push, this->getMasterChannelPort(_name)->push(_item));

#ifdef ACALSIM_STATISTICS
	ChannelPortManager::cost_stat.push(push_lat);
#endif  // ACALSIM_STATISTICS
}

void ChannelPortManager::handleInboundNotification() { ; }

}  // namespace acalsim
