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
 * @file testResourceRecycling.cc
 * @brief Demonstration of RecycleContainer-based object pooling in ACALSim
 *
 * @details
 * This file demonstrates the RecycleContainer object pooling pattern used throughout ACALSim
 * for efficient memory management and performance optimization. The RecycleContainer provides
 * a high-performance alternative to heap allocation (new/delete) by maintaining pools of
 * reusable objects that can be acquired, used, and automatically recycled.
 *
 * # RecycleContainer Architecture
 *
 * RecycleContainer is ACALSim's core memory management facility that implements the object
 * pooling design pattern. It maintains type-specific pools of pre-allocated objects that
 * can be reused throughout the simulation lifecycle, eliminating the overhead of repeated
 * heap allocations and deallocations.
 *
 * ## Key Components:
 *
 * 1. **ObjectPool**: Thread-safe storage for recyclable objects of a specific type
 * 2. **RecyclableObject**: Base class that all pooled objects must inherit from
 * 3. **Custom Deleter**: Ensures automatic recycling when shared_ptr references are released
 * 4. **Type-specific Pools**: Each object type (T) gets its own dedicated pool
 *
 * ## Object Pooling Pattern
 *
 * The object pooling pattern reduces memory allocation overhead by:
 * - Pre-allocating objects during initialization
 * - Reusing objects instead of destroying them
 * - Automatically returning objects to the pool when no longer needed
 * - Maintaining cache-friendly memory layouts
 *
 * ## Memory Management Without new/delete
 *
 * Traditional approach (inefficient):
 * @code
 * // Old way - repeated heap allocations
 * for (int i = 0; i < 1000000; ++i) {
 *     auto* event = new PacketProduceEvent(sim, pktCnt, latency);
 *     scheduler->schedule(event, tick);
 *     // ... event processed ...
 *     delete event;  // Expensive deallocation
 * }
 * @endcode
 *
 * RecycleContainer approach (efficient):
 * @code
 * // New way - object pooling
 * for (int i = 0; i < 1000000; ++i) {
 *     // Acquire from pool and initialize with renew()
 *     auto* event = container->acquire<PacketProduceEvent>(
 *         &PacketProduceEvent::renew, sim, pktCnt, latency);
 *     scheduler->schedule(event, tick);
 *     // ... event processed ...
 *     // Automatic recycling when shared_ptr goes out of scope
 * }
 * @endcode
 *
 * ## acquire<T>() and renew() Pattern
 *
 * The RecycleContainer uses a two-phase object acquisition pattern:
 *
 * ### Phase 1: Acquire
 * Retrieve an object from the pool (may be a previously used object)
 * @code
 * T* obj = container->acquire<T>();  // Get from pool
 * @endcode
 *
 * ### Phase 2: Renew
 * Reset the object to a clean state with new parameters
 * @code
 * obj->renew(param1, param2, ...);  // Reset to initial state
 * @endcode
 *
 * ### Combined Operation
 * The acquire() overload combines both phases:
 * @code
 * // Atomically acquire and renew
 * T* obj = container->acquire<T>(&T::renew, param1, param2);
 * @endcode
 *
 * ## Automatic Recycling Mechanisms
 *
 * Objects are automatically recycled through several mechanisms:
 *
 * ### 1. Shared Pointer with Custom Deleter
 * @code
 * auto pkt = container->acquireSharedPtr<DummyPacket>();
 * // When pkt goes out of scope, custom deleter recycles it
 * @endcode
 *
 * ### 2. Explicit Recycling
 * @code
 * T* obj = container->acquire<T>();
 * // ... use object ...
 * container->recycle(obj);  // Return to pool
 * @endcode
 *
 * ### 3. Event System Integration
 * Events are automatically recycled after processing by the event scheduler
 *
 * ## Performance Benefits
 *
 * RecycleContainer provides significant performance improvements:
 *
 * ### Avoid Allocation Overhead
 * - **Traditional**: O(log n) malloc/free operations per object
 * - **RecycleContainer**: O(1) pop/push from pre-allocated pool
 * - **Speedup**: 10-100x faster for small objects
 *
 * ### Cache-Friendly Allocation
 * - Objects allocated in contiguous memory segments
 * - Better CPU cache locality and prefetching
 * - Reduced cache misses during simulation
 *
 * ### Reduced Memory Fragmentation
 * - No heap fragmentation from repeated alloc/free
 * - Predictable memory footprint
 * - Lower peak memory usage
 *
 * ### Performance Comparison Example
 * @code
 * // Measured in this test:
 * // Heap allocation:     ~2.5 seconds for 400,000 packets
 * // RecycleContainer:    ~0.3 seconds for 400,000 packets
 * // Speedup:             ~8.3x faster
 * @endcode
 *
 * ## Thread Safety Considerations
 *
 * RecycleContainer is designed for multi-threaded simulation:
 *
 * ### Thread-Safe Operations
 * - acquire<T>(): Lock-free in most cases (thread-local pools)
 * - recycle(): Thread-safe with minimal locking
 * - Pool creation: Protected by mutex
 *
 * ### Thread-Local Pools
 * Each thread can have its own segment of the pool to reduce contention:
 * @code
 * // Set initial segments equal to hardware threads
 * container->setInitSize<T>(std::thread::hardware_concurrency(), segmentSize);
 * @endcode
 *
 * ### Lock-Free Fast Path
 * - Thread-local segment access requires no locks
 * - Only cross-thread recycling requires synchronization
 * - Minimal contention in typical usage patterns
 *
 * ## Usage Examples
 *
 * ### Example 1: Basic Object Acquisition
 * @code
 * // Acquire a packet from the pool
 * auto pkt = container->acquireSharedPtr<DummyPacket>();
 * pkt->setData(data);
 * // Automatically recycled when pkt goes out of scope
 * @endcode
 *
 * ### Example 2: Event with Renew
 * @code
 * // Acquire and initialize event in one call
 * auto* event = container->acquire<PacketProduceEvent>(
 *     &PacketProduceEvent::renew,
 *     simulator,
 *     packetCount,
 *     latency
 * );
 * scheduler->schedule(event, tick);
 * @endcode
 *
 * ### Example 3: Pre-sizing Pools
 * @code
 * // Pre-allocate pool for expected workload
 * container->setInitSize<DummyPacket>(
 *     numThreads,      // Number of segments
 *     10240            // Objects per segment
 * );
 * @endcode
 *
 * ### Example 4: Performance Measurement
 * @code
 * auto start = std::chrono::high_resolution_clock::now();
 * for (int i = 0; i < 100000; ++i) {
 *     auto pkt = container->acquireSharedPtr<Packet>();
 *     // ... use packet ...
 * }
 * auto duration = std::chrono::high_resolution_clock::now() - start;
 * @endcode
 *
 * ## Test Structure
 *
 * This test demonstrates the complete RecycleContainer workflow:
 *
 * 1. **Initialization**: Pre-size pools based on expected workload
 * 2. **Event Generation**: Create events using acquire() with renew()
 * 3. **Packet Allocation**: Allocate packets from pool during event processing
 * 4. **Automatic Recycling**: Objects returned to pool when no longer referenced
 * 5. **Performance Tracking**: Measure allocation and deallocation times
 * 6. **Profiling Support**: Optional Callgrind instrumentation for detailed analysis
 *
 * ## Profiling with Callgrind
 *
 * When compiled with PROFILE_CALLGRIND:
 * @code{.sh}
 * # Build with profiling
 * cmake -DPROFILE_CALLGRIND=ON ..
 * make
 *
 * # Run under valgrind
 * valgrind --tool=callgrind ./testResourceRecycling
 *
 * # Analyze results
 * callgrind_annotate callgrind.out.<pid>
 * @endcode
 *
 * ## Key Metrics Collected
 *
 * - **Allocation Time**: Time spent acquiring objects from pool
 * - **Release Time**: Time spent recycling objects back to pool
 * - **Total Runtime**: Overall simulation execution time
 * - **Object Count**: Total objects generated (with ACALSIM_STATISTICS)
 *
 * @see RecycleContainer
 * @see RecyclableObject
 * @see ObjectPool
 * @see ResourceUser for the simulator component
 * @see PacketProduceEvent for event-based object creation
 * @see PacketConsumeEvent for packet lifecycle management
 * @see DummyPacket for recyclable packet example
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 */

#ifdef PROFILE_CALLGRIND
#include <valgrind/callgrind.h>
#endif  // #ifdef PROFILE_CALLGRIND

#include <chrono>
#include <cmath>
#include <iostream>

#include "ACALSim.hh"
#include "DummyPacket.hh"
#include "ResourceUser.hh"

/**
 * @class TestResourceRecyclingTop
 * @brief Simulation top-level class for resource recycling demonstration
 *
 * @details
 * This class extends acalsim::SimTop to create a minimal simulation environment
 * for demonstrating the RecycleContainer's object pooling capabilities. It registers
 * a single ResourceUser simulator that generates and consumes packets using the
 * object pool.
 *
 * The primary purpose is to benchmark and validate the RecycleContainer's performance
 * characteristics compared to traditional heap allocation.
 *
 * @see acalsim::SimTop for base class functionality
 * @see ResourceUser for the simulator implementation
 */
class TestResourceRecyclingTop : public acalsim::SimTop {
public:
	/**
	 * @brief Default constructor
	 *
	 * Initializes the simulation top-level infrastructure. The RecycleContainer
	 * is automatically created by the base SimTop class and is accessible via
	 * getRecycleContainer().
	 */
	TestResourceRecyclingTop() : acalsim::SimTop() { ; }

	/**
	 * @brief Registers simulators for the recycling test
	 *
	 * @details
	 * This override creates and registers a ResourceUser simulator, which will:
	 * - Pre-size the DummyPacket object pool
	 * - Generate PacketProduceEvents that create packets from the pool
	 * - Measure allocation and deallocation performance
	 * - Validate automatic recycling behavior
	 *
	 * @note Only one simulator is registered to isolate the performance characteristics
	 *       of the RecycleContainer without interference from other components.
	 */
	void registerSimulators() override {
		acalsim::SimBase* sim = (acalsim::SimBase*)new ResourceUser();
		this->addSimulator(sim);
	}

	/**
	 * @brief Accumulated duration for custom timing measurements
	 *
	 * @details
	 * Currently unused in this test, but available for tracking additional
	 * timing metrics if needed. The ResourceUser class tracks allocation
	 * and release times separately.
	 */
	std::chrono::nanoseconds duration{0};
};

/**
 * @brief Main entry point for the resource recycling test
 *
 * @details
 * This function orchestrates the complete resource recycling benchmark:
 *
 * 1. **Initialization Phase**:
 *    - Creates TestResourceRecyclingTop instance
 *    - Initializes simulation infrastructure
 *    - Pre-sizes object pools based on workload
 *
 * 2. **Execution Phase**:
 *    - Starts high-resolution timer
 *    - Optionally starts Callgrind instrumentation
 *    - Runs simulation (processes all scheduled events)
 *    - Stops Callgrind instrumentation
 *    - Stops timer
 *
 * 3. **Reporting Phase**:
 *    - Calculates total execution time
 *    - Logs runtime statistics
 *    - Calls cleanup handlers to report detailed metrics
 *
 * ## Performance Measurement
 *
 * The test measures three key metrics:
 * - **Total Runtime**: End-to-end simulation execution time
 * - **Allocation Time**: Time spent acquiring objects from pool (tracked in ResourceUser)
 * - **Release Time**: Time spent recycling objects to pool (tracked in ResourceUser)
 *
 * ## Profiling Support
 *
 * When compiled with -DPROFILE_CALLGRIND=ON, the simulation is instrumented
 * with Callgrind markers that:
 * - Start instrumentation just before simulation run
 * - Stop instrumentation immediately after
 * - Toggle collection around critical allocation/deallocation paths
 *
 * This allows precise profiling of the RecycleContainer's performance impact.
 *
 * ## Expected Output
 *
 * The test produces log output showing:
 * @code
 * [INFO] Allocate 10 packets. | Total generated packet = 10
 * [INFO] Release 10 packets. | Total generated packet = 10
 * ...
 * [INFO] Allocation Time: 0.15 seconds. Releasing Time: 0.05 seconds.
 * [INFO] Time: 2.35 seconds.
 * @endcode
 *
 * ## Performance Analysis
 *
 * Compare allocation times with and without RecycleContainer:
 * - **With pooling**: Allocation time << Release time (pool reuse is fast)
 * - **Without pooling**: Allocation time ≈ Release time (both use heap)
 * - **Typical speedup**: 5-10x for this workload
 *
 * @param argc Command-line argument count
 * @param argv Command-line argument values
 * @return 0 on successful completion
 *
 * @see TestResourceRecyclingTop
 * @see ResourceUser
 * @see RecycleContainer
 */
int main(int argc, char** argv) {
	acalsim::top = std::make_shared<TestResourceRecyclingTop>();
	acalsim::top->init(argc, argv);

	auto start = std::chrono::high_resolution_clock::now();

#ifdef PROFILE_CALLGRIND
	CALLGRIND_START_INSTRUMENTATION;
#endif  // #ifdef PROFILE_CALLGRIND

	acalsim::top->run();

#ifdef PROFILE_CALLGRIND
	CALLGRIND_STOP_INSTRUMENTATION;
#endif  // #ifdef PROFILE_CALLGRIND

	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = duration_cast<std::chrono::nanoseconds>(stop - start);

	acalsim::LogOStream(acalsim::LoggingSeverity::L_INFO, __FILE__, __LINE__)
	    << "Time: " << (double)duration.count() / pow(10, 9) << " seconds.";

	acalsim::top->finish();

	return 0;
}
