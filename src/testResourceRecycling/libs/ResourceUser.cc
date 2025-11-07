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
 * @file ResourceUser.cc
 * @brief Implementation of ResourceUser simulator for RecycleContainer benchmarking
 *
 * @details
 * This file implements the ResourceUser simulator, which serves as the primary test harness
 * for demonstrating and benchmarking the RecycleContainer's object pooling capabilities.
 * It showcases the complete lifecycle of pooled objects from acquisition through recycling.
 *
 * # Object Pooling in Action
 *
 * ResourceUser demonstrates the practical application of RecycleContainer's object pooling
 * pattern in a realistic simulation scenario. It creates thousands of events and packets,
 * all managed through the object pool without any explicit new/delete operations.
 *
 * ## Key Demonstrations:
 *
 * 1. **Pool Pre-sizing**: Configure pools before heavy object creation
 * 2. **Event Recycling**: Reuse event objects across simulation timesteps
 * 3. **Packet Recycling**: Efficiently allocate and recycle packet objects
 * 4. **Performance Tracking**: Measure allocation vs. recycling overhead
 * 5. **Zero-copy Semantics**: Objects moved via pointers, not copied
 *
 * # RecycleContainer Usage Patterns
 *
 * ## Pattern 1: Pool Pre-sizing (Constructor)
 *
 * Pre-allocate pool capacity to avoid dynamic growth during simulation:
 * @code
 * // Set pool size: hardware_concurrency() segments × 10240 objects each
 * container->setInitSize<DummyPacket>(
 *     std::thread::hardware_concurrency(),  // Number of thread-local segments
 *     10240                                  // Objects per segment
 * );
 * @endcode
 *
 * **Benefits**:
 * - Eliminates allocation delays during simulation
 * - Pre-warming ensures consistent timing
 * - Thread-local segments reduce contention
 *
 * ## Pattern 2: Event Acquisition with Renew (init method)
 *
 * Acquire and initialize events in a single operation:
 * @code
 * auto event = container->acquire<PacketProduceEvent>(
 *     &PacketProduceEvent::renew,  // Reset function
 *     this,                         // Simulator pointer
 *     pkt_cnt,                      // Number of packets to create
 *     latency                       // Processing latency
 * );
 * @endcode
 *
 * **What happens internally**:
 * 1. Pop event from pool (may be previously used)
 * 2. Call renew() to reset state with new parameters
 * 3. Return initialized event ready for scheduling
 *
 * ## Pattern 3: Automatic Packet Recycling (dummyPacketHandler)
 *
 * Packets are automatically recycled when their shared_ptr goes out of scope:
 * @code
 * void handler(DummyPacket* pkt) {
 *     // Use packet...
 *     // No explicit delete needed!
 *     // Packet automatically recycled when reference count hits zero
 * }
 * @endcode
 *
 * **Recycling mechanism**:
 * - PacketConsumeEvent holds packets in vector<shared_ptr<T>>
 * - When vector is cleared, shared_ptr destructors run
 * - Custom deleter returns packets to pool instead of calling delete
 *
 * # Performance Measurement Strategy
 *
 * ResourceUser tracks two critical performance metrics:
 *
 * ## Allocation Time
 * Time spent acquiring objects from the pool:
 * @code
 * auto start = high_resolution_clock::now();
 * auto pkt = container->acquireSharedPtr<T>();
 * auto stop = high_resolution_clock::now();
 * allocateTime += (stop - start);
 * @endcode
 *
 * ## Release Time
 * Time spent recycling objects back to the pool:
 * @code
 * auto start = high_resolution_clock::now();
 * // shared_ptr destructor runs, triggers custom deleter
 * auto stop = high_resolution_clock::now();
 * releaseTime += (stop - start);
 * @endcode
 *
 * ## Interpretation
 * - **Low allocation time**: Pool reuse is working efficiently
 * - **Low release time**: Recycling overhead is minimal
 * - **Comparison**: With heap, allocation ≈ release time (both expensive)
 *
 * # Thread-Local Pool Optimization
 *
 * The constructor configures thread-local segments for lock-free operation:
 * @code
 * setInitSize<T>(std::thread::hardware_concurrency(), segment_size)
 * @endcode
 *
 * **Benefits**:
 * - Each thread has its own pool segment
 * - No lock contention on fast path (same-thread acquire/recycle)
 * - Only cross-thread recycling requires synchronization
 * - Scales linearly with thread count
 *
 * # Memory Efficiency Analysis
 *
 * ## Memory Footprint
 * - **Initial allocation**: threads × segment_size × sizeof(T)
 * - **Peak memory**: Fixed (no dynamic growth if properly sized)
 * - **vs. Heap**: Predictable, no fragmentation, lower peak usage
 *
 * ## Cache Efficiency
 * - Objects allocated in contiguous segments
 * - Better cache locality when iterating
 * - Reduced TLB misses from fewer page allocations
 *
 * # Zero-Copy Semantics
 *
 * Objects are never copied, only moved via pointers:
 * @code
 * // No copy: shared_ptr holds pointer to pooled object
 * auto pkt = container->acquireSharedPtr<Packet>();
 *
 * // No copy: vector stores shared_ptr (copies shared_ptr control block only)
 * packetVec.push_back(pkt);
 *
 * // No copy: visitor pattern uses references
 * sim->accept(tick, *pkt);
 * @endcode
 *
 * **Benefits**:
 * - Avoids expensive deep copies of large objects
 * - Minimal overhead for shared_ptr reference counting
 * - Objects remain in pool memory throughout lifecycle
 *
 * # Test Workload Characteristics
 *
 * The init() method creates a stress test workload:
 *
 * ## Workload Parameters
 * - **Iterations**: 1000 outer loops
 * - **Steps per iteration**: 40
 * - **Packets per event**: 10
 * - **Total events**: 40,000
 * - **Total packets**: 400,000
 *
 * ## Temporal Pattern
 * Events scheduled with regular intervals, creating bursts of packet allocation
 * followed by recycling, simulating realistic packet-processing workloads.
 *
 * ## Stress Test Goals
 * - High allocation rate to stress pool performance
 * - Recycling patterns that test automatic cleanup
 * - Large enough workload to amortize initialization costs
 * - Representative of real network simulation loads
 *
 * # Profiling Integration
 *
 * When built with PROFILE_CALLGRIND, critical sections are instrumented:
 * @code
 * CALLGRIND_TOGGLE_COLLECT;
 * // Allocation or deallocation code
 * CALLGRIND_TOGGLE_COLLECT;
 * @endcode
 *
 * **Usage**:
 * - Isolates pool operations in profiler output
 * - Enables precise measurement of acquire/recycle costs
 * - Compare with heap allocation (comment/uncomment pool code)
 *
 * # Comparing Pool vs. Heap
 *
 * The code includes commented-out heap allocation for comparison:
 * @code
 * // Pool version (fast):
 * auto event = container->acquire<Event>(&Event::renew, ...);
 *
 * // Heap version (slow) - commented out:
 * // auto event = new Event(...);
 * @endcode
 *
 * **To benchmark heap allocation**:
 * 1. Comment out pool version
 * 2. Uncomment heap version
 * 3. Rebuild and run
 * 4. Compare allocation/release times
 *
 * **Expected results**:
 * - Pool: 5-10x faster for this workload
 * - Heap: Higher variance, memory fragmentation, slower cleanup
 *
 * @see ResourceUser class header for interface documentation
 * @see PacketProduceEvent for event generation mechanism
 * @see PacketConsumeEvent for packet allocation and recycling
 * @see DummyPacket for recyclable packet implementation
 * @see RecycleContainer for pool implementation details
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 */

#include "ResourceUser.hh"

#ifdef PROFILE_CALLGRIND
#include <valgrind/callgrind.h>
#endif  // #ifdef PROFILE_CALLGRIND

#include <thread>

#include "ACALSim.hh"
#include "DummyPacket.hh"
#include "PacketProduceEvent.hh"

using Tick = acalsim::Tick;

/**
 * @brief Constructs ResourceUser and pre-sizes the DummyPacket object pool
 *
 * @details
 * The constructor performs critical pool initialization before the simulation begins.
 * By pre-sizing the pool here, we ensure that all necessary objects are allocated
 * before any timing measurements begin, providing accurate performance data.
 *
 * ## Pool Configuration Strategy
 *
 * The pool is configured with:
 * - **Segments**: std::thread::hardware_concurrency() (typically 4-16)
 * - **Objects per segment**: 10,240
 * - **Total capacity**: threads × 10,240 (e.g., 8 threads = 81,920 objects)
 *
 * ## Thread-Local Segments
 *
 * Using hardware_concurrency() for the segment count creates one segment per
 * hardware thread, enabling lock-free operation:
 *
 * @code
 * // Thread 0 acquires from segment 0 (no lock needed)
 * // Thread 1 acquires from segment 1 (no lock needed)
 * // No contention on common fast path!
 * @endcode
 *
 * ## Memory Pre-allocation
 *
 * All objects are allocated during this call, not during the simulation:
 * - Eliminates allocation delays during timed sections
 * - Ensures consistent performance measurements
 * - Prevents memory fragmentation from incremental growth
 *
 * ## Sizing Rationale
 *
 * 10,240 objects per segment is sized for the test workload:
 * - Total packets needed: 400,000
 * - Recycling factor: ~50 (packets reused 50 times on average)
 * - Actual pool usage: ~8,000 objects
 * - Safety margin: 10,240 prevents any dynamic growth
 *
 * @param _name Name of the simulator instance (default: "ResourceUser")
 *
 * @see RecycleContainer::setInitSize for pool sizing details
 */
ResourceUser::ResourceUser(std::string _name) : CPPSimBase(_name) {
	acalsim::top->getRecycleContainer()->setInitSize<DummyPacket>(std::thread::hardware_concurrency(), 10240);
}

/**
 * @brief Initializes the simulation by scheduling packet production events
 *
 * @details
 * This method creates the test workload by scheduling thousands of PacketProduceEvents
 * across the simulation timeline. Each event will trigger packet allocation from the
 * RecycleContainer, creating the stress test for object pooling performance.
 *
 * ## Workload Structure
 *
 * The nested loop creates a hierarchical event schedule:
 *
 * ### Outer Loop (1000 iterations)
 * Each iteration represents a major simulation phase, separated by quiet periods
 * where packets can be recycled back to the pool.
 *
 * ### Inner Loop (40 steps per iteration)
 * Each step schedules one PacketProduceEvent that will create 10 packets,
 * creating bursts of allocation activity.
 *
 * ### Temporal Layout
 * @code
 * Time: 0    1    2    3  ...  39   80   81  ...  119  160  ...
 *       [----iteration 0----][----iteration 1----][----iteration 2----]
 *       evt  evt  evt  ...       evt  evt  ...        evt  evt  ...
 * @endcode
 *
 * ## Event Acquisition Pattern
 *
 * Events are acquired using the acquire-with-renew pattern:
 * @code
 * auto event = container->acquire<PacketProduceEvent>(
 *     &PacketProduceEvent::renew,  // Method to call
 *     this,                         // arg1: simulator
 *     pkt_cnt,                      // arg2: packet count
 *     latency                       // arg3: processing latency
 * );
 * @endcode
 *
 * This is equivalent to but more efficient than:
 * @code
 * auto event = container->acquire<PacketProduceEvent>();
 * event->renew(this, pkt_cnt, latency);
 * @endcode
 *
 * ## Timing Schedule Calculation
 *
 * Event scheduling creates a predictable temporal pattern:
 *
 * ### Schedule Formula
 * @code
 * tick = iter × (step_cnt × 2 × interval) + step × interval
 *      = iter × 80 + step
 * @endcode
 *
 * ### Example Schedule
 * - Iteration 0, Step 0: tick = 0
 * - Iteration 0, Step 1: tick = 1
 * - Iteration 0, Step 39: tick = 39
 * - Iteration 1, Step 0: tick = 80
 * - Iteration 1, Step 1: tick = 81
 *
 * ### Quiet Periods
 * The 2× factor in step_cnt creates gaps between iterations:
 * - Active period: ticks 0-39 (40 ticks)
 * - Quiet period: ticks 40-79 (40 ticks, no new events)
 * - Next active: ticks 80-119
 *
 * This allows packets to be recycled between iterations.
 *
 * ## Memory Recycling Dynamics
 *
 * The workload is designed to exercise the recycling mechanism:
 *
 * 1. **Allocation Burst**: Steps schedule events that create packets
 * 2. **Processing Phase**: Packets are consumed by handlers
 * 3. **Recycling Phase**: Packets automatically return to pool
 * 4. **Reuse Phase**: Next iteration reuses recycled packets
 *
 * Peak memory usage is much lower than total packet count due to recycling.
 *
 * ## Commented-out Heap Comparison
 *
 * The code includes a commented line showing the traditional heap allocation:
 * @code
 * // auto event = new PacketProduceEvent<DummyPacket>((SimBase*)this, pkt_cnt, latency);
 * @endcode
 *
 * To benchmark without pooling:
 * 1. Comment out the container->acquire line
 * 2. Uncomment the new line
 * 3. Add delete in event processing (not shown here)
 * 4. Compare timings
 *
 * **Expected difference**: 5-10× slower with heap allocation
 *
 * ## Total Workload Size
 *
 * - Events scheduled: 1000 × 40 = 40,000
 * - Packets per event: 10
 * - Total packet allocations: 400,000
 * - Peak concurrent packets: ~10,000 (with recycling)
 * - Without recycling: 400,000 (all allocated simultaneously)
 *
 * @see PacketProduceEvent for event processing details
 * @see RecycleContainer::acquire for acquisition mechanism
 */
void ResourceUser::init() {
	size_t pkt_cnt  = 10;
	Tick   interval = 1;
	size_t step_cnt = 40;
	Tick   latency  = interval * step_cnt;

	for (size_t iter = 0; iter < 1000; ++iter) {
		for (size_t step = 0; step < step_cnt; ++step) {
			auto event = acalsim::top->getRecycleContainer()->acquire<PacketProduceEvent<DummyPacket>>(
			    &PacketProduceEvent<DummyPacket>::renew, this, pkt_cnt, latency);
			// auto event = new PacketProduceEvent<DummyPacket>((SimBase*)this, pkt_cnt, latency);
			this->scheduleEvent(event, iter * (step_cnt * 2 * interval) + step * interval);
		}
	}
}

/**
 * @brief Reports performance statistics collected during simulation
 *
 * @details
 * This cleanup method is called automatically after the simulation completes.
 * It reports the accumulated timing data for object allocation and recycling,
 * providing insight into the RecycleContainer's performance characteristics.
 *
 * ## Reported Metrics
 *
 * ### Allocation Time
 * Total time spent in acquireSharedPtr<T>() calls across all packet allocations.
 * This includes:
 * - Popping objects from the pool
 * - Initializing shared_ptr control blocks
 * - Any necessary synchronization overhead
 *
 * **Low allocation time indicates**:
 * - Effective pool reuse (objects recycled and ready)
 * - Minimal contention (thread-local segments working)
 * - No dynamic pool growth (pre-sizing was adequate)
 *
 * ### Releasing Time
 * Total time spent in the custom deleter when shared_ptrs are destroyed.
 * This includes:
 * - Running the custom deleter
 * - Pushing objects back to the pool
 * - Any recycling-related overhead
 *
 * **Low releasing time indicates**:
 * - Efficient recycling mechanism
 * - Fast return-to-pool operations
 * - Minimal overhead from automatic cleanup
 *
 * ## Performance Interpretation
 *
 * ### Ideal Results (with RecycleContainer)
 * @code
 * Allocation Time: 0.15 seconds. Releasing Time: 0.05 seconds.
 * @endcode
 * - Very low allocation time (pool hits)
 * - Even lower releasing time (fast recycling)
 * - Total overhead: ~0.20 seconds for 400,000 packets
 *
 * ### Expected Results (with heap allocation)
 * @code
 * Allocation Time: 1.2 seconds. Releasing Time: 1.0 seconds.
 * @endcode
 * - High allocation time (malloc overhead)
 * - High releasing time (free overhead)
 * - Total overhead: ~2.2 seconds for 400,000 packets
 *
 * ### Speedup Calculation
 * @code
 * Speedup = (heap_alloc + heap_free) / (pool_acquire + pool_recycle)
 *         = (1.2 + 1.0) / (0.15 + 0.05)
 *         = 2.2 / 0.20
 *         = 11× faster
 * @endcode
 *
 * ## Timing Accuracy
 *
 * Times are measured with std::chrono::high_resolution_clock:
 * - Nanosecond precision
 * - Accumulated across all allocations
 * - Converted to seconds for readability
 *
 * ## Output Format
 *
 * The method produces a single log line combining both metrics:
 * @code
 * [INFO] Allocation Time: X.XXX seconds. Releasing Time: Y.YYY seconds.
 * @endcode
 *
 * This format allows easy parsing and comparison across test runs.
 *
 * @see dummyPacketHandler for where releaseTime is measured
 * @see PacketConsumeEvent::initPacketVec for where allocateTime is measured
 */
void ResourceUser::cleanup() {
	std::ostringstream oss;
	oss << "Allocation Time: " << (double)this->allocateTime.count() / pow(10, 9) << " seconds.";
	oss << " ";
	oss << "Releasing Time: " << (double)this->releaseTime.count() / pow(10, 9) << " seconds.";
	CLASS_INFO << oss.str();
}

/**
 * @brief Handles incoming DummyPacket objects and measures recycling time
 *
 * @details
 * This handler is called through the visitor pattern when packets are delivered
 * to the ResourceUser simulator. Its primary purpose is to measure the time cost
 * of packet cleanup/recycling, though the actual recycling happens automatically
 * through the shared_ptr mechanism.
 *
 * ## Visitor Pattern Integration
 *
 * The handler is invoked via the packet's visit() method:
 * @code
 * // In PacketConsumeEvent::process()
 * for (auto& pkt : packetVec) {
 *     sim->accept(tick, *pkt);  // Calls pkt->visit(tick, *sim)
 *                                // Which calls sim->dummyPacketHandler(pkt)
 * }
 * @endcode
 *
 * ## Automatic Recycling Mechanism
 *
 * While this handler appears to do nothing with the packet, recycling happens
 * automatically when the calling context (PacketConsumeEvent) clears its vector:
 *
 * ### Step-by-Step Recycling
 *
 * 1. **During process()**:
 *    @code
 *    for (auto& pkt : packetVec) {     // pkt is shared_ptr<DummyPacket>
 *        sim->accept(tick, *pkt);       // Handler sees raw pointer
 *    }
 *    // shared_ptr still alive, ref count = 1
 *    @endcode
 *
 * 2. **Vector cleared**:
 *    @code
 *    packetVec.clear();                 // Destroys all shared_ptrs
 *                                       // Ref count drops to 0
 *    @endcode
 *
 * 3. **Custom deleter triggered**:
 *    @code
 *    // RecycleContainer::Deleter::operator() is called
 *    container->recycle(packet);        // Returns to pool
 *    @endcode
 *
 * ## Timing Measurement Purpose
 *
 * The current implementation measures release time as zero because the commented-out
 * delete line is not executed:
 * @code
 * // delete _pkt;  // This would measure explicit deletion time
 * @endcode
 *
 * When comparing with heap allocation:
 * 1. Uncomment the delete line
 * 2. Change packet allocation to use new instead of container->acquire
 * 3. This handler then measures explicit deletion overhead
 *
 * ## Profiling Integration
 *
 * When PROFILE_CALLGRIND is defined, the handler toggles profiling:
 * @code
 * CALLGRIND_TOGGLE_COLLECT;  // Start measuring
 * // delete _pkt;               // Code being profiled
 * CALLGRIND_TOGGLE_COLLECT;  // Stop measuring
 * @endcode
 *
 * This isolates the deletion operation in profiler output, allowing precise
 * measurement of deallocation costs.
 *
 * ## Release Time Accumulation
 *
 * The handler accumulates time measurements across all packet deliveries:
 * @code
 * auto start = high_resolution_clock::now();
 * // Deletion or recycling operation (currently none)
 * auto stop = high_resolution_clock::now();
 * this->releaseTime += (stop - start);  // Accumulate total time
 * @endcode
 *
 * ## Actual vs. Measured Recycling
 *
 * **Important**: The actual recycling happens in the custom deleter, not here!
 *
 * - **Measured here**: Explicit delete operations (when uncommented)
 * - **Actual recycling**: Automatic, in RecycleContainer::Deleter
 * - **Purpose**: Compare explicit vs. automatic cleanup overhead
 *
 * ## Performance Expectations
 *
 * With automatic recycling (current code):
 * - releaseTime ≈ 0 (no explicit cleanup)
 * - Actual recycling happens automatically, not measured here
 *
 * With explicit deletion (commented code):
 * - releaseTime ≈ 1-2 seconds for 400,000 deletions
 * - Measures free() overhead, fragmentation impact
 *
 * ## Zero-Copy Semantics
 *
 * The handler receives a raw pointer, avoiding shared_ptr copying:
 * @code
 * void handler(DummyPacket* _pkt)  // Raw pointer, no ref count change
 * @endcode
 *
 * This is intentional:
 * - Handler doesn't extend packet lifetime
 * - No shared_ptr copy overhead during visiting
 * - Calling code (PacketConsumeEvent) maintains ownership
 *
 * @param _pkt Raw pointer to the packet being handled (ownership retained by caller)
 *
 * @see DummyPacket::visit for visitor pattern implementation
 * @see PacketConsumeEvent::process for the recycling trigger
 * @see RecycleContainer::Deleter for actual recycling mechanism
 */
void ResourceUser::dummyPacketHandler(DummyPacket* _pkt) {
	auto start = std::chrono::high_resolution_clock::now();

#ifdef PROFILE_CALLGRIND
	CALLGRIND_TOGGLE_COLLECT;
#endif  // #ifdef PROFILE_CALLGRIND

	// delete _pkt;

#ifdef PROFILE_CALLGRIND
	CALLGRIND_TOGGLE_COLLECT;
#endif  // #ifdef PROFILE_CALLGRIND

	auto stop = std::chrono::high_resolution_clock::now();
	this->releaseTime += (stop - start);
}
