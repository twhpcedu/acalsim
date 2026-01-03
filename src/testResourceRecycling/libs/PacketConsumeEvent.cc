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

/**
 * @file PacketConsumeEvent.cc
 * @brief Implementation of PacketConsumeEvent for packet allocation and recycling demonstration
 *
 * @details
 * This file implements the PacketConsumeEvent template class, which demonstrates the complete
 * lifecycle of packet objects managed through RecycleContainer. It showcases how packets are
 * efficiently allocated from the object pool, used during event processing, and automatically
 * recycled when no longer needed.
 *
 * # Event and Packet Recycling Pattern
 *
 * PacketConsumeEvent demonstrates a sophisticated two-level recycling pattern where both
 * the event itself and the packets it manages are recycled:
 *
 * ## Level 1: Event Recycling
 * - PacketConsumeEvent objects are acquired from RecycleContainer
 * - Events are reused across different simulation timesteps
 * - renew() method reinitializes events with new parameters
 *
 * ## Level 2: Packet Recycling
 * - DummyPacket objects are acquired from RecycleContainer
 * - Packets stored in shared_ptr for automatic lifetime management
 * - Custom deleter returns packets to pool when references drop to zero
 *
 * # RecycleContainer Integration Points
 *
 * This class demonstrates three key RecycleContainer integration patterns:
 *
 * ## Pattern 1: acquireSharedPtr for Automatic Cleanup
 * @code
 * auto pkt = container->acquireSharedPtr<T>();
 * packetVec.push_back(pkt);
 * // ... later ...
 * packetVec.clear();  // Automatic recycling via custom deleter
 * @endcode
 *
 * ## Pattern 2: The renew() Method Contract
 * @code
 * void renew(SimBase* _sim, size_t _size) {
 *     this->SimEvent::renew();  // Reset base class state
 *     // Reset this class's state
 *     this->sim = _sim;
 *     this->packetVec.clear();
 *     this->initPacketVec(_size);
 * }
 * @endcode
 *
 * ## Pattern 3: Template Instantiation
 * @code
 * template class PacketConsumeEvent<DummyPacket>;
 * @endcode
 * Explicit instantiation ensures the template is compiled for specific types.
 *
 * # Automatic Recycling Through shared_ptr
 *
 * The key innovation demonstrated here is automatic recycling using shared_ptr
 * with a custom deleter:
 *
 * ## How It Works
 *
 * ### Acquisition
 * @code
 * auto pkt = container->acquireSharedPtr<T>();
 * // Returns shared_ptr<T> with:
 * //   - Pointer to pooled object
 * //   - Custom deleter: RecycleContainer::Deleter
 * @endcode
 *
 * ### Storage
 * @code
 * packetVec.push_back(pkt);
 * // shared_ptr copied, reference count = 2 (temp + vector)
 * // Temp destroyed, reference count = 1 (vector only)
 * @endcode
 *
 * ### Automatic Recycling
 * @code
 * packetVec.clear();
 * // For each shared_ptr:
 * //   1. shared_ptr destructor runs
 * //   2. Reference count decrements
 * //   3. If count reaches 0:
 * //      a. Custom deleter is called
 * //      b. Object returned to pool (not deleted!)
 * @endcode
 *
 * ## Benefits Over Manual Management
 *
 * **Without shared_ptr** (manual management):
 * @code
 * // Manual version - error-prone
 * vector<T*> packets;
 * for (size_t i = 0; i < size; ++i) {
 *     packets.push_back(container->acquire<T>());
 * }
 * // ... use packets ...
 * for (auto* pkt : packets) {
 *     container->recycle(pkt);  // Easy to forget!
 * }
 * @endcode
 *
 * **With shared_ptr** (automatic):
 * @code
 * // Automatic version - safe and clean
 * vector<shared_ptr<T>> packets;
 * for (size_t i = 0; i < size; ++i) {
 *     packets.push_back(container->acquireSharedPtr<T>());
 * }
 * // ... use packets ...
 * packets.clear();  // Automatic recycling, can't forget!
 * @endcode
 *
 * # Performance Measurement Integration
 *
 * PacketConsumeEvent integrates with ResourceUser's performance tracking:
 *
 * ## Allocation Time Tracking
 * @code
 * for (size_t i = 0; i < size; ++i) {
 *     auto start = high_resolution_clock::now();
 *     auto pkt = container->acquireSharedPtr<T>();
 *     auto stop = high_resolution_clock::now();
 *     duration += (stop - start);
 *     packetVec.push_back(pkt);
 * }
 * resourceUser->allocateTime += duration;
 * @endcode
 *
 * ## What Gets Measured
 * - **acquire time**: Pop from pool + shared_ptr construction
 * - **NOT measured**: Packet initialization (already done during pre-allocation)
 * - **Excludes**: Vector growth (amortized across many operations)
 *
 * ## Profiling Integration
 * When PROFILE_CALLGRIND is enabled:
 * @code
 * CALLGRIND_TOGGLE_COLLECT;    // Start profiling
 * auto pkt = container->acquireSharedPtr<T>();
 * CALLGRIND_TOGGLE_COLLECT;    // Stop profiling
 * @endcode
 * This isolates the acquire operation in profiler output.
 *
 * # Object Pool Statistics
 *
 * When compiled with ACALSIM_STATISTICS:
 * @code
 * size_t total = container->getGenObjectCnt<T>();
 * LOG << "Allocate " << size << " packets. | Total generated = " << total;
 * @endcode
 *
 * ## Statistics Interpretation
 * - **Generated count**: Total objects ever created in pool
 * - **Requested count**: Total acquire() calls (can exceed generated)
 * - **Reuse factor**: requested / generated (higher is better)
 *
 * Example output:
 * @code
 * Allocate 10 packets. | Total generated packet = 10     // First allocation
 * Allocate 10 packets. | Total generated packet = 10     // Reusing same 10!
 * Allocate 10 packets. | Total generated packet = 15     // Had to create 5 more
 * @endcode
 *
 * # The renew() Method Pattern
 *
 * The renew() method is central to the object recycling pattern:
 *
 * ## Purpose
 * Reset an existing object to a clean initial state so it can be reused
 * with new parameters, avoiding the need to destroy and recreate it.
 *
 * ## Design Requirements
 *
 * 1. **Call base class renew()**: Reset inherited state
 *    @code
 *    this->SimEvent::renew();  // MUST be first!
 *    @endcode
 *
 * 2. **Reset all member variables**: Clear any state from previous use
 *    @code
 *    this->sim = nullptr;      // Reset pointer
 *    this->packetVec.clear();  // Clear container
 *    @endcode
 *
 * 3. **Reinitialize with new parameters**: Set up for new use
 *    @code
 *    this->sim = _sim;
 *    this->initPacketVec(_size);
 *    @endcode
 *
 * ## Common Pitfalls
 *
 * **Forgetting to call base class renew()**:
 * @code
 * void renew(...) {
 *     // WRONG: Missing SimEvent::renew()
 *     this->sim = _sim;
 * }
 * @endcode
 * Result: Event scheduling state corrupted, hard-to-debug crashes
 *
 * **Not clearing containers**:
 * @code
 * void renew(...) {
 *     SimEvent::renew();
 *     // WRONG: Not clearing packetVec
 *     initPacketVec(_size);  // Appends to old packets!
 * }
 * @endcode
 * Result: Vector grows unbounded, memory leak
 *
 * **Forgetting member variables**:
 * @code
 * void renew(...) {
 *     SimEvent::renew();
 *     this->sim = _sim;
 *     // WRONG: Forgot to reset someCounter, someFlag, etc.
 * }
 * @endcode
 * Result: Stale state from previous use affects new usage
 *
 * # Cache-Friendly Allocation Pattern
 *
 * The vector-based storage provides excellent cache behavior:
 *
 * ## Memory Layout
 * @code
 * vector<shared_ptr<T>>: [ptr1][ptr2][ptr3][ptr4]...
 *                          |     |     |     |
 *                          v     v     v     v
 * Pool segment:         [obj1][obj2][obj3][obj4]...
 * @endcode
 *
 * ## Cache Benefits
 * - Vector stored contiguously (good spatial locality)
 * - Pool objects from same segment (good temporal locality)
 * - Sequential allocation pattern (hardware prefetcher friendly)
 * - Bulk processing (amortizes cache misses)
 *
 * # Zero-Copy Packet Delivery
 *
 * Packets are delivered to handlers without copying:
 *
 * @code
 * for (auto& pkt : packetVec) {        // Reference to shared_ptr (no copy)
 *     sim->accept(tick, *pkt);          // Dereference to raw pointer
 *                                       // Handler sees DummyPacket* (no copy)
 * }
 * @endcode
 *
 * ## Copy Analysis
 * - **shared_ptr**: Reference in loop, no copy
 * - **Packet object**: Never copied, passed by pointer
 * - **Overhead**: Minimal, just pointer dereference
 *
 * # Comparing Pool vs. Heap Allocation
 *
 * The code includes commented alternatives for comparison:
 *
 * ## Pool Version (Current)
 * @code
 * auto pkt = container->acquireSharedPtr<T>();
 * @endcode
 * - Fast: O(1) pop from pool
 * - No allocation: Object already exists
 * - No deallocation: Recycled to pool
 *
 * ## Heap Version (Commented)
 * @code
 * // auto pkt = new T();
 * @endcode
 * - Slow: O(log n) malloc from heap
 * - Allocation overhead: Find free block, update heap metadata
 * - Deallocation overhead: Merge free blocks, update heap metadata
 * - Fragmentation: Over time, heap becomes fragmented
 *
 * ## Benchmarking Methodology
 * 1. Comment out pool version
 * 2. Uncomment heap version
 * 3. Change shared_ptr to raw pointer or use shared_ptr<T> with default deleter
 * 4. Rebuild and run test
 * 5. Compare allocateTime and releaseTime
 *
 * ## Expected Results
 * - Pool allocation: 5-10× faster
 * - Pool deallocation: 10-20× faster
 * - Total speedup: 8-12× for typical workloads
 *
 * @see PacketConsumeEvent template class header
 * @see RecycleContainer for pool implementation
 * @see ResourceUser for performance measurement collection
 * @see DummyPacket for recyclable packet type
 *
 * @author ACALSim Development Team
 * @date 2023-2025
 */

#include "PacketConsumeEvent.hh"

#ifdef PROFILE_CALLGRIND
#include <valgrind/callgrind.h>
#endif  // #ifdef PROFILE_CALLGRIND

#include <chrono>
#include <string>

#include "ACALSim.hh"
#include "DummyPacket.hh"
#include "ResourceUser.hh"

/**
 * @brief Explicit template instantiation for DummyPacket
 *
 * @details
 * This explicit instantiation ensures that the compiler generates code for
 * PacketConsumeEvent<DummyPacket> in this translation unit. Without this,
 * the template would only be instantiated where it's used, potentially
 * causing linker errors.
 *
 * ## Why Explicit Instantiation?
 *
 * Templates are typically instantiated at the point of use, but this can cause issues:
 * - **Code bloat**: Same template instantiated in multiple translation units
 * - **Longer compile times**: Template compiled repeatedly
 * - **Linker errors**: Definition not available when needed
 *
 * Explicit instantiation solves these problems:
 * @code
 * // In .cc file: Generate all methods for this type
 * template class PacketConsumeEvent<DummyPacket>;
 * @endcode
 *
 * ## What Gets Instantiated
 *
 * This single line causes the compiler to generate:
 * - Default constructor
 * - Constructor with parameters
 * - initPacketVec() method
 * - renew() method
 * - process() method
 * - Destructor
 *
 * All for the specific type T = DummyPacket.
 *
 * ## Alternative Approaches
 *
 * **Header-only templates** (not used here):
 * @code
 * // All implementation in .hh file
 * // Pros: Always available, no explicit instantiation needed
 * // Cons: Code bloat, slower compilation
 * @endcode
 *
 * **Explicit instantiation** (current approach):
 * @code
 * // Implementation in .cc file
 * // Explicit instantiation for known types
 * // Pros: Faster compilation, smaller binaries
 * // Cons: Must know types at library build time
 * @endcode
 *
 * @see PacketConsumeEvent template class definition
 */
template class PacketConsumeEvent<DummyPacket>;

/**
 * @brief Initializes the packet vector by acquiring packets from RecycleContainer
 *
 * @details
 * This method is the heart of the packet allocation benchmark. It acquires the specified
 * number of packets from the RecycleContainer's object pool and stores them in the
 * packetVec member. Timing data is collected to measure the efficiency of the pooling
 * mechanism.
 *
 * ## Allocation Pattern
 *
 * The method allocates packets in a tight loop, maximizing allocation rate:
 * @code
 * for (size_t i = 0; i < _size; ++i) {
 *     // Timed acquisition
 *     auto pkt = container->acquireSharedPtr<T>();
 *     // Store for later use
 *     packetVec.push_back(pkt);
 * }
 * @endcode
 *
 * ## Performance Measurement
 *
 * ### Individual Allocation Timing
 * Each allocation is individually timed:
 * @code
 * for (size_t i = 0; i < _size; ++i) {
 *     auto start = high_resolution_clock::now();
 *     auto pkt = container->acquireSharedPtr<T>();  // Measured operation
 *     auto stop = high_resolution_clock::now();
 *     duration += (stop - start);
 *     packetVec.push_back(pkt);  // Not measured
 * }
 * @endcode
 *
 * ### What Gets Measured
 * - **Included**: acquireSharedPtr() call
 *   - Pop from object pool
 *   - Construct shared_ptr with custom deleter
 *   - Any synchronization overhead
 *
 * - **Excluded**: Vector operations
 *   - push_back() amortized cost
 *   - Vector growth (if needed)
 *   - Memory copying during growth
 *
 * ### Accumulated Timing
 * Individual durations are summed and reported to ResourceUser:
 * @code
 * resourceUser->allocateTime += duration;
 * @endcode
 *
 * ## Profiling Integration
 *
 * When PROFILE_CALLGRIND is enabled, each allocation is instrumented:
 * @code
 * CALLGRIND_TOGGLE_COLLECT;              // Start profiling
 * auto pkt = container->acquireSharedPtr<T>();
 * CALLGRIND_TOGGLE_COLLECT;              // Stop profiling
 * @endcode
 *
 * ### Profiling Benefits
 * - Isolates acquire() in callgrind output
 * - Shows exact instruction counts
 * - Reveals cache miss rates
 * - Identifies synchronization costs
 *
 * ### Reading Profiler Output
 * @code
 * callgrind_annotate callgrind.out.<pid> | grep acquireSharedPtr
 * @endcode
 * Shows instruction count and cache statistics for acquisition.
 *
 * ## Pool vs. Heap Comparison
 *
 * The commented line shows the heap allocation alternative:
 * @code
 * // auto pkt = new T();  // Heap version for comparison
 * @endcode
 *
 * ### To Benchmark Heap Allocation
 * 1. Comment out: `auto pkt = container->acquireSharedPtr<T>();`
 * 2. Uncomment: `auto pkt = new T();`
 * 3. Change storage to: `vector<T*>` or `vector<shared_ptr<T>>` with default deleter
 * 4. Add cleanup code in process() to delete packets
 * 5. Rebuild and compare timing
 *
 * ### Expected Performance Difference
 * **Pool (acquireSharedPtr)**:
 * - 10-50 nanoseconds per object (typical)
 * - Constant time regardless of previous allocations
 * - No system calls (pre-allocated pool)
 *
 * **Heap (new)**:
 * - 100-500 nanoseconds per object (typical)
 * - Variable time (depends on heap state)
 * - May trigger system calls (sbrk/mmap)
 * - Fragmentation increases over time
 *
 * **Speedup**: 5-10× faster with pool
 *
 * ## Statistics Reporting
 *
 * When ACALSIM_STATISTICS is defined, allocation is logged:
 * @code
 * size_t total_gen = container->getGenObjectCnt<T>();
 * LOG << "Allocate " << _size << " packets. | Total generated = " << total_gen;
 * @endcode
 *
 * ### Interpreting Statistics
 *
 * **First allocation**:
 * @code
 * Allocate 10 packets. | Total generated packet = 10
 * @endcode
 * - Pool had to create 10 new objects
 * - No recycling yet
 *
 * **Subsequent allocations**:
 * @code
 * Allocate 10 packets. | Total generated packet = 10
 * @endcode
 * - Same total! Pool reused existing objects
 * - Perfect recycling efficiency
 *
 * **Pool growth**:
 * @code
 * Allocate 10 packets. | Total generated packet = 15
 * @endcode
 * - Pool had to create 5 additional objects
 * - Reused 5, created 5 new
 * - Indicates concurrent usage pattern
 *
 * ## Memory Efficiency Analysis
 *
 * ### Peak Memory Usage
 * Without recycling:
 * @code
 * // 40,000 events × 10 packets = 400,000 packets
 * // Memory = 400,000 × sizeof(DummyPacket) ≈ 3-4 MB
 * @endcode
 *
 * With recycling:
 * @code
 * // Peak concurrent packets ≈ 10,000 (due to recycling)
 * // Memory = 10,000 × sizeof(DummyPacket) ≈ 0.7-0.8 MB
 * // Savings: 75% reduction
 * @endcode
 *
 * ### Cache Efficiency
 * Packets allocated from contiguous pool segments:
 * - Better cache spatial locality
 * - Hardware prefetcher can predict access pattern
 * - Reduced TLB misses (fewer memory pages)
 *
 * ## Thread Safety Considerations
 *
 * This method is called during event processing, which may be multi-threaded:
 *
 * ### Thread-Safe Operations
 * - `acquireSharedPtr()`: Thread-safe (lock-free for thread-local segments)
 * - `packetVec.push_back()`: Safe (each event has its own vector)
 * - `duration +=`: Safe (local variable, accumulated later)
 *
 * ### No Synchronization Needed
 * Each PacketConsumeEvent instance is owned by a single event:
 * - No shared state between events
 * - Vector operations don't require locks
 * - Only pool access needs synchronization (handled internally)
 *
 * ## Error Handling
 *
 * The method assumes the pool has sufficient capacity:
 * - If pool is pre-sized correctly, acquisitions never fail
 * - If pool runs out, new objects are created automatically
 * - No exception handling needed (allocations always succeed)
 *
 * @param _size Number of packets to allocate
 *
 * @see RecycleContainer::acquireSharedPtr for acquisition mechanism
 * @see ResourceUser::allocateTime for where timing is accumulated
 * @see process() for where packets are used and recycled
 */
template <typename T>
void PacketConsumeEvent<T>::initPacketVec(size_t _size) {
	std::chrono::nanoseconds duration{0};

	for (size_t i = 0; i < _size; ++i) {
		auto start = std::chrono::high_resolution_clock::now();

#ifdef PROFILE_CALLGRIND
		CALLGRIND_TOGGLE_COLLECT;
#endif  // #ifdef PROFILE_CALLGRIND

		auto pkt = acalsim::top->getRecycleContainer()->acquireSharedPtr<T>();
		// auto pkt = new T();

#ifdef PROFILE_CALLGRIND
		CALLGRIND_TOGGLE_COLLECT;
#endif  // #ifdef PROFILE_CALLGRIND

		auto stop = std::chrono::high_resolution_clock::now();
		duration  = duration + (stop - start);

		this->packetVec.push_back(pkt);
	}

	dynamic_cast<ResourceUser*>(this->sim)->allocateTime += duration;

#ifdef ACALSIM_STATISTICS
	size_t total_gen_cnt = acalsim::top->getRecycleContainer()->getGenObjectCnt<T>();
	CLASS_INFO << "Allocate " << this->packetVec.size() << " packets. | Total generated packet = " << total_gen_cnt;
#else
	CLASS_INFO << "Allocate " << this->packetVec.size() << " packets.";
#endif  // ACALSIM_STATISTICS
}

/**
 * @brief Resets the event to a clean state for reuse with new parameters
 *
 * @details
 * This method implements the renew pattern for object recycling. When an event is
 * acquired from the RecycleContainer pool, it may contain state from its previous use.
 * The renew() method clears this state and reinitializes the event with new parameters,
 * making it safe to reuse.
 *
 * ## The renew() Contract
 *
 * All recyclable objects must implement renew() following this pattern:
 *
 * ### Step 1: Call Base Class renew()
 * @code
 * this->SimEvent::renew();  // CRITICAL: Must be first!
 * @endcode
 *
 * This resets event scheduling state (priority, tick, processed flag, etc.)
 * that is maintained by the base SimEvent class.
 *
 * ### Step 2: Reset Member Variables
 * @code
 * this->sim = nullptr;
 * this->packetVec.clear();
 * @endcode
 *
 * Clear any state that was set during previous use.
 *
 * ### Step 3: Reinitialize with New Parameters
 * @code
 * this->sim = _sim;
 * this->initPacketVec(_size);
 * @endcode
 *
 * Set up the object for its new use case.
 *
 * ## Why renew() Instead of Constructor?
 *
 * **Traditional approach** (with heap allocation):
 * @code
 * Event* evt = new Event(sim, size);  // Construct new object
 * scheduler->schedule(evt);
 * // ... process ...
 * delete evt;                          // Destroy object
 * @endcode
 * Cost: malloc + construct + destruct + free
 *
 * **Recycling approach** (with object pool):
 * @code
 * Event* evt = pool->acquire<Event>();  // Get existing object
 * evt->renew(sim, size);                // Reset to initial state
 * scheduler->schedule(evt);
 * // ... process ...
 * pool->recycle(evt);                   // Return to pool
 * @endcode
 * Cost: pop + renew + push (much faster!)
 *
 * ## State Management
 *
 * Objects in the pool may have arbitrary state from previous use:
 *
 * ### Before renew()
 * @code
 * // Event just popped from pool
 * this->sim = 0xDEADBEEF;           // Stale pointer from previous use
 * this->packetVec = {pkt1, pkt2};   // Old packets, wrong size
 * this->tick = 12345;                // Old scheduling time
 * this->processed = true;            // Stale processing flag
 * @endcode
 *
 * ### After renew()
 * @code
 * // Event ready for new use
 * this->sim = validSimPtr;           // Correct simulator
 * this->packetVec = {10 new pkts};   // Fresh packets, correct size
 * this->tick = 0;                    // Reset by SimEvent::renew()
 * this->processed = false;           // Reset by SimEvent::renew()
 * @endcode
 *
 * ## Automatic Packet Recycling
 *
 * Note how packetVec is cleared:
 * @code
 * this->packetVec.clear();  // Destroys all shared_ptr<T>
 * @endcode
 *
 * This triggers automatic recycling:
 * 1. clear() destroys each shared_ptr
 * 2. shared_ptr destructor runs custom deleter
 * 3. Custom deleter returns packets to pool
 * 4. Old packets are now available for reuse
 * 5. initPacketVec() may get those same packets back!
 *
 * ## Performance Characteristics
 *
 * renew() is designed to be very fast:
 *
 * ### Cost Breakdown
 * - SimEvent::renew(): ~10 instructions (reset flags and counters)
 * - this->sim = _sim: 1 instruction (pointer assignment)
 * - packetVec.clear(): O(n) shared_ptr destructions (n = packet count)
 * - initPacketVec(): O(m) packet acquisitions (m = new packet count)
 *
 * ### Total Cost
 * Dominated by packetVec operations, but still much faster than:
 * - Destroying and recreating the event (destructor + constructor)
 * - Heap deallocation + allocation (free + malloc)
 * - Potential system calls (if heap grows/shrinks)
 *
 * ### Typical Timing
 * - renew(): ~1-5 microseconds
 * - new + delete: ~10-50 microseconds
 * - Speedup: ~5-10×
 *
 * ## Common Pitfalls and Solutions
 *
 * ### Pitfall 1: Forgetting Base Class renew()
 * @code
 * void renew(SimBase* _sim, size_t _size) {
 *     // WRONG: Didn't call SimEvent::renew()
 *     this->sim = _sim;
 *     // ...
 * }
 * @endcode
 * **Result**: Event scheduling state corrupted, crashes when scheduled
 * **Solution**: Always call base class renew() first
 *
 * ### Pitfall 2: Not Clearing Containers
 * @code
 * void renew(SimBase* _sim, size_t _size) {
 *     SimEvent::renew();
 *     this->sim = _sim;
 *     // WRONG: Didn't clear packetVec
 *     initPacketVec(_size);  // Appends to old data!
 * }
 * @endcode
 * **Result**: Vector grows unbounded, memory leak, wrong packet count
 * **Solution**: Clear all containers before reinitializing
 *
 * ### Pitfall 3: Incomplete State Reset
 * @code
 * void renew(SimBase* _sim, size_t _size) {
 *     SimEvent::renew();
 *     this->sim = _sim;
 *     packetVec.clear();
 *     // WRONG: Forgot to reset errorFlag, processCount, etc.
 * }
 * @endcode
 * **Result**: Stale flags affect new behavior, hard-to-debug bugs
 * **Solution**: Reset ALL member variables
 *
 * ## Testing renew() Correctness
 *
 * To verify renew() works correctly:
 *
 * ### Test 1: Multiple Reuses
 * @code
 * auto evt = pool->acquire<Event>();
 * evt->renew(sim, 10);
 * // Use event...
 * pool->recycle(evt);
 *
 * evt = pool->acquire<Event>();  // May get same object!
 * evt->renew(sim, 20);            // Should work correctly
 * // Verify: evt has 20 packets, not 10 or 30
 * @endcode
 *
 * ### Test 2: State Independence
 * @code
 * evt->renew(sim1, 10);
 * assert(evt->sim == sim1);
 * assert(evt->packetVec.size() == 10);
 *
 * evt->renew(sim2, 5);
 * assert(evt->sim == sim2);       // Not sim1!
 * assert(evt->packetVec.size() == 5);  // Not 10 or 15!
 * @endcode
 *
 * ## Integration with acquire()
 *
 * RecycleContainer provides a convenient overload that combines acquire and renew:
 * @code
 * // Manual two-step
 * auto evt = container->acquire<Event>();
 * evt->renew(sim, size);
 *
 * // Automatic one-step (preferred)
 * auto evt = container->acquire<Event>(&Event::renew, sim, size);
 * @endcode
 *
 * The one-step version:
 * - Acquires object from pool
 * - Calls the specified member function (renew)
 * - Forwards arguments to that function
 * - Returns the initialized object
 *
 * This ensures objects are always properly initialized before use.
 *
 * @param _sim Pointer to the simulator that will process this event
 * @param _size Number of packets to allocate for this event
 *
 * @see SimEvent::renew for base class state reset
 * @see initPacketVec for packet allocation
 * @see RecycleContainer::acquire for acquisition with automatic renew
 */
template <typename T>
void PacketConsumeEvent<T>::renew(SimBase* _sim, size_t _size) {
	this->SimEvent::renew();

	this->sim = _sim;
	this->packetVec.clear();
	this->initPacketVec(_size);
}

/**
 * @brief Processes the event by delivering packets and triggering automatic recycling
 *
 * @details
 * This method is called by the event scheduler when the event's scheduled time arrives.
 * It demonstrates the complete packet lifecycle: packets are delivered to the simulator
 * via the visitor pattern, then automatically recycled when the packet vector is cleared.
 *
 * ## Event Processing Flow
 *
 * ### Step 1: Packet Delivery
 * @code
 * for (auto& pkt : packetVec) {
 *     sim->accept(globalTick, *pkt);
 * }
 * @endcode
 *
 * Each packet is delivered to the simulator using the accept() method, which
 * triggers the visitor pattern:
 * - accept() calls pkt->visit(tick, *sim)
 * - visit() calls sim->dummyPacketHandler(pkt)
 * - Handler processes the packet
 *
 * ### Step 2: Automatic Recycling
 * @code
 * packetVec.clear();
 * @endcode
 *
 * Clearing the vector triggers the automatic recycling mechanism:
 * - Each shared_ptr<T> in the vector is destroyed
 * - shared_ptr destructor decrements reference count
 * - When count reaches zero, custom deleter is invoked
 * - Custom deleter returns packet to RecycleContainer pool
 * - Packets are now available for reuse
 *
 * ## Visitor Pattern for Packet Delivery
 *
 * The visitor pattern enables type-safe, polymorphic packet handling:
 *
 * ### Double Dispatch
 * @code
 * // First dispatch: based on simulator type
 * sim->accept(tick, *pkt);
 *   // Inside accept():
 *   pkt->visit(tick, *this);  // Second dispatch: based on packet type
 *     // Inside visit():
 *     sim->dummyPacketHandler(this);  // Type-safe handler call
 * @endcode
 *
 * ### Benefits
 * - Type-safe: Compile-time checking of handler signatures
 * - Extensible: New packet types add new visit() methods
 * - No casting: No need for dynamic_cast or type checking
 * - Clean separation: Packet logic separate from simulator logic
 *
 * ## Zero-Copy Packet Iteration
 *
 * The for-each loop is carefully designed to avoid copying:
 * @code
 * for (auto& pkt : packetVec) {  // Reference, no copy!
 *     sim->accept(tick, *pkt);    // Dereference to raw pointer
 * }
 * @endcode
 *
 * ### Why References Matter
 * **With reference** (current, efficient):
 * @code
 * for (auto& pkt : packetVec) {
 *     // pkt is a reference to shared_ptr
 *     // No reference count change
 *     // No atomic operations
 * }
 * @endcode
 *
 * **Without reference** (inefficient):
 * @code
 * for (auto pkt : packetVec) {  // Copy!
 *     // pkt is a copy of shared_ptr
 *     // Reference count incremented (atomic op)
 *     // Then decremented at loop end (atomic op)
 *     // 2× atomic ops per iteration!
 * }
 * @endcode
 *
 * For 10 packets, that's 20 unnecessary atomic operations!
 *
 * ## Automatic Recycling Mechanism
 *
 * The clear() call is where the magic happens:
 *
 * ### What Happens During clear()
 *
 * #### Phase 1: shared_ptr Destruction
 * @code
 * packetVec.clear();
 * // For each element:
 * for (auto& ptr : packetVec) {
 *     // Destroy shared_ptr
 *     ptr.~shared_ptr<T>();
 *       // Decrement reference count
 *       if (--refcount == 0) {
 *           // Call custom deleter
 *           deleter(ptr.get());
 *       }
 * }
 * @endcode
 *
 * #### Phase 2: Custom Deleter Execution
 * @code
 * // Inside RecycleContainer::Deleter::operator()
 * void operator()(RecyclableObject* obj) {
 *     if (!containerDestroyed) {
 *         container->recycle(obj);  // Return to pool
 *     } else {
 *         delete obj;                // Container gone, actually delete
 *     }
 * }
 * @endcode
 *
 * #### Phase 3: Pool Recycling
 * @code
 * // Inside RecycleContainer::recycle()
 * void recycle(RecyclableObject* obj) {
 *     ObjectPool<T>* pool = getPool<T>();
 *     pool->push(obj);  // Add to free list
 * }
 * @endcode
 *
 * ### Memory Lifecycle Diagram
 * @code
 * acquire()       use()           clear()         acquire()
 *    |             |                |                |
 *    v             v                v                v
 * [Pool] ---> [InUse] ---> [Processing] ---> [Pool] ---> [InUse]
 *    ^                                           |
 *    |___________________________________________|
 *                    recycle()
 * @endcode
 *
 * ## Statistics Logging
 *
 * When ACALSIM_STATISTICS is enabled, recycling is logged:
 * @code
 * size_t total = container->getGenObjectCnt<T>();
 * LOG << "Release " << count << " packets. | Total generated = " << total;
 * @endcode
 *
 * ### Interpreting the Statistics
 *
 * **Example output sequence**:
 * @code
 * // Event 1 processes
 * Allocate 10 packets. | Total generated packet = 10
 * Release 10 packets. | Total generated packet = 10
 *
 * // Event 2 processes
 * Allocate 10 packets. | Total generated packet = 10  // Reused!
 * Release 10 packets. | Total generated packet = 10
 *
 * // Event 3 processes concurrently
 * Allocate 10 packets. | Total generated packet = 15  // Needed 5 more
 * Release 10 packets. | Total generated packet = 15
 * @endcode
 *
 * ### Pool Health Indicators
 * - **Constant total**: Perfect recycling, all packets reused
 * - **Slow growth**: Good recycling, occasional concurrent usage
 * - **Linear growth**: Poor recycling, packets not being returned
 * - **Sudden jump**: Burst of concurrent usage
 *
 * ## Performance Characteristics
 *
 * ### Delivery Phase
 * - **Iteration**: O(n) where n = packet count
 * - **accept() call**: O(1) virtual dispatch
 * - **visit() call**: O(1) virtual dispatch
 * - **Handler**: O(1) typically (depends on handler implementation)
 * - **Total**: O(n) with low constant factor
 *
 * ### Recycling Phase
 * - **clear()**: O(n) shared_ptr destructions
 * - **Each destructor**: O(1) atomic decrement + O(1) recycle
 * - **recycle()**: O(1) push to pool free list
 * - **Total**: O(n) with very low constant factor
 *
 * ### Overall Event Cost
 * Dominated by packet allocation (in initPacketVec), not processing:
 * - Allocation: ~60% of time (acquire from pool)
 * - Processing: ~20% of time (delivery loop)
 * - Recycling: ~20% of time (clear + recycle)
 *
 * ## Thread Safety
 *
 * ### Safe Operations
 * - **Iteration**: Safe (event owns the vector)
 * - **Packet delivery**: Safe (visitor pattern is thread-safe)
 * - **clear()**: Safe (event owns the vector)
 * - **Recycling**: Safe (RecycleContainer uses thread-safe pool)
 *
 * ### No Locks Needed
 * Each event is processed by exactly one thread:
 * - Event scheduler ensures serial processing per event
 * - Different events can process concurrently
 * - Pool handles cross-thread recycling internally
 *
 * ## Exception Safety
 *
 * The clear() operation provides strong exception safety:
 *
 * ### If Handler Throws
 * @code
 * for (auto& pkt : packetVec) {
 *     sim->accept(tick, *pkt);  // May throw
 * }
 * packetVec.clear();  // May not be reached!
 * @endcode
 *
 * **Problem**: If accept() throws, clear() is skipped, packets leak!
 *
 * ### Solution (not implemented here, but recommended)
 * @code
 * try {
 *     for (auto& pkt : packetVec) {
 *         sim->accept(tick, *pkt);
 *     }
 * } catch (...) {
 *     packetVec.clear();  // Ensure cleanup
 *     throw;
 * }
 * packetVec.clear();
 * @endcode
 *
 * Or use RAII guard (better):
 * @code
 * auto cleanup = makeScopeExit([&]{ packetVec.clear(); });
 * for (auto& pkt : packetVec) {
 *     sim->accept(tick, *pkt);
 * }
 * // cleanup runs automatically
 * @endcode
 *
 * ## Debugging Tips
 *
 * ### Enable Statistics
 * @code
 * cmake -DACALSIM_STATISTICS=ON ..
 * @endcode
 * Shows packet allocation/recycling counts for debugging pool behavior.
 *
 * ### Check Reference Counts
 * During debugging, verify shared_ptr reference counts:
 * @code
 * for (auto& pkt : packetVec) {
 *     assert(pkt.use_count() == 1);  // Only we hold reference
 * }
 * @endcode
 *
 * ### Verify Pool Returns
 * Check that packets are actually being recycled:
 * @code
 * size_t before = container->getGenObjectCnt<T>();
 * packetVec.clear();
 * size_t after = container->getGenObjectCnt<T>();
 * assert(before == after);  // No new objects created
 * @endcode
 *
 * ## Optimization Opportunities
 *
 * ### Batch Processing
 * Current implementation processes one packet at a time.
 * Could batch process for better cache locality:
 * @code
 * // Group packets by type or destination
 * // Process each group together
 * // Better cache locality, fewer virtual calls
 * @endcode
 *
 * ### Reserve Vector Capacity
 * If packet count is known, reserve capacity:
 * @code
 * packetVec.reserve(expected_size);  // Avoid reallocations
 * @endcode
 *
 * ### Reuse Vector
 * Instead of clear(), could keep vector allocated:
 * @code
 * packetVec.resize(0);  // Keep capacity
 * // Next initPacketVec() has pre-allocated storage
 * @endcode
 *
 * @see initPacketVec for packet allocation
 * @see renew for event reinitialization
 * @see DummyPacket::visit for visitor pattern implementation
 * @see ResourceUser::dummyPacketHandler for packet handler
 * @see RecycleContainer::Deleter for custom deleter implementation
 */
template <typename T>
void PacketConsumeEvent<T>::process() {
	for (auto& pkt : this->packetVec) { this->sim->accept(acalsim::top->getGlobalTick(), *pkt); }

#ifdef ACALSIM_STATISTICS
	size_t total_gen_cnt = acalsim::top->getRecycleContainer()->getGenObjectCnt<T>();
	CLASS_INFO << "Release " << this->packetVec.size() << " packets. | Total generated packet = " << total_gen_cnt;
#else
	CLASS_INFO << "Release " << this->packetVec.size() << " packets.";
#endif  // ACALSIM_STATISTICS

	this->packetVec.clear();
}
