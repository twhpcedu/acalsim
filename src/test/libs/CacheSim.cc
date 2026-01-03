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
 * @file CacheSim.cc
 * @brief Cache simulator implementation for memory hierarchy modeling
 *
 * This file implements the **CacheSim** simulator component, which serves as the memory
 * hierarchy endpoint in the test architecture. It demonstrates cache simulation concepts
 * including event-driven cache operations, request processing, and upstream response paths.
 * This serves as a foundation for implementing realistic cache models with hit/miss logic,
 * replacement policies, and coherence protocols.
 *
 * **Role in System Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                            CacheSim                                      │
 * │                     (Cache Simulator)                                    │
 * │                     (CPPSimBase derived)                                 │
 * │                                                                           │
 * │  Responsibilities:                                                        │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Simulate cache behavior (hit/miss/replacement)                  │ │
 * │  │ 2. Process memory access requests from NOC                         │ │
 * │  │ 3. Generate independent cache events for testing                   │ │
 * │  │ 4. Serve as memory hierarchy endpoint                              │ │
 * │  │ 5. Provide data storage simulation                                 │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Connection Topology:                                                     │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │                                                                     │ │
 * │  │  Upstream: "USNOC" ◄────── NocSim                                  │ │
 * │  │    - Receives memory requests from NOC                             │ │
 * │  │    - Entry point for cache access operations                       │ │
 * │  │    - Potential response path (not fully implemented)               │ │
 * │  │                                                                     │ │
 * │  │  Downstream: (none in current implementation)                      │ │
 * │  │    - Could connect to lower-level cache (L2, L3)                   │ │
 * │  │    - Could connect to main memory simulator                        │ │
 * │  │                                                                     │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Cache Structure (Conceptual):                                            │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ Cache Lines: [Tag | Valid | Dirty | Data | LRU bits]              │ │
 * │  │                                                                     │ │
 * │  │ Current: Simplified model (no actual cache structure)              │ │
 * │  │ Future:  Full cache hierarchy with:                                │ │
 * │  │          - Set-associative organization                            │ │
 * │  │          - Replacement policies (LRU, LFU, Random)                 │ │
 * │  │          - Write policies (write-back, write-through)              │ │
 * │  │          - Coherence protocol support (MESI, MOESI)                │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Event Queue:                                                             │
 * │  [CacheEvent_1@tick3] [CacheEvent_2@tick5] ... [CacheEvent_9@tick19]   │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Cache Simulation Flow:**
 * ```
 * Initialization (init() called by framework):
 *   Step 1: Loop i = 1 to 9
 *     ├─► Create CacheEvent(std::to_string(i))
 *     │   - Event name: "CacheEvent_1", "CacheEvent_2", etc.
 *     │   - Simulates periodic cache operations
 *     │
 *     ├─► Calculate target tick: i * 2 + 1
 *     │   - Same scheduling pattern as other simulators
 *     │   - Ticks: 3, 5, 7, 9, 11, 13, 15, 17, 19
 *     │   - Represents periodic cache maintenance/monitoring
 *     │
 *     └─► scheduleEvent(cache_event, targetTick)
 *         - Framework inserts event into cache event queue
 *         - These events run independently of memory requests
 *
 * Simulation Phase (event processing):
 *   Type 1: Independent Cache Events (scheduled in init)
 *     Tick 3:
 *       └─► CacheEvent_1::process() invoked
 *           - Logs: "CacheEvent Processed."
 *           - Represents internal cache operations
 *           - Could be: prefetching, writeback, maintenance
 *
 *   Type 2: Memory Request Events (from NOC, if implemented)
 *     Future Enhancement:
 *       └─► CacheReqEvent::process() invoked
 *           - Check cache for hit/miss
 *           - Update cache state
 *           - Generate response event
 *           - Send response back via upstream connection
 *
 * Cleanup Phase (cleanup() called by framework):
 *   - Release dynamic memory allocations
 *   - Clear event queue
 *   - Dump cache statistics (hit rate, miss rate, etc.)
 *   - Free cache data structures
 * ```
 *
 * **Event Scheduling Timeline:**
 * ```
 * Cache Event Timeline:
 *
 * Tick:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
 *
 * Cache Init Events:
 *              ●     ●     ●     ●     ●     ●     ●     ●     ●
 *              │     │     │     │     │     │     │     │     │
 *         Periodic cache operations (maintenance, monitoring, prefetch)
 *
 * Memory Requests (future):
 *                             ●     ●     ●     ●     ●     ●     ●
 *                       (from NOC via upstream connection)
 *
 * Combined View:
 *              ●     ●     ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
 *              │     │     │  │  │  │  │  │  │  │  │  │  │  │  │  │
 *         Init │     │     │  │  │  │  │  │  │  │  │  │  │  │  │  │
 *              └─────┴─────┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
 *                                 │  │  │  │  │  │  │
 *                      Requests   │  │  │  │  │  │  │
 *                                 └──┴──┴──┴──┴──┴──┴──
 * ```
 *
 * **CacheEvent Creation:**
 * ```cpp
 * // Constructor signature from CacheEvent.hh:
 * CacheEvent(std::string name)
 *
 * // Creation in init():
 * CacheEvent* cache_event = new CacheEvent(
 *     std::to_string(i)     // Event name suffix: "1", "2", ..., "9"
 * );
 * // Full event name becomes: "CacheEvent_1", "CacheEvent_2", etc.
 *
 * // Key properties:
 * - Simple event model (no address or data fields currently)
 * - Fixed scheduling pattern
 * - Placeholder for realistic cache operations
 * ```
 *
 * **Realistic Cache Model Extension:**
 * ```cpp
 * class CacheSim : public CPPSimBase {
 * private:
 *     // Cache configuration
 *     struct CacheConfig {
 *         uint32_t cacheSize;      // Total cache size in bytes
 *         uint32_t lineSize;       // Cache line size (typically 64B)
 *         uint32_t associativity;  // N-way set associative
 *         uint32_t numSets;        // Number of cache sets
 *         std::string replacementPolicy;  // "LRU", "LFU", "Random"
 *         std::string writePolicy;        // "write-back", "write-through"
 *     } config;
 *
 *     // Cache line structure
 *     struct CacheLine {
 *         bool valid;              // Valid bit
 *         bool dirty;              // Dirty bit (for write-back)
 *         uint64_t tag;            // Address tag
 *         uint64_t lruCounter;     // For LRU replacement
 *         std::vector<uint8_t> data;  // Actual data storage
 *     };
 *
 *     // Cache organization
 *     std::vector<std::vector<CacheLine>> cacheSets;  // Sets x Ways
 *
 *     // Statistics
 *     struct CacheStats {
 *         uint64_t totalAccesses;
 *         uint64_t hits;
 *         uint64_t misses;
 *         uint64_t writebacks;
 *         uint64_t evictions;
 *     } stats;
 *
 * public:
 *     void init() override {
 *         // Initialize cache structure
 *         config.numSets = config.cacheSize /
 *                         (config.lineSize * config.associativity);
 *
 *         cacheSets.resize(config.numSets);
 *         for (auto& set : cacheSets) {
 *             set.resize(config.associativity);
 *             for (auto& line : set) {
 *                 line.valid = false;
 *                 line.dirty = false;
 *                 line.data.resize(config.lineSize);
 *             }
 *         }
 *     }
 *
 *     bool accessCache(uint64_t address, bool isWrite, uint8_t* data) {
 *         // Extract tag and set index
 *         uint64_t setIndex = (address / config.lineSize) % config.numSets;
 *         uint64_t tag = address / (config.lineSize * config.numSets);
 *
 *         stats.totalAccesses++;
 *
 *         // Search for cache hit
 *         auto& set = cacheSets[setIndex];
 *         for (auto& line : set) {
 *             if (line.valid && line.tag == tag) {
 *                 stats.hits++;
 *                 updateLRU(setIndex, &line);
 *
 *                 if (isWrite) {
 *                     line.dirty = true;
 *                     memcpy(line.data.data(), data, config.lineSize);
 *                 } else {
 *                     memcpy(data, line.data.data(), config.lineSize);
 *                 }
 *                 return true;  // Cache hit
 *             }
 *         }
 *
 *         // Cache miss - need to fetch from memory
 *         stats.misses++;
 *         handleCacheMiss(address, setIndex, tag, isWrite, data);
 *         return false;  // Cache miss
 *     }
 *
 *     void handleCacheMiss(uint64_t address, uint64_t setIndex,
 *                         uint64_t tag, bool isWrite, uint8_t* data) {
 *         // Find victim line using replacement policy
 *         CacheLine* victim = selectVictim(setIndex);
 *
 *         // Writeback if dirty
 *         if (victim->valid && victim->dirty) {
 *             stats.writebacks++;
 *             // Schedule writeback event to memory
 *         }
 *
 *         // Fetch new line from memory (schedule memory request event)
 *         // For now, just update the cache line
 *         victim->valid = true;
 *         victim->tag = tag;
 *         victim->dirty = isWrite;
 *         updateLRU(setIndex, victim);
 *
 *         stats.evictions++;
 *     }
 *
 *     CacheLine* selectVictim(uint64_t setIndex) {
 *         auto& set = cacheSets[setIndex];
 *
 *         // LRU replacement policy
 *         if (config.replacementPolicy == "LRU") {
 *             auto victim = std::min_element(set.begin(), set.end(),
 *                 [](const CacheLine& a, const CacheLine& b) {
 *                     return a.lruCounter < b.lruCounter;
 *                 });
 *             return &(*victim);
 *         }
 *
 *         // Random replacement
 *         return &set[rand() % set.size()];
 *     }
 *
 *     void updateLRU(uint64_t setIndex, CacheLine* accessedLine) {
 *         accessedLine->lruCounter = top->getGlobalTick();
 *     }
 *
 *     void cleanup() override {
 *         // Dump statistics
 *         double hitRate = (double)stats.hits / stats.totalAccesses * 100.0;
 *         double missRate = (double)stats.misses / stats.totalAccesses * 100.0;
 *
 *         std::cout << "Cache Statistics:" << std::endl;
 *         std::cout << "  Total Accesses: " << stats.totalAccesses << std::endl;
 *         std::cout << "  Hits: " << stats.hits << " (" << hitRate << "%)" << std::endl;
 *         std::cout << "  Misses: " << stats.misses << " (" << missRate << "%)" << std::endl;
 *         std::cout << "  Writebacks: " << stats.writebacks << std::endl;
 *         std::cout << "  Evictions: " << stats.evictions << std::endl;
 *     }
 * };
 * ```
 *
 * **Cache Coherence Protocol Extension:**
 * ```cpp
 * // MESI protocol states
 * enum class CoherenceState {
 *     MODIFIED,    // Modified (dirty, exclusive)
 *     EXCLUSIVE,   // Exclusive (clean, exclusive)
 *     SHARED,      // Shared (clean, multiple copies)
 *     INVALID      // Invalid (not present)
 * };
 *
 * struct CoherentCacheLine : CacheLine {
 *     CoherenceState state;
 *     std::vector<int> sharers;  // List of cache IDs with shared copies
 * };
 *
 * class CoherentCacheSim : public CacheSim {
 * private:
 *     int cacheId;
 *     std::vector<CoherentCacheSim*> peerCaches;
 *
 * public:
 *     void handleCoherenceRequest(uint64_t address, CoherenceOp op) {
 *         // Handle coherence messages: Read, ReadX, Upgrade, Invalidate
 *         switch (op) {
 *             case READ:
 *                 handleRead(address);
 *                 break;
 *             case READX:
 *                 handleReadExclusive(address);
 *                 break;
 *             case INVALIDATE:
 *                 handleInvalidate(address);
 *                 break;
 *         }
 *     }
 *
 *     void handleRead(uint64_t address) {
 *         auto line = findCacheLine(address);
 *         if (line && (line->state == MODIFIED || line->state == EXCLUSIVE)) {
 *             // Transition to SHARED
 *             if (line->state == MODIFIED) {
 *                 // Writeback data
 *                 line->dirty = false;
 *             }
 *             line->state = SHARED;
 *             // Send data to requester
 *         }
 *     }
 *
 *     void handleInvalidate(uint64_t address) {
 *         auto line = findCacheLine(address);
 *         if (line) {
 *             if (line->state == MODIFIED) {
 *                 // Writeback before invalidating
 *             }
 *             line->state = INVALID;
 *             line->valid = false;
 *         }
 *     }
 * };
 * ```
 *
 * **Multi-Level Cache Hierarchy:**
 * ```cpp
 * class CacheSim : public CPPSimBase {
 * private:
 *     CacheSim* lowerLevelCache;  // Pointer to L2 cache
 *     bool isL1;                  // true for L1, false for L2/L3
 *
 * public:
 *     void handleMiss(uint64_t address) {
 *         if (lowerLevelCache) {
 *             // Forward request to lower level
 *             CacheReqEvent* reqEvent = new CacheReqEvent(address);
 *             lowerLevelCache->scheduleEvent(reqEvent,
 *                                           top->getGlobalTick() + l2Latency);
 *         } else {
 *             // This is LLC, forward to memory
 *             MemoryReqEvent* memEvent = new MemoryReqEvent(address);
 *             memory->scheduleEvent(memEvent,
 *                                  top->getGlobalTick() + memLatency);
 *         }
 *     }
 *
 *     void handleFillResponse(uint64_t address, uint8_t* data) {
 *         // Fill cache line with data from lower level
 *         installCacheLine(address, data);
 *
 *         if (isL1) {
 *             // Forward response to processor/NOC
 *             CacheRespEvent* respEvent = new CacheRespEvent(address, data);
 *             upstream->scheduleEvent(respEvent, top->getGlobalTick() + 1);
 *         }
 *     }
 * };
 * ```
 *
 * **Cache Performance Metrics:**
 * ```cpp
 * struct DetailedCacheStats {
 *     // Access patterns
 *     uint64_t readAccesses;
 *     uint64_t writeAccesses;
 *     uint64_t readHits;
 *     uint64_t writeHits;
 *
 *     // Latency tracking
 *     uint64_t totalAccessLatency;
 *     uint64_t totalMissLatency;
 *     std::map<uint64_t, uint64_t> latencyHistogram;
 *
 *     // Replacement analysis
 *     std::map<std::string, uint64_t> evictionReasons;
 *
 *     // Spatial locality
 *     uint64_t consecutiveLineAccesses;
 *
 *     // Temporal locality
 *     std::map<uint64_t, Tick> lastAccessTime;
 *
 *     double getHitRate() {
 *         return (double)(readHits + writeHits) /
 *                (readAccesses + writeAccesses) * 100.0;
 *     }
 *
 *     double getAverageAccessLatency() {
 *         return (double)totalAccessLatency /
 *                (readAccesses + writeAccesses);
 *     }
 * };
 * ```
 *
 * **Comparison: Simple Model vs. Realistic Cache:**
 *
 * | Feature              | This Example | Realistic Cache         |
 * |----------------------|--------------|-------------------------|
 * | Cache Structure      | None         | Set-associative arrays  |
 * | Hit/Miss Logic       | None         | Tag comparison          |
 * | Replacement Policy   | None         | LRU, LFU, Random        |
 * | Write Policy         | None         | Write-back/through      |
 * | Coherence Protocol   | None         | MESI, MOESI             |
 * | Multi-level Support  | None         | L1, L2, L3 hierarchy    |
 * | Statistics           | None         | Hit rate, latency, etc. |
 * | Address Translation  | None         | Tag, set, offset        |
 *
 * **Usage in Test Framework:**
 * ```cpp
 * // In test.cc:
 * SimBase* cacheSim = (SimBase*)new CacheSim("Cache Simulator");
 * this->addSimulator(cacheSim);
 *
 * // Connect NocSim → CacheSim
 * nocSim->addDownStream(cacheSim, "DSCache");
 * cacheSim->addUpStream(nocSim, "USNOC");
 *
 * // Framework calls:
 * // 1. init()    - schedules 9 independent cache events
 * // 2. run()     - processes events (both init and routed)
 * // 3. cleanup() - releases resources, dumps statistics
 * ```
 *
 * **Integration with NOC:**
 * ```
 * Request Flow (Future Enhancement):
 *
 * Tick 13:
 *   NOC forwards request to Cache:
 *     NocSim creates CacheReqEvent
 *       ├─► Get cache via getDownStream("DSCache")
 *       └─► cache->scheduleEvent(reqEvent, currentTick + 1)
 *
 * Tick 14:
 *   CacheSim processes request:
 *     CacheReqEvent::process()
 *       ├─► Check cache for hit/miss
 *       ├─► If hit: return data immediately
 *       └─► If miss: fetch from lower level, then respond
 *
 * Response Path:
 *   CacheSim sends response back:
 *     Create CacheRespEvent
 *       ├─► Get NOC via getUpStream("USNOC")
 *       └─► noc->scheduleEvent(respEvent, currentTick + cacheLatency)
 * ```
 *
 * **Design Rationale:**
 * - Simple event-based structure provides foundation
 * - No actual cache logic keeps initial implementation clean
 * - Extensible design allows adding realistic cache behavior
 * - Independent events demonstrate cache-internal operations
 * - Serves as template for memory hierarchy modeling
 *
 * @see test.cc Main test framework and system topology
 * @see CacheEvent.cc Cache event processing implementation
 * @see CacheEvent.hh Cache event class definition
 * @see NocSim.cc Upstream NOC simulator
 * @see TrafficGenerator.cc Original traffic source
 * @see CPPSimBase Base class for C++ simulators
 * @see SimEvent Base class for all events
 */

#include "CacheSim.hh"

#include "CacheEvent.hh"

void CacheSim::init() {
	// TODO: Should schedule the events into event queue.
	for (Tick i = 1; i < 10; ++i) {
		// Schedule the event for testing.
		CacheEvent* cache_event = new CacheEvent(std::to_string(i));
		scheduleEvent(cache_event, i * 2 + 1);
	}
}

void CacheSim::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}
