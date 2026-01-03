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
 * @file CacheEvent.cc
 * @brief Cache event implementation for memory hierarchy simulation
 *
 * This file implements the **CacheEvent** class, which represents cache operations and
 * memory access events in the cache simulator. It demonstrates **simple event processing**,
 * serves as the endpoint event in the multi-simulator chain, and provides a foundation
 * for implementing realistic cache access modeling.
 *
 * **Event Role and Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                          CacheEvent                                      │
 * │                      (SimEvent derived)                                  │
 * │                                                                           │
 * │  Purpose:                                                                 │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Represent cache operations (hit/miss/prefetch/writeback)       │ │
 * │  │ 2. Process memory access requests                                  │ │
 * │  │ 3. Serve as terminal event in simulator chain                      │ │
 * │  │ 4. Demonstrate minimal event processing pattern                    │ │
 * │  │ 5. Provide extensibility for realistic cache modeling              │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Event Types:                                                             │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ Current: Self-scheduled cache maintenance events                   │ │
 * │  │   - Periodic operations (e.g., prefetch, stats collection)         │ │
 * │  │   - No actual cache logic                                          │ │
 * │  │   - Simple logging for demonstration                               │ │
 * │  │                                                                     │ │
 * │  │ Future: Memory access events                                       │ │
 * │  │   - Read requests (load operations)                                │ │
 * │  │   - Write requests (store operations)                              │ │
 * │  │   - Writeback events (dirty line evictions)                        │ │
 * │  │   - Prefetch events (speculative loads)                            │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Key Members:                                                             │
 * │  - _name: Event name (e.g., "CacheEvent_1")                              │
 * │                                                                           │
 * │  Future Extensions:                                                       │
 * │  - address: Memory address being accessed                                │
 * │  - size: Access size in bytes                                            │
 * │  - type: Access type (READ, WRITE, PREFETCH, WRITEBACK)                 │
 * │  - data: Data payload for writes                                         │
 * │  - callback: Completion notification (for upstream response)             │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Processing Flow:**
 * ```
 * Current Implementation (Self-Scheduled):
 *
 * Tick 3 (example - when CacheEvent_1 is scheduled):
 *   Step 1: Framework invokes process()
 *     └─► CacheEvent::process() called
 *
 *   Step 2: Log event processing
 *     └─► CLASS_INFO << "CacheEvent Processed."
 *     └─► Output: [Cache Simulator] CacheEvent Processed.
 *
 *   Step 3: Processing complete
 *     └─► No cache operations performed
 *     └─► No response generated
 *     └─► Event lifecycle ends
 *
 * Future Implementation (Memory Access):
 *
 * Tick N (when cache request arrives from NOC):
 *   Step 1: Framework invokes process()
 *     └─► CacheEvent::process() called
 *
 *   Step 2: Log access
 *     └─► CLASS_INFO << "Processing " << accessType
 *                    << " to address 0x" << address
 *
 *   Step 3: Cache lookup
 *     └─► bool hit = cacheLookup(address)
 *     └─► if (hit):
 *           └─► Extract data from cache line
 *           └─► Log cache hit
 *         else:
 *           └─► Select victim line (LRU/LFU)
 *           └─► Writeback if dirty
 *           └─► Fetch from lower level
 *           └─► Install new line
 *           └─► Log cache miss
 *
 *   Step 4: Generate response (future)
 *     └─► Create response event
 *     └─► Schedule to upstream NOC
 *     └─► Include data and latency
 *
 * Flow Diagram (Future):
 *   NOC Request                    Cache Processing                Response
 *       │                                │                             │
 *   ┌───▼────────┐              ┌────────▼──────────┐         ┌───────▼─────┐
 *   │ NocEvent   │              │  CacheEvent       │         │  NocResp    │
 *   │ schedules  │──────────────►│  - process()     │────────►│  Event      │
 *   │ CacheEvent │              │  - lookup()       │         │             │
 *   └────────────┘              │  - hit/miss logic │         └─────────────┘
 *                               │  - response gen   │
 *                               └───────────────────┘
 * ```
 *
 * **process() Method Implementation:**
 * ```cpp
 * void CacheEvent::process() {
 *     // Current implementation:
 *     CLASS_INFO << "CacheEvent Processed.";
 *     // Logs: [Cache Simulator] CacheEvent Processed.
 *     // [Cache Simulator] is the name from CacheSim construction
 * }
 *
 * // Minimal implementation demonstrates:
 * // - Event execution in cache simulator context
 * // - Logging with CLASS_INFO macro
 * // - Placeholder for cache logic
 * // - Template for future extensions
 * ```
 *
 * **Event Name Construction:**
 * ```
 * Name Format: "CacheEvent_" + supplied_name
 *
 * Examples (from CacheSim::init()):
 *   Input: "1"    → Output: "CacheEvent_1"
 *   Input: "5"    → Output: "CacheEvent_5"
 *   Input: "9"    → Output: "CacheEvent_9"
 *
 * Future naming conventions:
 *   Access-based:     "CacheRead_0x1000", "CacheWrite_0x2000"
 *   Type-based:       "Prefetch_Line42", "Writeback_Set7"
 *   Operation-based:  "LoadHit", "StoreMiss", "Eviction"
 *
 * Name Usage:
 *   - Debugging cache behavior
 *   - Filtering cache events in logs
 *   - Performance analysis (hit/miss patterns)
 *   - Trace generation for validation
 * ```
 *
 * **Constructor Details:**
 * ```cpp
 * // From CacheEvent.hh:
 * CacheEvent(std::string name)
 *     : SimEvent(),
 *       _name{"CacheEvent_" + name}
 * {}
 *
 * // Simple constructor:
 * // - Only requires event name
 * // - No address, size, or type (future enhancement)
 * // - No callback mechanism (terminal event)
 * // - Uses default SimEvent memory management (auto-cleanup)
 *
 * // Usage (from CacheSim::init()):
 * CacheEvent* cache_event = new CacheEvent(std::to_string(i));
 * // i = 1: _name = "CacheEvent_1"
 * // i = 2: _name = "CacheEvent_2"
 * // etc.
 * ```
 *
 * **Extending for Realistic Cache Simulation:**
 * ```cpp
 * // Enhanced CacheEvent with memory access details
 * class CacheEvent : public SimEvent {
 * private:
 *     // Memory access properties
 *     uint64_t address;        // Physical/virtual address
 *     uint32_t size;           // Access size in bytes
 *     AccessType type;         // READ, WRITE, PREFETCH, WRITEBACK
 *     uint8_t* data;           // Data payload (for writes)
 *
 *     // Cache operation context
 *     int requestorId;         // Source PE/core ID
 *     int cacheLevel;          // L1, L2, or L3
 *     bool isExclusive;        // For coherence protocols
 *
 *     // Response handling
 *     std::function<void(uint8_t*)> responseCallback;
 *     SimBase* upstreamSim;    // For sending responses
 *
 *     // Performance tracking
 *     Tick issueTime;
 *     bool recordStats;
 *
 * public:
 *     CacheEvent(uint64_t _addr, uint32_t _size, AccessType _type)
 *         : SimEvent(),
 *           address(_addr),
 *           size(_size),
 *           type(_type),
 *           data(nullptr),
 *           requestorId(0),
 *           cacheLevel(1),
 *           isExclusive(false),
 *           responseCallback(nullptr),
 *           upstreamSim(nullptr),
 *           issueTime(0),
 *           recordStats(true)
 *     {
 *         _name = "CacheEvent_" + accessTypeToString(type) +
 *                 "_" + std::to_string(address);
 *     }
 *
 *     void process() override {
 *         CLASS_INFO << "Processing " << accessTypeToString(type)
 *                    << " request to address 0x" << std::hex << address
 *                    << std::dec << " (" << size << " bytes)";
 *
 *         // Step 1: Extract cache line address
 *         uint64_t lineAddr = address & ~(CACHE_LINE_SIZE - 1);
 *
 *         // Step 2: Cache lookup
 *         CacheLine* line = cacheLookup(lineAddr);
 *         bool hit = (line != nullptr);
 *
 *         // Step 3: Handle hit or miss
 *         if (hit) {
 *             handleCacheHit(line);
 *         } else {
 *             handleCacheMiss(lineAddr);
 *         }
 *
 *         // Step 4: Record statistics
 *         if (recordStats) {
 *             updateCacheStats(type, hit);
 *         }
 *
 *         // Step 5: Send response if callback exists
 *         if (responseCallback && hit) {
 *             responseCallback(line->data);
 *         } else if (!hit) {
 *             // Miss - need to fetch from lower level
 *             scheduleLowerLevelAccess(lineAddr);
 *         }
 *     }
 *
 * private:
 *     void handleCacheHit(CacheLine* line) {
 *         CLASS_INFO << "Cache HIT at address 0x" << std::hex << address;
 *
 *         // Update access bits
 *         line->lruCounter = top->getGlobalTick();
 *
 *         // Perform operation
 *         if (type == WRITE) {
 *             line->dirty = true;
 *             if (data) {
 *                 memcpy(&line->data[address % CACHE_LINE_SIZE],
 *                        data, size);
 *             }
 *         } else if (type == READ) {
 *             if (data) {
 *                 memcpy(data, &line->data[address % CACHE_LINE_SIZE],
 *                        size);
 *             }
 *         }
 *     }
 *
 *     void handleCacheMiss(uint64_t lineAddr) {
 *         CLASS_INFO << "Cache MISS at address 0x" << std::hex << address;
 *
 *         // Select victim
 *         CacheLine* victim = selectVictimLine(lineAddr);
 *
 *         // Writeback if dirty
 *         if (victim->valid && victim->dirty) {
 *             scheduleWriteback(victim);
 *         }
 *
 *         // Allocate new line
 *         victim->valid = true;
 *         victim->tag = getTag(lineAddr);
 *         victim->dirty = (type == WRITE);
 *         victim->lruCounter = top->getGlobalTick();
 *     }
 *
 *     void scheduleLowerLevelAccess(uint64_t lineAddr) {
 *         // Create event for lower-level cache or memory
 *         if (lowerLevelCache) {
 *             CacheEvent* lowerEvent = new CacheEvent(
 *                 lineAddr,
 *                 CACHE_LINE_SIZE,
 *                 READ
 *             );
 *             lowerEvent->setCallback([this](uint8_t* fetchedData) {
 *                 this->handleFill(fetchedData);
 *             });
 *             lowerLevelCache->scheduleEvent(lowerEvent,
 *                 top->getGlobalTick() + LOWER_LEVEL_LATENCY);
 *         } else {
 *             // Access main memory
 *             scheduleMemoryAccess(lineAddr);
 *         }
 *     }
 * };
 * ```
 *
 * **Cache Access Patterns:**
 * ```cpp
 * // Pattern 1: Read Access
 * uint8_t buffer[64];
 * CacheEvent* readEvent = new CacheEvent(
 *     0x1000,                    // Address
 *     8,                         // 8-byte read
 *     READ                       // Access type
 * );
 * readEvent->setDataBuffer(buffer);  // Where to store result
 * cacheSimulator->scheduleEvent(readEvent, currentTick + 1);
 *
 * // Pattern 2: Write Access
 * uint8_t writeData[64] = {...};
 * CacheEvent* writeEvent = new CacheEvent(
 *     0x2000,                    // Address
 *     64,                        // Full cache line
 *     WRITE                      // Access type
 * );
 * writeEvent->setData(writeData);    // Data to write
 * cacheSimulator->scheduleEvent(writeEvent, currentTick + 1);
 *
 * // Pattern 3: Prefetch
 * CacheEvent* prefetchEvent = new CacheEvent(
 *     predictedAddress,          // Predicted next access
 *     CACHE_LINE_SIZE,           // Prefetch full line
 *     PREFETCH                   // Speculative load
 * );
 * prefetchEvent->setLowPriority(true);  // Don't interfere with demands
 * cacheSimulator->scheduleEvent(prefetchEvent, currentTick + 5);
 *
 * // Pattern 4: Writeback
 * CacheEvent* wbEvent = new CacheEvent(
 *     evictedLineAddr,           // Address of victim
 *     CACHE_LINE_SIZE,           // Full line size
 *     WRITEBACK                  // Writeback operation
 * );
 * wbEvent->setData(evictedLineData);
 * lowerLevel->scheduleEvent(wbEvent, currentTick + 1);
 * ```
 *
 * **Integration with Multi-Level Cache Hierarchy:**
 * ```
 * L1 Cache Miss Flow:
 *
 * Tick 10: L1 CacheEvent processes
 *   └─► Miss detected
 *       └─► Create L2 CacheEvent
 *           └─► Schedule @ L1 + L2_LATENCY
 *
 * Tick 20: L2 CacheEvent processes
 *   └─► Hit detected
 *       └─► Invoke L1 callback with data
 *           └─► L1 installs cache line
 *               └─► L1 responds to requester
 *
 * Alternative: L2 Miss
 * Tick 20: L2 CacheEvent processes
 *   └─► Miss detected
 *       └─► Create L3/Memory event
 *           └─► Wait for response
 *               └─► Chain of callbacks back to L1
 *
 * Callback Chain:
 *   Memory → L3 callback → L2 callback → L1 callback → Requester
 * ```
 *
 * **Cache Coherence Event Handling:**
 * ```cpp
 * // Coherence-aware CacheEvent
 * class CacheEvent : public SimEvent {
 * private:
 *     CoherenceOp coherenceOp;  // READ, READX, UPGRADE, INVALIDATE
 *     std::vector<int> sharers; // List of sharers (for broadcasts)
 *
 * public:
 *     void process() override {
 *         if (coherenceOp == INVALIDATE) {
 *             handleInvalidate();
 *         } else if (coherenceOp == READX) {
 *             handleReadExclusive();
 *         } else {
 *             // Normal read/write
 *             handleNormalAccess();
 *         }
 *     }
 *
 * private:
 *     void handleInvalidate() {
 *         CacheLine* line = findLine(address);
 *         if (line && line->state != INVALID) {
 *             if (line->state == MODIFIED) {
 *                 // Writeback before invalidation
 *                 scheduleWriteback(line);
 *             }
 *             line->state = INVALID;
 *             line->valid = false;
 *             CLASS_INFO << "Invalidated line at 0x" << std::hex << address;
 *         }
 *     }
 *
 *     void handleReadExclusive() {
 *         // Request exclusive ownership
 *         // Broadcast invalidates to sharers
 *         for (int sharerId : sharers) {
 *             sendInvalidate(sharerId, address);
 *         }
 *         // Transition to MODIFIED state
 *         acquireExclusiveOwnership();
 *     }
 * };
 * ```
 *
 * **Performance Monitoring Extensions:**
 * ```cpp
 * void CacheEvent::process() {
 *     Tick startTick = top->getGlobalTick();
 *
 *     // Perform cache operation
 *     bool hit = performAccess();
 *
 *     Tick endTick = top->getGlobalTick();
 *     Tick latency = endTick - startTick;
 *
 *     // Record detailed statistics
 *     cacheStats.recordAccess(
 *         type,           // READ, WRITE, etc.
 *         hit,            // Hit or miss
 *         latency,        // Access latency
 *         address,        // Address (for locality analysis)
 *         requestorId     // Source (for per-core stats)
 *     );
 *
 *     // Update heat map for visualization
 *     cacheStats.updateHeatMap(address, hit);
 * }
 * ```
 *
 * **Comparison: Current vs. Realistic Implementation:**
 *
 * | Feature              | Current            | Realistic                |
 * |----------------------|--------------------|--------------------------|
 * | Event Data           | Name only          | Address, size, type      |
 * | Cache Logic          | None               | Hit/miss detection       |
 * | Response Generation  | None               | Callback-based           |
 * | Multi-level Support  | None               | Recursive events         |
 * | Coherence Protocol   | None               | MESI/MOESI states        |
 * | Statistics           | None               | Detailed metrics         |
 * | Memory Management    | Auto               | Manual with callbacks    |
 *
 * **Position in Event Chain:**
 * ```
 * Event Flow Sequence:
 *
 * TrafficGenerator                NOC                     Cache
 *       │                          │                         │
 *   TrafficEvent              NocEvent                 CacheEvent
 *       │                          │                         │
 *   Generates                  Routes                  Processes
 *   traffic                    packets                 memory ops
 *       │                          │                         │
 *   ────┴─────────────────────────┴─────────────────────────┴────►
 *      Upstream                 Intermediate              Downstream
 *      (Source)                 (Routing)                 (Endpoint)
 *
 * CacheEvent is the terminal event:
 * - No further event creation (current implementation)
 * - Could create response events (future)
 * - Could access lower levels (L2, memory)
 * ```
 *
 * **Usage in Test Framework:**
 * ```cpp
 * // In CacheSim::init():
 * for (Tick i = 1; i < 10; ++i) {
 *     CacheEvent* cache_event = new CacheEvent(std::to_string(i));
 *     scheduleEvent(cache_event, i * 2 + 1);
 * }
 *
 * // Framework execution:
 * // Tick 3: CacheEvent_1 processes → logs "CacheEvent Processed."
 * // Tick 5: CacheEvent_2 processes → logs "CacheEvent Processed."
 * // ... continues for all 9 events
 *
 * // Expected output pattern:
 * // [Cache Simulator] CacheEvent Processed.
 * // [Cache Simulator] CacheEvent Processed.
 * // (repeated 9 times at ticks 3, 5, 7, 9, 11, 13, 15, 17, 19)
 * ```
 *
 * **Design Rationale:**
 * - Minimal process() demonstrates event execution pattern
 * - Simple structure provides clean extension point
 * - Terminal event (no downstream creation) simplifies initial design
 * - Logging shows cache simulator context
 * - Serves as foundation for realistic cache modeling
 *
 * @see test.cc Main test framework and topology
 * @see CacheEvent.hh CacheEvent class definition
 * @see CacheSim.cc Cache simulator creating self-scheduled events
 * @see NocEvent.cc Upstream NOC events (future: will create CacheEvents)
 * @see TrafficEvent.cc Original traffic source events
 * @see SimEvent Base class for all events
 */

#include "CacheEvent.hh"

void CacheEvent::process() { CLASS_INFO << "CacheEvent Processed."; }
