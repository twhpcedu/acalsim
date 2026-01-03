<!--
Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Advanced Scheduler Optimization Plan: Lock-Free Queues and Work-Stealing

## Executive Summary

This document outlines potential advanced optimizations for ThreadManagerV1's task scheduling, building on the successful busy-wait elimination optimization. After comprehensive research, we recommend a **Per-Simulator Partitioned Queue** approach as the most practical next step, with existing experimental implementations (ThreadManagerV6) available for evaluation.

## Current State

### ThreadManagerV1 Performance
- **Baseline**: 302 voluntary context switches (8 threads)
- **After busy-wait optimization**: ~193-236 voluntary CS (~22-36% reduction)
- **Main bottleneck**: Global `taskQueueMutex` contention

### Key Constraints
1. **Few tasks, many threads**: 3 simulators with 8 worker threads
2. **Priority ordering required**: Tasks must execute in `next_execution_cycle` order
3. **Dynamic priorities**: Task priorities change after execution
4. **Two-phase synchronization**: All threads must complete iteration before next cycle

### Existing Experimental Code (ThreadManagerV6)
The codebase already contains experimental implementations:
- `LockFreePriorityQueue.hh`: Atomic operations with fine-grained mutex for heap operations
- `FineGrainedConcurrentTaskQueue.hh`: `shared_mutex` (reader-writer lock) per operation

---

## Recommended Approaches (Ranked by Feasibility)

### Approach 1: Per-Simulator Partitioned Queues (HIGHEST PRIORITY)

**Concept**: Divide the global queue into 3 independent queues (one per simulator)

**Architecture**:
```cpp
class TaskManagerPartitioned {
    UpdateablePriorityQueue queues[3];  // One per simulator
    std::mutex queueMutex[3];           // Per-simulator lock

    Task getTask(int threadID) {
        int mySimID = threadID % 3;  // Simple affinity

        // Fast path: check own queue
        if (queues[mySimID].hasReadyTask(tick)) {
            std::lock_guard lock(queueMutex[mySimID]);
            return queues[mySimID].extractTop();
        }

        // Steal path: try others (round-robin)
        for (int i = 1; i < 3; ++i) {
            int victim = (mySimID + i) % 3;
            if (trySteal(victim, task)) return task;
        }
        return goToSleep();
    }
};
```

**Benefits**:
- Reduces lock contention from "8 threads on 1 lock" to "~3 threads on 3 locks"
- Reuses existing `UpdateablePriorityQueue` code
- Simple to implement and test
- Maintains priority ordering per simulator

**Drawbacks**:
- May not balance load if simulators have uneven work
- Requires thread-to-simulator affinity logic
- Still needs global sync at phase boundaries

**Expected Improvement**: 25-35% additional reduction in context switches
**Implementation Effort**: 3-5 days

---

### Approach 2: Fine-Grained Locking (ThreadManagerV6 Style)

**Concept**: Use `shared_mutex` to allow concurrent reads, exclusive writes

**Existing Implementation**: `FineGrainedConcurrentTaskQueue.hh`

**Key Features**:
- `shared_lock` for `hasReadyTask()`, `top()` - multiple threads can read
- `unique_lock` for `pop()`, `push()` - exclusive access for writes
- Atomic priority updates with O(1) lookup via `index_map`

**Code Pattern**:
```cpp
bool hasReadyTask(uint64_t priority) {
    processUpdates();
    std::shared_lock<std::shared_mutex> lock(vector_mutex);  // Read lock
    return !nodes.empty() && nodes[0]->priority <= priority;
}

void pop() {
    std::unique_lock<std::shared_mutex> lock(vector_mutex);  // Write lock
    // ... remove top element
}
```

**Benefits**:
- Allows concurrent `hasReadyTask()` checks
- Reduces contention for read-heavy workloads
- Already implemented and tested

**Drawbacks**:
- `shared_mutex` has higher overhead than `mutex` on some platforms
- Still global queue (all threads compete)
- Complex edge cases in concurrent heap operations

**Expected Improvement**: 15-20% additional reduction
**Implementation Effort**: 1 week (mostly testing)

---

### Approach 3: Lock-Free Priority Updates (ThreadManagerV6 Style)

**Concept**: Use atomic operations for priority updates without full queue lock

**Existing Implementation**: `LockFreePriorityQueue.hh`

**Key Features**:
- `std::atomic<uint64_t> priority` per node
- `std::atomic<bool> needsUpdate` flag for lazy updates
- `version_counter` for change notification
- Mutex only for vector resize operations

**Code Pattern**:
```cpp
void update(int simID, uint64_t newPriority) {
    auto target = findNode(simID);

    // Atomic priority update - no lock needed
    uint64_t old = target->priority.exchange(newPriority, std::memory_order_acq_rel);
    target->needsUpdate.store(true, std::memory_order_release);

    // Sift operation still needs coordination
    if (newPriority < old) siftUp(target->index);
    else siftDown(target->index);
}
```

**Benefits**:
- O(1) priority updates without queue lock
- Lazy update processing reduces overhead
- Good for frequent priority changes

**Drawbacks**:
- Complex memory ordering requirements
- Still needs mutex for structural changes (push/pop)
- ABA problem potential during concurrent sift operations

**Expected Improvement**: 10-15% additional reduction
**Implementation Effort**: Already implemented, needs validation

---

### Approach 4: Work-Stealing (NOT RECOMMENDED)

**Why Not Suitable for ACALSim**:

1. **Priority ordering conflict**: Work-stealing optimizes for local cache, not global priority
2. **Few tasks**: Only 3 tasks makes stealing overhead > benefit
3. **Dynamic priorities**: Stealing requires stable task ownership
4. **Two-phase model**: Global sync negates work-stealing benefits

**Better Alternative**: Per-simulator partitioning (Approach 1) provides similar benefits without breaking priority semantics.

---

### Approach 5: Skip-List Based Queue (FUTURE CONSIDERATION)

**When to Consider**: If task count grows to 50+ simulators

**Libraries**:
- **libcds**: Production-ready, C++11, header-only
- **CSLPQ**: Used in DCSim simulator

**Current Assessment**: Overkill for 3 tasks. Skip-list overhead dominates at low contention.

---

## Implementation Roadmap

### Phase 1: Evaluate ThreadManagerV6 (Week 1)

1. Add V6 to test suite configuration
2. Benchmark against V1 with identical workloads
3. Measure:
   - Context switches
   - Lock contention (via profiling)
   - Throughput (tasks/second)

```bash
# Add test for V6
./test --threadmanager V6 --threads 8
```

### Phase 2: Per-Simulator Partitioning (Week 2-3)

If V6 doesn't provide sufficient improvement:

1. Create `TaskManagerV1Partitioned` class
2. Implement per-simulator queues
3. Add simple thread affinity
4. Implement steal logic
5. Benchmark and validate

### Phase 3: Production Validation (Week 4)

1. Run full regression tests
2. Profile with realistic workloads
3. Document performance characteristics
4. Update user documentation

---

## Metrics to Track

| Metric | Current (V1) | Target |
|--------|--------------|--------|
| Voluntary CS (8 threads) | ~193-236 | <150 |
| Lock hold time | ~100-500ns | <50ns |
| Lock contention % | High | <20% |
| Throughput | Baseline | +25% |

---

## Risk Assessment

| Approach | Risk Level | Mitigation |
|----------|------------|------------|
| Per-Simulator Partitioning | LOW | Reuses existing code |
| Fine-Grained Locking (V6) | MEDIUM | Already tested |
| Lock-Free Updates | MEDIUM-HIGH | Complex memory ordering |
| Work-Stealing | HIGH | Not recommended |
| Skip-List | LOW (deferred) | For future consideration |

---

## Decision Matrix

| Factor | Partitioned | Fine-Grained | Lock-Free | Work-Stealing |
|--------|-------------|--------------|-----------|---------------|
| Implementation Effort | Low | Medium | High | Very High |
| Expected Benefit | High | Medium | Medium | Low |
| Risk | Low | Medium | High | Very High |
| Fits ACALSim Model | Yes | Yes | Yes | No |
| **Recommendation** | **First** | Second | Third | Avoid |

---

## Conclusion

**Recommended Path Forward**:

1. **Immediate**: Benchmark existing ThreadManagerV6 implementations
2. **If V6 insufficient**: Implement Per-Simulator Partitioned Queues
3. **Long-term**: Consider skip-list based queues if simulator count grows significantly

The key insight is that ACALSim's constraint of only 3 simulator tasks means the bottleneck is **lock contention**, not algorithmic complexity. Partitioning eliminates contention without adding complexity.

---

## References

- ThreadManagerV6 Implementation: `include/sim/ThreadManagerV6/`
- Context Switch Optimization Doc: `docs/context-switch-optimization.md`
- Lock-Free Algorithms: https://www.1024cores.net/home/lock-free-algorithms/queues/priority-queues
- libcds Library: https://github.com/khizmax/libcds
- Work-Stealing Paper: https://www.sciencedirect.com/science/article/pii/S0743731596901070
