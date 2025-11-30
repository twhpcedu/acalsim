<!--
Copyright 2023-2025 Playlab/ACAL

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

# ThreadManagerV1 Context Switch Optimization

## Overview

This document describes the context-switching issues identified in ThreadManagerV1 and the optimizations implemented to reduce unnecessary context switches during parallel simulation.

## Background

Context switches occur when the operating system suspends one thread and resumes another. There are two types:

- **Voluntary context switches**: Occur when a thread voluntarily yields the CPU (e.g., waiting on a mutex, condition variable, or I/O)
- **Involuntary context switches**: Occur when the OS preempts a thread (e.g., time slice expiration)

Excessive context switches degrade performance because:
1. CPU cache invalidation
2. TLB flushes
3. Pipeline stalls
4. Memory bandwidth consumption

## Identified Issues

### Issue #1: Busy-Wait Spin Loop at Worker Thread Startup

**Location**: `TaskManagerV1.inl`, scheduler() function

**Problem**: Worker threads used a busy-wait spin loop to wait for the ThreadManager to enter the running state:

```cpp
// BEFORE: Busy-wait spin loop
while (!this->getThreadManager()->isRunning()) { ; }
```

This causes:
- CPU cycles wasted on spinning
- Potential priority inversion
- Unnecessary context switches as threads spin and yield

**Root Cause**: The original implementation assumed busy-waiting would be brief, but in practice the startup sequence can take significant time, causing threads to spin unnecessarily.

### Issue #2: Thundering Herd Effect from notify_all()

**Location**: `ThreadManagerV1.inl`, startPhase1() function

**Problem**: Using `notify_all()` to wake worker threads causes all threads to wake up simultaneously, even though only a subset may have work to do:

```cpp
this->getTaskManager()->newTaskAvailableCondVar.notify_all();
```

This causes:
- All threads wake and compete for the taskQueueMutex
- Most threads find no work and go back to sleep immediately
- Unnecessary context switches for threads that won't do work

**Root Cause**: The two-phase synchronization model requires all threads to participate in each phase, making selective waking complex. The alternative `notify_one()` approach was attempted but caused deadlocks when threads hadn't entered the wait state yet.

### Issue #3: Lock Contention on taskQueueMutex

**Location**: `TaskManagerV1.inl`, scheduler() function

**Problem**: The taskQueueMutex is held during:
- Checking for ready tasks
- Popping tasks from the queue
- Pushing completed tasks back to the queue

**Analysis**: The current implementation already releases the lock during task execution, which is the most critical optimization. Further lock scope reduction was deemed unnecessary as it would add complexity without significant benefit.

## Implemented Optimization

### Optimization #1: Replace Busy-Wait with Condition Variable

**Files Modified**:
- `include/sim/ThreadManager.hh` - Made `startRunning()` virtual
- `include/sim/ThreadManagerV1/TaskManagerV1.hh` - Added `runningCondVar` and `runningMutex`
- `include/sim/ThreadManagerV1/TaskManagerV1.inl` - Replaced busy-wait with condition variable wait
- `include/sim/ThreadManagerV1/ThreadManagerV1.hh` - Added `startRunning()` override
- `include/sim/ThreadManagerV1/ThreadManagerV1.inl` - Implemented `startRunning()` to notify waiting threads

**Code Changes**:

```cpp
// AFTER: Condition variable wait
{
    std::unique_lock<std::mutex> lock(this->runningMutex);
    this->runningCondVar.wait(lock, [this] {
        return this->getThreadManager()->isRunning();
    });
}
```

And in `startRunning()`:

```cpp
void ThreadManagerV1<T>::startRunning() {
    // Set the running flag first
    this->running = true;

    // Notify all waiting worker threads
    {
        std::lock_guard<std::mutex> lock(this->getTaskManager()->runningMutex);
    }
    this->getTaskManager()->runningCondVar.notify_all();
}
```

**Benefits**:
- Threads sleep efficiently while waiting instead of burning CPU cycles
- Proper signaling ensures threads wake up exactly when needed
- Reduces voluntary context switches from repeated yield/wake cycles

## Results

### Test Configuration
- Test: ACALSim test suite with NocSim, CacheSim, TrafficGenerator
- Platform: Docker container (debian-pytorch-workspace)
- Measurement: `getrusage()` for context switch counting

### Baseline Measurements (Before Optimization)

| Configuration | Voluntary CS | Involuntary CS |
|--------------|--------------|----------------|
| V1, 4 threads | 116 | 8 |
| V1, 8 threads | 302 | 5 |

### Optimized Measurements (After Optimization)

| Configuration | Voluntary CS | Involuntary CS | Improvement |
|--------------|--------------|----------------|-------------|
| V1, 4 threads | ~116 | ~7 | ~0% (within variance) |
| V1, 8 threads | ~193 | ~7 | **~36% reduction** |

### Analysis

The optimization shows significant improvement for higher thread counts:
- **4 threads**: Results are within measurement variance, showing no regression
- **8 threads**: 36% reduction in voluntary context switches

The larger improvement with more threads is expected because:
1. More threads means more instances of the busy-wait loop being eliminated
2. The thundering herd effect from `notify_all()` is more pronounced with more threads
3. With fewer tasks than threads (3 simulators vs 8 threads), more threads were spinning in the busy-wait loop

## Future Optimization Opportunities

### 1. notify_one() with Sleep Tracking (Not Implemented)

An attempt was made to replace `notify_all()` with `notify_one()` calls to reduce thundering herd:

```cpp
// Track sleeping threads
std::atomic<int> nSleepingThreads{0};

// In wait path:
nSleepingThreads.fetch_add(1);
condvar.wait(...);
nSleepingThreads.fetch_sub(1);

// In startPhase1:
int nSleeping = nSleepingThreads.load();
for (int i = 0; i < nSleeping; ++i) {
    condvar.notify_one();
}
```

**Issue**: This caused deadlocks when:
- First phase starts before threads enter wait state
- Sleep counter shows 0, so no threads are notified
- Threads then enter wait but miss the notification

**Mitigation**: A fallback to `notify_all()` when `nSleepingThreads == 0` was tried but still had timing issues.

### 2. Lock-Free Task Queue

Replace the mutex-protected priority queue with a lock-free data structure to reduce lock contention.

### 3. Work-Stealing Scheduler

Implement per-thread task queues with work-stealing to reduce global synchronization.

## Measuring Context Switches

Context switches are measured using `getrusage()`:

```cpp
#include <sys/resource.h>

static struct rusage g_rusage_start;

void SimTopBase::init() {
    // ... initialization code ...
    getrusage(RUSAGE_SELF, &g_rusage_start);
}

void SimTopBase::finish() {
    struct rusage rusage_end;
    getrusage(RUSAGE_SELF, &rusage_end);
    long voluntary_cs = rusage_end.ru_nvcsw - g_rusage_start.ru_nvcsw;
    long involuntary_cs = rusage_end.ru_nivcsw - g_rusage_start.ru_nivcsw;
    std::cout << "[SimTopBase] Context Switches during simulation:\n";
    std::cout << "[SimTopBase]   Voluntary context switches: " << voluntary_cs << "\n";
    std::cout << "[SimTopBase]   Involuntary context switches: " << involuntary_cs << "\n";
    // ... cleanup code ...
}
```

## Conclusion

The busy-wait replacement optimization provides measurable context switch reduction, particularly for configurations with more threads than tasks. The optimization is conservative and maintains correctness while improving efficiency.

The more aggressive `notify_one()` optimization was not implemented due to timing/synchronization complexities that could cause deadlocks. Future work could explore this with more sophisticated tracking mechanisms or alternative synchronization patterns.
