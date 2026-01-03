# Thread Manager and Task Manager - Developer Document

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

[toc]

---

- Author: Jen-Chien Chang \<jenchien@twhpcedu.org\>
- Date: 2024/08/24

---

## Running Simulators with Multiple Threads

A primary objective of ACALSim is to enhance the efficiency of multi-simulator co-simulations through multi-threaded execution. A straightforward method to achieve this is by allocating a separate thread to each simulator. However, as the number of simulators increases, the frequency of context switching in processor cores also rises, because the number of software threads significantly exceeds the number of available hardware threads.

### Thread Pooling

Thread pooling is a design pattern frequently employed in concurrent software programming. Its purpose is to minimize the overhead associated with:

- Repeated creation and destruction of threads.
- Frequent scheduling and context-switching among an excessive number of threads.

A basic thread pooling implementation usually consists of a task queue and a thread pool.

<p>
    <div align="center">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Thread_pool.svg" />
		<br />
        Source: <a href="https://en.wikipedia.org/wiki/Thread_pool">Thread pool - Wikipedia</a>
    </div>
</p>

In the diagram above, all the threads in the pool act as "workers" designated to complete tasks from the task queue collectively.

- Each thread initially picks up a task from the queue to begin work.
- After completing a task, a thread retrieves another task from the queue.
- When the task queue is empty, idle threads wait for new tasks to arrive.

In the following sections, we will explain how ACALSim leverages this design pattern to enhance scalability by efficiently managing numerous simulator tasks.

## Thread Manager and Task Manager Classes

`ThreadManager` and `TaskManager` are abstraction layer classes designed to encapsulate all the stuff about dynamic thread allocation of simulation tasks. They accomplish two purposes:

- Provides `SimTop` with a consistent interface to control the execution of simulators and monitor their status.
- Enable different thread and management mechanisms to be implemented for analysis and further study.

### Dual Phases Execution Model of ACALSim

According to the design of `SimTop::run()` executed by the control thread, the top-level simulation loop is divided into two phases, each with distinct goals:

- **Phase #1**: Simulators perform their internal operations for the current time tick in parallel. The execution order of simulators is not predetermined.
- **Phase #2**: `SimTop` (the control thread) synchronizes message and data exchange among simulators, updates their statuses, determines the next simulation time tick, and decides whether the simulation should terminate.

The interface of `ThreadManager` is designed based on this pattern, allowing `SimTop` to seamlessly trigger the appropriate operations and access necessary information through various detailed implementations.

### Optimization Opportunity of Event-based Simulation

ACALSim manages simulation tasks using an event-driven programming model, which allows it to accelerate simulations by executing only the essential simulator logic during each time tick. A simulator needs to be executed if:

- At least one event expires at the current time tick.
- At least one inbound packet arrives via its `SlaveChannelPort` (i.e. via a `SimChannel`) or `SlavePort` (i.e. via a [`SimPort`]).

All variants of `ThreadManager` and `TaskManager` implementations can improve their overall efficiency by taking advantage of this characteristic.

[`SimPort`]: ../for-users/simport.md

## `ThreadManager` API Overview

This section offers developers a comprehensive guide for implementing a custom `ThreadManager` variant. For complete API details defined in `ThreadManager`, please refer to the API documentation (currently under development).

> **Note**: When developing multi-threaded software, it is crucial to carefully determine which methods will be executed by which threads. All virtual methods listed below are executed by the control thread.

### Enum Declaration

- The status of each worker thread. It is declared for all task scheduler designers to record the status of each worker thread for debugging.
    ```cpp
    enum class ThreadStatus { InActive, Ready, Sleep, Terminated };
    ```

### Virtual Methods

- Prepare the `ThreadManager` and `TaskManager` instances before initializing simulators or creating worker threads.
    ```cpp
    virtual void preSimInit() = 0;
    ```
- Trigger all worker threads to execute the simulators' routines for the current time tick.
    ```cpp
    virtual void startPhase1() = 0;
    ```
- Wait for all worker threads to complete their tasks for the current time tick.
    ```cpp
    virtual void finishPhase1() = 0;
    ```
- Prepare for the start of the 2nd phase of the current iteration.
    ```cpp
    virtual void startPhase2() = 0;
    ```
- Execute `ThreadManager`-specific and `TaskManager`-specific tasks before concluding the second phase.
    ```cpp
    virtual void finishPhase2() = 0;
    ```
- Execute the `bool SimBase::interIterationUpdate()` method for all simulators.
    ```cpp
    virtual void runInterIterationUpdate() = 0;
    ```
    > **Note**: Theoretically, the `interIterationUpdate()` method of simulators only needs to be executed in certain situations. However, we have not yet defined the specific criteria for this. Optimizing this aspect is left as future work.
- Terminate all worker threads once the simulation is complete.
    ```cpp
    virtual void terminateAllThreads() = 0;
    ```
- Perform `ThreadManager`-specific and `TaskManager`-specific cleanup routines after the simulation completes.
    ```cpp
    virtual void postSimCleanup() = 0;
    ```

### Auxiliary Methods

- Get the number of worker threads that have been launched.
    ```cpp
    size_t getNumThreads() const;
    ```
- Return whether the simulation is currently in progress or has either not started or already completed.
    ```cpp
    bool isRunning() const;
    ```
    - The return value depends on the `ThreadManager::running` atomic boolean variable.
    - This flag is set to `true` by `SimTop` and reset to `false` by each `ThreadManager` implementation.
- Get the number of simulators.
    ```cpp
    size_t getNumSimulators() const;
    ```
- Provide a constant reference to the `std::unordered_map` containing pointers to all simulators.
    ```cpp
    const std::unordered_map<std::string, SimBase*>& getAllSimulators() const;
    ```
- Check if all simulators have completed.
    ```cpp
    bool isAllSimulatorDone() const;
    ```
    - This method relies on the value of `ThreadManager::pSimulatorActiveBitMask`, where each bit corresponds to a simulator and is managed by that simulator.

## `TaskManager` API Overview

### Virtual Methods

- Initialize the `TaskManager` instance after all worker threads have been launched.
   ```cpp
   virtual void init();
   ```
- Add a `Task` object to the `TaskManager`.
   ```cpp
   virtual void addTask(const std::shared_ptr<Task>& task) = 0;
   ```
   - `Task` is a data structure representing a job that a worker thread will execute. Developers can extend this structure to customize its contents.
- Retrieve a task that is ready for execution.
   ```cpp
   virtual std::shared_ptr<Task> getReadyTask() = 0;
   ```
- Get the next execution time for the simulation.
   ```cpp
   virtual Tick getNextSimTick() = 0;
   ```
   - The value is defined as the nearest time tick when at least one simulator has something to do.
- Task scheduling routine executed by each worker thread.
   ```cpp
   virtual void scheduler() = 0;
   ```
- Terminate a thread and update the necessary data.
   ```cpp
   virtual void terminateThread() = 0;
   ```

### Auxiliary Methods

- Get the number of simulators registered in the `ThreadManager` instance.
   ```cpp
   int getNTasks() const;
   ```
- Set the status of a worker thread. The enum class `ThreadStatus` is introduced [here](#Enum-Declaration).
   ```cpp
   void setWorkerStatus(uint64_t _tid, ThreadStatus _status);
   ```
- Get the status of a specific worker thread.
   ```cpp
   ThreadStatus getWorkerStatus(uint64_t _tid) const;
   ```

### Auxiliary Variables

- An atomic variable shared by all worker threads, representing the number of threads currently in the synchronization phase. This variable is used by designers to manage synchronization across threads.
    ```cpp
    std::atomic<int> nFinishedThreads;
    ```

## Empty Base Class `Task`

The `Task` class is intended to represent a "task" processed by a worker thread. Since different task scheduler designs may require unique information and methods, the `Task` class is left empty to allow for customization in all `ThreadManager` and `TaskManager` variants.

## Responsibilities of Derived Classes

The `ThreadManager` and `TaskManager` are designed to provide the essential infrastructure that enables developers to create custom variants tailored to specific strategies. Any custom variant must implement the following core functionalities:

- Thread Manager
    - Initiate worker threads to execute all simulators with pending operations during Phase #1 of the current time tick.
    - Block the control thread until all tasks for Phase #1 of the current time tick are complete.
    - Execute the `SimBase::interIterationUpdate()` function for all simulators during Phase #2 of each iteration.
- Task Manager
    - Collect all tasks for execution.
    - Identify the nearest time tick at which at least one simulator has pending operations.
        > **Note**: The next simulation time tick for each simulator can be retrieved using `SimBase::getNextSimTick()`.
    - Provide tasks that are ready for execution at the current time tick.
    - The task scheduling routine that is executed by all worker threads.

> **Note**: The terminology "task" mentioned here usually refers to the `SimBase::stepWrapperBase()` of a simulator.

In the next section, we will introduce how each available `ThreadManager` and `TaskManager` design implements these features in separate documents.

## Available ThreadManager Implementations

ACALSim provides several ThreadManager implementations validated for production use, as well as experimental versions for research purposes.

### Production Versions

The following ThreadManager implementations are production-ready and validated:

| Name | Alias | Best For | Documentation |
|------|-------|----------|---------------|
| **PriorityQueue** | V1 | Sparse activation patterns (e.g., DGXSim) | [Documentation](./thread-manager-v1.md) |
| **Barrier** | V2 | C++20 barrier-based synchronization | Coming soon |
| **PrebuiltTaskList** | V3 | Memory-intensive workloads (e.g., GPUSim) | Coming soon |
| **LocalTaskQueue** | V6 | Lock-optimized version of PriorityQueue | Coming soon |

**Usage example:**
```bash
# Using descriptive names (recommended)
./my_simulation --threadmanager PriorityQueue
./my_simulation --threadmanager PrebuiltTaskList

# Using numeric aliases (backward compatible)
./my_simulation --threadmanager 1
./my_simulation --threadmanager 3
```

### Experimental Versions

The following versions are research prototypes not validated for production use. They are located in `include/sim/experimental/` and should only be used for research purposes:

- **V4**: Dedicated thread per simulator (no thread pooling)
- **V5**: Simplified V1 with active bit mask removed
- **V7**: Hybrid approach combining C++20 barriers with pre-built task lists
- **V8**: Experimental refinement of V3

For details about experimental versions, see `include/sim/experimental/README.md`.

### Choosing the Right ThreadManager

1. **Profile your workload** - Use ACALSim's profiling tools to understand your simulation patterns
2. **Check simulation characteristics**:
   - **Sparse activation** (many simulators, few active per tick) → **PriorityQueue**
   - **Memory-intensive** (high event/packet frequency) → **PrebuiltTaskList**
   - **V1 with lock contention** → **LocalTaskQueue**
3. **Run experiments** - Test different versions with your specific workload

### References

- Figure 5 & 7: Performance comparison of PriorityQueue, PrebuiltTaskList, and LocalTaskQueue
- Section IV: Thread Manager Specialization
