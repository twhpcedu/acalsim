# PriorityQueue ThreadManager (ThreadManagerV1)

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

[toc]

---

- Author: Jen-Chien Chang \<jenchien@twhpcedu.org\>
- Date: 2024/08/24
- Updated: 2025/11/06 (renamed to PriorityQueue)

---

## Overview

The **PriorityQueue** ThreadManager (also known as ThreadManagerV1) is the default and most commonly used ThreadManager implementation in ACALSim. It is optimized for workloads with **sparse activation patterns**, where many simulators exist but only a subset is active at any given time tick (e.g., DGXSim).

**When to use:**
- Default choice for most simulations
- Workloads with sparse activation patterns
- When you need dynamic task scheduling with priority-based execution

**Location:** `include/sim/ThreadManagerV1/`

## Preface

As outlined in the [Thread Manager and Task Manager - Developer Document](./thread-manager.md), the `ThreadManagerV1` (PriorityQueue) and `TaskManagerV1` must implement specific features to effectively orchestrate the multi-threaded simulation in ACALSim.

- Thread Manager
    - Initiate worker threads to execute all simulators with pending operations during Phase #1 of the current time tick.
    - Block the control thread until all tasks for Phase #1 of the current time tick are complete.
    - Execute the `SimBase::interIterationUpdate()` function for all simulators during Phase #2 of each iteration.
- Task Manager
    - Collect all tasks for execution.
    - Identify the nearest time tick at which at least one simulator has pending operations.
    - Provide tasks that are ready for execution at the current time tick.
    - The task scheduling routine that executed by all worker threads.

In this document, we will introduce the key building blocks of the `V1` variant and demonstrate how these two derived classes meet the requirements mentioned above.

## Building Blocks

### `TaskFunctor` Class

```cpp
struct TaskFunctor {
    SimBase*    simbase = nullptr;
    std::string simBaseName;

    void        operator()();
    Tick        getSimNextTick() const;
    Tick        getGlobalTick() const;
    Tick        isReadyToTerminate() const;
    std::string getSimBaseName() const;
    int         getSimBaseId() const;
};
```

The primary purpose of the `TaskFunctor` class is to simplify the execution of simulation routines within worker threads. It does this by encapsulating a simulation task and providing a callable interface that executes operations on a specific `SimBase` instance. This design allows worker threads to execute and manage tasks without dealing with the underlying simulator-specific logic, streamlining the overall process.

#### Structure

The `TaskFunctor` class contains the following members:

- `simbase`: A pointer to a `SimBase` object, representing the simulator instance associated with the task.
- `simBaseName`: A `std::string` holding the name of the simulator, providing a convenient way to identify the associated simulator.

#### Key Methods

- `void operator()()`
    - This is the core method of the `TaskFunctor` class, enabling the object to be called like a function.
    - When invoked by a worker thread, it triggers the `stepWrapperBase()` method of the associated `SimBase` instance, which executes the current step in the simulation process.
- `Tick getSimNextTick()`
    - Retrieves the next scheduled simulation tick for the associated simulator by calling `getSimNextTick()` on the `SimBase` instance.
    - This is essential for the `TaskManagerV1` to determine when the next simulation event should occur.
- `Tick getGlobalTick()`
    - Gets the current global simulation tick from the `SimTop` instance that the pointer `top` refers to.
- `Tick isReadyToTerminate()`
    - Checks if the whole simulation is ready to terminate by calling `isReadyToTerminate()` of the `SimTop` instance that the pointer `top` refers to.
- `std::string getSimBaseName() const`
    - Returns the name of the associated simulator.
- `int getSimBaseId()`
    - Retrieves the unique identifier of the associated simulator by calling `getID()` on the `SimBase` instance.

### `Task` Class

```cpp
struct Task {
    // Function to execute
    TaskFunctor functor;

    // Next execution time
    Tick next_execution_cycle;

    int id;

    Task() : next_execution_cycle(0), id(0) { ; }

    // Constructor to initialize the task
    Task(SimBase* base, std::string name) {
        functor.simbase      = base;
        functor.simBaseName  = name;
        next_execution_cycle = 0;
        id                   = functor.getSimBaseId();
    }

    // Operator() to execute the task
    void operator()() {
        functor();
        updateNextTick();
    }

    void updateNextTick() {
        if (!functor.isReadyToTerminate()) {
            next_execution_cycle = functor.getSimNextTick();
        } else {
            next_execution_cycle = functor.getGlobalTick() + 1;
        }
    }

    bool operator<(const Task& other) const { return next_execution_cycle < other.next_execution_cycle; }
    bool operator==(const Task& other) const { return next_execution_cycle == other.next_execution_cycle; }

    friend std::ostream& operator<<(std::ostream& os, const Task& t);
};
```

The `Task` class streamlines the execution of simulation routines by integrating both task management and scheduling. It allows worker threads to execute tasks without dealing with simulator-specific logic, while also managing the timing of task execution through the `next_execution_cycle` attribute.

#### Structure

The `Task` class consists of the following key components:

- `TaskFunctor functor`: Encapsulates the simulation logic to be executed, allowing worker threads to invoke the simulation routine without dealing with simulator-specific details.
- `Tick next_execution_cycle`: Represents the next time tick at which the task should be executed. This allows for precise control over task scheduling.
- `int id`: The unique identifier of the task, corresponding to the `SimBase` instance's ID.

#### Constructor

- Default Constructor
    - Initializes a `Task` with default values. Both the `next_execution_cycle` and the `id` are initialized to 0.
- Parameterized Constructor
    - Initializes a `Task` with a given `SimBase` instance and a corresponding name. This constructor assigns the `SimBase` instance to the functor and sets the `id` to the unique identifier of the simulator (i.e. `SimBase`).
    - The `next_execution_cycle` is initialized to 0, indicating that the task is ready to be executed at the first simulation tick.

#### Key Methods

- `void operator()()`
    - This method makes the `Task` class callable like a function.
    - When invoked, it triggers the execution of the encapsulated `TaskFunctor` and subsequently updates the next execution time for the task using `updateNextTick()`.
    - This ensures that after each execution, the task is rescheduled based on the simulator's state.
- `void updateNextTick()`
    - Updates the `next_execution_cycle` based on the state of the simulator.
    - If the co-simulation **is not ready** to terminate, it retrieves the next scheduled simulation tick via `functor.getSimNextTick()` and assigns it to `next_execution_cycle`.
    - If the co-simulation **is ready** to terminate, it assigns `functor.getGlobalTick() + 1` to the `next_execution_cycle`. This ensures that the task will be executed the next time tick to process its `ExitEvent`.
- Comparison Operators (`<` and `==`)
    - The `Task` class overloads these two operators to allow for easy comparison based on the `next_execution_cycle`.
    - This is crucial for task scheduling, enabling tasks to be ordered and prioritized according to their scheduled execution time.

### Task Queue - `UpdateablePriorityQueue`

This is a custom data structure designed to efficiently manage tasks based on their priority, where priority represents the next simulation time tick for a task. According to the design of ACALSim, the next simulation time tick of a simulator (i.e. `SimBase`) is determined by its two types of status.

1. The nearest time tick that needs to process events in the simulator's event queue.
2. Whether there is inbound communication via `ChannelPort` or `SimPort`. If so, the simulator needs to be executed in the next time tick.

Since the first one can be determined right after the routine of the current iteration is finished while the second one cannot be identified until the Phase #2 of the current iteration, it is a common use case for the task queue to insert a task first and update its porition later on. In addition, another common operation againest the task queue is to query when is the next time tick that has operations to do. The `UpdateablePriorityQueue` class is designed to fulfill these operations.

#### Key Methods

> **Note**: In this section, only the important public methods are introduced.

- `push(const T& value, uint64_t priority)`
    - Adds a new task to the priority queue with the given priority.
    - The task is immediately positioned based on its priority using a heap structure.
- `update(int simID, uint64_t newPriority)`
    - Updates the priority of an existing task identified by simID.
- `hasReadyTask(uint64_t priority)`
    - Checks if there is a task in the queue with a priority less than or equal to the provided value, indicating that the task is ready to be executed.
- `top()`
    - Retrieves the task with the highest priority (smallest priority value) without removing it from the queue.
    - Any pending updates are processed to ensure that the top task is accurate.
- `pop()`
    - Removes the task with the highest priority from the queue.
    - This operation also processes any pending updates to maintain the integrity of the priority queue.

## Task Scheduling Routine for Worker Threads

The `TaskManagerV1::scheduler()` function is responsible for scheduling and executing tasks across multiple worker threads in ACALSim. This function manages synchronization using various mechanisms to ensure orderly execution and coordination between threads.

> **Note**: All the variables mentioned in this section belong to `TaskManagerV1` unless stated otherwise.

### Synchronization Mechanisms

- Conditional Variables
    - `newTaskAvailableCondVar`: This condition variable is used to notify worker threads when a new iteration begins, allowing them to start processing tasks.
    - `workerThreadsDoneCondVar`: This is used to signal the control thread once all worker threads have completed processing for the current iteration.
- Mutexes
    - `taskQueueMutex`: Protects access to the shared `taskQueue`, ensuring that only one thread can modify it at a time.
    - `taskAvailableMutex`: Guards the sleeping mechanism of worker threads and control thread, ensuring threads properly go to sleep and wake up in a synchronized manner.
- Atomic Variables
    - `startPhase1`: Controls whether threads can be woken up for a new iteration. It ensures that threads are only activated when the system is ready to proceed.
    - `allThreadsDone`: Indicates when all worker threads have finished processing for the current iteration, ensuring the control thread can safely proceed to the next iteration.

### Scheduler Method Design

The `scheduler()` method is designed to manage task execution and synchronization for each worker thread in a multi-threaded environment. The whole design aims to minimize idle time and ensure smooth task execution across iterations in the simulation.

- **Initialization Stage**: Upon startup, the thread waits until the simulation is running and then changes its status to Ready, indicating that it is prepared to start processing tasks.
- **Task Scheduling Loop**: The core of the function is a loop that continues until the thread is set to terminate. In each iteration:
    - The thread acquires a lock on the `taskQueue` to check if there are any tasks ready to execute.
    - If a task is ready, it is popped from the queue and executed immediately. Once finished, the task may be rescheduled for the next execution cycle, and the thread continues to process other tasks.
- **Sleep Mechanism**:
    - If no tasks are available, the thread increments the `TaskManager::nFinishedThreads` counter and potentially puts itself to sleep, waiting for the next iteration.
    - The last thread to complete processing signals the control thread by setting `allThreadsDone` to `true` and notifying the control thread via `workerThreadsDoneCondVar`.
- **Wake-up Mechanism**: When a new iteration begins, all sleeping threads are woken up using the `newTaskAvailableCondVar` condition variable, and they proceed to check for available tasks again.
- **Termination Handling**:
    - The function gracefully handles termination by checking the local flag `readyToTerminate`.
    - If all tasks have been processed and termination is signaled, the thread exits the loop and calls the `terminateThread()` function before stopping its execution.

## Operations in Each Simulation Phase

### Initialization

The `ThreadManagerV1` implementation leaves the `void preSimInit()` function empty. No pre-simulation initialization is needed before worker threads start.

### Phase #1

- `void startPhase1()`
    - The following variables are reset.
        ```cpp
        this->getTaskManager()->nFinishedThreads = 0;
        this->getTaskManager()->startPhase1.store(true);
        this->getTaskManager()->allThreadsDone.store(false);
        ```
    - All worker threads are notified to start executing simulators.
        ```cpp
        this->getTaskManager()->newTaskAvailableCondVar.notify_all();
        ```
- `void finishPhase1()`
    - The control thread is blocked by `TaskManagerV1::workerThreadsDoneCondVar` until `TaskManagerV1::allThreadsDone` becomes `true`.
        ```cpp
        std::unique_lock<std::mutex> lock(this->getTaskManager()->taskAvailableMutex);

        this->getTaskManager()->workerThreadsDoneCondVar.wait(lock, [this] {
            return this->getTaskManager()->allThreadsDone.load();
        });
        ```

### Phase #2

- `void startPhase2()`: This method performs no operations.
- `void runInterIterationUpdate()`: The `ThreadManagerV1` executes `SimBase::interIterationUpdate()` of all simulators and update their positions in the task queue if needed.
- `void finishPhase2()`: This method performs no operations.

### Clean Up

- `void terminateAllThreads()`
    - Sets `ThreadManager::running` to `false`.
    - Waits for all worker threads to stop using `std::thread::join()`.
- `void postSimCleanup()`: This method performs no operations.
