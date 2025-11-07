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

#pragma once

#include "external/gem5/Event.hh"

namespace acalsim {

/**
 * @file Task.hh
 * @brief Base classes for parallel task execution in multi-threaded simulation
 *
 * @details
 * This file defines the infrastructure for task-based parallelism in ACALSim's
 * multi-threaded simulation framework. Tasks represent units of work that can
 * be executed by worker threads in the ThreadManager.
 *
 * **Task-Based Parallelism Model:**
 * ```
 * Main Thread                Worker Threads
 *     |                         |  |  |
 *     |--- Create Tasks ------->|  |  |
 *     |                         |  |  |
 *     |                      [Execute]  |
 *     |                         |  [Execute]
 *     |                         |  |  [Execute]
 *     |<-- Synchronize ---------|--|--|
 *     |                         |  |  |
 * ```
 *
 * **Key Components:**
 *
 * - **Task**: Abstract base class for work units
 * - **TaskFunctor**: Callable wrapper that executes a SimBase simulator
 *
 * **Use Cases:**
 *
 * | Scenario | Description |
 * |----------|-------------|
 * | **Parallel Simulation** | Execute multiple simulators concurrently |
 * | **Load Balancing** | Distribute work across worker threads |
 * | **NoC Simulation** | Parallel router execution |
 * | **Multi-Core Models** | Independent core simulation |
 *
 * @see TaskManager for task queue management
 * @see ThreadManager for thread pool execution
 * @see SimBase for simulator interface
 * @since ACALSim 0.1.0
 */

// Forward declaration
class SimBase;

/**
 * @class Task
 * @brief Abstract base class for executable tasks
 *
 * @details
 * Task provides a minimal interface for work units that can be executed
 * by worker threads. Derived classes implement specific task logic.
 *
 * **Design Pattern:**
 * ```
 * class MyTask : public Task {
 * public:
 *     void execute() {
 *         // Task-specific logic
 *     }
 * };
 * ```
 *
 * **Lifecycle:**
 * ```
 * 1. Create:  new MyTask()
 * 2. Enqueue: taskManager.add(task)
 * 3. Execute: Worker thread picks up task
 * 4. Delete:  Task deleted after completion
 * ```
 *
 * @note Currently a base class with no methods - extended by derived tasks
 * @note Virtual destructor enables polymorphic deletion
 *
 * @code{.cpp}
 * // Example: Custom task for data processing
 * class ProcessDataTask : public Task {
 * public:
 *     ProcessDataTask(int* data, size_t size)
 *         : data(data), size(size) {}
 *
 *     void execute() {
 *         for (size_t i = 0; i < size; ++i) {
 *             data[i] = data[i] * 2;  // Process data
 *         }
 *     }
 *
 * private:
 *     int* data;
 *     size_t size;
 * };
 * @endcode
 *
 * @see TaskFunctor for simulator-based tasks
 */
class Task {
public:
	/**
	 * @brief Default constructor
	 */
	Task() {}

	/**
	 * @brief Virtual destructor for polymorphic deletion
	 * @note Enables safe deletion through base class pointer
	 */
	virtual ~Task() {}
};

/**
 * @struct TaskFunctor
 * @brief Callable functor that executes a SimBase simulator as a task
 *
 * @details
 * TaskFunctor wraps a SimBase simulator into a callable object that can be
 * executed by worker threads. It provides the interface for the ThreadManager
 * to execute simulation steps and query simulation state.
 *
 * **Execution Model:**
 * ```
 * TaskFunctor functor;
 * functor.simbase = mySimulator;
 * functor.simBaseName = "CPU0";
 *
 * // Worker thread calls:
 * functor();  // Executes one simulation step
 * ```
 *
 * **State Queries:**
 * ```
 * Tick nextTick = functor.getSimNextTick();  // When next event
 * Tick globalTick = functor.getGlobalTick(); // Current global time
 * bool done = functor.isReadyToTerminate();  // Simulation complete?
 * ```
 *
 * **Thread Safety:**
 * - **operator()**: Not thread-safe - one thread executes per simulator
 * - **Query Methods**: Read-only - safe to call from main thread
 * - **simbase Pointer**: Not owned - caller manages lifetime
 *
 * @code{.cpp}
 * // Example: Create task functors for parallel execution
 * std::vector<TaskFunctor> tasks;
 *
 * for (auto* sim : simulators) {
 *     TaskFunctor functor;
 *     functor.simbase = sim;
 *     functor.simBaseName = sim->getName();
 *     tasks.push_back(functor);
 * }
 *
 * // ThreadManager executes tasks in parallel
 * threadManager.executeTasks(tasks);
 *
 * // Query state after execution
 * for (const auto& task : tasks) {
 *     Tick nextTick = task.getSimNextTick();
 *     LOG_DEBUG << task.getSimBaseName() << " next: " << nextTick;
 * }
 * @endcode
 *
 * @note simbase pointer is not owned - do not delete
 * @note Functor is copied - keep lightweight
 *
 * @see SimBase for simulator interface
 * @see ThreadManager for parallel execution
 */
struct TaskFunctor {
	/** @brief Pointer to the simulator to execute (not owned) */
	SimBase* simbase = nullptr;

	/** @brief Name of the simulator for debugging/logging */
	std::string simBaseName;

	/**
	 * @brief Execute one simulation step
	 *
	 * @note Called by worker thread to run simulator
	 * @note Advances simulation by processing next event
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * TaskFunctor task;
	 * task.simbase = mySim;
	 * task();  // Execute one step
	 * @endcode
	 */
	void operator()();

	/**
	 * @brief Get the tick of the simulator's next scheduled event
	 *
	 * @return Tick when next event will fire (or MaxTick if none)
	 *
	 * @note Used by ThreadManager for synchronization
	 * @note Returns gem5::MaxTick if event queue is empty
	 *
	 * @code{.cpp}
	 * Tick nextEvent = functor.getSimNextTick();
	 * if (nextEvent < globalBarrier) {
	 *     // Can execute this simulator
	 * }
	 * @endcode
	 */
	Tick getSimNextTick() const;

	/**
	 * @brief Get the simulator's current global tick
	 *
	 * @return Current simulation time
	 *
	 * @note Used for synchronization and logging
	 * @note All simulators must stay synchronized
	 *
	 * @code{.cpp}
	 * Tick currentTime = functor.getGlobalTick();
	 * LOG_INFO << "Sim at tick: " << currentTime;
	 * @endcode
	 */
	Tick getGlobalTick() const;

	/**
	 * @brief Check if the simulator is ready to terminate
	 *
	 * @return Tick value indicating readiness (0 = not ready, non-zero = ready)
	 *
	 * @note Used by ThreadManager to detect completion
	 * @note All simulators must be ready before terminating
	 *
	 * @code{.cpp}
	 * if (functor.isReadyToTerminate()) {
	 *     LOG_INFO << "Simulator completed";
	 * }
	 * @endcode
	 */
	Tick isReadyToTerminate() const;

	/**
	 * @brief Get the simulator's name
	 *
	 * @return Simulator name string
	 *
	 * @note Used for debugging and logging
	 *
	 * @code{.cpp}
	 * std::string name = functor.getSimBaseName();
	 * LOG_DEBUG << "Executing: " << name;
	 * @endcode
	 */
	std::string getSimBaseName() const { return simBaseName; }

	/**
	 * @brief Get the simulator's unique ID
	 *
	 * @return Simulator identifier
	 *
	 * @note Used for tracking and profiling
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * int id = functor.getSimBaseId();
	 * performanceStats[id].recordExecution();
	 * @endcode
	 */
	int getSimBaseId() const;
};

}  // end of namespace acalsim
