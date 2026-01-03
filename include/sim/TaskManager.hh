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

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "external/gem5/Event.hh"
#include "sim/Task.hh"
#include "sim/ThreadManager.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

/**
 * @file TaskManager.hh
 * @brief Abstract base class for managing task queues and coordinating with thread pools
 *
 * @details
 * TaskManager orchestrates task-based parallelism by managing task queues and
 * coordinating with ThreadManager for worker thread execution. It provides the
 * interface for task scheduling, load balancing, and synchronization in multi-threaded
 * simulation environments.
 *
 * **Task Management Architecture:**
 * ```
 * TaskManager (task queue + scheduling)
 *      |
 *      |--- addTask() → [Task Queue]
 *      |
 *      |--- getReadyTask() ← Worker Thread 0
 *      |--- getReadyTask() ← Worker Thread 1
 *      |--- getReadyTask() ← Worker Thread 2
 *      |
 *      |--- scheduler(_tidx) → Per-thread scheduling logic
 *      |
 *      |--- terminateThread() → Synchronization barrier
 *      |
 *      ↓
 * ThreadManager (thread pool execution)
 * ```
 *
 * **Key Features:**
 *
 * - **Abstract Interface**: Pure virtual methods for derived implementations
 * - **Thread Coordination**: Manages worker thread status and completion
 * - **Task Queue Management**: Add, retrieve, and schedule tasks
 * - **Load Balancing**: Distribute tasks across worker threads
 * - **Statistics Collection**: Optional profiling of task execution
 * - **Synchronization**: Track finished threads for barrier synchronization
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Implementation |
 * |----------|-------------|----------------|
 * | **Parallel Simulation** | Multiple simulators in parallel | FIFO task queue |
 * | **Load Balancing** | Dynamic work distribution | Work-stealing queue |
 * | **Priority Scheduling** | Execute high-priority tasks first | Priority queue |
 * | **Dependency Tracking** | Respect task dependencies | DAG-based scheduler |
 * | **Timed Execution** | Schedule tasks at specific ticks | Time-based priority |
 *
 * **Inheritance Hierarchy:**
 * ```
 * HashableType
 *    ↑
 *    | (virtual)
 *    |
 * TaskManager (abstract base)
 *    ↑
 *    |
 * ├── FIFOTaskManager (FIFO queue)
 * ├── PriorityTaskManager (priority-based)
 * ├── WorkStealingTaskManager (load balancing)
 * └── Custom implementations...
 * ```
 *
 * **Task Lifecycle:**
 * ```
 * 1. Creation:     Create tasks (e.g., TaskFunctor)
 * 2. Addition:     addTask(task) → Insert into queue
 * 3. Scheduling:   scheduler(tid) → Coordinate execution
 * 4. Retrieval:    getReadyTask() → Worker fetches task
 * 5. Execution:    Worker executes task
 * 6. Completion:   terminateThread() when queue empty
 * 7. Barrier:      All threads terminate → Simulation advances
 * ```
 *
 * **Synchronization Model:**
 * ```
 * Worker Thread 0:  [Execute] [Execute] [Execute] → terminateThread()
 * Worker Thread 1:  [Execute] [Execute] → terminateThread()
 * Worker Thread 2:  [Execute] [Execute] [Execute] [Execute] → terminateThread()
 *                                                   ↓
 *                   [Barrier: All threads finished] → Continue simulation
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | addTask() | Varies | Implementation-dependent (queue insert) |
 * | getReadyTask() | Varies | Implementation-dependent (queue pop) |
 * | terminateThread() | O(1) | Atomic increment |
 * | getWorkerStatus() | O(1) | Direct array access |
 * | setWorkerStatus() | O(1) | Direct array write |
 *
 * @code{.cpp}
 * // Example: FIFO Task Manager implementation
 * class FIFOTaskManager : public TaskManager {
 * public:
 *     FIFOTaskManager(const std::string& name) : TaskManager(name) {}
 *
 *     void addTask(const std::shared_ptr<Task>& task) override {
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         taskQueue.push(task);
 *     }
 *
 *     std::shared_ptr<Task> getReadyTask() override {
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         if (taskQueue.empty()) {
 *             return nullptr;
 *         }
 *         auto task = taskQueue.front();
 *         taskQueue.pop();
 *         return task;
 *     }
 *
 *     Tick getNextSimTick() override {
 *         // Return earliest scheduled tick from all tasks
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         if (taskQueue.empty()) {
 *             return MaxTick;
 *         }
 *         return taskQueue.front()->getNextTick();
 *     }
 *
 *     void scheduler(const size_t _tidx) override {
 *         // Worker thread main loop
 *         while (true) {
 *             auto task = getReadyTask();
 *             if (!task) {
 *                 terminateThread();
 *                 break;
 *             }
 *
 *             // Execute task
 *             (*task)();
 *
 *             // Update statistics if enabled
 * #ifdef ACALSIM_STATISTICS
 *             collectTaskExecStatistics(*task, task->getName());
 * #endif
 *         }
 *     }
 *
 *     void terminateThread() override {
 *         nFinishedThreads.fetch_add(1, std::memory_order_release);
 *     }
 *
 * private:
 *     std::queue<std::shared_ptr<Task>> taskQueue;
 *     std::mutex queueMutex;
 * };
 *
 * // Example: Priority-based Task Manager
 * class PriorityTaskManager : public TaskManager {
 * public:
 *     PriorityTaskManager(const std::string& name) : TaskManager(name) {}
 *
 *     void addTask(const std::shared_ptr<Task>& task) override {
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         // Priority queue: tasks with earlier tick execute first
 *         taskQueue.push(task);
 *     }
 *
 *     std::shared_ptr<Task> getReadyTask() override {
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         if (taskQueue.empty()) {
 *             return nullptr;
 *         }
 *         auto task = taskQueue.top();
 *         taskQueue.pop();
 *         return task;
 *     }
 *
 *     Tick getNextSimTick() override {
 *         std::lock_guard<std::mutex> lock(queueMutex);
 *         if (taskQueue.empty()) {
 *             return MaxTick;
 *         }
 *         return taskQueue.top()->getNextTick();
 *     }
 *
 *     void scheduler(const size_t _tidx) override {
 *         // Retrieve and execute tasks in priority order
 *         while (auto task = getReadyTask()) {
 *             (*task)();
 *         }
 *         terminateThread();
 *     }
 *
 *     void terminateThread() override {
 *         nFinishedThreads.fetch_add(1, std::memory_order_release);
 *     }
 *
 * private:
 *     struct TaskComparator {
 *         bool operator()(const std::shared_ptr<Task>& a,
 *                        const std::shared_ptr<Task>& b) {
 *             return a->getNextTick() > b->getNextTick();
 *         }
 *     };
 *     std::priority_queue<std::shared_ptr<Task>,
 *                        std::vector<std::shared_ptr<Task>>,
 *                        TaskComparator> taskQueue;
 *     std::mutex queueMutex;
 * };
 *
 * // Example: Using TaskManager with ThreadManager
 * class ParallelSimulation {
 * public:
 *     ParallelSimulation(int nThreads) {
 *         // Create thread manager
 *         threadMgr = new ThreadManagerBase(nThreads);
 *
 *         // Create task manager
 *         taskMgr = new FIFOTaskManager("TaskMgr");
 *
 *         // Link them together
 *         taskMgr->linkThreadManager(threadMgr);
 *         threadMgr->setTaskManager(taskMgr);
 *
 *         // Initialize
 *         taskMgr->init();
 *     }
 *
 *     void addSimulator(SimBase* sim) {
 *         // Create task functor for simulator
 *         auto task = std::make_shared<TaskFunctor>();
 *         task->simbase = sim;
 *         task->simBaseName = sim->getName();
 *
 *         // Add to task manager
 *         taskMgr->addTask(task);
 *     }
 *
 *     void run() {
 *         // Launch worker threads to execute tasks
 *         threadMgr->launchWorkers();
 *
 *         // Wait for all tasks to complete
 *         threadMgr->wait();
 *     }
 *
 * private:
 *     ThreadManagerBase* threadMgr;
 *     FIFOTaskManager* taskMgr;
 * };
 * @endcode
 *
 * @note TaskManager is abstract - must derive to implement scheduling policy
 * @note Virtual inheritance from HashableType supports diamond inheritance
 * @note Thread-safe methods must be implemented by derived classes
 *
 * @warning Ensure proper synchronization in derived implementations
 * @warning terminateThread() must use atomic operations
 * @warning getReadyTask() must be thread-safe across worker threads
 *
 * @see ThreadManager for thread pool management
 * @see Task for task abstraction
 * @see TaskFunctor for simulator-based tasks
 * @since ACALSim 0.1.0
 */

/**
 * @class TaskManager
 * @brief Abstract base class for task queue management and thread coordination
 *
 * @details
 * TaskManager provides the interface for managing task queues in multi-threaded
 * simulations. Derived classes implement specific scheduling policies (FIFO,
 * priority, work-stealing, etc.) and queue data structures.
 *
 * **Design Pattern:**
 * - **Abstract Base Class**: Defines interface, no concrete implementation
 * - **Strategy Pattern**: Different scheduling algorithms via derivation
 * - **Friend Class**: ThreadManagerBase has privileged access
 *
 * **Virtual Inheritance:**
 * Uses virtual inheritance from HashableType to prevent diamond inheritance issues.
 *
 * @note All queue operations must be implemented by derived classes
 * @note Thread safety is responsibility of derived implementations
 */
class TaskManager : virtual public HashableType {
	/** @brief Friend class for privileged access to internals */
	friend class ThreadManagerBase;

public:
	/**
	 * @brief Construct a TaskManager with a name
	 *
	 * @param _name Name of this task manager instance
	 *
	 * @note Initializes with no thread manager linked
	 * @note Sets finished thread count to 0
	 * @note Sets task count to 0
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * class MyTaskManager : public TaskManager {
	 * public:
	 *     MyTaskManager() : TaskManager("MyTaskMgr") {}
	 * };
	 * @endcode
	 */
	TaskManager(const std::string& _name) : threadManager(nullptr), nFinishedThreads(0), nTasks(0) {}

	/**
	 * @brief Virtual destructor for safe polymorphic deletion
	 *
	 * @note Implementation in TaskManager.inl
	 */
	inline ~TaskManager();

	/**
	 * @brief Link this TaskManager to a ThreadManager
	 *
	 * @param manager Pointer to ThreadManagerBase to link
	 *
	 * @note Must be called before init()
	 * @note Enables access to worker thread status and simulators
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * TaskManager* taskMgr = new FIFOTaskManager("TaskMgr");
	 * ThreadManagerBase* threadMgr = new ThreadManagerBase(4);
	 * taskMgr->linkThreadManager(threadMgr);
	 * @endcode
	 */
	void linkThreadManager(ThreadManagerBase* manager) { this->threadManager = manager; }

	/**
	 * @brief Set the total number of tasks
	 *
	 * @param n Total number of tasks to execute
	 *
	 * @note Used for progress tracking and statistics
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * taskMgr->setNTasks(simulators.size());
	 * @endcode
	 */
	void setNTasks(int n) { nTasks = n; }

	/**
	 * @brief Get the total number of tasks
	 *
	 * @return int Total task count
	 *
	 * @note Returns value set by setNTasks()
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * int total = taskMgr->getNTasks();
	 * LOG_INFO << "Managing " << total << " tasks";
	 * @endcode
	 */
	int getNTasks() const { return nTasks; }

	/**
	 * @brief Get the status of a specific worker thread
	 *
	 * @param _tid Thread ID (index into worker array)
	 * @return ThreadStatus Current status of worker thread
	 *
	 * @note Delegates to ThreadManager's workerStatus array
	 * @note Complexity: O(1) - direct array access
	 *
	 * @code{.cpp}
	 * ThreadStatus status = taskMgr->getWorkerStatus(0);
	 * if (status == ThreadStatus::IDLE) {
	 *     // Worker 0 is idle
	 * }
	 * @endcode
	 */
	ThreadStatus getWorkerStatus(uint64_t _tid) const { return this->threadManager->workerStatus[_tid]; }

	/**
	 * @brief Set the status of a specific worker thread
	 *
	 * @param _tid Thread ID (index into worker array)
	 * @param _status New status to set
	 *
	 * @note Delegates to ThreadManager's workerStatus array
	 * @note Not thread-safe - use with caution
	 * @note Complexity: O(1) - direct array write
	 *
	 * @code{.cpp}
	 * taskMgr->setWorkerStatus(0, ThreadStatus::BUSY);
	 * @endcode
	 */
	void setWorkerStatus(uint64_t _tid, ThreadStatus _status) { this->threadManager->workerStatus[_tid] = _status; }

	/**
	 * @brief Get reference to all simulators from ThreadManager
	 *
	 * @return const HashVector<std::string, SimBase*>& Simulator registry
	 *
	 * @note Provides access to all registered simulators
	 * @note Complexity: O(1) - reference return
	 *
	 * @code{.cpp}
	 * const auto& sims = taskMgr->getSimulators();
	 * for (const auto& [name, sim] : sims) {
	 *     LOG_INFO << "Simulator: " << name;
	 * }
	 * @endcode
	 */
	const HashVector<std::string, SimBase*>& getSimulators() const { return this->threadManager->simulators; }

	/**
	 * @brief Terminate current thread and increment finished counter (pure virtual)
	 *
	 * @details
	 * Called by worker threads when no more tasks available. Increments
	 * nFinishedThreads atomically to coordinate barrier synchronization.
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note MUST use atomic operations on nFinishedThreads
	 * @note Typically called at end of scheduler() loop
	 *
	 * @code{.cpp}
	 * void MyTaskManager::terminateThread() override {
	 *     nFinishedThreads.fetch_add(1, std::memory_order_release);
	 * }
	 * @endcode
	 */
	virtual void terminateThread() = 0;

	/**
	 * @brief Add a task to the task queue (pure virtual)
	 *
	 * @param task Shared pointer to task to add
	 *
	 * @details
	 * Inserts task into queue based on scheduling policy. Implementation
	 * must be thread-safe if called from multiple threads.
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note Implementation must provide thread safety
	 * @note Complexity varies by queue type (typically O(1) or O(log n))
	 *
	 * @code{.cpp}
	 * void FIFOTaskManager::addTask(const std::shared_ptr<Task>& task) override {
	 *     std::lock_guard<std::mutex> lock(queueMutex);
	 *     taskQueue.push(task);
	 * }
	 * @endcode
	 */
	virtual void addTask(const std::shared_ptr<Task>& task) = 0;

	/**
	 * @brief Retrieve next ready task from queue (pure virtual)
	 *
	 * @return std::shared_ptr<Task> Next task to execute, or nullptr if none
	 *
	 * @details
	 * Retrieves and removes next task based on scheduling policy.
	 * Returns nullptr when queue is empty. Implementation must be
	 * thread-safe for concurrent worker access.
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note MUST be thread-safe - called by multiple workers
	 * @note Returns nullptr when no tasks available
	 * @note Complexity varies by queue type (typically O(1) or O(log n))
	 *
	 * @code{.cpp}
	 * std::shared_ptr<Task> FIFOTaskManager::getReadyTask() override {
	 *     std::lock_guard<std::mutex> lock(queueMutex);
	 *     if (taskQueue.empty()) return nullptr;
	 *     auto task = taskQueue.front();
	 *     taskQueue.pop();
	 *     return task;
	 * }
	 * @endcode
	 */
	virtual std::shared_ptr<Task> getReadyTask() = 0;

	/**
	 * @brief Get tick of next scheduled task (pure virtual)
	 *
	 * @return Tick Next simulation tick, or MaxTick if no tasks
	 *
	 * @details
	 * Returns earliest tick among all queued tasks. Used by ThreadManager
	 * for synchronization barriers. Returns MaxTick when queue is empty.
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note Used for time-based synchronization
	 * @note Should be thread-safe if accessed concurrently
	 *
	 * @code{.cpp}
	 * Tick MyTaskManager::getNextSimTick() override {
	 *     std::lock_guard<std::mutex> lock(queueMutex);
	 *     if (taskQueue.empty()) return MaxTick;
	 *     return taskQueue.front()->getNextTick();
	 * }
	 * @endcode
	 */
	virtual Tick getNextSimTick() = 0;

	/**
	 * @brief Main scheduler loop for worker threads (pure virtual)
	 *
	 * @param _tidx Worker thread index
	 *
	 * @details
	 * Entry point for worker threads. Continuously retrieves and executes
	 * tasks until queue is empty, then calls terminateThread().
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note Runs in worker thread context
	 * @note Typically loops: getReadyTask() → execute → repeat
	 * @note Must call terminateThread() before returning
	 *
	 * @code{.cpp}
	 * void MyTaskManager::scheduler(const size_t _tidx) override {
	 *     while (auto task = getReadyTask()) {
	 *         (*task)();  // Execute task
	 *     }
	 *     terminateThread();
	 * }
	 * @endcode
	 */
	virtual void scheduler(const size_t _tidx) = 0;

	/**
	 * @brief Initialize task manager after threading setup
	 *
	 * @note Virtual method with inline implementation
	 * @note Called after linkThreadManager() and thread pool launch
	 * @note Default implementation in TaskManager.inl
	 * @note Override for custom initialization
	 *
	 * @code{.cpp}
	 * void MyTaskManager::init() override {
	 *     TaskManager::init();  // Call base
	 *     // Custom initialization
	 *     taskQueue.reserve(getNTasks());
	 * }
	 * @endcode
	 */
	inline virtual void init();

	/**
	 * @brief Get pointer to linked ThreadManager
	 *
	 * @return ThreadManagerBase* Pointer to thread manager, or nullptr
	 *
	 * @note Returns nullptr if linkThreadManager() not called
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * ThreadManagerBase* threadMgr = taskMgr->getThreadManager();
	 * if (threadMgr) {
	 *     int nWorkers = threadMgr->getNumWorkers();
	 * }
	 * @endcode
	 */
	virtual ThreadManagerBase* getThreadManager() const { return this->threadManager; }

protected:
	/**
	 * @brief Collect execution statistics for a task functor
	 *
	 * @tparam TaskFunctor Type of task functor
	 * @param _task Task functor to profile
	 * @param _sim_name Name of simulator for categorization
	 * @return double Execution time in seconds
	 *
	 * @note Only active when ACALSIM_STATISTICS defined
	 * @note Implementation in TaskManager.inl
	 * @note Records per-simulator execution statistics
	 *
	 * @code{.cpp}
	 * #ifdef ACALSIM_STATISTICS
	 * double execTime = collectTaskExecStatistics(taskFunctor, "CPU0");
	 * #endif
	 * @endcode
	 */
	template <typename TaskFunctor>
	inline double collectTaskExecStatistics(TaskFunctor& _task, const std::string& _sim_name);

	/**
	 * @brief Collect execution statistics for function-object pair
	 *
	 * @tparam T Object type
	 * @tparam Func Function type
	 * @param _func Function to execute and profile
	 * @param _obj Object context
	 * @param _sim_name Name for categorization
	 * @return double Execution time in seconds
	 *
	 * @note Only active when ACALSIM_STATISTICS defined
	 * @note Implementation in TaskManager.inl
	 * @note Alternative interface for method profiling
	 */
	template <typename T, typename Func>
	inline double collectTaskExecStatistics(const Func& _func, T* const& _obj, const std::string& _sim_name);

	/** @brief Pointer to linked ThreadManager (not owned) */
	ThreadManagerBase* threadManager = nullptr;

	/**
	 * @brief Atomic counter of finished worker threads
	 * @details Incremented by terminateThread(), used for barrier sync
	 */
	std::atomic<int> nFinishedThreads;

#ifdef ACALSIM_STATISTICS
	/**
	 * @brief Per-simulator execution time statistics
	 * @details Categorized by simulator name, accumulates execution times
	 */
	CategorizedStatistics<std::string, double, StatisticsMode::AccumulatorWithSize, false> execPeriodStatistic;
#endif  // ACALSIM_STATISTICS

private:
	/** @brief Total number of tasks (for tracking/statistics) */
	int nTasks;
};

}  // end of namespace acalsim

#include "TaskManager.inl"
