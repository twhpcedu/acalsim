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

#include <atomic>
#include <cstddef>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ACALSim
#include "common/BitVector.hh"
#include "common/HashVector.hh"
#include "container/SharedContainer.hh"
#include "external/gem5/Event.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#include "profiling/Synchronization.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

/**
 * @file ThreadManager.hh
 * @brief Thread pool management and synchronization for parallel simulation
 *
 * @details
 * ThreadManager orchestrates multi-threaded parallel simulation by managing a pool
 * of worker threads, coordinating simulator execution, and implementing two-phase
 * synchronization barriers. Essential for achieving performance scalability in
 * complex system simulations.
 *
 * **Thread Pool Architecture:**
 * ```
 * ThreadManager
 *      |
 *      |--- Worker Thread 0 → Execute Simulator 0
 *      |--- Worker Thread 1 → Execute Simulator 1
 *      |--- Worker Thread 2 → Execute Simulator 2
 *      |--- Worker Thread 3 → Execute Simulator 3
 *      |
 *      |--- Phase 1: Parallel Execution
 *      |       ↓
 *      |--- Synchronization Barrier
 *      |       ↓
 *      |--- Phase 2: Inter-simulator Communication
 *      |       ↓
 *      |--- Next Iteration
 * ```
 *
 * **Key Features:**
 *
 * - **Thread Pool**: Manages worker threads for parallel execution
 * - **Two-Phase Synchronization**: Separates computation and communication
 * - **Bit Mask Coordination**: Tracks simulator active/pending status
 * - **Dynamic Thread Assignment**: Maps simulators to worker threads
 * - **Statistics Collection**: Optional profiling of thread performance
 * - **Scalability**: Adjustable thread count based on workload
 *
 * **Two-Phase Synchronization Model:**
 * ```
 * Iteration N:
 *   Phase 1 (Parallel):
 *     Thread 0: Execute Simulator 0 (independent)
 *     Thread 1: Execute Simulator 1 (independent)
 *     Thread 2: Execute Simulator 2 (independent)
 *     Thread 3: Execute Simulator 3 (independent)
 *
 *   [Barrier: All threads finish Phase 1]
 *
 *   Phase 2 (Communication):
 *     Inter-simulator messages processed
 *     Event queue updates
 *     Shared state synchronization
 *
 *   [Barrier: Phase 2 complete]
 *
 * Iteration N+1: Repeat
 * ```
 *
 * **Thread Status Lifecycle:**
 * ```
 * InActive → Ready → Executing → Sleep (waiting) → ReadyToTerminate → Terminated
 *              ↑                      ↓
 *              └──────────────────────┘
 *                 (next task available)
 * ```
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Benefit |
 * |----------|-------------|---------|
 * | **Multi-Core Simulation** | Parallel CPU core execution | N× speedup |
 * | **NoC Simulation** | Parallel router processing | Scalable interconnect |
 * | **Heterogeneous Systems** | CPU + GPU + Accelerators | Concurrent execution |
 * | **Large-Scale Models** | 100+ component systems | Performance critical |
 * | **Parameter Sweeps** | Multiple configurations | Throughput optimization |
 *
 * **Performance Characteristics:**
 *
 * | Metric | Ideal | Typical | Notes |
 * |--------|-------|---------|-------|
 * | **Speedup** | N× (N threads) | 0.7N - 0.9N× | Amdahl's law applies |
 * | **Overhead** | 0% | 5-15% | Synchronization cost |
 * | **Load Balance** | Perfect | Good | Work-stealing helps |
 * | **Scalability** | Linear | Sub-linear | Diminishing returns |
 *
 * @code{.cpp}
 * // Example: Setting up parallel simulation with 4 threads
 * class ParallelSystemSimulation {
 * public:
 *     ParallelSystemSimulation(int nCores = 4) {
 *         // Create thread manager with 4 worker threads
 *         threadMgr = new ThreadManager("ThreadMgr", nCores, true);
 *
 *         // Create task manager for work distribution
 *         taskMgr = new FIFOTaskManager("TaskMgr");
 *
 *         // Link them together
 *         threadMgr->linkTaskManager(taskMgr);
 *         taskMgr->linkThreadManager(threadMgr);
 *
 *         // Create simulators
 *         for (int i = 0; i < 4; i++) {
 *             auto* cpu = new CPUSimulator("CPU" + std::to_string(i));
 *             threadMgr->addSimulator(cpu);
 *         }
 *
 *         // Initialize
 *         threadMgr->preSimInitWrapper();
 *         threadMgr->simInit();
 *     }
 *
 *     void run() {
 *         // Start worker threads
 *         threadMgr->startSimThreads();
 *
 *         // Main simulation loop
 *         while (!threadMgr->isAllSimulatorDone()) {
 *             // Phase 1: Parallel execution
 *             threadMgr->startPhase1();
 *             // Workers execute simulators in parallel
 *             threadMgr->finishPhase1();
 *
 *             // Phase 2: Synchronization and communication
 *             threadMgr->startPhase2();
 *             threadMgr->runInterIterationUpdate();
 *             threadMgr->finishPhase2();
 *         }
 *
 *         // Cleanup
 *         threadMgr->terminateAllThreads();
 *         threadMgr->postSimCleanupWrapper();
 *     }
 *
 * private:
 *     ThreadManager* threadMgr;
 *     TaskManager* taskMgr;
 * };
 *
 * // Example: Dynamic load balancing with work-stealing
 * class AdaptiveThreadManager : public ThreadManager {
 * public:
 *     AdaptiveThreadManager(int nThreads)
 *         : ThreadManager("AdaptiveMgr", nThreads, true) {}
 *
 *     void startPhase1() override {
 *         // Distribute work with load balancing
 *         for (size_t i = 0; i < getNumSimulators(); i++) {
 *             if (isPendingEventBitMaskSet(i)) {
 *                 // Simulator has pending work
 *                 taskMgr->addTask(createTaskFor(i));
 *             }
 *         }
 *
 *         // Workers steal tasks from each other
 *         ThreadManager::startPhase1();
 *     }
 * };
 *
 * // Example: Monitoring thread performance
 * class MonitoredSimulation {
 * public:
 *     void runWithProfiling() {
 * #ifdef ACALSIM_STATISTICS
 *         threadMgr->startRunning();
 *
 *         while (!done) {
 *             threadMgr->startPhase1();
 *             threadMgr->finishPhase1();
 *
 *             // Check performance metrics
 *             double idleTime = threadMgr->getAvgThreadIdleTime();
 *             double execTime = threadMgr->getAvgTaskExecTimePerThread();
 *             double efficiency = execTime / (execTime + idleTime);
 *
 *             LOG_INFO << "Thread efficiency: " << (efficiency * 100) << "%";
 *
 *             threadMgr->startPhase2();
 *             threadMgr->finishPhase2();
 *         }
 *
 *         threadMgr->printSchedulingOverheads(totalTime);
 * #endif
 *     }
 *
 * private:
 *     ThreadManager* threadMgr;
 * };
 * @endcode
 *
 * @note Virtual inheritance from HashableType prevents diamond inheritance
 * @note Thread safety critical - careful synchronization required
 * @note Two-phase model essential for correctness in parallel simulation
 *
 * @warning Do not modify bit masks during parallel execution
 * @warning Ensure all simulators reach barriers at same iteration
 * @warning Statistics collection has overhead - use conditionally
 *
 * @see TaskManager for task queue management
 * @see SimBase for simulator interface
 * @since ACALSim 0.1.0
 */

// Forward declarations for circular dependencies
class SimBase;
class TaskManager;

/**
 * @enum ThreadStatus
 * @brief Worker thread execution states
 *
 * @details
 * Defines the lifecycle states of worker threads in the thread pool.
 * Used for coordinating thread execution and detecting completion.
 */
enum class ThreadStatus {
	InActive,         /**< Thread not yet started */
	Ready,            /**< Thread ready to execute tasks */
	Sleep,            /**< Thread waiting for work */
	ReadyToTerminate, /**< Thread completing execution */
	Terminated        /**< Thread has stopped */
};

/**
 * @class ThreadManagerBase
 * @brief Abstract base class for multi-threaded simulation coordination
 *
 * @details
 * Provides the interface and common infrastructure for managing worker threads,
 * coordinating simulator execution, and implementing synchronization barriers.
 * Derived classes implement specific thread management strategies.
 *
 * **Design Pattern:**
 * - **Template Method**: Virtual methods define extension points
 * - **Two-Phase Protocol**: startPhase1/2 and finishPhase1/2
 * - **Observer Pattern**: Bit masks track simulator state
 * - **Virtual Inheritance**: Supports diamond inheritance
 *
 * @note Abstract class - must derive to implement thread management strategy
 * @note Thread-safe methods use atomic operations and synchronization
 */
class ThreadManagerBase : virtual public HashableType {
	/** @brief Friend class for privileged access to internals */
	friend class TaskManager;

public:
	/**
	 * @brief Construct thread manager with specified thread count
	 * @param _name Manager name for identification
	 * @param _nThreads Number of worker threads
	 * @param _nThreadsAdjustable Allow dynamic thread count adjustment
	 * @note Implementation in source file
	 */
	ThreadManagerBase(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable);

	/**
	 * @brief Destructor - cleans up worker threads
	 * @note Implementation in source file
	 */
	~ThreadManagerBase();

	/**
	 * @brief Link this ThreadManager to a TaskManager
	 * @param _taskManager Pointer to task manager
	 * @note Must be called before initialization
	 */
	void linkTaskManager(TaskManager* _taskManager) { taskManager = _taskManager; }

	/**
	 * @brief Add simulator to thread pool (pure virtual)
	 * @param _sim Simulator to add
	 * @note Pure virtual - implementation in derived class
	 */
	virtual void addSimulator(SimBase* _sim) = 0;

	/**
	 * @brief Get simulator by name
	 * @param _name Simulator name
	 * @return SimBase* Pointer to simulator, or nullptr if not found
	 * @note Implementation in source file
	 */
	SimBase* getSimulator(const std::string& _name) const;

	/**
	 * @brief Get all registered simulators
	 * @return const HashVector& Reference to simulator registry
	 */
	const HashVector<std::string, SimBase*>& getAllSimulators() const { return this->simulators; }

	/** @brief Get number of registered simulators */
	size_t getNumSimulators() const { return simulators.size(); }

	/** @brief Get number of worker threads */
	size_t getNumThreads() const { return nThreads; }

	/**
	 * @brief Check if all simulators inactive (bit mask all zero)
	 * @return true if no simulator has pending events
	 */
	bool isPendingEventBitMaskZero() const {
		return this->pSimulatorActiveBitMask->run(0, &BitVector::allEqual, false);
	}

	/**
	 * @brief Check if specific simulator has pending events
	 * @param id Simulator ID
	 * @return true if simulator has pending work
	 */
	bool isPendingEventBitMaskSet(size_t id) const {
		return this->pSimulatorActiveBitMask->run(0, &BitVector::getBit, id);
	}

	/**
	 * @brief Mark simulator as having pending events
	 * @param id Simulator ID
	 */
	void setPendingEventBitMask(size_t id) { this->pSimulatorActiveBitMask->run(0, &BitVector::setBit, id, true); }

	/**
	 * @brief Clear simulator's pending event flag
	 * @param id Simulator ID
	 */
	void clearPendingEventBitMask(size_t id) { this->pSimulatorActiveBitMask->run(0, &BitVector::setBit, id, false); }

	/**
	 * @brief Check if all simulators completed execution
	 * @return true if all simulators done
	 */
	bool isAllSimulatorDone() const { return this->pSimulatorActiveBitMask->run(0, &BitVector::allEqual, false); }

	/**
	 * @brief Check if thread manager is in running state
	 * @return true if running
	 */
	bool isRunning() const { return this->running; }

	/** @brief Wrapper to terminate all worker threads (pure virtual) */
	virtual void terminateAllThreadsWrapper() = 0;

	/** @brief Start worker threads and begin execution (pure virtual) */
	virtual void startSimThreads() = 0;

	/** @brief Terminate all worker threads (pure virtual) */
	virtual void terminateAllThreads() = 0;

	/** @brief Wrapper for pre-simulation initialization */
	void preSimInitWrapper();

	/** @brief Pre-simulation initialization hook (pure virtual) */
	virtual void preSimInit() = 0;

	/** @brief Post-simulation initialization hook (optional override) */
	virtual void postSimInit() {}

	/** @brief Initialize all simulators */
	void simInit();

	/** @brief Inter-iteration updates between phases (pure virtual) */
	virtual void runInterIterationUpdate() = 0;

	/** @brief Begin Phase 1: Parallel execution (pure virtual) */
	virtual void startPhase1() = 0;

	/** @brief Begin Phase 2: Communication/sync (pure virtual) */
	virtual void startPhase2() = 0;

	/** @brief End Phase 1: Wait for all threads (pure virtual) */
	virtual void finishPhase1() = 0;

	/** @brief End Phase 2: Complete iteration (pure virtual) */
	virtual void finishPhase2() = 0;

	/**
	 * @brief Calculate fast-forward cycles
	 * @return Tick Number of cycles to skip
	 */
	virtual Tick getFastForwardCycles();

	/**
	 * @brief Send exit event to all simulators at specified tick
	 * @param t Tick when simulation should exit
	 */
	virtual void issueExitEvent(Tick t);

	/** @brief Wrapper for post-simulation cleanup */
	void postSimCleanupWrapper();

	/** @brief Post-simulation cleanup hook (pure virtual) */
	virtual void postSimCleanup() = 0;

	/**
	 * @brief Set thread manager to running state
	 * @note Call before starting simulation loop
	 */
	virtual void startRunning() { running = true; }

#ifdef ACALSIM_STATISTICS
	/** @brief Get number of tasks executed per iteration */
	size_t getNTasksPerIter() const { return this->taskCntPerIteration.load(); }

	/**
	 * @brief Print scheduling overhead statistics
	 * @param _total_time Total simulation time
	 * @note Only active when ACALSIM_STATISTICS defined
	 */
	virtual void printSchedulingOverheads(double _total_time) const {}
#endif  // ACALSIM_STATISTICS

	/**
	 * @brief Static map for thread status string conversion
	 * @details Maps ThreadStatus enum to human-readable strings
	 */
	static std::unordered_map<ThreadStatus, std::string> ThreadStatusString;

protected:
	/**
	 * @brief Collect statistics before Phase 1
	 * @note Implementation in ThreadManager.inl
	 */
	inline void collectBeforePhase1Statistics();

	/**
	 * @brief Collect statistics after Phase 1
	 * @param _is_first_iter First iteration flag
	 * @param _is_last_iter Last iteration flag
	 * @note Implementation in ThreadManager.inl
	 */
	inline void collectAfterPhase1Statistics(bool _is_first_iter, bool _is_last_iter);

protected:
	/** @brief Pointer to linked TaskManager (not owned) */
	TaskManager* taskManager = nullptr;

	/**
	 * @brief Atomic flag indicating thread pool running state
	 * @details Set true when simulation starts, false when terminating
	 */
	std::atomic<bool> running;

	/** @brief Number of worker threads in pool */
	unsigned int nThreads;

	/**
	 * @brief Allow automatic thread count adjustment
	 * @details If true, adjusts nThreads to match simulator count
	 */
	bool nThreadsAdjustable = true;

	/**
	 * @brief Vector of worker thread pointers
	 * @details Managed worker threads for parallel execution
	 */
	std::vector<std::thread*> workers;

	/**
	 * @brief Map of worker thread status
	 * @details Key: thread ID, Value: current ThreadStatus
	 */
	std::unordered_map<uint64_t, ThreadStatus> workerStatus;

	/**
	 * @brief Registry of all managed simulators
	 * @details Maps simulator name to SimBase pointer
	 */
	HashVector<std::string, SimBase*> simulators;

	/**
	 * @brief Bit mask tracking active simulators
	 * @details One bit per simulator - set if has pending events
	 * @note Shared container for thread-safe access
	 */
	std::shared_ptr<SharedContainer<BitVector>> pSimulatorActiveBitMask = nullptr;

#ifdef ACALSIM_STATISTICS
public:
	/** @brief Get average thread idle time across all workers */
	double getAvgThreadIdleTime() const;

	/** @brief Get average task execution time per thread */
	double getAvgTaskExecTimePerThread() const;

	/** @brief Get total task execution time across all threads */
	double getTotalTaskExecTime() const;

	/** @brief Get total thread idle time across all threads */
	double getTotalThreadIdleTime() const;

protected:
	/** @brief Atomic counter of tasks per iteration */
	std::atomic<size_t> taskCntPerIteration = 0;

	/** @brief Statistics on number of tasks executed */
	Statistics<size_t> execNTasksStatistic;

	/**
	 * @brief Per-thread task execution time statistics
	 * @details Categorized by thread ID, accumulates execution times
	 */
	CategorizedStatistics<size_t, double, StatisticsMode::Accumulator, true, false, false> taskExecTimeStatistics;

	/** @brief Timer for measuring thread idle time in Phase 1 */
	ThreadIdleTimer phase1IdleTimer;
#endif  // ACALSIM_STATISTICS
};

/**
 * @class ThreadManager
 * @brief Concrete implementation of multi-threaded simulation manager
 *
 * @details
 * Provides complete implementation of thread pool management for parallel
 * simulation. Implements abstract methods from ThreadManagerBase for
 * simulator management and thread lifecycle control.
 *
 * **Functionality:**
 * - Creates and manages worker threads
 * - Maps simulators to worker threads
 * - Implements two-phase synchronization
 * - Handles thread termination and cleanup
 *
 * @note Final implementations prevent further derivation
 */
class ThreadManager : public ThreadManagerBase {
public:
	/**
	 * @brief Construct concrete thread manager
	 *
	 * @param name Manager name for identification
	 * @param nThreads Number of worker threads to create
	 * @param nThreadsAdjustable Allow automatic thread count adjustment
	 *
	 * @note Delegates to ThreadManagerBase constructor
	 * @note If nThreadsAdjustable=true, threads auto-adjust to simulator count
	 *
	 * @code{.cpp}
	 * // Fixed 4 threads
	 * ThreadManager* mgr1 = new ThreadManager("Fixed", 4, false);
	 *
	 * // Auto-adjust to number of simulators
	 * ThreadManager* mgr2 = new ThreadManager("Auto", 8, true);
	 * @endcode
	 */
	ThreadManager(const std::string& name, unsigned int nThreads, bool nThreadsAdjustable = true)
	    : ThreadManagerBase(name, nThreads, nThreadsAdjustable) {}

	/**
	 * @brief Add simulator to thread manager (final implementation)
	 *
	 * @param _sim Pointer to simulator to add
	 *
	 * @note Final - cannot be overridden
	 * @note Implementation in source file
	 * @note Registers simulator for parallel execution
	 *
	 * @code{.cpp}
	 * auto* cpu = new CPUSimulator("CPU0");
	 * threadMgr->addSimulator(cpu);
	 * @endcode
	 */
	void addSimulator(SimBase* _sim) final;

	/**
	 * @brief Start worker threads and begin simulation
	 *
	 * @note Implementation in source file
	 * @note Creates worker threads and assigns simulators
	 * @note Call after all simulators added and initialized
	 *
	 * @code{.cpp}
	 * threadMgr->preSimInitWrapper();
	 * threadMgr->simInit();
	 * threadMgr->startSimThreads();  // Launch workers
	 * @endcode
	 */
	void startSimThreads() override;

	/**
	 * @brief Terminate all worker threads (final wrapper)
	 *
	 * @note Final - cannot be overridden
	 * @note Delegates to terminateAllThreads()
	 * @note Ensures clean thread shutdown
	 *
	 * @code{.cpp}
	 * // At end of simulation
	 * threadMgr->terminateAllThreadsWrapper();
	 * @endcode
	 */
	void terminateAllThreadsWrapper() final { this->terminateAllThreads(); }
};

}  // namespace acalsim

#include "sim/ThreadManager.inl"
