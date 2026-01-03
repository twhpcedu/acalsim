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
 * @file ThreadManager.cc
 * @brief ThreadManager implementation - worker thread pool coordination and lifecycle
 *
 * This file implements ThreadManager, responsible for managing the thread pool that
 * executes SimBase instances in parallel during Phase 1 of ACALSim's two-phase
 * synchronization model.
 *
 * **Thread Pool Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                      ThreadManager (Control Thread)                    │
 * │                                                                        │
 * │  Thread Pool: 8 workers (0-7)    Activity Bitmask: 0b10110101        │
 * │  Hardware Threads: 16             CPU Affinity: Enabled               │
 * └────────────┬───────────────────────────────────────────────────────────┘
 *              │
 *              ├─ startPhase1() ────────────────────────────────────┐
 *              │                                                     │
 *              ▼                                                     ▼
 * ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
 * │  Worker Thread 0       │  │  Worker Thread 1       │  │  Worker Thread 7       │
 * │  CPU: Core 0           │  │  CPU: Core 1           │  │  CPU: Core 7           │
 * │  Status: Ready         │  │  Status: Ready         │  │  Status: Sleep         │
 * ├────────────────────────┤  ├────────────────────────┤  ├────────────────────────┤
 * │  scheduler(0):         │  │  scheduler(1):         │  │  scheduler(7):         │
 * │   while (running) {    │  │   while (running) {    │  │   while (running) {    │
 * │     task = getTask()   │  │     task = getTask()   │  │     wait_for_task()    │
 * │     task->execute()    │  │     task->execute()    │  │   }                    │
 * │   }                    │  │   }                    │  │                        │
 * └────────────────────────┘  └────────────────────────┘  └────────────────────────┘
 *              │                        │                          │
 *              └────────────────────────┴──────────────────────────┘
 *                                       │
 *                                       ▼
 *              finishPhase1() ──── All workers complete tasks ────
 * ```
 *
 * **Thread Pool Lifecycle:**
 * ```
 * SimTop Initialization:
 *   │
 *   ├─ 1. ThreadManager Constructor
 *   │     └─ Store nThreads (from hardware_concurrency or CLI)
 *   │
 *   ├─ 2. preSimInitWrapper()
 *   │     └─ Create simulatorActiveBitMask (track active SimBase instances)
 *   │
 *   ├─ 3. simInit()
 *   │     └─ Call initWrapper() on all registered SimBase instances
 *   │
 *   ├─ 4. startSimThreads()
 *   │     ├─ Adjust nThreads if needed (min(nThreads, nSimulators))
 *   │     ├─ For each thread:
 *   │     │   ├─ Create std::thread running TaskManager::scheduler(thread_idx)
 *   │     │   ├─ Set CPU affinity (pin to specific core)
 *   │     │   └─ Initialize workerStatus[tid] = InActive
 *   │     └─ Call taskManager->init()
 *   │
 *   └─ 5. SimTop::run() - Main simulation loop
 *         ├─ startPhase1() → workers execute tasks
 *         ├─ finishPhase1() → wait for completion
 *         ├─ startPhase2() → synchronization
 *         └─ finishPhase2() → advance clock
 *
 * Simulation Termination:
 *   │
 *   ├─ terminateAllThreadsWrapper()
 *   │     └─ Signal all workers to exit scheduler loop
 *   │
 *   └─ postSimCleanupWrapper()
 *         ├─ Call cleanup() on all SimBase instances
 *         └─ Join and delete worker threads
 * ```
 *
 * **CPU Affinity Pinning:**
 * ```cpp
 * // Why CPU affinity matters:
 * // 1. Cache locality - thread stays on same core, preserves L1/L2 cache
 * // 2. NUMA awareness - memory access patterns optimized
 * // 3. Reduced context switching - OS scheduler respects affinity
 *
 * Example with 8 workers on 16 hardware threads:
 *   Worker 0 → Core 0   (cpuset: 0b0000000000000001)
 *   Worker 1 → Core 1   (cpuset: 0b0000000000000010)
 *   Worker 2 → Core 2   (cpuset: 0b0000000000000100)
 *   Worker 3 → Core 3   (cpuset: 0b0000000000001000)
 *   ...
 *   Worker 7 → Core 7   (cpuset: 0b0000000010000000)
 *
 * // pthread_setaffinity_np ensures worker i stays on core (i % n_hw_threads)
 * ```
 *
 * **Thread Status State Machine:**
 * ```
 *                    startSimThreads()
 *                           │
 *                           ▼
 *                     ┌──────────┐
 *                     │ InActive │ ◄─── Initial state
 *                     └──────────┘
 *                           │
 *                startPhase1()
 *                           │
 *                           ▼
 *                     ┌──────────┐
 *        ┌────────────┤  Ready   │◄──────────┐
 *        │            └──────────┘           │
 *        │                  │                │
 *        │         No tasks available        │
 *        │                  │          Task completed
 *        │                  ▼                │
 *        │            ┌──────────┐           │
 *        │            │  Sleep   │           │
 *        │            └──────────┘           │
 *        │                  │                │
 *        │         New task arrives          │
 *        │                  │                │
 *        └──────────────────┴────────────────┘
 *                           │
 *              terminateAllThreadsWrapper()
 *                           │
 *                           ▼
 *                     ┌──────────┐
 *                     │Terminated│ ◄─── Final state
 *                     └──────────┘
 * ```
 *
 * **Worker Thread Adaptive Scheduling:**
 * - If nThreads > nSimulators: Reduce thread count to nSimulators (avoid idle threads)
 * - If nThreadsAdjustable == false: User override, keep configured count
 * - Trade-off: More threads → better load balance, but higher context switch overhead
 *
 * **Performance Statistics (ACALSIM_STATISTICS):**
 * - **Per-thread metrics**: Task execution time, idle time (phase1IdleTimer)
 * - **Global metrics**: Total executed tasks, tasks per iteration (avg/max)
 * - **Overhead breakdown**: Scheduling overhead = phase1_time - exec_time - idle_time
 * - **Parallelism tracking**: Average activation ratio, parallel degree distribution
 *
 * **Integration with TaskManager:**
 * ```
 * ThreadManager (Thread Pool)     TaskManager (Task Queue)
 *        │                                 │
 *        │  1. startPhase1()               │
 *        ├─────────────────────────────────►
 *        │                                 │ createTasks()
 *        │                                 │ distributeTasks()
 *        │                                 │
 *        │  2. Worker threads:             │
 *        │     scheduler(thread_idx)       │
 *        ├─────────────────────────────────►
 *        │                                 │ getNextTask(thread_idx)
 *        │◄─────────────────────────────────
 *        │     TaskFunctor*                │
 *        │                                 │
 *        │  3. Execute task                │
 *        │     task->execute()             │
 *        │     (calls SimBase::stepWrapper)│
 *        │                                 │
 *        │  4. finishPhase1()              │
 *        │     Wait for all tasks done     │
 *        └─────────────────────────────────┘
 * ```
 *
 * **Fast-Forward Clock Delegation:**
 * - getFastForwardCycles() delegates to taskManager->getNextSimTick()
 * - TaskManager queries all active SimBase instances for next event
 * - Returns minimum tick across all simulators
 * - SimTop uses this to skip idle cycles (no activity between events)
 *
 * **Simulator Registration Flow:**
 * ```cpp
 * // In SimTop initialization:
 * auto sim1 = new MySimulator("cpu");
 * auto sim2 = new MemoryController("mem");
 * addSimulator(sim1);  // ThreadManager::addSimulator()
 * addSimulator(sim2);  // Assigns unique ID, stores in HashVector
 *
 * // Each SimBase gets:
 * // - Unique ID: 0, 1, 2, ... (sequential)
 * // - Entry in simulatorActiveBitMask
 * // - Name-based lookup in simulators HashVector
 * ```
 *
 * @see ThreadManager.hh For complete interface documentation
 * @see TaskManager.cc For task scheduling implementation
 * @see SimTop.cc For main simulation loop coordination
 * @see ThreadManagerV3.cc For recommended production implementation
 */

#include "sim/ThreadManager.hh"

#include <pthread.h>
#include <sched.h>

#include <thread>

#include "external/gem5/Event.hh"
#include "sim/SimBase.hh"
#include "sim/SimTop.hh"
#include "sim/TaskManager.hh"

namespace acalsim {

ThreadManagerBase::ThreadManagerBase(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable)
    : nThreads(_nThreads), running(false), nThreadsAdjustable(_nThreadsAdjustable) {}

ThreadManagerBase::~ThreadManagerBase() {
#ifdef ACALSIM_STATISTICS
	LABELED_STATISTICS("ThreadManager") << "Active Iterations = " << this->execNTasksStatistic.size()
	                                    << " (except for the first and the last iterations)";
	LABELED_STATISTICS("ThreadManager") << "Executed Tasks = " << this->execNTasksStatistic.sum()
	                                    << " (except for the first and the last iterations)";
	LABELED_STATISTICS("ThreadManager") << "Tasks Per Iteration (avg) = " << this->execNTasksStatistic.avg()
	                                    << " (except for the first and the last iterations)";
	LABELED_STATISTICS("ThreadManager") << "Tasks Per Iteration (max) = " << this->execNTasksStatistic.max()
	                                    << " (except for the first and the last iterations)";
#endif  // ACALSIM_STATISTICS

	for (auto& sim : this->simulators) { delete sim; }
}

void ThreadManager::addSimulator(SimBase* _sim) {
	auto name = _sim->getName();
	_sim->setID(this->simulators.size());

	auto existing = this->simulators.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!existing, "ThreadManagerBase `" + name + "` already exists!");
	this->simulators.insert(std::make_pair(name, _sim));
}

SimBase* ThreadManagerBase::getSimulator(const std::string& _name) const {
	auto iter = this->simulators.getUMapRef().find(_name);
	CLASS_ASSERT_MSG(iter != this->simulators.getUMapRef().end(), "The simulator \'" + _name + "\' does not exist.");
	return iter->second.get();
}

void ThreadManagerBase::preSimInitWrapper() {
	this->preSimInit();

	this->pSimulatorActiveBitMask = std::make_shared<SharedContainer<BitVector>>();
	this->pSimulatorActiveBitMask->add(simulators.size(), false);
}

void ThreadManager::startSimThreads() {
	MT_DEBUG_CLASS_INFO << "number of simulator: " + std::to_string(simulators.size());

	this->nThreads = (this->nThreadsAdjustable && this->nThreads > this->simulators.size()) ? this->simulators.size()
	                                                                                        : this->nThreads;

	size_t n_hw_threads = std::thread::hardware_concurrency();

#ifdef ACALSIM_STATISTICS
	LABELED_STATISTICS("ThreadManager") << "Launches " << this->nThreads << " software threads for "
	                                    << this->simulators.size() << " simulators.";
	this->phase1IdleTimer.resize(this->nThreads);
#endif

	// Initialize worker threads
	this->workers.reserve(this->nThreads);
	for (size_t thread_idx = 0; thread_idx < this->nThreads; ++thread_idx) {
		auto thread = new std::thread(&TaskManager::scheduler, this->taskManager, thread_idx);

		// Set the CPU affinity of the spawned worker thread
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(thread_idx % n_hw_threads, &cpuset);
		pthread_setaffinity_np(thread->native_handle(), sizeof(cpuset), &cpuset);

		this->workers.emplace_back(thread);
		auto tid                = static_cast<uint64_t>(std::hash<std::thread::id>()(thread->get_id()));
		this->workerStatus[tid] = ThreadStatus::InActive;

#ifdef ACALSIM_STATISTICS
		this->taskExecTimeStatistics.addEntry(thread_idx);
#endif  // ACALSIM_STATISTICS
	}

	this->taskManager->init();
}

Tick ThreadManagerBase::getFastForwardCycles() { return this->taskManager->getNextSimTick(); }

void ThreadManagerBase::issueExitEvent(Tick t) {
	//  Send the terminate signal to each simulator
	for (auto& sim : this->simulators) { sim->issueExitEvent(t); }
}

void ThreadManagerBase::simInit() {
	for (auto& sim : this->simulators) { sim->initWrapper(); }
}

void ThreadManagerBase::postSimCleanupWrapper() {
	this->postSimCleanup();

	for (auto& sim : this->simulators) { sim->cleanup(); }
	for (auto& worker : this->workers) { delete worker; }
}

#ifdef ACALSIM_STATISTICS
double ThreadManagerBase::getAvgThreadIdleTime() const { return this->phase1IdleTimer.getSum() / this->nThreads; }

double ThreadManagerBase::getAvgTaskExecTimePerThread() const {
	return this->taskExecTimeStatistics.sum() / this->nThreads;
}

double ThreadManagerBase::getTotalTaskExecTime() const { return this->taskExecTimeStatistics.sum(); }

double ThreadManagerBase::getTotalThreadIdleTime() const { return this->phase1IdleTimer.getSum(); }
#endif  // ACALSIM_STATISTICS

std::unordered_map<ThreadStatus, std::string> ThreadManagerBase::ThreadStatusString = {
    {ThreadStatus::InActive, "InActive"},
    {ThreadStatus::Ready, "Ready"},
    {ThreadStatus::Sleep, "Sleep"},
    {ThreadStatus::Terminated, "Terminated"}};

}  // namespace acalsim
