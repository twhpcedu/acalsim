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

/**
 * @file ThreadManagerV4.hh
 * @warning EXPERIMENTAL: This version was not validated in published research.
 *          Not recommended for production use.
 *
 * @note For production use, see:
 *       - ThreadManagerV1 (PriorityQueue): Default, sparse activation patterns
 *       - ThreadManagerV3 (PrebuiltTaskList): Memory-intensive workloads
 *       - ThreadManagerV6 (LocalTaskQueue): Lock-optimized V1
 *
 * @details ThreadManagerV4 implements a dedicated thread per simulator approach.
 *          This design was explored but not included in published research.
 *          It may have performance limitations compared to thread pooling approaches.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <string>

// ACALSim
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV1/TaskManagerV1.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class TaskManagerV4;

template <typename TClass>
class ThreadManagerV4 : public TClass {
	friend class TaskManagerV4<TClass>;

public:
	ThreadManagerV4(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable)
	    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

	virtual ~ThreadManagerV4() {}

	// Pre-Simulation Initialization
	void preSimInit() override;

	// Post simulator intiailization
	void postSimInit() override;

	// Configure the mapping between simulations and threads
	void startSimThreads() override;

	Tick getFastForwardCycles() override;

	// Performance inter-iteration updates (similar to the register update in hardware modeling)
	void runInterIterationUpdate() override;

	// Terminate all threads
	void terminateAllThreads() override;

	// end of Phase 1 synchronization
	void startPhase1() override;

	// end of Phase 2 synchronization
	void startPhase2() override;

	// end of Phase 1 synchronization
	void finishPhase1() override;

	// end of Phase 2 synchronization
	void finishPhase2() override;

	// Post-Simulation Cleanup
	void postSimCleanup() override {}

private:
	TaskManagerV1<TClass>* getTaskManager() { return static_cast<TaskManagerV1<TClass>*>(this->taskManager); }

	// Each simulator would be allocated a thread
	void runSimOnThread(SimBase* sim);

	// Conditional variable for the control thread to notify each simulator thread
	std::condition_variable nextIterationReadyCondVar;

	// Conditional variable for the simulator threads to notify the control thread
	std::condition_variable iterationDoneCondVar;

	// used with the conditional variable.
	std::mutex threadManagerLock;

	// Number of workers that have finished phase 1
	std::atomic<int> nRemainingWorkers;

	std::unordered_map<SimBase*, bool> nextIterationReady;
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV4/ThreadManagerV4.inl"
