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
 * @file ThreadManagerV2.hh
 * @brief Defines the ThreadManagerV2 class for managing threads and synchronization.
 *
 * This file contains the definition of the ThreadManagerV2 class, which is responsible
 * for managing threads, synchronization, and control during simulation. It inherits
 * from ThreadManager and includes methods for initialization, iteration updates,
 * thread termination, phase synchronization, and cleanup.
 */

#pragma once

#include <barrier>
#include <string>

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

// class definition forwarding
template <typename T>
class TaskManagerV2;

template <typename TClass>
class ThreadManagerV2 : public TClass {
	friend class TaskManagerV2<TClass>;

public:
	/**
	 * @brief Constructor for ThreadManagerV2.
	 * @param _name The name of the ThreadManagerV2 instance.
	 * @param _nThreads The number of worker threads to manage.
	 */
	ThreadManagerV2(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable)
	    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

	virtual ~ThreadManagerV2() { delete this->workerThreadsDoneBarrier; }

	/// @brief Pre-Simulation Initialization
	void preSimInit() override;

	/// @brief Performance inter-iteration updates (similar to the register update in hardware modeling)
	void runInterIterationUpdate() override;

	/// @brief Terminate all threads
	void terminateAllThreads() override;

	/// @brief Start of Phase 1 synchronization
	void startPhase1() override;

	/// @brief Start of Phase 2 synchronization
	void startPhase2() override;

	/// @brief End of Phase 1 synchronization
	void finishPhase1() override;

	/// @brief End of Phase 2 synchronization
	void finishPhase2() override;

	/// @brief Post-Simulation Cleanup (placeholder implementation)
	void postSimCleanup() override { ; }

	/**
	 * @brief Get the barrier object used to synchronize worker threads at specific points.
	 * @return Pointer to the barrier object.
	 */
	std::barrier<void (*)(void)>* getWorkerThreadsDoneBarrier() { return this->workerThreadsDoneBarrier; }

private:
	std::barrier<void (*)(void)>* workerThreadsDoneBarrier;  // A global barrier for all simulation threads.

	TaskManagerV2<TClass>* getTaskManager() { return static_cast<TaskManagerV2<TClass>*>(this->taskManager); }

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent retrieving the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqManipTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/ThreadManagerV2/ThreadManagerV2.inl"
