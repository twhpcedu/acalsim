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

#include <string>

// ACALSim
#include "profiling/Statistics.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class TaskManagerV1;

template <typename TClass>
class ThreadManagerV1 : public TClass {
	friend class TaskManagerV1<TClass>;

public:
	ThreadManagerV1(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable = true)
	    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

	virtual ~ThreadManagerV1() override = default;

	void startSimThreads() override;

	/**
	 * @brief Override to notify workers waiting on the running condition variable
	 */
	void startRunning() override;

	// Pre-Simulation Initialization
	void preSimInit() override {}

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

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent operating the task queue
	CategorizedStatistics<size_t, double, StatisticsMode::Accumulator, true, false, false> tqOperationTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/ThreadManagerV1/ThreadManagerV1.inl"
