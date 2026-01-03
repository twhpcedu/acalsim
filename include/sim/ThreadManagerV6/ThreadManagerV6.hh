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

#include <string>

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

// class definition forwarding
template <typename T>
class TaskManagerV6;

template <typename TClass>
class ThreadManagerV6 : public TClass {
	friend class TaskManagerV6<TClass>;

public:
	ThreadManagerV6(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable = true)
	    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

	virtual ~ThreadManagerV6() {}

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
	TaskManagerV6<TClass>* getTaskManager() const { return static_cast<TaskManagerV6<TClass>*>(this->taskManager); }

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent waiting for partitioning TaskManagerV6::taskQueue to TaskManagerV6::ThreadLocalQueue
	Statistics<double, StatisticsMode::Accumulator, true> globalTqPartitionTimeStatistics;

	// The time spent waiting for the TaskManagerV6::ThreadLocalQueue to be ready
	Statistics<double, StatisticsMode::Accumulator, true> localTqPrepareTimeStatistics;

	// The time spent operating the local task queue
	Statistics<double, StatisticsMode::Accumulator, true> localTqManipTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/ThreadManagerV6/ThreadManagerV6.inl"
