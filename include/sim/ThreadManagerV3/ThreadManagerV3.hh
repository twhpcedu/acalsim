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

// ACALSim
#include "sim/ThreadManagerV3/TaskManagerV3.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename TClass>
class ThreadManagerV3 : public TClass {
	friend class TaskManagerV3<TClass>;

public:
	ThreadManagerV3(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable);

	~ThreadManagerV3() { ; }

	// Pre-Simulation Initialization
	void preSimInit() override;

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
	void postSimCleanup() override { ; }

	TaskManagerV3<TClass>* getTaskManager() const { return static_cast<TaskManagerV3<TClass>*>(this->taskManager); }

private:
	std::shared_ptr<typename TaskManagerV3<TClass>::TaskList> stepTaskList                 = nullptr;
	std::shared_ptr<typename TaskManagerV3<TClass>::TaskList> interIterationUpdateTaskList = nullptr;

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent preparing the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqPreparationTimeStatistics;

	// The time spent retrieving the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqRetrievalTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/ThreadManagerV3/ThreadManagerV3.inl"
