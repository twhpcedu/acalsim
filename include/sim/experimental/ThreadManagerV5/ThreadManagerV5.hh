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
 * @file ThreadManagerV5.hh
 * @warning EXPERIMENTAL: This version was not validated in published research.
 *          Not recommended for production use.
 *
 * @note For production use, see:
 *       - ThreadManagerV1 (PriorityQueue): Default, sparse activation patterns
 *       - ThreadManagerV3 (PrebuiltTaskList): Memory-intensive workloads
 *       - ThreadManagerV6 (LocalTaskQueue): Lock-optimized V1
 *
 * @details ThreadManagerV5 is a simplified version of V1 with active bit mask removed.
 *          This design was explored but not included in published research.
 */

#pragma once

#include <string>

namespace acalsim {

// class definition forwarding
template <typename T>
class TaskManagerV5;

template <typename TClass>
class ThreadManagerV5 : public TClass {
	friend class TaskManagerV5<TClass>;

public:
	ThreadManagerV5(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable = true)
	    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

	virtual ~ThreadManagerV5() {}

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
	TaskManagerV5<TClass>* getTaskManager() const { return static_cast<TaskManagerV5<TClass>*>(this->taskManager); }

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV5/ThreadManagerV5.inl"
