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

#include <cmath>

#include "TaskManager.hh"

namespace acalsim {

TaskManager::~TaskManager() {
#ifdef ACALSIM_STATISTICS
	for (auto& [name, statistic] : this->execPeriodStatistic) {
		LABELED_STATISTICS("TaskManager")
		    << "\"" << name << "\" uses " << statistic->avg() << " us on average per iteration. It was active in "
		    << statistic->size() << " iterations.";
	}
#endif  // ACALSIM_STATISTICS
}

void TaskManager::init() {
#ifdef ACALSIM_STATISTICS
	for (auto& [name, _] : this->getSimulators().getUMapRef()) { this->execPeriodStatistic.addEntry(name); }
#endif  // ACALSIM_STATISTICS
}

template <typename TaskFunctor>
double TaskManager::collectTaskExecStatistics(TaskFunctor& _task, const std::string& _sim_name) {
#ifdef ACALSIM_STATISTICS
	auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

	_task();  // execute task

#ifdef ACALSIM_STATISTICS
	auto   stop = std::chrono::high_resolution_clock::now();
	double time = (double)(stop - start).count() / pow(10, 3);

	this->getThreadManager()->taskCntPerIteration += 1;

	this->execPeriodStatistic.getEntry(_sim_name)->push(time);

	return time;
#endif  // ACALSIM_STATISTICS

	return 0;
}

template <typename T, typename Func>
double TaskManager::collectTaskExecStatistics(const Func& _func, T* const& _obj, const std::string& _sim_name) {
#ifdef ACALSIM_STATISTICS
	auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

	((_obj)->*_func)();  // execute task

#ifdef ACALSIM_STATISTICS
	auto   stop = std::chrono::high_resolution_clock::now();
	double time = (double)(stop - start).count() / pow(10, 3);

	this->getThreadManager()->taskCntPerIteration += 1;

	this->execPeriodStatistic.getEntry(_sim_name)->push(time);

	return time;
#endif  // ACALSIM_STATISTICS

	return 0;
}

}  // namespace acalsim
