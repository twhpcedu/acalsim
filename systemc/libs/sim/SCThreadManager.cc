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

#include "sim/SCThreadManager.hh"

#include <systemc>
#include <thread>

#include "sim/SCSimBase.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

SCThreadManager::SCThreadManager(const std::string& name, unsigned int _nThreads, bool _nThreadsAdjustable)
    : ThreadManagerBase(name, _nThreads, _nThreadsAdjustable), scSim(nullptr), scThread(nullptr), systemcExist(false) {}

void SCThreadManager::addSimulator(SimBase* _sim) {
	if (_sim->isSystemC()) this->registerSCSimulator(dynamic_cast<SCSimBase*>(_sim));

	auto name = _sim->getName();
	_sim->setID(this->simulators.size());

	auto existing = this->simulators.getUMapRef().contains(name);
	CLASS_ASSERT_MSG(!existing, "ThreadManagerBase `" + name + "` already exists!");
	this->simulators.insert(std::make_pair(name, _sim));
}

void SCThreadManager::startSimThreads() {
	MT_DEBUG_CLASS_INFO << "number of simulator: " << simulators.size();

	this->nThreads = (this->nThreadsAdjustable && this->nThreads > this->simulators.size()) ? this->simulators.size()
	                                                                                        : this->nThreads;

#ifdef ACALSIM_STATISTICS
	LABELED_STATISTICS("ThreadManager") << "Launches " << this->nThreads << " software threads for "
	                                    << this->simulators.size() << " simulators.";
	this->phase1IdleTimer.resize(this->nThreads);
#endif

	// Initialize worker threads
	this->workers.reserve(this->nThreads);
	for (size_t i = 0; i < this->workers.capacity(); ++i) {
		auto thread = new std::thread(&TaskManager::scheduler, this->taskManager, i);
		this->workers.emplace_back(thread);
		auto tid                = static_cast<uint64_t>(std::hash<std::thread::id>()(thread->get_id()));
		this->workerStatus[tid] = ThreadStatus::InActive;

#ifdef ACALSIM_STATISTICS
		this->taskExecTimeStatistics.addEntry(i);
#endif  // ACALSIM_STATISTICS
	}

	// Create SystemC thread.
	if (this->isSystemcExist()) { this->startSCSimThread(); }

	taskManager->init();
}

int SCThreadManager::getSCSimId() const { return this->scSim->getID(); }

const std::string SCThreadManager::getSCSimName() const { return this->scSim->getName(); }

void SCThreadManager::registerSCSimulator(SCSimBase* sim) {
	this->systemcExist = true;
	this->scSim        = sim;
}

void SCThreadManager::startSCSimThread() { this->scThread = new std::thread(&SCSimBase::scMainRunner, this->scSim); }

}  // end of namespace acalsim
