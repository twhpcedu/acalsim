
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

#include "sim/SimBase.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManager.hh"
#include "sim/experimental/ThreadManagerV4/TaskManagerV4.hh"
#include "sim/experimental/ThreadManagerV4/ThreadManagerV4.hh"

namespace acalsim {

template <typename T>
void ThreadManagerV4<T>::preSimInit() {
	// Each simulator has one thread
	this->nThreads = this->simulators.size();

	nRemainingWorkers = 0;
}

template <typename T>
void ThreadManagerV4<T>::postSimInit() {
	// Update Pending Event Bit Mask before each simulator thread is launched.
	runInterIterationUpdate();
}

template <typename T>
void ThreadManagerV4<T>::startSimThreads() {
	// Initialize worker threads
	this->workers.reserve(this->nThreads);

	for (auto& sim : this->simulators) {
		auto thread = new std::thread(&ThreadManagerV4::runSimOnThread, this, sim);
		this->workers.emplace_back(thread);

		auto tid                = static_cast<uint64_t>(std::hash<std::thread::id>()(thread->get_id()));
		this->workerStatus[tid] = ThreadStatus::InActive;

		nextIterationReady.insert({sim, false});
	}

	this->taskManager->init();
}

template <typename T>
Tick ThreadManagerV4<T>::getFastForwardCycles() {
	// overwrite this function because we don't want to call into TaskManager

	Tick nextTick = -1;

	for (auto& sim : this->simulators) {
		if (!top->isPendingEventBitMaskSet(sim->getID())) { continue; }
		Tick nextSimTick = sim->getSimNextTick();
		nextTick         = (nextSimTick < nextTick) ? nextSimTick : nextTick;
	}
	VERBOSE_CLASS_INFO << "Next Tick is: " << nextTick;
	return nextTick;
}

template <typename T>
void ThreadManagerV4<T>::runInterIterationUpdate() {
	// Set the pendingEventBitMask for each simulator
	for (auto& sim : this->simulators) { sim->interIterationUpdate(); }
}

template <typename T>
void ThreadManagerV4<T>::terminateAllThreads() {
	// Set this to false so we can schedule events before current tick
	this->running = false;

	MT_DEBUG_CLASS_INFO << "Waiting for all worker threads to terminate themselves...";

	for (auto& worker : this->workers) {
		ASSERT(worker->joinable());
		if (worker->joinable()) worker->join();
	}
}

template <typename T>
void ThreadManagerV4<T>::startPhase1() {
	MT_DEBUG_CLASS_INFO << "[v4 sync] ThreadManagerV4::startPhase1() begin";

	std::unique_lock<std::mutex> lock(threadManagerLock);

	ASSERT(nRemainingWorkers == 0);

	for (auto& sim : this->simulators) {
		ASSERT(nextIterationReady[sim] == false);

		// Only set the simulators with pending events to run
		// ExitEvents are scheduled after interIterationUpdate() in SimTop, so the bit mask would not be set.
		if (top->isPendingEventBitMaskSet(sim->getID()) || top->getGlobalTick() == 0 || top->isReadyToTerminate()) {
			nRemainingWorkers++;
			nextIterationReady[sim] = true;
		}
	}

	lock.unlock();

	this->collectBeforePhase1Statistics();

	// Wake up all worker threads
	nextIterationReadyCondVar.notify_all();

	MT_DEBUG_CLASS_INFO << "[v4 sync] ThreadManagerV4::startPhase1() end";
}

template <typename T>
void ThreadManagerV4<T>::finishPhase1() {
	MT_DEBUG_CLASS_INFO << "[v4 sync] ThreadManagerV4::finishPhase1() begin";

	std::unique_lock<std::mutex> lock(threadManagerLock);
	iterationDoneCondVar.wait(lock, [this] {
		// The second condition is to make sure that if
		// control thread goes too quickly before the worker threads get to restart the iteration
		// we wouldn't pass this sync point again

		return (nRemainingWorkers == 0);
	});

	lock.unlock();

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());

	MT_DEBUG_CLASS_INFO << "[v4 sync] ThreadManagerV4::finishPhase1() end.";
}

template <typename T>
void ThreadManagerV4<T>::startPhase2() {}

template <typename T>
void ThreadManagerV4<T>::finishPhase2() {}

template <typename T>
void ThreadManagerV4<T>::runSimOnThread(SimBase* sim) {
	while (!this->isRunning()) { ; }

	bool readyToTerminate = false;

	while (!readyToTerminate) {
		std::unique_lock<std::mutex> lock(threadManagerLock);

		// Wait for all the other workers to be done and then the next iteration to start
		nextIterationReadyCondVar.wait(lock, [this, sim] { return nextIterationReady[sim]; });

		nextIterationReady[sim] = false;

		lock.unlock();

		MT_DEBUG_CLASS_INFO << "[v4 sync] Simulator " << sim->getName() << " enters phase 1.";

		// Phase 1 starts

		readyToTerminate = top->isReadyToTerminate();  // This var would only be written in phase 2. No lock needed.

		((TaskManagerV4<T>*)(this->taskManager))
		    ->collectTaskExecStatistics(&SimBase::stepWrapperBase, sim, sim->getName());

		// Phase 1 ends

		// Last worker finishes Phase 1
		if (--nRemainingWorkers == 0) {
			MT_DEBUG_CLASS_INFO << "[v4 sync] Simulator (" << sim->getName()
			                    << ") is the last simulator to finish phase 1";

			// notify control thread
			iterationDoneCondVar.notify_one();

		} else {
			MT_DEBUG_CLASS_INFO << "[v4 sync] Simulator (" << sim->getName() << ") finishes phase 1";
		}
	}

	MT_DEBUG_CLASS_INFO << "[v4 sync] Simulator " << sim->getName() << " finishes simulation";
}

}  // namespace acalsim
