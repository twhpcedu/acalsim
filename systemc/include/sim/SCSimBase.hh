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

#include <systemc>

#include "sim/SCInterface.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

#ifndef NO_LOGS
#define SC_MAIN_INFO    LABELED_INFO("SystemC")
#define SC_MAIN_WARNING LABELED_WARNING("SystemC")
#define SC_MAIN_ERROR   LABELED_ERROR("SystemC")
#else
#define SC_MAIN_INFO    acalsim::FakeLogOStream()
#define SC_MAIN_WARNING acalsim::FakeLogOStream()
#define SC_MAIN_ERROR   acalsim::FakeLogOStream()
#endif  // #ifndef NO_LOGS

class SCSimBase : public SimBase, public sc_core::sc_module {
public:
	sc_core::sc_clock*        clock = nullptr;
	sc_core::sc_signal<bool>* reset = nullptr;
	sc_core::sc_trace_file*   file  = nullptr;

	SCSimBase(std::string _name, int _cycleDuration = 10);
	~SCSimBase() {
		for (auto [_, sc_if] : this->interfaces) delete sc_if;
		delete this->systemcThreadBarrier;
	}

	bool isSystemC() const final { return true; }

	Tick getSimNextTick() final;

	void stepWrapperBase() final {
		this->stepWrapper();
		this->stepWrapperSC();
	}

	void stepWrapperSC();

	/**
	 * @brief This function runs as a SystemC thread dedicated to executing sc_main.
	 */
	void scMainRunner();

	// Set the cycle duration for a SystemC-type simulator.
	void setCycleDuration(int _cycleDuration) { this->cycleDuration = _cycleDuration; }

	// Get the cycle duration for a SystemC-type simulator
	int getCycleDuration() const { return this->cycleDuration; }

	// Add the interface for other SimBase to set/get signals.
	void addSCInterface(SCInterface* _interface, std::string _name);

	/**
	 * @brief Wait the scheduler to notify systemc to start current operations.
	 */
	void waitStepWrapperSC() { this->systemcThreadBarrier->arrive_and_wait(); }

	/**
	 * @brief Notify the scheduler that SCSimBase::scMainRunner() finish 1 iteration.
	 */
	void notifyStepWrapperSC() { this->systemcThreadBarrier->arrive_and_wait(); }

	/**
	 * @brief Synchronizes and updates simulation states between iterations.
	 *
	 * This function performs several methods:
	 * - Synchronizes the entries in the SlavePort with the entries in the MasterPort.
	 * - Moves entries from the inbound container to the outbound container.
	 * - Checks for pending requests in the inbound channels and SlavePort.
	 * - Checks if there are any pending events in the event queue.
	 * - Determines if there are any pending simulation activities.
	 * - Captures positive and negative edge transitions of pending activities.
	 *
	 * @return true if there is a pending request in the inbound channels or SlavePort, false otherwise.
	 */
	bool interIterationUpdate() final;

	// For a SystemC-type simulator, users can create their SystemC Module (SC_MODULE)
	// before starting the SystemC simulation in this virtual function.
	virtual void registerSystemCSim() = 0;

	// For a SystemC-type simulator, users can initialize/reset their SystemC Module (SC_MODULE)
	// before starting the SystemC simulation in this virtual function.
	virtual void initSystemCSim() = 0;

	// For a SystemC-type simulator, users can define actions to be taken
	// before the SystemC simulation starts at the current Tick in this virtual function.
	virtual void preSystemCSim() = 0;

	// For a SystemC-type simulator, users can define actions to be taken
	// after the SystemC simulation ends at the current Tick in this virtual function.
	virtual void postSystemCSim() = 0;

	void setTrace(std::string name);

	inline void step() final {}

	inline void intraIterationUpdateSC() {
		for (auto& [_, sc_if] : this->interfaces) sc_if->intraIterationUpdateSC();
	}

	virtual bool SimulationDone() = 0;

	// callback on arbitration win, allow users to override
	virtual void masterPortRetry(MasterPort* _port) {}

protected:
	/// @brief To specify a cycle duration for a SystemC-type simulator
	int cycleDuration = 1;

	std::unordered_map<std::string, SCInterface*> interfaces;

	/// @brief Barrier to synchronize SCSimBase::scMainRunner() and SCSimBase::stepWrapperSC()
	std::barrier<void (*)(void)>* systemcThreadBarrier;

	/**
	 * @brief Notify the SCSimBase::scMainRunner() to start the current iteration of systemc
	 */
	void notifySCMainRunner() { this->systemcThreadBarrier->arrive_and_wait(); }

	/**
	 * @brief Wait the SCSimBase::scMainRunner() to finish the current iteration of systemc
	 */
	void waitSCMainRunner() { this->systemcThreadBarrier->arrive_and_wait(); }
};

}  // end of namespace acalsim
