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

#include "sim/SCSimBase.hh"

#include <systemc>

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

// A global variable for SCSimBase::stepWrapperSystemC to send its pointer to sc_main. Note that the scope of this
// variable is limited in this file instead of the whole program since this file will not be included by any other
// files.
SCSimBase* scsim = nullptr;

SCSimBase::SCSimBase(std::string _name, int _cycleDuration)
    : SimBase(_name),
      sc_core::sc_module(sc_core::sc_module_name(_name.c_str())),
      cycleDuration(_cycleDuration),
      systemcThreadBarrier(new std::barrier<void (*)(void)>(2, []() {})) {}

void SCSimBase::stepWrapperSC() {
	// users can do something before the systemC simulation starts at the current Tick
	this->preSystemCSim();

	// notify SCSimBase::scMainRunner to go through current iteration of systemc
	this->notifySCMainRunner();

	// wait SCSimBase::scMainRunner to finish the current iteration of systemc
	this->waitSCMainRunner();

	// users can catch the response from SystemC to SCSimBase users can do something after the systemC simulation
	this->postSystemCSim();
}

void SCSimBase::scMainRunner() {
	// Push Simulator pointer to channel.
	scsim = this;
	// sc_elab_and_sim will call sc_main() function.
	sc_core::sc_elab_and_sim(0, nullptr);
}

Tick SCSimBase::getSimNextTick() { return Tick(top->getGlobalTick() + 1); }

void SCSimBase::addSCInterface(SCInterface* _interface, std::string _name) {
	bool is_present = this->interfaces.contains(_name);
	CLASS_ASSERT_MSG(!is_present, "SCSimBase : `" + _name + "` is present in SCSimBase::interfaces!");
	_interface->file = this->file;
	_interface->clock(*this->clock);
	_interface->reset(*this->reset);
	_interface->setSCSimulator(this);
	this->interfaces.insert(std::make_pair(_name, _interface));
}

bool SCSimBase::interIterationUpdate() {
	// inbound request in Inbound SimChannel and SlavePort.
	// hasInboundChannelReq() shows whether there is a pending request in the inbound channels
	// hasPendingActivityInSimPort() shows whether there is a pending request in the slavePorts
	bool inboundRequestNotDone = this->hasInboundChannelReq() || this->hasPendingActivityInSimPort(true);

	// eventQueueEmpty() shows whether there is a pending event in the event queue
	// SimulationDone() function is a user-defined function that determines when the SystemC simulation is complete
	bool notDone = (!this->eventQueueEmpty() || inboundRequestNotDone || !this->SimulationDone());

	if (notDone && !this->hasPendingActivityLastIteration()) {  // capture positive edge transition
		// Only set PendingEventBitMask when it changes on the positive edge transition
		top->setPendingEventBitMask(this->getID());
		VERBOSE_CLASS_INFO << "SetPendingEventBitMask";
	}

	if (this->hasPendingActivityLastIteration() && !notDone) {  // capture negative edge transition
		// Only clear PendingEventBitMask when it changes on the negative edge transition
		top->clearPendingEventBitMask(this->getID());
		VERBOSE_CLASS_INFO << "ClearPendingEventBitMask";
	}
	this->setInboundChannelReqLastIteration(this->hasInboundChannelReq());
	this->setPendingActivityLastIteration(notDone);

	this->clearHasPendingActivityInSimPortFlag();

	return inboundRequestNotDone;
}

void SCSimBase::setTrace(std::string name) {
	sc_core::sc_trace(this->file, *this->clock, "clock");
	sc_core::sc_trace(this->file, this->reset, "reset");
	for (auto& [if_name, sc_if] : this->interfaces) { sc_if->setTraceWrapper(this->getName() + "." + if_name + "."); }
}

}  // end of namespace acalsim

int sc_main(int argc, char** argv) {
	// Get SystemC Simulator (SCSimBase) Pointer
	acalsim::SCSimBase* simulator = acalsim::scsim;

	SC_MAIN_INFO << "int sc_main(int argc, char **argv) starts.";

	simulator->file  = sc_core::sc_create_vcd_trace_file("trace");
	simulator->clock = new sc_core::sc_clock("clock", simulator->getCycleDuration(), sc_core::SC_PS);
	simulator->reset = new sc_core::sc_signal<bool>("reset");

	// For SystemC-type simulator, user can create their SystemC Module (SC_MODULE)
	// before the the systemC simulation Initiation in this virtual function
	simulator->registerSystemCSim();

	simulator->setTrace(simulator->getName() + ".");

	// For SystemC-type simulator, user can init/reset their SystemC Module (SC_MODULE)
	// before the the systemC simulation Initiation in this virtual function
	simulator->initSystemCSim();

	bool isReadyToTerminate = false;

	while (true) {
		simulator->waitStepWrapperSC();

		// Let SystemC Process the activity at the current time.
		while (sc_core::sc_pending_activity_at_current_time()) { sc_core::sc_start(sc_core::SC_ZERO_TIME); }
		sc_core::sc_start(simulator->getCycleDuration(), sc_core::SC_PS);

		simulator->intraIterationUpdateSC();

		isReadyToTerminate = acalsim::top->isReadyToTerminate();

		simulator->notifyStepWrapperSC();

		if (isReadyToTerminate) break;
	}
	sc_core::sc_close_vcd_trace_file(simulator->file);
	delete simulator->clock;
	delete simulator->reset;

	sc_core::sc_stop();

	return 0;
}
