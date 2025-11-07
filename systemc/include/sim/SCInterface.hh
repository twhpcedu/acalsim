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

#include <memory>
#include <queue>
#include <string>
#include <systemc>

#include "packet/SCSimPacket.hh"
#include "utils/sc_rv.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

class SCSimBase;

class SCInterface : public sc_core::sc_module, virtual public HashableType {
	friend class SCSimBase;

public:
	SCInterface(std::string _name);

	void setTraceWrapper(std::string _name);

	void setSCSimulator(SCSimBase* _sim) { this->scSimulator = _sim; }

	SCSimBase* getSCSimulator() const { return this->scSimulator; }

	void configSimPort(MasterPort* _mPort, SlavePort* _sPort) { this->mPort = _mPort, this->sPort = _sPort; }

	inline void updateOutReady() { this->rv_signal_out.ready.write(this->mPort->isPushReady()); }

	inline void updateInValid() { this->rv_signal_in.valid.write(this->sPort->isPopValid()); }

	inline virtual void updateInReady() = 0;

	inline virtual void updateOutValid() = 0;

	void intraIterationUpdateSC();

	inline void inPacketHandler(Tick when, SCSimPacket* packet) { this->setInputs(packet); }

	void outPacketHandler();

	// Set the inbound signals based on the request packet.
	virtual void setInputs(SCSimPacket* _packet) = 0;

	// Get the SimPacket when the Valid Signal raised (call by SCSimBase::getOutBoundValid)
	virtual SimPacket* getOutputs() = 0;

	// Set the trace signals in the file for SystemC simulation.
	inline virtual void setTrace(std::string _name) {}

	std::string getName() const { return this->name(); }

	void accept(Tick when, SCSimPacket& pkt) { pkt.visit(when, *this); }

protected:
	SCSimBase* scSimulator = nullptr;

	sc_core::sc_trace_file* file = nullptr;

	sc_signal_rv rv_signal_in;
	sc_signal_rv rv_signal_out;

	sc_core::sc_in_clk   clock;
	sc_core::sc_in<bool> reset;

private:
	MasterPort* mPort = nullptr;
	SlavePort*  sPort = nullptr;
};

}  // end of namespace acalsim
