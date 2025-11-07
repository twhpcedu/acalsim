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

#include "sim/SCInterface.hh"

// ACALSim SystemC Library
#include "packet/SCSimPacket.hh"
#include "sim/SCSimBase.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

SCInterface::SCInterface(std::string _name)
    : sc_core::sc_module(_name.c_str()),
      rv_signal_in("rv_signal_in"),
      rv_signal_out("rv_signal_out"),
      reset((_name + "_scif_reset").c_str()),
      clock((_name + "_scid_clock").c_str()) {
	this->rv_signal_in.ready.write(false);
	this->rv_signal_in.valid.write(false);
	this->rv_signal_in.handshake.write(false);
	this->rv_signal_out.ready.write(false);
	this->rv_signal_out.ready.write(false);
	this->rv_signal_out.handshake.write(false);

	SC_METHOD(outPacketHandler);
	sensitive << this->rv_signal_out.handshake;

	SC_METHOD(updateInValid);
	sensitive << this->clock.pos();

	SC_METHOD(updateOutReady);
	sensitive << this->clock.pos();

	SC_METHOD(updateInReady);
	sensitive << this->clock.pos();

	SC_METHOD(updateOutValid);
	sensitive << this->clock.pos();
}

void SCInterface::setTraceWrapper(std::string _name) {
	sc_core::sc_trace(this->file, this->clock, _name + "clock");
	sc_core::sc_trace(this->file, this->reset, _name + "reset");
	this->rv_signal_in.setTrace(this->file, _name + "rv_signal_in.");
	this->rv_signal_out.setTrace(this->file, _name + "rv_signal_out.");
	this->setTrace(_name);
}

void SCInterface::outPacketHandler() {
	if (this->reset.read() == 0 && this->rv_signal_out.handshake.read() == 1) {
		if (auto packet = this->getOutputs()) { this->mPort->push(packet); }
	}
}

void SCInterface::intraIterationUpdateSC() {
	// Check if the slave port is valid for pop packet.
	if (this->sPort->isPopValid()) {
		// Verify the handshake signal
		if (this->rv_signal_in.handshake.read()) {
			// Attempt to process the packet
			if (auto packet = dynamic_cast<SCSimPacket*>(this->sPort->pop())) {
				this->accept(top->getGlobalTick(), *packet);
			} else {
				// Log an error for invalid packet types
				LABELED_ERROR(this->name()) << "SCInterface - Failed to process packet: Invalid packet type. "
				                            << "Expected a packet inheriting from SCSimPacket.";
			}
		} else {
			MT_DEBUG_CLASS_INFO << "SCInterface - Handshake signal not asserted. Packet processing skipped.";
		}
	}
}

}  // end of namespace acalsim
