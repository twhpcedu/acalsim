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

#include <string>
#include <systemc>

#include "ACALSimSC.hh"
using namespace acalsim;

class SC_MAC;

class MacInterface : public SCInterface {
private:
	SC_MAC*                  mac;
	int                      transactionID;
	sc_core::sc_signal<bool> enable;
	static int               outstandingReqs;

protected:
	/* A x B + C = REG -> store in DFF reg */
	sc_core::sc_signal<sc_dt::sc_uint<4>> A, B;
	sc_core::sc_signal<sc_dt::sc_uint<8>> C, MUL_Out;
	sc_core::sc_signal<sc_dt::sc_uint<9>> ADD_Out, D;

public:
	MacInterface(std::string _name);
	void       setSubmodule();
	void       setTrace(std::string _name) override;
	void       setInputs(SCSimPacket* packet) override;
	SimPacket* getOutputs() override;
	void       updateOutValid() override {}
	void       updateInReady() override { this->rv_signal_in.ready.write(true); }
	int        getOutstandingReqs() { return this->outstandingReqs; }
};
