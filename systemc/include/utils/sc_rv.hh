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

#include <systemc>

namespace acalsim {

class sc_and : public sc_core::sc_module {
public:
	sc_and(sc_core::sc_module_name _name = "And") : sc_core::sc_module(_name), in1("in1"), in2("in2"), out("out") {
		SC_METHOD(do_and);
		sensitive << in1 << in2;
	}

	void do_and() { out.write(in1.read() & in2.read()); }

	void setTrace(sc_core::sc_trace_file* file, std::string name) {
		sc_core::sc_trace(file, this->in1, name + "in1");
		sc_core::sc_trace(file, this->in2, name + "in2");
		sc_core::sc_trace(file, this->out, name + "out");
	}

	sc_core::sc_in<bool>  in1, in2;
	sc_core::sc_out<bool> out;
};

class sc_signal_rv : public sc_core::sc_module {
public:
	sc_signal_rv(sc_core::sc_module_name name)
	    : sc_core::sc_module(name), valid("valid"), ready("ready"), handshake("handshake") {
		handshake_and.in1(this->ready);
		handshake_and.in2(this->valid);
		handshake_and.out(this->handshake);
	}

	void setTrace(sc_core::sc_trace_file* file, std::string name) {
		sc_core::sc_trace(file, this->ready, name + "ready");
		sc_core::sc_trace(file, this->valid, name + "valid");
		sc_core::sc_trace(file, this->valid, name + "handshake");
		this->handshake_and.setTrace(file, name + "and.");
	}

	sc_core::sc_signal<bool> ready;
	sc_core::sc_signal<bool> valid;
	sc_core::sc_signal<bool> handshake;

protected:
	sc_and handshake_and;
};

}  // end of namespace acalsim
