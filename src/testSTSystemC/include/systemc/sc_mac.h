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

#ifndef SRC_TESTSTSYSTEMC_INCLUDE_SYSTEMC_SCMAC_H_
#define SRC_TESTSTSYSTEMC_INCLUDE_SYSTEMC_SCMAC_H_

#include <systemc>

SC_MODULE(SC_MUL) {
	sc_core::sc_in_clk                  clock;
	sc_core::sc_in<bool>                mul_enable;
	sc_core::sc_in<sc_dt::sc_uint<4> >  in1, in2;
	sc_core::sc_out<sc_dt::sc_uint<8> > out, out_mul;
	sc_core::sc_out<bool>               add_enable;

	void run() {
		while (1) {
			// std::cout << "\n[MUL] If there is activity at current time: "
			//           << sc_core::sc_pending_activity_at_current_time() << std::endl;
			// std::cout << "[MUL] If there is activity at future time: " <<
			// sc_core::sc_pending_activity_at_future_time()
			//           << std::endl;
			// std::cout << "[MUL] real cycle   : " << sc_core::sc_time_stamp().to_double() / 10 << std::endl;
			// std::cout << "[MUL] next activity: " << sc_core::sc_time_to_pending_activity() << std::endl;
			// std::cout << "[MUL] value        : " << out << std::endl;
			if (mul_enable) {
				sc_core::wait();
				out.write(in1.read() * in2.read());
				out_mul.write(in1.read() * in2.read());
				add_enable.write(mul_enable.read());
			} else
				sc_core::wait();
		}
	}

	SC_CTOR(SC_MUL)
	    : clock("clock"),
	      mul_enable("mul_enable"),
	      in1("in1"),
	      in2("in2"),
	      out("out"),
	      out_mul("out_mul"),
	      add_enable("add_enable") {
		SC_THREAD(run);
		sensitive << clock.pos();
	}
};

SC_MODULE(SC_ADD) {
	sc_core::sc_in_clk                  clock;
	sc_core::sc_in<bool>                add_enable;
	sc_core::sc_in<sc_dt::sc_uint<8> >  in1, in2;
	sc_core::sc_out<sc_dt::sc_uint<9> > out, out_add;
	sc_core::sc_out<bool>               dff_enable;

	void run() {
		while (1) {
			// std::cout << "\n[ADD] If there is activity at current time: "
			//           << sc_core::sc_pending_activity_at_current_time() << std::endl;
			// std::cout << "[ADD] If there is activity at future time: " <<
			// sc_core::sc_pending_activity_at_future_time()
			//           << std::endl;
			// std::cout << "[ADD] real cycle   : " << sc_core::sc_time_stamp().to_double() / 10 << std::endl;
			// std::cout << "[ADD] next activity: " << sc_core::sc_time_to_pending_activity() << std::endl;
			// std::cout << "[ADD] value        : " << out << std::endl;
			if (add_enable) {
				sc_core::wait();
				out.write(in1.read() + in2.read());
				out_add.write(in1.read() + in2.read());
				dff_enable.write(add_enable.read());
			} else
				sc_core::wait();
		}
	}

	SC_CTOR(SC_ADD)
	    : clock("clock"),
	      add_enable("add_enable"),
	      in1("in1"),
	      in2("in2"),
	      out("out"),
	      out_add("out_add"),
	      dff_enable("dff_enable") {
		SC_THREAD(run);
		sensitive << clock.pos();
	}
};

SC_MODULE(SC_DFF) {
	sc_core::sc_in_clk                  clock;
	sc_core::sc_in<bool>                dff_enable;
	sc_core::sc_in<sc_dt::sc_uint<9> >  in;
	sc_core::sc_out<sc_dt::sc_uint<9> > out;
	sc_core::sc_out<bool>               done;

	int reg_val;

	void write_reg() {
		while (1) {
			// std::cout << "\n[DFF] If there is activity at current time: "
			//           << sc_core::sc_pending_activity_at_current_time() << std::endl;
			// std::cout << "[DFF] If there is activity at future time: " <<
			// sc_core::sc_pending_activity_at_future_time()
			//           << std::endl;
			// std::cout << "[DFF] real cycle   : " << sc_core::sc_time_stamp().to_double() / 10 << std::endl;
			// std::cout << "[DFF] next activity: " << sc_core::sc_time_to_pending_activity() << std::endl;
			// std::cout << "[DFF] value        : " << out << std::endl;
			if (dff_enable) {
				sc_core::wait();
				reg_val = in.read();
				out.write(in.read());
				done.write(dff_enable.read());
			} else
				sc_core::wait();
		}
	}

	int read_reg() { return reg_val; }

	SC_CTOR(SC_DFF) {
		SC_THREAD(write_reg);
		sensitive << clock.pos();
	}
};

SC_MODULE(SC_MAC) {
	sc_core::sc_in_clk                  clock;
	sc_core::sc_in<bool>                mac_enable;
	sc_core::sc_in<sc_dt::sc_uint<4> >  in1, in2;
	sc_core::sc_in<sc_dt::sc_uint<8> >  in3;
	sc_core::sc_out<sc_dt::sc_uint<8> > out1;
	sc_core::sc_out<sc_dt::sc_uint<9> > out2, out3;
	sc_core::sc_out<bool>               top_done;

	SC_MUL* MUL_module;
	SC_ADD* ADD_module;
	SC_DFF* DFF_module;

	sc_core::sc_signal<sc_dt::sc_uint<8> > mul_2_add_reg;
	sc_core::sc_signal<sc_dt::sc_uint<9> > add_2_dff_reg;

	sc_core::sc_signal<bool> mul_2_add_flag;
	sc_core::sc_signal<bool> add_2_dff_flag;

	void write_input(int A, int B, int C) {}

	SC_CTOR(SC_MAC)
	    : clock("clock"),
	      mac_enable("dff_enable"),
	      in1("in1"),
	      in2("in2"),
	      in3("in3"),
	      out1("out1"),
	      out2("out2"),
	      out3("out3"),
	      top_done("top_done"),
	      mul_2_add_reg("mul_2_add_reg"),
	      add_2_dff_reg("add_2_dff_reg"),
	      mul_2_add_flag("mul_2_add_flag"),
	      add_2_dff_flag("add_2_dff_flag") {
		MUL_module = new SC_MUL("MUL_module");
		ADD_module = new SC_ADD("ADD_module");
		DFF_module = new SC_DFF("DFF_module");

		MUL_module->clock(clock);
		MUL_module->mul_enable(mac_enable);
		MUL_module->in1(in1);
		MUL_module->in2(in2);
		MUL_module->out(mul_2_add_reg);
		MUL_module->out_mul(out1);
		MUL_module->add_enable(mul_2_add_flag);

		ADD_module->clock(clock);
		ADD_module->add_enable(mul_2_add_flag);
		ADD_module->in1(in3);
		ADD_module->in2(mul_2_add_reg);
		ADD_module->out(add_2_dff_reg);
		ADD_module->out_add(out2);
		ADD_module->dff_enable(add_2_dff_flag);

		DFF_module->clock(clock);
		DFF_module->dff_enable(add_2_dff_flag);
		DFF_module->in(add_2_dff_reg);
		DFF_module->out(out3);
		DFF_module->done(top_done);

		sensitive << clock.pos();
	}
};

#endif  // SRC_TESTSTSYSTEMC_INCLUDE_SYSTEMC_SCMAC_H_
