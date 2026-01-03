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

/**
 * @file MacInterface.cc
 * @brief SystemC interface wrapper for MAC hardware module integration
 *
 * @details
 * This file implements the SCInterface-derived wrapper that bridges between ACALSim's
 * packet-based communication model and SystemC's signal-based hardware model. It demonstrates
 * the essential pattern for integrating arbitrary SystemC modules into the ACALSim framework.
 *
 * ## Overview
 *
 * MacInterface serves as a translation layer that:
 * - Wraps the SC_MAC SystemC module (actual hardware)
 * - Converts MacInPacket (ACALSim) to SystemC signals (A, B, C)
 * - Converts SystemC signals (D) to MacOutPacket (ACALSim)
 * - Manages clock signal connection
 * - Implements ready-valid handshake protocol
 * - Provides VCD trace file generation
 * - Tracks outstanding transactions
 *
 * This interface exemplifies the **Adapter design pattern** where:
 * - ACALSim expects packet-based communication
 * - SystemC hardware uses signal-based communication
 * - MacInterface adapts between the two models
 *
 * ## Architecture
 *
 * ```
 * MacInterface (SCInterface)
 *     |
 *     +-- SC_MAC* mac                    (SystemC hardware module)
 *     |
 *     +-- sc_signal<bool> enable         (Control signal)
 *     |
 *     +-- sc_signal<sc_uint<4>> A, B     (Input data - 4-bit)
 *     +-- sc_signal<sc_uint<8>> C        (Input data - 8-bit)
 *     +-- sc_signal<sc_uint<8>> MUL_Out  (Intermediate - multiply result)
 *     +-- sc_signal<sc_uint<9>> ADD_Out  (Intermediate - add result)
 *     +-- sc_signal<sc_uint<9>> D        (Output data - 9-bit result)
 *     |
 *     +-- sc_clock clock                 (Inherited from SCInterface)
 *     +-- RV_Signal rv_signal_in         (Ready-valid input handshake)
 *     +-- RV_Signal rv_signal_out        (Ready-valid output handshake)
 * ```
 *
 * ## SCInterface Base Class
 *
 * **SCInterface** provides the foundation for SystemC module wrappers:
 *
 * **Inherited Infrastructure**:
 * - sc_clock clock: SystemC clock signal (managed by framework)
 * - RV_Signal rv_signal_in: Input handshake (valid, ready)
 * - RV_Signal rv_signal_out: Output handshake (valid, ready)
 * - sc_trace_file* file: VCD trace file handle
 *
 * **Virtual Interface Methods**:
 * - setInputs(SCSimPacket*): Convert packet to input signals
 * - getOutputs(): Convert output signals to packet
 * - setSubmodule(): Bind SystemC module ports
 * - setTrace(string): Configure VCD signal tracing
 * - updateOutValid(): Update output valid signal (optional)
 * - updateInReady(): Update input ready signal (optional)
 *
 * **Lifecycle Integration**:
 * - Constructor: Initialize signals, create SystemC module
 * - setSubmodule(): Connect module ports to signals
 * - setInputs(): Receive packet, write signals
 * - Clock cycle: SystemC module executes
 * - getOutputs(): Read signals, create packet
 *
 * ## SystemC Signal Types
 *
 * MacInterface uses specific SystemC signal types matching hardware:
 *
 * **Input Signals** (4-bit and 8-bit):
 * ```cpp
 * sc_signal<sc_uint<4>> A, B;  // 4-bit unsigned (0-15)
 * sc_signal<sc_uint<8>> C;     // 8-bit unsigned (0-255)
 * ```
 *
 * **Intermediate Signals**:
 * ```cpp
 * sc_signal<sc_uint<8>> MUL_Out;   // Multiply result (up to 15×15=225)
 * sc_signal<sc_uint<9>> ADD_Out;   // Add result (up to 225+255=480)
 * ```
 *
 * **Output Signal**:
 * ```cpp
 * sc_signal<sc_uint<9>> D;  // Final result (up to 511, needs 9 bits)
 * ```
 *
 * **Control Signals**:
 * ```cpp
 * sc_signal<bool> enable;           // Start MAC operation
 * rv_signal_out.valid (bool)        // Result ready (done signal)
 * ```
 *
 * **Why sc_uint**:
 * - Fixed-width unsigned integer type
 * - Synthesizable hardware representation
 * - Prevents overflow issues
 * - Matches hardware bit-width constraints
 *
 * ## Ready-Valid Handshake Protocol
 *
 * MacInterface implements a standard ready-valid protocol:
 *
 * **Input Channel (rv_signal_in)**:
 * - **valid**: Driven by sender (currently unused in this example)
 * - **ready**: Driven by MacInterface (always true - always ready)
 * - Purpose: Flow control for incoming packets
 *
 * **Output Channel (rv_signal_out)**:
 * - **valid**: Driven by SC_MAC module (done signal)
 * - **ready**: Driven by receiver (currently unused)
 * - Purpose: Indicate result availability
 *
 * **Protocol Behavior**:
 * ```
 * Transfer occurs when: valid AND ready == true
 *
 * Input:  ready=1 (always accepting)
 * Output: valid=1 when computation complete
 * ```
 *
 * **Why Always Ready**:
 * - Simple unbuffered interface
 * - One transaction at a time
 * - No backpressure needed
 * - Suitable for this example
 *
 * ## Outstanding Request Tracking
 *
 * MacInterface uses a static counter for load tracking:
 *
 * **Static Variable**:
 * ```cpp
 * static int outstandingReqs = 0;
 * ```
 *
 * **Incremented** (in setInputs):
 * - When new packet received
 * - Indicates transaction in-flight
 * - Used for simulation completion detection
 *
 * **Decremented** (in getOutputs):
 * - When result packet created
 * - Indicates transaction completed
 * - Zero means all work done
 *
 * **Static Rationale**:
 * - Shared across all MacInterface instances
 * - Tracks total system load
 * - Enables MacSim::SimulationDone() to check completion
 *
 * ## MAC Computation Pipeline
 *
 * The SC_MAC module performs a 3-cycle pipelined MAC operation:
 *
 * **Cycle 1**: Multiply
 * - Read inputs: A, B
 * - Compute: MUL_Out = A × B
 * - Store in register
 *
 * **Cycle 2**: Add
 * - Read: MUL_Out, C
 * - Compute: ADD_Out = MUL_Out + C
 * - Store in register
 *
 * **Cycle 3**: Store
 * - Read: ADD_Out
 * - Store: D = ADD_Out
 * - Assert: done signal (rv_signal_out.valid)
 *
 * **Enable Signal**:
 * - Asserted by setInputs() when packet arrives
 * - Starts computation pipeline
 * - Deasserted by getOutputs() after result collected
 *
 * ## Constructor Implementation
 *
 * MacInterface::MacInterface(std::string _name):
 *
 * **Signal Initialization**:
 * 1. Create signal objects with names (A, B, C, etc.)
 * 2. Initialize all signals to safe default values:
 *    - enable = false (idle state)
 *    - rv_signal_in.ready = true (accepting input)
 *    - rv_signal_out.valid = false (no output yet)
 *    - All data signals = 0
 *
 * **SystemC Module Creation**:
 * ```cpp
 * sc_module_name module_name = "MAC_module";
 * this->mac = new SC_MAC(module_name);
 * ```
 *
 * **Port Binding**:
 * ```cpp
 * this->setSubmodule();  // Connect SC_MAC ports to signals
 * ```
 *
 * **Why Initialize Signals**:
 * - Prevents undefined behavior
 * - Ensures known initial state
 * - Avoids 'X' (unknown) values in simulation
 * - Required for proper SystemC semantics
 *
 * ## setSubmodule() - Port Binding
 *
 * This method connects SC_MAC module ports to interface signals:
 *
 * ```cpp
 * void MacInterface::setSubmodule() {
 *     this->mac->clock(this->clock);           // Clock input
 *     this->mac->mac_enable(this->enable);     // Control input
 *     this->mac->in1(this->A);                 // Data input
 *     this->mac->in2(this->B);                 // Data input
 *     this->mac->in3(this->C);                 // Data input
 *     this->mac->out1(this->MUL_Out);          // Intermediate output
 *     this->mac->out2(this->ADD_Out);          // Intermediate output
 *     this->mac->out3(this->D);                // Final output
 *     this->mac->top_done(rv_signal_out.valid); // Done signal
 * }
 * ```
 *
 * **Port Binding Syntax**:
 * - module->port_name(signal_object)
 * - Establishes connection for signal propagation
 * - Must be done before simulation starts
 *
 * **Required for**:
 * - Clock distribution
 * - Input signal delivery
 * - Output signal collection
 * - Control signal communication
 *
 * ## setInputs() - Packet to Signal Conversion
 *
 * Converts incoming MacInPacket to SystemC input signals:
 *
 * **Implementation**:
 * 1. Increment outstanding request counter
 * 2. Cast generic packet to MacInPacket type
 * 3. Extract InBoundData from packet's shared container
 * 4. Store transaction ID for response correlation
 * 5. Write data to signals: A, B, C
 * 6. Assert enable signal to trigger computation
 * 7. Free packet memory (ownership transferred)
 *
 * ```cpp
 * void setInputs(SCSimPacket* packet) {
 *     outstandingReqs++;
 *     auto in_packet = dynamic_cast<MacInPacket*>(packet);
 *     InBoundData* in_data = in_packet->getData()->get(0).get();
 *     this->transactionID = in_data->id;
 *     A.write(in_data->A);
 *     B.write(in_data->B);
 *     C.write(in_data->C);
 *     enable.write(true);
 *     free(in_packet);
 * }
 * ```
 *
 * **Signal Write Semantics**:
 * - signal.write(value): Updates signal value
 * - Change becomes effective on next delta cycle
 * - SC_MAC reads values on clock edge
 *
 * ## getOutputs() - Signal to Packet Conversion
 *
 * Converts SystemC output signals to MacOutPacket:
 *
 * **Implementation**:
 * 1. Check if result ready (rv_signal_out.valid)
 * 2. If not ready, return nullptr (no packet yet)
 * 3. Decrement outstanding request counter
 * 4. Deassert enable signal (acknowledge completion)
 * 5. Read D signal value (result)
 * 6. Create shared container with OutBoundData
 * 7. Create MacOutPacket with container
 * 8. Return packet to framework
 *
 * ```cpp
 * SimPacket* getOutputs() {
 *     if (rv_signal_out.valid.read()) {
 *         outstandingReqs--;
 *         enable.write(false);
 *         auto container = std::make_shared<SharedContainer<OutBoundData>>();
 *         container->add(transactionID, D.read().to_int());
 *         return new MacOutPacket(container);
 *     }
 *     return nullptr;
 * }
 * ```
 *
 * **Signal Read Semantics**:
 * - signal.read(): Returns current signal value
 * - to_int(): Converts sc_uint to C++ int
 * - Valid only after clock edge propagation
 *
 * ## setTrace() - VCD Waveform Generation
 *
 * Configures SystemC trace file for debugging:
 *
 * ```cpp
 * void setTrace(std::string _name) {
 *     sc_trace(this->file, this->clock, _name + "clock");
 *     sc_trace(this->file, this->enable, _name + "enable");
 *     sc_trace(this->file, this->A, _name + "A");
 *     sc_trace(this->file, this->B, _name + "B");
 *     // ... trace all relevant signals
 * }
 * ```
 *
 * **Trace Parameters**:
 * - file: VCD file handle (from SCInterface base)
 * - signal: Signal object to trace
 * - name: Signal name in VCD file
 *
 * **Generated Waveforms**:
 * - Can be viewed in GTKWave or similar tools
 * - Shows signal values over time
 * - Essential for debugging timing issues
 * - Validates hardware behavior
 *
 * ## Signal Lifecycle Example
 *
 * Complete transaction sequence:
 *
 * ```
 * Time   Event                              Signals
 * ----   -----                              -------
 * T0     Packet arrives                     enable=0
 * T1     setInputs() called                 A=5, B=3, C=7, enable=1
 * T2     Clock edge (cycle 1)               MUL_Out=15
 * T3     Clock edge (cycle 2)               ADD_Out=22
 * T4     Clock edge (cycle 3)               D=22, valid=1
 * T5     getOutputs() called                enable=0, packet created
 * T6     Packet sent to TGSim               valid=0 (auto-cleared)
 * ```
 *
 * ## Transaction ID Correlation
 *
 * Maintains request-response association:
 *
 * **Storage**:
 * ```cpp
 * int transactionID;  // Member variable
 * ```
 *
 * **Capture** (in setInputs):
 * ```cpp
 * this->transactionID = in_data->id;
 * ```
 *
 * **Propagation** (in getOutputs):
 * ```cpp
 * container->add(this->transactionID, D.read().to_int());
 * ```
 *
 * **Purpose**:
 * - Correlates responses to requests
 * - Enables out-of-order completion (if needed)
 * - Facilitates verification in TGSim
 *
 * ## Memory Management
 *
 * MacInterface follows specific ownership rules:
 *
 * **Input Packet**:
 * - Received as parameter to setInputs()
 * - Ownership transferred to MacInterface
 * - Freed after extracting data: free(in_packet)
 *
 * **Output Packet**:
 * - Created in getOutputs()
 * - Ownership transferred to caller (framework)
 * - Eventually freed by TGSim after processing
 *
 * **Shared Container**:
 * - Created with std::make_shared
 * - Reference-counted
 * - Automatically freed when packet deleted
 *
 * **SystemC Module**:
 * - Created in constructor: new SC_MAC
 * - Managed by SystemC kernel
 * - Cleanup handled by SystemC framework
 *
 * ## Extending MacInterface
 *
 * To create a custom SystemC interface wrapper:
 *
 * 1. Inherit from SCInterface
 * 2. Declare SystemC module pointer
 * 3. Declare all necessary signals
 * 4. Initialize signals in constructor
 * 5. Create SystemC module instance
 * 6. Implement setSubmodule() for port binding
 * 7. Implement setInputs() for packet→signal conversion
 * 8. Implement getOutputs() for signal→packet conversion
 * 9. Optionally implement setTrace() for debugging
 *
 * Example for custom ALU:
 * ```cpp
 * class ALUInterface : public SCInterface {
 * private:
 *     SC_ALU* alu;
 *     sc_signal<sc_uint<32>> operand_a, operand_b;
 *     sc_signal<sc_uint<32>> result;
 *     sc_signal<sc_uint<3>> opcode;
 *
 * public:
 *     ALUInterface(std::string name) : SCInterface(name) {
 *         alu = new SC_ALU("alu_module");
 *         operand_a.write(0);
 *         operand_b.write(0);
 *         opcode.write(0);
 *         setSubmodule();
 *     }
 *
 *     void setSubmodule() override {
 *         alu->clk(clock);
 *         alu->a(operand_a);
 *         alu->b(operand_b);
 *         alu->op(opcode);
 *         alu->result(result);
 *     }
 *
 *     void setInputs(SCSimPacket* pkt) override {
 *         auto alu_pkt = dynamic_cast<ALUPacket*>(pkt);
 *         operand_a.write(alu_pkt->getA());
 *         operand_b.write(alu_pkt->getB());
 *         opcode.write(alu_pkt->getOp());
 *         free(alu_pkt);
 *     }
 *
 *     SimPacket* getOutputs() override {
 *         return new ALUResultPacket(result.read());
 *     }
 * };
 * ```
 *
 * ## Key Design Patterns
 *
 * **Adapter Pattern**:
 * - Adapts SystemC signal interface to ACALSim packet interface
 * - Allows incompatible interfaces to work together
 *
 * **Facade Pattern**:
 * - Provides simplified interface to complex SystemC subsystem
 * - Hides SystemC complexity from ACALSim framework
 *
 * **Static Variable for Global State**:
 * - outstandingReqs tracks system-wide load
 * - Enables coordinated termination
 * - Simple but effective for single-threaded simulation
 *
 * @see MacSim.cc for SystemC simulator using this interface
 * @see MacPacket.hh for packet type definitions
 * @see TGSim.cc for traffic generator creating packets
 * @see testSTSystemC.cc for complete system integration
 * @see SCInterface for base class documentation
 * @see SCSimBase for SystemC simulator framework
 */

#include "MacInterface.hh"

#include "MacPacket.hh"
#include "systemc/sc_mac.h"

int MacInterface::outstandingReqs = 0;

MacInterface::MacInterface(std::string _name)
    : SCInterface(_name), A("A"), B("B"), C("C"), MUL_Out("MUL_Out"), ADD_Out("ADD_Out"), D("D") {
	sc_core::sc_module_name module_name = "MAC_module";
	this->mac                           = new SC_MAC(module_name);

	this->enable.write(false);
	this->rv_signal_in.ready.write(true);
	this->rv_signal_out.valid.write(false);
	this->A.write(0);
	this->B.write(0);
	this->C.write(0);
	this->MUL_Out.write(0);
	this->ADD_Out.write(0);
	this->D.write(0);

	this->setSubmodule();
}

void MacInterface::setSubmodule() {
	this->mac->clock(this->clock);
	this->mac->mac_enable(this->enable);
	this->mac->in1(this->A);
	this->mac->in2(this->B);
	this->mac->in3(this->C);
	this->mac->out1(this->MUL_Out);
	this->mac->out2(this->ADD_Out);
	this->mac->out3(this->D);
	this->mac->top_done(this->rv_signal_out.valid);
}

void MacInterface::setTrace(std::string _name) {
	CLASS_ASSERT(this->file);
	sc_core::sc_trace(this->file, this->clock, _name + "clock");
	sc_core::sc_trace(this->file, this->enable, _name + "enable");
	sc_core::sc_trace(this->file, this->A, _name + "A");
	sc_core::sc_trace(this->file, this->B, _name + "B");
	sc_core::sc_trace(this->file, this->C, _name + "C");
	sc_core::sc_trace(this->file, this->MUL_Out, _name + "MUL_Out");
	sc_core::sc_trace(this->file, this->ADD_Out, _name + "ADD_Out");
	sc_core::sc_trace(this->file, this->D, _name + "D");
}

void MacInterface::setInputs(SCSimPacket* packet) {
	LABELED_INFO("2") << this->getName() << " | Inbound : setInputs";
	this->outstandingReqs++;
	auto         in_packet = dynamic_cast<MacInPacket*>(packet);
	InBoundData* in_data   = in_packet->getData()->get(0).get();
	this->transactionID    = in_data->id;
	A.write(in_data->A);
	B.write(in_data->B);
	C.write(in_data->C);
	enable.write(true);
	free(in_packet);
}

SimPacket* MacInterface::getOutputs() {
	if (this->rv_signal_out.valid.read()) {
		LABELED_INFO("3") << this->getName() << " | Outbound : getOutput";
		this->outstandingReqs--;
		enable.write(false);
		auto container = std::make_shared<SharedContainer<OutBoundData>>();
		container->add(this->transactionID, this->D.read().to_int());
		auto out_packet = new MacOutPacket(container);
		return out_packet;
	}
	return nullptr;
}
