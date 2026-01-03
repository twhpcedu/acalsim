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
 * @file testSTSystemC.cc
 * @brief SystemC integration example demonstrating hybrid ACALSim + SystemC simulation
 *
 * @details
 * This file provides a comprehensive example of integrating SystemC hardware models with
 * the ACALSim simulation framework. It demonstrates a complete MAC (Multiply-Accumulate)
 * simulation system combining SystemC cycle-accurate hardware modeling with event-driven
 * software simulation using a traffic generator pattern.
 *
 * ## Overview
 *
 * This example showcases a hybrid simulation architecture that combines:
 * - **SystemC Hardware Model**: Cycle-accurate MAC computation unit
 * - **ACALSim Event Simulation**: Traffic generator for stimulus and verification
 * - **Port-Based Communication**: Master/Slave port connections between simulators
 * - **Packet-Based Data Transfer**: Custom packet types for typed communication
 *
 * The simulation implements a MAC operation (A × B + C = D) where:
 * - Traffic generator (TGSim) sends input data (A, B, C)
 * - MAC hardware (MacSim) performs cycle-accurate computation
 * - Results are verified against expected values
 *
 * ## Architecture
 *
 * ```
 * SCSimTop (TestSTSystemC)
 *     |
 *     +-- MacSim (SCSimBase - SystemC simulator)
 *     |   |
 *     |   +-- MacInterface (SCInterface - SystemC wrapper)
 *     |       |
 *     |       +-- SC_MAC (SystemC module - actual hardware)
 *     |
 *     +-- TGSim (CPPSimBase - C++ event simulator)
 *         |
 *         +-- Traffic generation and verification logic
 *
 * Port Connections:
 *     MacSim.sc_top1-m <--> TGSim.sc_top1-s
 *     MacSim.sc_top2-m <--> TGSim.sc_top2-s
 *     TGSim.sc_top1-m  <--> MacSim.sc_top1-s
 *     TGSim.sc_top2-m  <--> MacSim.sc_top2-s
 * ```
 *
 * ## SystemC Integration Components
 *
 * ### SCSimTop (SystemC Simulation Top)
 * - Manages multiple simulators (both SystemC and pure C++)
 * - Coordinates SystemC time advancement with event simulation
 * - Provides unified initialization and execution control
 * - Handles command-line argument parsing
 *
 * ### SCSimBase (SystemC Simulator Base)
 * - Base class for SystemC-integrated simulators
 * - Manages SystemC module lifecycle
 * - Synchronizes SystemC clock with ACALSim tick counter
 * - Provides hooks for SystemC-specific initialization:
 *   - registerSystemCSim(): Create SystemC modules
 *   - initSystemCSim(): Initialize SystemC state
 *   - preSystemCSim(): Pre-cycle processing
 *   - postSystemCSim(): Post-cycle processing
 *
 * ### CPPSimBase (C++ Simulator Base)
 * - Pure C++ event-driven simulator (no SystemC)
 * - Lightweight alternative for non-hardware components
 * - Event-based execution model
 * - Suitable for traffic generators, monitors, testbenches
 *
 * ### SCInterface (SystemC Interface Wrapper)
 * - Bridges ACALSim packets to SystemC signals
 * - Manages SystemC clock and signal connections
 * - Implements ready-valid handshake protocol
 * - Provides trace file generation support
 * - Key methods:
 *   - setInputs(): Convert packet to SystemC signals
 *   - getOutputs(): Convert SystemC signals to packet
 *   - setSubmodule(): Connect SystemC module ports
 *
 * ## Hybrid Simulation Model
 *
 * This example demonstrates a hybrid simulation approach:
 *
 * **Event-Driven Components (TGSim)**:
 * - Use discrete event scheduling
 * - Trigger on packet arrivals
 * - No fixed time step
 * - Efficient for irregular activity
 *
 * **Clock-Driven Components (MacSim)**:
 * - Advance on clock edges
 * - Fixed cycle duration (configurable via --cycleduration)
 * - SystemC time in picoseconds (SC_PS)
 * - Cycle-accurate hardware behavior
 *
 * The framework automatically synchronizes these models by:
 * - Converting event times to SystemC time
 * - Advancing SystemC clock in discrete steps
 * - Propagating packets across simulator boundaries
 *
 * ## Port-Based Communication
 *
 * Simulators communicate using a port-based messaging system:
 *
 * **Master Ports**:
 * - Initiate transactions (send packets)
 * - Push packets to connected slave ports
 * - Example: `this->pushToMasterPort("sc_top1-m", packet)`
 *
 * **Slave Ports**:
 * - Receive transactions (receive packets)
 * - Buffer incoming packets in FIFO queues
 * - Example: `port->pop()` to retrieve packet
 *
 * **Port Connection**:
 * ```cpp
 * SimPortManager::ConnectPort(master_sim, slave_sim,
 *                              "master_port_name", "slave_port_name");
 * ```
 *
 * **Upstream/Downstream Connections**:
 * - Define simulation topology and packet routing
 * - Enable visitor pattern for packet handling
 * - Example: `mac->addDownStream(tg, "label")`
 *
 * ## Packet Visitor Pattern
 *
 * The simulation uses the visitor design pattern for packet handling:
 *
 * 1. Packet is pushed through a port
 * 2. Receiving simulator's accept() method is called
 * 3. Packet's visit() method dispatches to appropriate handler
 * 4. Handler processes packet based on type
 *
 * ```cpp
 * class MacOutPacket : public SimPacket {
 *     void visit(Tick when, SimBase& simulator) override {
 *         if (auto tg = dynamic_cast<TGSim*>(&simulator)) {
 *             tg->macOutPacketHandler(when, *this);
 *         }
 *     }
 * };
 * ```
 *
 * This allows type-safe, extensible packet handling without switch statements.
 *
 * ## SystemC Clock and Timing
 *
 * **Clock Signal**:
 * - Generated by SCInterface base class
 * - Period specified by cycleDuration parameter
 * - Default: 10 picoseconds (10 SC_PS)
 * - Configurable via --cycleduration command-line argument
 *
 * **Time Conversion**:
 * - ACALSim ticks are abstract time units
 * - SystemC time is physical time (picoseconds)
 * - Conversion: `sc_time = tick × cycleDuration SC_PS`
 *
 * **Clock Advancement**:
 * - SCSimBase automatically advances SystemC clock
 * - One tick = one clock cycle
 * - SystemC processes execute during sc_start()
 *
 * ## Ready-Valid Handshake Protocol
 *
 * The example implements a ready-valid handshake for flow control:
 *
 * **Input Channel (rv_signal_in)**:
 * - valid: Data is valid (driven by sender)
 * - ready: Receiver can accept (driven by receiver)
 * - Transfer occurs when both valid and ready are high
 *
 * **Output Channel (rv_signal_out)**:
 * - valid: Result is valid (driven by MAC)
 * - ready: Consumer can accept (driven by traffic generator)
 *
 * This protocol prevents data loss and provides backpressure control.
 *
 * ## Three-Step Implementation Pattern
 *
 * ### Step 1: Create Simulator Classes
 *
 * Define simulator classes inheriting from appropriate base classes:
 *
 * ```cpp
 * // SystemC hardware simulator
 * class MacSim : public SCSimBase {
 *     void registerSystemCSim() override {
 *         // Create SystemC interface wrappers
 *     }
 * };
 *
 * // C++ event simulator
 * class TGSim : public CPPSimBase {
 *     void step() override {
 *         // Process events
 *     }
 * };
 * ```
 *
 * ### Step 2: Create Top-Level Simulation Class
 *
 * Inherit from SCSimTop and register all simulators:
 *
 * ```cpp
 * class TestSTSystemC : public SCSimTop {
 *     void registerSimulators() override {
 *         // Create simulators
 *         MacSim* mac = new MacSim("MAC", cycleDuration);
 *         TGSim* tg = new TGSim("TG");
 *
 *         // Register with framework
 *         this->addSimulator(mac);
 *         this->addSimulator(tg);
 *
 *         // Connect ports
 *         SimPortManager::ConnectPort(mac, tg, "port-m", "port-s");
 *
 *         // Define topology
 *         mac->addDownStream(tg, "label");
 *     }
 * };
 * ```
 *
 * ### Step 3: Instantiate and Run
 *
 * Create top-level object and execute simulation lifecycle:
 *
 * ```cpp
 * int main(int argc, char** argv) {
 *     top = std::make_shared<TestSTSystemC>();
 *     top->init(argc, argv);        // Initialize
 *     // Inject initial traffic
 *     top->run();                   // Execute
 *     top->finish();                // Cleanup
 * }
 * ```
 *
 * ## Simulation Execution Flow
 *
 * 1. **Initialization**:
 *    - Parse command-line arguments (--cycleduration, etc.)
 *    - Create MacSim and TGSim instances
 *    - Register simulators with framework
 *    - Connect master/slave ports
 *    - Initialize SystemC modules and signals
 *
 * 2. **Traffic Injection**:
 *    - Generate random test data (A, B, C)
 *    - Create MacInPacket with input data
 *    - Push packet to MAC via master port
 *
 * 3. **MAC Processing**:
 *    - SCInterface receives packet via slave port
 *    - Convert packet data to SystemC signals (A, B, C)
 *    - Assert enable signal to start MAC operation
 *    - SystemC module computes A × B + C over multiple cycles
 *    - Result stored in register D
 *    - Assert done signal when complete
 *
 * 4. **Result Collection**:
 *    - SCInterface detects done signal (rv_signal_out.valid)
 *    - Convert SystemC signal D to MacOutPacket
 *    - Push packet to traffic generator via master port
 *
 * 5. **Verification**:
 *    - TGSim receives MacOutPacket
 *    - Compare result D with expected value (A × B + C)
 *    - Log PASS/FAIL status
 *    - Inject next transaction if not complete
 *
 * 6. **Termination**:
 *    - Continue until transaction count reaches limit (100)
 *    - Check all outstanding requests completed
 *    - Finalize and cleanup
 *
 * ## Command-Line Arguments
 *
 * The simulation accepts custom arguments via CLI11:
 *
 * **--cycleduration** (default: 10):
 * - SystemC clock period in picoseconds
 * - Controls simulation timing granularity
 * - Higher values = slower but more detailed timing
 * - Example: `./testSTSystemC --cycleduration 100`
 *
 * Standard ACALSim arguments also available:
 * - --help: Display help message
 * - --verbose: Enable verbose logging
 * - --trace: Generate waveform traces
 *
 * ## Differences from Pure C++ Simulation (testSTSim)
 *
 * | Aspect | testSTSim (Pure C++) | testSTSystemC (Hybrid) |
 * |--------|---------------------|------------------------|
 * | Base Class | STSimBase | SCSimBase / CPPSimBase |
 * | Top Class | STSim<T> | SCSimTop |
 * | Timing Model | Abstract ticks | SystemC time (picoseconds) |
 * | Hardware Modeling | Event-driven | Cycle-accurate (SystemC) |
 * | Clock Support | None | SystemC clock signals |
 * | Signal Support | None | sc_signal, sc_in, sc_out |
 * | Module Type | SimModule | sc_module (SystemC) |
 * | Use Case | Software modeling | Hardware/software co-simulation |
 *
 * ## When to Use This Approach
 *
 * Use SystemC integration (this template) when:
 * - Modeling cycle-accurate hardware components
 * - Integrating existing SystemC IP blocks
 * - Requiring pin-level signal accuracy
 * - Hardware/software co-simulation needed
 * - Generating waveform traces for debug
 * - Validating RTL before synthesis
 *
 * Use pure C++ (testSTSim) when:
 * - High-level architectural exploration
 * - Software performance modeling
 * - Avoiding SystemC compilation overhead
 * - Event-driven behavior sufficient
 * - Rapid prototyping required
 *
 * ## Code Example: Adding Custom SystemC Module
 *
 * ```cpp
 * // 1. Define SystemC module
 * class MyHardware : public sc_module {
 * public:
 *     sc_in<bool> clk;
 *     sc_in<int> data_in;
 *     sc_out<int> data_out;
 *
 *     SC_CTOR(MyHardware) {
 *         SC_METHOD(compute);
 *         sensitive << clk.pos();
 *     }
 *
 *     void compute() {
 *         data_out.write(data_in.read() * 2);
 *     }
 * };
 *
 * // 2. Create interface wrapper
 * class MyInterface : public SCInterface {
 * private:
 *     MyHardware* hw;
 *     sc_signal<int> in_sig, out_sig;
 *
 * public:
 *     MyInterface(std::string name) : SCInterface(name) {
 *         hw = new MyHardware("hw_module");
 *         hw->clk(this->clock);
 *         hw->data_in(in_sig);
 *         hw->data_out(out_sig);
 *     }
 *
 *     void setInputs(SCSimPacket* pkt) override {
 *         // Convert packet to signals
 *         in_sig.write(pkt->getValue());
 *     }
 *
 *     SimPacket* getOutputs() override {
 *         // Convert signals to packet
 *         return new MyPacket(out_sig.read());
 *     }
 * };
 *
 * // 3. Create simulator
 * class MySim : public SCSimBase {
 * public:
 *     MySim(std::string name, int cycle) : SCSimBase(name, cycle) {}
 *
 *     void registerSystemCSim() override {
 *         auto iface = new MyInterface("my_iface");
 *         this->addSCInterface(iface, "my_iface");
 *     }
 * };
 * ```
 *
 * ## Related Classes
 *
 * **Core Framework Classes**:
 * - SCSimTop: Top-level SystemC simulation manager
 * - SCSimBase: Base for SystemC-integrated simulators
 * - CPPSimBase: Base for pure C++ simulators
 * - SCInterface: SystemC module wrapper interface
 * - SimPortManager: Port connection management
 *
 * **Example-Specific Classes**:
 * - MacSim: MAC hardware simulator (SCSimBase)
 * - TGSim: Traffic generator (CPPSimBase)
 * - MacInterface: MAC SystemC interface (SCInterface)
 * - MacPacket: Base packet template
 * - MacInPacket: Input packet (A, B, C)
 * - MacOutPacket: Output packet (D)
 *
 * ## Trace File Generation
 *
 * SystemC supports VCD waveform generation for debugging:
 *
 * ```cpp
 * void MacInterface::setTrace(std::string name) {
 *     sc_trace(this->file, this->clock, name + "clock");
 *     sc_trace(this->file, this->A, name + "A");
 *     sc_trace(this->file, this->B, name + "B");
 *     // ... trace other signals
 * }
 * ```
 *
 * Generated traces can be viewed in waveform viewers like GTKWave.
 *
 * ## Outstanding Request Tracking
 *
 * The example demonstrates transaction tracking for completion detection:
 *
 * ```cpp
 * void setInputs(SCSimPacket* pkt) override {
 *     outstandingReqs++;  // Increment on request
 * }
 *
 * SimPacket* getOutputs() override {
 *     if (done) {
 *         outstandingReqs--;  // Decrement on response
 *     }
 * }
 *
 * bool SimulationDone() override {
 *     return outstandingReqs == 0;
 * }
 * ```
 *
 * This ensures all transactions complete before simulation terminates.
 *
 * @see MacSim.cc for SystemC simulator implementation
 * @see TGSim.cc for C++ traffic generator
 * @see MacInterface.cc for SystemC interface wrapper
 * @see MacPacket.cc for packet implementation
 * @see testSTSim.cc for pure C++ simulation example
 * @see SCSimBase for SystemC simulator base class
 * @see SCSimTop for SystemC simulation manager
 * @see SCInterface for SystemC module wrapper
 * @see CPPSimBase for C++ simulator base class
 */

/* --------------------------------------------------------------------------------------
 *  A test template to demonstrate how to create your own simulation using this framework
 *  Step 1. Inherit SimBase to create your own simulator classes
 *  Step 2. Inherit SimTop to create your own top-level simulation class
 *          Add all the simulators one by one using the SimTop::addSimulation() API
 *  Step 3. instantiate a top-level simulation instance and call the following APIs in turn
 *          1) SimTop::init(); //Pre-Simulation Initialization
 *          2) SimTop::run();  //Simulation main loop
 *          3) SimTop::finish(); // Post-Simulation cleanup
 * --------------------------------------------------------------------------------------*/

#include "ACALSimSC.hh"
using namespace acalsim;

#include <cstdlib>
#include <unordered_map>

// Step 1 include header files of the simulator classes
#include "MacPacket.hh"
#include "MacSim.hh"
#include "TGSim.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestSTSystemC : public SCSimTop {
public:
	TestSTSystemC() : SCSimTop() {}

	void registerCLIArguments() override {
		this->getCLIApp()
		    ->add_option("--cycleduration", this->cycleDuration,
		                 "A cycle duration for a SystemC-type simulator : SC_PS")
		    ->default_val(this->cycleDuration);
	}

	void registerSimulators() override {
		// 1. Create simulators
		this->mac = new MacSim("MAC", this->cycleDuration);
		this->tg  = new TGSim("TG_sim");

		// 2. register Simulators
		this->addSimulator(this->mac);
		this->addSimulator(this->tg);

		// 3. Connect MasterPort and SlavePort
		SimPortManager::ConnectPort(this->mac, this->tg, "sc_top1-m", "sc_top1-s");
		SimPortManager::ConnectPort(this->mac, this->tg, "sc_top2-m", "sc_top2-s");
		SimPortManager::ConnectPort(this->tg, this->mac, "sc_top1-m", "sc_top1-s");
		SimPortManager::ConnectPort(this->tg, this->mac, "sc_top2-m", "sc_top2-s");

		// 4. connect simulators (DownStream)
		this->mac->addDownStream(this->tg, "DSMAC");
		// 5. connect simulators (UpStream)
		this->tg->addUpStream(this->mac, "USTG");
	}

	void injectTraffic() { this->tg->injectTraffic(); }

private:
	MacSim* mac;
	TGSim*  tg;

	int cycleDuration = 10;
};

int main(int argc, char** argv) {
	std::srand(std::time(nullptr));
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestSTSystemC>();
	top->init(argc, argv);
	std::static_pointer_cast<TestSTSystemC>(top)->injectTraffic();
	top->run();
	top->finish();

	return 0;
}
