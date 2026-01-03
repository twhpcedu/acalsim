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
 * @file MacSim.cc
 * @brief SystemC-based MAC (Multiply-Accumulate) hardware simulator
 *
 * @details
 * This file implements a SystemC-integrated simulator that wraps cycle-accurate MAC hardware
 * modules. It demonstrates how to create an SCSimBase-derived simulator that bridges ACALSim's
 * event-driven framework with SystemC's cycle-accurate simulation model.
 *
 * ## Overview
 *
 * MacSim (MAC Simulator) is an SCSimBase-derived class that:
 * - Manages SystemC MAC hardware instances (SC_MAC modules)
 * - Provides ACALSim port interface for packet-based communication
 * - Synchronizes SystemC clock with ACALSim tick counter
 * - Converts ACALSim packets to/from SystemC signals
 * - Supports multiple parallel MAC channels (sc_top1, sc_top2)
 * - Tracks outstanding requests for completion detection
 *
 * This simulator exemplifies **hardware/software co-simulation** where:
 * - Hardware behavior modeled in SystemC (cycle-accurate)
 * - Software/testbench uses event-driven simulation (TGSim)
 * - Bridge layer (MacInterface) converts between domains
 *
 * ## Architecture
 *
 * ```
 * MacSim (SCSimBase)
 *     |
 *     +-- SCInterface #1 (sc_top1)
 *     |   |
 *     |   +-- SC_MAC module (SystemC hardware)
 *     |   +-- Clock signal (SystemC)
 *     |   +-- Data signals: A, B, C (inputs)
 *     |   +-- Data signals: MUL_Out, ADD_Out, D (outputs)
 *     |   +-- Control signals: enable, done
 *     |
 *     +-- SCInterface #2 (sc_top2)
 *         |
 *         +-- SC_MAC module (SystemC hardware)
 *         +-- [Similar signal structure]
 *
 * Port Infrastructure:
 *     Master Ports: sc_top1-m, sc_top2-m (send results)
 *     Slave Ports:  sc_top1-s, sc_top2-s (receive inputs)
 * ```
 *
 * ## SCSimBase Integration
 *
 * **SCSimBase** is the base class for SystemC-integrated simulators:
 *
 * **Key Features**:
 * - Manages SystemC module lifecycle
 * - Automatically advances SystemC clock (sc_start)
 * - Provides hooks for SystemC-specific operations
 * - Synchronizes with ACALSim's global tick counter
 * - Supports trace file generation (VCD waveforms)
 *
 * **Lifecycle Hooks**:
 * - registerSystemCSim(): Create and register SystemC interface wrappers
 * - initSystemCSim(): Initialize SystemC state before simulation
 * - preSystemCSim(): Pre-cycle processing (before clock edge)
 * - postSystemCSim(): Post-cycle processing (after clock edge)
 * - SimulationDone(): Check termination conditions
 *
 * **Clock Management**:
 * - cycleDuration: SystemC clock period in picoseconds
 * - Passed to constructor: MacSim(name, cycleDuration)
 * - Default value: 10 ps (configurable via --cycleduration)
 * - Each ACALSim tick advances SystemC by one clock cycle
 *
 * ## SCInterface Wrapper Pattern
 *
 * MacSim uses SCInterface wrappers to encapsulate SystemC modules:
 *
 * **Purpose of SCInterface**:
 * - Bridges packet-based (ACALSim) to signal-based (SystemC) communication
 * - Manages SystemC module instantiation and port binding
 * - Converts data formats (packets ↔ signals)
 * - Implements ready-valid handshake protocol
 * - Provides signal tracing infrastructure
 *
 * **Interface Methods**:
 * - setInputs(SCSimPacket*): Convert packet to SystemC input signals
 * - getOutputs(): Convert SystemC output signals to packet
 * - setSubmodule(): Bind SystemC module ports to signals
 * - setTrace(): Configure VCD signal tracing
 *
 * ## Dual-Channel Architecture
 *
 * MacSim demonstrates multi-channel parallel processing:
 *
 * **Two Independent Channels**:
 * - sc_top1: First MAC instance
 * - sc_top2: Second MAC instance
 * - Each has dedicated ports, signals, and SystemC module
 * - Operate concurrently for parallel execution
 *
 * **Channel Creation**:
 * ```cpp
 * auto sc_top1 = new MacInterface("sc_top1");
 * auto sc_top2 = new MacInterface("sc_top2");
 * ```
 *
 * **Port Configuration**:
 * ```cpp
 * sc_top1->configSimPort(getMasterPort("sc_top1-m"),
 *                        getSlavePort("sc_top1-s"));
 * ```
 *
 * **Interface Registration**:
 * ```cpp
 * this->addSCInterface(sc_top1, "sc_top1");
 * this->addSCInterface(sc_top2, "sc_top2");
 * ```
 *
 * ## Port-Based Communication
 *
 * MacSim provides ACALSim ports for inter-simulator communication:
 *
 * **Slave Ports** (Input - receive requests):
 * - sc_top1-s: Receive MacInPacket with input data (A, B, C)
 * - sc_top2-s: Receive MacInPacket for second channel
 * - Depth: 1 (minimal buffering)
 * - Created in constructor via addSlavePort()
 *
 * **Master Ports** (Output - send responses):
 * - sc_top1-m: Send MacOutPacket with result (D)
 * - sc_top2-m: Send MacOutPacket for second channel
 * - Created in constructor via addMasterPort()
 *
 * **Port Naming Convention**:
 * - Format: "channel_name-direction"
 * - -m suffix: Master port (output)
 * - -s suffix: Slave port (input)
 * - Consistent naming enables easy connection
 *
 * ## SystemC Lifecycle Management
 *
 * MacSim implements SCSimBase lifecycle hooks:
 *
 * **registerSystemCSim()**:
 * - Called during initialization phase
 * - Creates MacInterface wrapper instances
 * - Binds interfaces to ports
 * - Registers interfaces with framework
 * - Purpose: Build SystemC module hierarchy
 *
 * **initSystemCSim()**:
 * - Called after registerSystemCSim()
 * - Currently empty (no additional initialization needed)
 * - Purpose: Initialize SystemC state/registers
 * - Example uses: Reset signal assertion, memory initialization
 *
 * **preSystemCSim()**:
 * - Called before each clock cycle
 * - Currently empty
 * - Purpose: Pre-cycle signal updates
 * - Example uses: Combinational logic evaluation, input sampling
 *
 * **postSystemCSim()**:
 * - Called after each clock cycle
 * - Currently empty
 * - Purpose: Post-cycle result collection
 * - Example uses: Output reading, state checking
 *
 * **SimulationDone()**:
 * - Checks if simulation should terminate
 * - Returns true when all outstanding requests complete
 * - Queries MacInterface::outstandingReqs counter
 * - Ensures all in-flight transactions finish
 *
 * ## Outstanding Request Tracking
 *
 * MacSim uses a static counter for completion tracking:
 *
 * **Implementation**:
 * ```cpp
 * bool MacSim::SimulationDone() {
 *     return ((MacInterface*)(interfaces.begin()->second))->getOutstandingReqs() == 0;
 * }
 * ```
 *
 * **Counter Semantics**:
 * - Incremented when MacInterface receives input packet (setInputs)
 * - Decremented when MacInterface produces output packet (getOutputs)
 * - Zero indicates all transactions completed
 *
 * **Static Variable**:
 * - Shared across all MacInterface instances
 * - Tracks total system load
 * - Enables graceful simulation termination
 *
 * ## Packet-to-Signal Conversion Flow
 *
 * Data flows through MacSim in this sequence:
 *
 * **Inbound Path** (TGSim → MacSim):
 * 1. MacInPacket arrives at slave port (sc_top1-s or sc_top2-s)
 * 2. SCSimBase framework detects packet in port
 * 3. Calls corresponding SCInterface::setInputs(packet)
 * 4. MacInterface extracts data: A, B, C values
 * 5. Writes to SystemC signals: A.write(), B.write(), C.write()
 * 6. Asserts enable signal to trigger MAC computation
 * 7. Increments outstandingReqs counter
 *
 * **Processing** (SystemC domain):
 * 1. Clock edge triggers SC_MAC process
 * 2. MAC reads input signals: A, B, C
 * 3. Computes: MUL_Out = A × B (cycle 1)
 * 4. Computes: ADD_Out = MUL_Out + C (cycle 2)
 * 5. Stores: D = ADD_Out (cycle 3)
 * 6. Asserts done signal (rv_signal_out.valid)
 *
 * **Outbound Path** (MacSim → TGSim):
 * 1. SCSimBase framework calls SCInterface::getOutputs()
 * 2. MacInterface checks done signal (rv_signal_out.valid)
 * 3. If done, reads D signal value
 * 4. Creates MacOutPacket with result
 * 5. Deasserts enable signal
 * 6. Decrements outstandingReqs counter
 * 7. SCSimBase pushes packet to master port
 * 8. Packet delivered to TGSim for verification
 *
 * ## Clock-Driven vs. Event-Driven Synchronization
 *
 * MacSim operates in a hybrid timing model:
 *
 * **SystemC Domain (Clock-Driven)**:
 * - SC_MAC module operates on clock edges
 * - Fixed cycle duration (10 ps default)
 * - Synchronous sequential logic
 * - Processes triggered by clock.posedge()
 *
 * **ACALSim Domain (Event-Driven)**:
 * - Packets trigger events
 * - No fixed time step outside SystemC
 * - Discrete event scheduling
 * - Events processed at scheduled ticks
 *
 * **Synchronization Bridge**:
 * - SCSimBase maps ACALSim ticks to SystemC time
 * - Each tick = one cycleDuration period
 * - sc_start(cycleDuration, SC_PS) advances clock
 * - Maintains causality across domains
 *
 * ## Multi-Instance SystemC Modules
 *
 * MacSim demonstrates instantiating multiple SystemC modules:
 *
 * **Benefits**:
 * - Parallel hardware execution
 * - Load distribution across instances
 * - Independent state per channel
 * - Scalability testing
 *
 * **Instance Isolation**:
 * - Each MacInterface has separate SC_MAC module
 * - Independent signal sets (A, B, C, D, enable, done)
 * - Separate clock connections (same clock signal)
 * - No shared state between instances
 *
 * **Management**:
 * - Stored in interfaces map (inherited from SCSimBase)
 * - Accessed by name: interfaces["sc_top1"]
 * - Iterated for collective operations
 *
 * ## Integration with ACALSim Framework
 *
 * MacSim integrates with ACALSim through:
 *
 * **Base Class Inheritance**:
 * ```cpp
 * class MacSim : public SCSimBase { ... };
 * ```
 *
 * **Constructor Parameters**:
 * ```cpp
 * MacSim(std::string name, int _cycleDuration)
 *     : SCSimBase(name, _cycleDuration) { ... }
 * ```
 *
 * **Port Registration**:
 * ```cpp
 * this->addSlavePort("sc_top1-s", 1);   // Input queue depth 1
 * this->addMasterPort("sc_top1-m");     // Output (unbuffered)
 * ```
 *
 * **Framework Callbacks**:
 * - init(): Standard simulator initialization
 * - cleanup(): Standard simulator cleanup
 * - registerSystemCSim(): SystemC-specific setup
 * - SimulationDone(): Custom termination logic
 *
 * ## Code Example: Adding Third MAC Channel
 *
 * To add a third parallel MAC instance:
 *
 * ```cpp
 * // In constructor
 * MacSim::MacSim(std::string name, int _cycleDuration)
 *     : SCSimBase(name, _cycleDuration) {
 *     // Existing channels
 *     this->addSlavePort("sc_top1-s", 1);
 *     this->addSlavePort("sc_top2-s", 1);
 *     this->addMasterPort("sc_top1-m");
 *     this->addMasterPort("sc_top2-m");
 *
 *     // Add third channel
 *     this->addSlavePort("sc_top3-s", 1);
 *     this->addMasterPort("sc_top3-m");
 * }
 *
 * // In registerSystemCSim()
 * void MacSim::registerSystemCSim() {
 *     auto sc_top1 = new MacInterface("sc_top1");
 *     auto sc_top2 = new MacInterface("sc_top2");
 *     auto sc_top3 = new MacInterface("sc_top3");  // New instance
 *
 *     sc_top1->configSimPort(getMasterPort("sc_top1-m"),
 *                            getSlavePort("sc_top1-s"));
 *     sc_top2->configSimPort(getMasterPort("sc_top2-m"),
 *                            getSlavePort("sc_top2-s"));
 *     sc_top3->configSimPort(getMasterPort("sc_top3-m"),
 *                            getSlavePort("sc_top3-s"));
 *
 *     this->addSCInterface(sc_top1, "sc_top1");
 *     this->addSCInterface(sc_top2, "sc_top2");
 *     this->addSCInterface(sc_top3, "sc_top3");
 * }
 *
 * // In TestSTSystemC::registerSimulators()
 * SimPortManager::ConnectPort(mac, tg, "sc_top3-m", "sc_top3-s");
 * SimPortManager::ConnectPort(tg, mac, "sc_top3-m", "sc_top3-s");
 * ```
 *
 * ## When to Use SCSimBase
 *
 * Use SCSimBase (like MacSim) for:
 * - Cycle-accurate hardware modeling (RTL-level)
 * - Integrating existing SystemC IP cores
 * - Pin-accurate digital circuit simulation
 * - Hardware/software co-simulation
 * - Clock-driven synchronous logic
 * - Hardware verification before synthesis
 * - Performance analysis of hardware designs
 *
 * ## Differences from CPPSimBase
 *
 * | Aspect | SCSimBase (MacSim) | CPPSimBase (TGSim) |
 * |--------|-------------------|-------------------|
 * | Timing Model | Clock-driven (SystemC) | Event-driven |
 * | Dependencies | Requires SystemC library | Pure C++ |
 * | Accuracy | Cycle-accurate | Functional |
 * | Module Type | sc_module | None (pure C++) |
 * | Signals | sc_signal, sc_in, sc_out | None |
 * | Clock | SystemC clock (sc_clock) | No clock |
 * | Overhead | Higher (SystemC kernel) | Lower |
 * | Use Case | Hardware modeling | Software/testbench |
 *
 * ## Key Methods
 *
 * **MacSim(std::string name, int _cycleDuration)**:
 * - Constructor with name and clock period
 * - Creates slave ports for input (depth 1)
 * - Creates master ports for output
 * - Initializes SCSimBase with cycleDuration
 *
 * **void registerSystemCSim()**:
 * - Creates MacInterface instances (sc_top1, sc_top2)
 * - Binds interfaces to master/slave ports
 * - Registers interfaces with framework
 * - Called during initialization
 *
 * **void initSystemCSim()**:
 * - Empty in this implementation
 * - Can be used for reset sequence
 * - Called after registerSystemCSim()
 *
 * **void preSystemCSim()**:
 * - Empty in this implementation
 * - Called before each clock cycle
 * - Can be used for input sampling
 *
 * **void postSystemCSim()**:
 * - Empty in this implementation
 * - Called after each clock cycle
 * - Can be used for output collection
 *
 * **bool SimulationDone()**:
 * - Checks if all transactions complete
 * - Queries outstandingReqs from first interface
 * - Returns true when counter reaches zero
 * - Enables graceful termination
 *
 * ## Extending MacSim
 *
 * To create a custom SystemC simulator:
 *
 * 1. Inherit from SCSimBase
 * 2. Define constructor with cycleDuration parameter
 * 3. Add slave/master ports in constructor
 * 4. Create custom SCInterface wrappers for your SystemC modules
 * 5. Implement registerSystemCSim() to instantiate interfaces
 * 6. Optionally override init/pre/post SystemCSim hooks
 * 7. Implement SimulationDone() for termination logic
 *
 * Example for custom hardware:
 * ```cpp
 * class MyHWSim : public SCSimBase {
 * public:
 *     MyHWSim(std::string name, int cycle) : SCSimBase(name, cycle) {
 *         addSlavePort("input", 4);   // Deeper buffering
 *         addMasterPort("output");
 *     }
 *
 *     void registerSystemCSim() override {
 *         auto iface = new MyInterface("hw_instance");
 *         iface->configSimPort(getMasterPort("output"),
 *                              getSlavePort("input"));
 *         this->addSCInterface(iface, "hw_instance");
 *     }
 * };
 * ```
 *
 * @see MacInterface.cc for SystemC interface implementation
 * @see TGSim.cc for pure C++ traffic generator
 * @see MacPacket.hh for packet type definitions
 * @see testSTSystemC.cc for top-level integration
 * @see SCSimBase for SystemC simulator base class
 * @see SCInterface for SystemC module wrapper base class
 * @see CPPSimBase for C++ event simulator comparison
 */

#include "MacSim.hh"

#include "MacInterface.hh"

MacSim::MacSim(std::string name, int _cycleDuration) : SCSimBase(name, _cycleDuration) {
	// 2. Create SlavePort in Simulator
	this->addSlavePort("sc_top1-s", 1);
	this->addSlavePort("sc_top2-s", 1);

	// 3. Create MasterPort in Simulator
	this->addMasterPort("sc_top1-m");
	this->addMasterPort("sc_top2-m");
}

void MacSim::registerSystemCSim() {
	auto sc_top1 = new MacInterface("sc_top1");
	auto sc_top2 = new MacInterface("sc_top2");
	sc_top1->configSimPort(this->getMasterPort("sc_top1-m"), this->getSlavePort("sc_top1-s"));
	sc_top2->configSimPort(this->getMasterPort("sc_top2-m"), this->getSlavePort("sc_top2-s"));
	this->addSCInterface(sc_top1, "sc_top1");
	this->addSCInterface(sc_top2, "sc_top2");
}

bool MacSim::SimulationDone() { return ((MacInterface*)(this->interfaces.begin()->second))->getOutstandingReqs() == 0; }

void MacSim::initSystemCSim() {}

void MacSim::preSystemCSim() {}

void MacSim::postSystemCSim() {}
