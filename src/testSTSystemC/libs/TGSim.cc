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
 * @file TGSim.cc
 * @brief Traffic Generator simulator for testing SystemC MAC module integration
 *
 * @details
 * This file implements a pure C++ event-driven traffic generator that serves as a testbench
 * for the SystemC-based MAC (Multiply-Accumulate) hardware simulator. It demonstrates how to
 * create a CPPSimBase-derived simulator that communicates with SystemC components through
 * the ACALSim port-based messaging infrastructure.
 *
 * ## Overview
 *
 * TGSim (Traffic Generator Simulator) is a CPPSimBase-derived class that:
 * - Generates random test stimulus (A, B, C values)
 * - Sends input packets to MAC hardware via master ports
 * - Receives result packets via slave ports
 * - Verifies computation correctness (A × B + C = D)
 * - Provides automated testing with transaction tracking
 *
 * This simulator exemplifies the **separation of concerns** design pattern where:
 * - Hardware modeling uses SystemC (MacSim)
 * - Software/testbench uses pure C++ event simulation (TGSim)
 * - Communication occurs through well-defined packet interfaces
 *
 * ## Architecture
 *
 * ```
 * TGSim (CPPSimBase)
 *     |
 *     +-- Master Ports (sc_top1-m, sc_top2-m)
 *     |   |
 *     |   +-- Send MacInPacket to MAC hardware
 *     |
 *     +-- Slave Ports (sc_top1-s, sc_top2-s)
 *         |
 *         +-- Receive MacOutPacket from MAC hardware
 * ```
 *
 * ## CPPSimBase vs. SCSimBase
 *
 * **CPPSimBase** (used by TGSim):
 * - Pure C++ event-driven simulation
 * - No SystemC dependencies
 * - Lightweight execution model
 * - Ideal for traffic generators, monitors, scoreboards
 * - Event-based stepping (step() called when packets arrive)
 * - No clock signals required
 *
 * **SCSimBase** (used by MacSim):
 * - SystemC-integrated simulation
 * - Manages SystemC module lifecycle
 * - Clock-driven execution
 * - Cycle-accurate hardware modeling
 * - Requires SystemC compilation
 *
 * Both can coexist in the same simulation under SCSimTop management.
 *
 * ## Port-Based Communication
 *
 * TGSim uses ACALSim's port infrastructure for bidirectional communication:
 *
 * **Master Ports** (Output - sends packets):
 * - Created via addMasterPort("port_name")
 * - Used to send packets: pushToMasterPort("port_name", packet)
 * - Connected to slave ports of other simulators
 * - Example: Sending test vectors to MAC
 *
 * **Slave Ports** (Input - receives packets):
 * - Created via addSlavePort("port_name", depth)
 * - Buffered FIFO with configurable depth
 * - Polled in step() method: if (port->isPopValid()) { ... }
 * - Example: Receiving results from MAC
 *
 * **Dual Channels**:
 * This example demonstrates dual-channel communication:
 * - sc_top1: Channel 1 for parallel testing
 * - sc_top2: Channel 2 for parallel testing
 * - Random channel selection for load balancing
 *
 * ## Packet Visitor Pattern
 *
 * TGSim uses the visitor pattern for type-safe packet handling:
 *
 * 1. MacOutPacket arrives via slave port
 * 2. accept() method called by framework
 * 3. Packet's visit() method dispatches to macOutPacketHandler()
 * 4. Handler extracts data and performs verification
 *
 * This pattern enables:
 * - Type-safe packet processing
 * - Extensible packet types without modifying receiver
 * - Clear separation of packet structure and handling logic
 *
 * ## Transaction Lifecycle
 *
 * Each transaction follows this lifecycle:
 *
 * **1. Generation** (injectTraffic()):
 * - Generate random test data (A, B, C)
 * - Create InBoundData object
 * - Store in verification queue (inDataQ)
 * - Wrap in MacInPacket
 * - Assign unique transaction ID
 *
 * **2. Transmission**:
 * - Select random channel (sc_top1 or sc_top2)
 * - Push packet to master port
 * - Packet delivered to MAC simulator
 *
 * **3. Processing**:
 * - MAC hardware computes A × B + C
 * - Result stored with transaction ID
 * - MacOutPacket created
 *
 * **4. Reception** (step()):
 * - Poll slave ports for incoming packets
 * - Extract MacOutPacket when available
 * - Trigger macOutPacketHandler()
 *
 * **5. Verification** (checkAns()):
 * - Retrieve original input from verification queue
 * - Compare expected (A × B + C) vs. actual (D)
 * - Log PASS/FAIL status
 * - Free resources
 * - Inject next transaction if not complete
 *
 * ## SharedContainer Pattern
 *
 * TGSim uses SharedContainer for efficient data management:
 *
 * ```cpp
 * auto container = std::make_shared<SharedContainer<InBoundData>>();
 * container->add(transactionID, A, B, C);
 * auto packet = new MacInPacket(container);
 * ```
 *
 * **Benefits**:
 * - Reference-counted memory management (shared_ptr)
 * - Avoids data copying across simulator boundaries
 * - Type-safe container for homogeneous data
 * - Efficient bulk data transfer
 *
 * **Container Operations**:
 * - add(): Append data element
 * - get(index): Retrieve element by index
 * - size(): Get element count
 * - Automatic cleanup when all references released
 *
 * ## Event-Driven Execution Model
 *
 * TGSim follows an event-driven execution pattern:
 *
 * **Initialization** (init()):
 * - Called once during simulation setup
 * - Currently empty (no initialization needed)
 *
 * **Stepping** (step()):
 * - Called by framework when events pending
 * - Iterates over all slave ports
 * - Processes available packets
 * - Triggers downstream handlers via accept()
 *
 * **Cleanup** (cleanup()):
 * - Called once during simulation teardown
 * - Currently empty (automatic resource cleanup)
 *
 * **No Fixed Clock**:
 * - Unlike MacSim, TGSim has no clock signal
 * - step() called asynchronously when packets arrive
 * - Timing driven by MAC completion, not fixed cycles
 *
 * ## Verification Strategy
 *
 * TGSim implements a self-checking testbench pattern:
 *
 * **Input Tracking**:
 * - Stores sent inputs in FIFO queue (inDataQ)
 * - Transaction ID correlates requests and responses
 * - In-order verification (FIFO ordering)
 *
 * **Computation Check**:
 * - Golden model: D_expected = A × B + C
 * - Compare with D_actual from MAC
 * - Boolean pass/fail result
 *
 * **Logging**:
 * - LABELED_INFO: Transaction details
 * - LABELED_WARNING: Test pass indication
 * - LABELED_ERROR: Test failure indication
 *
 * **Automated Testing**:
 * - Continues until transactionID reaches limit (100)
 * - Self-terminates when complete
 * - Suitable for regression testing
 *
 * ## Random Test Generation
 *
 * The traffic generator uses pseudo-random test vectors:
 *
 * ```cpp
 * int A = std::rand() % 15;  // Range: 0-14
 * int B = std::rand() % 15;  // Range: 0-14
 * int C = std::rand() % 15;  // Range: 0-14
 * ```
 *
 * **Seed Initialization**:
 * - Set in main(): std::srand(std::time(nullptr))
 * - Provides different test patterns each run
 * - Can be fixed for reproducible testing
 *
 * **Value Ranges**:
 * - Limited to prevent overflow in 4-bit/8-bit hardware
 * - A, B: 4-bit values (0-15)
 * - C: 8-bit value (0-255, but limited to 0-14)
 * - D: 9-bit result (up to 511)
 *
 * ## Dual-Channel Load Balancing
 *
 * TGSim demonstrates multi-channel communication:
 *
 * ```cpp
 * std::string port_name = std::rand() % 2 == 1 ? "sc_top1" : "sc_top2";
 * this->pushToMasterPort(port_name + "-m", packet);
 * ```
 *
 * **Purpose**:
 * - Test concurrent MAC instances
 * - Demonstrate parallel execution
 * - Validate port routing infrastructure
 *
 * **Channel Selection**:
 * - Random selection (50/50 distribution)
 * - Independent MAC execution paths
 * - Results merged in verification
 *
 * ## Memory Management
 *
 * TGSim follows specific memory management patterns:
 *
 * **Packet Ownership**:
 * - Sender allocates: `new MacInPacket(container)`
 * - Receiver frees: `free(out_packet)` in macOutPacketHandler()
 * - Transfer of ownership through ports
 *
 * **Container Lifecycle**:
 * - Created with shared_ptr (reference counted)
 * - Automatically freed when packet deleted
 * - No manual container cleanup needed
 *
 * **Input Queue Management**:
 * - InBoundData allocated: `new InBoundData(id, A, B, C)`
 * - Freed after verification: `free(inData)` in checkAns()
 * - FIFO ordering maintained
 *
 * ## Integration with SystemC Simulator
 *
 * TGSim integrates with MacSim (SystemC simulator) through:
 *
 * **Port Connections** (in TestSTSystemC::registerSimulators()):
 * ```cpp
 * SimPortManager::ConnectPort(mac, tg, "sc_top1-m", "sc_top1-s");
 * SimPortManager::ConnectPort(tg, mac, "sc_top1-m", "sc_top1-s");
 * ```
 *
 * **Topology Specification**:
 * ```cpp
 * mac->addDownStream(tg, "DSMAC");  // MAC -> TG routing
 * tg->addUpStream(mac, "USTG");     // TG -> MAC routing
 * ```
 *
 * **Packet Types**:
 * - MacInPacket: TG -> MAC (input data)
 * - MacOutPacket: MAC -> TG (computation results)
 *
 * ## Code Example: Processing Flow
 *
 * ```
 * 1. injectTraffic() called from main
 *    |
 *    v
 * 2. Generate random A, B, C
 *    |
 *    v
 * 3. Create MacInPacket(A, B, C)
 *    |
 *    v
 * 4. pushToMasterPort() -> sends to MAC
 *    |
 *    v
 * 5. MAC processes (cycle-accurate)
 *    |
 *    v
 * 6. MAC creates MacOutPacket(D)
 *    |
 *    v
 * 7. step() detects packet in slave port
 *    |
 *    v
 * 8. accept() triggers MacOutPacket::visit()
 *    |
 *    v
 * 9. macOutPacketHandler() called
 *    |
 *    v
 * 10. checkAns() verifies result
 *     |
 *     v
 * 11. Log PASS/FAIL
 *     |
 *     v
 * 12. injectTraffic() again (if not done)
 * ```
 *
 * ## Key Methods
 *
 * **TGSim(std::string name)**:
 * - Constructor with simulator name
 * - Creates master and slave ports
 * - Port depth set to 1 (minimal buffering)
 *
 * **void injectTraffic()**:
 * - Public interface for traffic generation
 * - Called externally to start/continue testing
 * - Generates one transaction per call
 *
 * **void step()**:
 * - Framework-called event processing
 * - Polls all slave ports for packets
 * - Triggers accept() for received packets
 *
 * **void macOutPacketHandler(Tick when, SimPacket& pkt)**:
 * - Visitor pattern handler for MacOutPacket
 * - Extracts result data (D value)
 * - Calls checkAns() for verification
 * - Frees packet memory
 * - Injects next traffic if not complete
 *
 * **bool checkAns(OutBoundData* outData, int id)**:
 * - Verification golden model
 * - Compares expected vs. actual
 * - Logs detailed values and result
 * - Returns true if pass, false if fail
 * - Manages inDataQ lifecycle
 *
 * ## When to Use CPPSimBase Pattern
 *
 * Use CPPSimBase (like TGSim) when creating:
 * - Traffic generators and stimulus sources
 * - Result checkers and scoreboards
 * - Protocol monitors and analyzers
 * - High-level software models
 * - Test harnesses and drivers
 * - Performance monitors
 * - Non-hardware components
 *
 * Advantages over SystemC:
 * - No compilation overhead
 * - Simpler event-driven model
 * - Easier debugging
 * - Faster simulation for non-hardware
 *
 * ## Extending TGSim
 *
 * To create a custom traffic generator:
 *
 * 1. Inherit from CPPSimBase
 * 2. Add master ports for outputs
 * 3. Add slave ports for inputs (if bidirectional)
 * 4. Implement traffic generation logic
 * 5. Define custom packet types
 * 6. Implement step() for packet reception
 * 7. Add verification/checking logic
 *
 * Example for memory traffic generator:
 * ```cpp
 * class MemTG : public CPPSimBase {
 * public:
 *     MemTG(std::string name) : CPPSimBase(name) {
 *         addMasterPort("mem_req");
 *         addSlavePort("mem_resp", 16);
 *     }
 *
 *     void generateLoad(uint64_t addr) {
 *         auto pkt = new LoadRequest(addr);
 *         pushToMasterPort("mem_req", pkt);
 *     }
 *
 *     void step() override {
 *         // Handle memory responses
 *     }
 * };
 * ```
 *
 * @see MacSim.cc for SystemC MAC hardware simulator
 * @see MacInterface.cc for SystemC interface wrapper
 * @see MacPacket.hh for packet type definitions
 * @see testSTSystemC.cc for top-level integration
 * @see CPPSimBase for C++ simulator base class
 * @see SCSimBase for SystemC simulator base class
 */

#include "TGSim.hh"

#include "MacPacket.hh"

TGSim::TGSim(std::string name) : CPPSimBase(name) {
	// 1. Create SlavePort in Simulator
	this->addSlavePort("sc_top1-s", 1);
	this->addSlavePort("sc_top2-s", 1);

	// 2. Create MasterPort in Simulator
	this->addMasterPort("sc_top1-m");
	this->addMasterPort("sc_top2-m");
}

void TGSim::init() {}

void TGSim::cleanup() {}

void TGSim::injectTraffic() {
	int A = std::rand() % 15;
	int B = std::rand() % 15;
	int C = std::rand() % 15;

	this->inDataQ.push(new InBoundData(this->transactionID, A, B, C));
	auto container = std::make_shared<SharedContainer<InBoundData>>();
	container->add(this->transactionID, A, B, C);

	std::string port_name = std::rand() % 2 == 1 ? "sc_top1" : "sc_top2";
	auto        packet    = new MacInPacket(container);

	this->pushToMasterPort(port_name + "-m", packet);

	this->transactionID++;
	LABELED_INFO("1") << this->getName() << " | Inbound : Schedule Event to MasterPort : `" + port_name + "`";
}

void TGSim::step() {
	for (auto& port : this->s_ports_) {
		if (port->isPopValid()) {
			auto packet = port->pop();
			this->accept(top->getGlobalTick(), *packet);
		}
	}
}

void TGSim::macOutPacketHandler(Tick when, SimPacket& pkt) {
	if (auto out_packet = dynamic_cast<MacOutPacket*>(&pkt)) {
		LABELED_INFO("4") << this->getName() << " | Inbound : Handle the Packet(MacOutPacket)";
		OutBoundData* outData = out_packet->getData()->get(0).get();
		this->checkAns(outData, out_packet->getID()) ? LABELED_WARNING(this->getName()) << "!!!!!! PASS !!!!!!"
		                                             : LABELED_ERROR(this->getName()) << "!!!!!! FAIL !!!!!!";
		INFO << "=============================================================================";
		free(out_packet);
		if (this->transactionID < 100) this->injectTraffic();
	}
}

bool TGSim::checkAns(OutBoundData* outData, int id) {
	InBoundData* inData = this->inDataQ.front();
	LABELED_INFO("5") << "checkAns = " << id << " | " << this->getName();
	LABELED_INFO("5") << "( Inbound) A value = " << inData->A << " | " << this->getName();
	LABELED_INFO("5") << "( Inbound) B value = " << inData->B << " | " << this->getName();
	LABELED_INFO("5") << "( Inbound) C value = " << inData->C << " | " << this->getName();
	LABELED_INFO("5") << "(Outbound) D value = " << outData->D << " | " << this->getName();
	bool status = inData->A * inData->B + inData->C == outData->D;
	this->inDataQ.pop();
	free(inData);
	return status;
}
