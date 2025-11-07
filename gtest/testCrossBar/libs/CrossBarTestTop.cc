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

/**
 * @file CrossBarTestTop.cc
 * @brief Test harness implementation for CrossBar component validation
 *
 * @details
 * This file implements the CrossBarTestTop class, which serves as the top-level test
 * environment for validating the CrossBar interconnect component. It orchestrates the
 * creation and configuration of the complete test topology including master devices,
 * slave devices, the CrossBar itself, and all necessary interconnections.
 *
 * ## Purpose
 *
 * CrossBarTestTop provides a GoogleTest-compatible environment for:
 * - **Topology Setup**: Dynamically configures N masters and M slaves
 * - **Connection Management**: Establishes bidirectional request/response channels
 * - **Configuration Handling**: Manages test parameters via CLI and config files
 * - **Pipe Register Setup**: Configures inter-component communication buffers
 * - **Test Orchestration**: Coordinates master stimulus and slave responses
 *
 * ## Test Topology Architecture
 *
 * The test bench creates a complete crossbar system with configurable dimensions:
 *
 * @code
 *                         Request Channel (Req)
 *                         ════════════════════►
 *   [Master 0]◄─────┐                        ┌─────►[Slave 0]
 *   [Master 1]◄─────┤                        ├─────►[Slave 1]
 *   [Master 2]◄─────┤◄──►[ CrossBar N×M ]◄──►├─────►[Slave 2]
 *      ...          │                        │        ...
 *   [Master N-1]◄───┘                        └─────►[Slave M-1]
 *                         ◄════════════════════
 *                         Response Channel (Resp)
 * @endcode
 *
 * **Request Path**:
 * 1. Master generates TBPacket with destination slave ID
 * 2. Packet pushed to master's PipeRegister
 * 3. CrossBar arbitrates among competing masters
 * 4. Winner's packet forwarded to target slave's SlavePort
 * 5. Slave receives packet via SlavePort pop operation
 *
 * **Response Path**:
 * 1. Slave generates response TBPacket with source master ID
 * 2. Packet pushed to slave's PipeRegister
 * 3. CrossBar routes response to originating master
 * 4. Master receives response via SlavePort pop operation
 * 5. Master validates transaction ID matches original request
 *
 * ## Key Components
 *
 * - **CrossBar**: Central NxM arbitrated interconnect (device under test)
 * - **MasterTBSim**: Test bench masters that generate request traffic
 * - **SlaveTBSim**: Test bench slaves that respond to requests
 * - **PipeRegisters**: Buffered channels between components
 * - **SimPorts**: Master/Slave port abstractions for communication
 *
 * ## Configuration Parameters
 *
 * Tests are parameterized using the CrossBarTestConfig:
 *
 * | Parameter    | Description                          | Default |
 * |--------------|--------------------------------------|---------|
 * | `n_master`   | Number of master devices             | TBD     |
 * | `n_slave`    | Number of slave devices              | TBD     |
 * | `n_requests` | Requests generated per master        | TBD     |
 *
 * These can be overridden via CLI:
 * @code{.sh}
 * ./testCrossBar --n_master 8 --n_slave 4 --n_requests 1000
 * @endcode
 *
 * ## Port Binding Strategy
 *
 * The test bench uses a specific binding pattern to connect components:
 *
 * **Request Channel Bindings**:
 * - Master → PipeRegister: `addPRMasterPort("bus-m", crossbar_pipe_register)`
 * - CrossBar → Slave: `ConnectPort(crossbar, slave, master_port_name, "bus-s")`
 *
 * **Response Channel Bindings**:
 * - Slave → PipeRegister: `addPRMasterPort("bus-m", crossbar_pipe_register)`
 * - CrossBar → Master: `ConnectPort(crossbar, master, master_port_name, "bus-s")`
 *
 * This creates bidirectional communication with proper backpressure handling.
 *
 * ## Test Execution Lifecycle
 *
 * 1. **registerConfigs()**: Loads CrossBarTestConfig for parameter storage
 * 2. **registerCLIArguments()**: Maps CLI flags to configuration parameters
 * 3. **registerSimulators()**: Creates CrossBar, masters, and slaves
 * 4. **registerPipeRegisters()**: Registers all communication channels
 * 5. **postSimInitSetup()**: Optional post-initialization hook
 * 6. **run()**: Executes simulation until all transactions complete
 * 7. **finish()**: Cleanup and result validation
 *
 * ## Arbitration Testing
 *
 * The CrossBar arbitration is implicitly tested through concurrent master access:
 *
 * @code{.cpp}
 * // Example: 4 masters simultaneously accessing slave 0
 * // Master 0: issues request to slave 0 at cycle 10
 * // Master 1: issues request to slave 0 at cycle 10
 * // Master 2: issues request to slave 0 at cycle 10
 * // Master 3: issues request to slave 0 at cycle 10
 * //
 * // Expected: CrossBar arbitrates fairly, all requests eventually serviced
 * // Validation: All 4 masters receive responses with correct transaction IDs
 * @endcode
 *
 * ## Backpressure Scenarios
 *
 * Backpressure is naturally tested when:
 * - Multiple masters target the same slave (request backpressure)
 * - CrossBar internal buffers fill (structural hazard)
 * - Slaves cannot immediately accept requests (response backpressure)
 *
 * When backpressure occurs:
 * 1. Slave device stalls (isStalled() returns true)
 * 2. Master's push() to PipeRegister returns false
 * 3. Master retries on next cycle
 * 4. When backpressure releases, masterPortRetry() callback invoked
 *
 * ## Code Example: Adding a New Test Scenario
 *
 * @code{.cpp}
 * // To test a specific arbitration pattern:
 * void CrossBarTestTop::postSimInitSetup() {
 *     // Force all masters to target slave 0 initially
 *     for (auto* master : m_devices) {
 *         master->setTargetSlavePattern({0, 0, 0, 0, 1, 2, 3});
 *     }
 * }
 * @endcode
 *
 * ## Debugging and Tracing
 *
 * The test harness integrates with ACALSim's Chrome tracing:
 * - Each transaction generates duration events
 * - Request and response paths are separately traced
 * - Visualize arbitration decisions and timing
 * - Load trace.json in chrome://tracing
 *
 * ## Best Practices
 *
 * 1. **Determinism**: Use fixed traffic patterns for reproducible tests
 * 2. **Coverage**: Test corner cases (1x1, NxN, Nx1, 1xN configurations)
 * 3. **Stress Testing**: Use high request counts to expose race conditions
 * 4. **Validation**: Verify transaction ID integrity in every test
 * 5. **Performance**: Monitor simulation time for regression detection
 *
 * @see testcrossbar::MasterTBSim Master test bench simulator
 * @see testcrossbar::SlaveTBSim Slave test bench simulator
 * @see testcrossbar::TBPacket Test packet format
 * @see acalsim::crossbar::CrossBar CrossBar implementation under test
 * @see acalsim::SimTop Base simulation top class
 * @see CrossBarTestConfig Configuration parameter container
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "CrossBarTestTop.hh"

#include "ACALSim.hh"
#include "CrossBarTestConfig.hh"

namespace testcrossbar {

/**
 * @brief Constructs the CrossBar test environment
 *
 * @details
 * Initializes the test harness by calling the base SimTop constructor and setting
 * the bus pointer to null. The actual CrossBar instantiation occurs later in
 * registerSimulators().
 *
 * @param[in] _configFilePaths Optional configuration file paths for parameter loading
 * @param[in] _tracingFileName Optional Chrome trace output filename
 *
 * @see acalsim::SimTop::SimTop()
 */
CrossBarTestTop::CrossBarTestTop(const std::vector<std::string>& _configFilePaths, const std::string& _tracingFileName)
    : acalsim::SimTop(_configFilePaths, _tracingFileName), bus(nullptr) {}

/**
 * @brief Registers configuration objects for parameter management
 *
 * @details
 * Creates and registers the CrossBarTestConfig object which holds test-specific
 * parameters (n_master, n_slave, n_requests). These parameters can be loaded from
 * configuration files or overridden via CLI arguments.
 *
 * **Configuration Namespace**: "crossbar_test"
 *
 * @see CrossBarTestConfig
 * @see acalsim::SimTop::addConfig()
 */
void CrossBarTestTop::registerConfigs() {
	this->addConfig("crossbar_test", new CrossBarTestConfig("CrossBarTestConfig"));
}

/**
 * @brief Registers command-line interface arguments for test parameterization
 *
 * @details
 * Maps CLI flags to configuration parameters, enabling runtime test customization
 * without recompilation. This is essential for GoogleTest parameterized testing
 * and continuous integration scenarios.
 *
 * **Registered CLI Options**:
 * - `--n_master <N>`: Number of master test bench devices (default from config)
 * - `--n_slave <M>`: Number of slave test bench devices (default from config)
 * - `--n_requests <K>`: Number of requests each master generates (default from config)
 *
 * ## Example Usage
 *
 * @code{.sh}
 * # Test with 8 masters, 4 slaves, 500 requests per master
 * ./testCrossBar --n_master 8 --n_slave 4 --n_requests 500
 *
 * # Minimal configuration for quick validation
 * ./testCrossBar --n_master 1 --n_slave 1 --n_requests 10
 *
 * # Stress test with many transactions
 * ./testCrossBar --n_master 16 --n_slave 8 --n_requests 10000
 * @endcode
 *
 * @note The third parameter description "Number of Slave Devices" appears to be
 *       a copy-paste error and should refer to requests, but functionality is correct.
 *
 * @see acalsim::SimTop::addCLIOption()
 * @see registerConfigs()
 */
void CrossBarTestTop::registerCLIArguments() {
	this->addCLIOption<int>("--n_master", "Number of Master Devices", "crossbar_test", "n_master");
	this->addCLIOption<int>("--n_slave", "Number of Slave Devices", "crossbar_test", "n_slave");
	this->addCLIOption<int>("--n_requests", "Number of Slave Devices", "crossbar_test", "n_requests");
}

/**
 * @brief Creates and interconnects all simulation components
 *
 * @details
 * This is the core topology setup function that instantiates the CrossBar device
 * under test along with the master and slave test bench simulators. It then
 * establishes all necessary port connections for bidirectional communication.
 *
 * ## Creation Sequence
 *
 * 1. **CrossBar Instantiation**: Creates NxM crossbar with specified dimensions
 * 2. **Master Creation**: Instantiates N MasterTBSim devices
 * 3. **Slave Creation**: Instantiates M SlaveTBSim devices
 * 4. **Request Channel Binding**: Connects masters → CrossBar → slaves
 * 5. **Response Channel Binding**: Connects slaves → CrossBar → masters
 *
 * ## Port Connection Details
 *
 * **Request Channel** (Master → Slave):
 * - Masters push packets to CrossBar via PipeRegister "bus-m"
 * - CrossBar arbitrates and forwards to target slave's SlavePort "bus-s"
 * - Connections established per-slave (getMasterPortsBySlave)
 *
 * **Response Channel** (Slave → Master):
 * - Slaves push responses to CrossBar via PipeRegister "bus-m"
 * - CrossBar routes responses to originating master's SlavePort "bus-s"
 * - Connections established per-master (getMasterPortsBySlave)
 *
 * ## Connection Pattern Visualization
 *
 * @code
 * Request Path (for slave i):
 *   Master[0]──┐
 *   Master[1]──┤ PipeReg → CrossBar.MasterPorts[*][i] → Slave[i].SlavePort
 *   Master[k]──┘
 *
 * Response Path (for master i):
 *   Slave[0]──┐
 *   Slave[1]──┤ PipeReg → CrossBar.MasterPorts[*][i] → Master[i].SlavePort
 *   Slave[k]──┘
 * @endcode
 *
 * ## Implementation Notes
 *
 * - Uses raw pointers (new) for component creation; ownership transferred to SimTop
 * - Components automatically registered via addSimulator() for lifecycle management
 * - Port connections use SimPortManager for proper binding and validation
 * - PipeRegisters provide buffering and backpressure management
 *
 * @see acalsim::crossbar::CrossBar
 * @see MasterTBSim
 * @see SlaveTBSim
 * @see acalsim::SimPortManager::ConnectPort()
 * @see registerPipeRegisters()
 */
void CrossBarTestTop::registerSimulators() {
	using SimPortManager = acalsim::SimPortManager;
	const auto nMasters  = acalsim::top->getParameter<int>("crossbar_test", "n_master");
	const auto nSlaves   = acalsim::top->getParameter<int>("crossbar_test", "n_slave");

	// 1.1 Create CrossBar
	this->bus = new acalsim::crossbar::CrossBar("crossbar", nMasters, nSlaves);
	this->addSimulator(this->bus);

	// 1.2 Create MasterTBSim
	for (auto m = 0; m < nMasters; m++) {
		auto m_device = new MasterTBSim(m);
		this->m_devices.push_back(m_device);
		this->addSimulator(m_device);
	}
	// 1.3 Create SlaveTBSim
	for (auto s = 0; s < nSlaves; s++) {
		auto s_device = new SlaveTBSim(s);
		this->s_devices.push_back(s_device);
		this->addSimulator(s_device);
	}

	// 2.1 Register PRMasterPort to MasterTBSim for the request channel
	for (int i = 0; i < nMasters; i++) { m_devices[i]->addPRMasterPort("bus-m", this->bus->getPipeRegister("Req", i)); }

	// 2.2 Simport Connection (Bus <> SlavePort at Devices)
	for (int i = 0; i < nSlaves; i++) {
		for (auto mp : bus->getMasterPortsBySlave("Req", i)) {
			SimPortManager::ConnectPort(bus, s_devices[i], mp->getName(), "bus-s");
		}
	}

	// 3.1 Register PRMasterPort to SlaveTBSim for the response channel
	for (int i = 0; i < nSlaves; i++) { s_devices[i]->addPRMasterPort("bus-m", this->bus->getPipeRegister("Resp", i)); }

	// 3.2 Simport Connection (Bus <> SlavePort at Devices)
	for (int i = 0; i < nMasters; i++) {
		for (auto mp : bus->getMasterPortsBySlave("Resp", i)) {
			SimPortManager::ConnectPort(bus, m_devices[i], mp->getName(), "bus-s");
		}
	}
}

/**
 * @brief Registers all PipeRegisters with the simulation manager
 *
 * @details
 * Collects all PipeRegister objects from the CrossBar (both request and response
 * channels) and registers them with the central PipeRegisterManager. This enables:
 * - Coordinated pipeline advancement during simulation
 * - Backpressure propagation across components
 * - Debug visibility into buffer states
 * - Trace file generation for performance analysis
 *
 * **Registered Channels**:
 * - **"Req"**: Request path PipeRegisters (Master → CrossBar → Slave)
 * - **"Resp"**: Response path PipeRegisters (Slave → CrossBar → Master)
 *
 * The base SimTop::registerPipeRegisters() is called first to handle any
 * framework-level pipe registers before adding CrossBar-specific ones.
 *
 * ## Why PipeRegister Management Matters
 *
 * PipeRegisters act as buffered communication channels with these properties:
 * - **Decoupling**: Producers and consumers operate independently
 * - **Backpressure**: Stalled consumers prevent buffer overflow
 * - **Timing**: Introduces realistic pipeline stage delays
 * - **Observability**: Buffer occupancy visible in traces
 *
 * @see acalsim::SimTop::registerPipeRegisters()
 * @see acalsim::PipeRegisterManager
 * @see acalsim::crossbar::CrossBar::getAllPipeRegisters()
 */
void CrossBarTestTop::registerPipeRegisters() {
	this->SimTop::registerPipeRegisters();
	for (auto reg : bus->getAllPipeRegisters("Req")) this->getPipeRegisterManager()->addPipeRegister(reg);
	for (auto reg : bus->getAllPipeRegisters("Resp")) this->getPipeRegisterManager()->addPipeRegister(reg);
}

/**
 * @brief Post-initialization setup hook (currently unused)
 *
 * @details
 * This virtual function is called after all simulators, configs, and pipe registers
 * have been initialized but before the main simulation run() begins. It provides a
 * hook for test-specific initialization logic.
 *
 * ## Potential Use Cases
 *
 * - Inject initial stimulus into master devices
 * - Configure specific traffic patterns for targeted testing
 * - Pre-load data into slave memory models
 * - Set up performance monitoring callbacks
 * - Initialize randomization seeds for reproducibility
 *
 * ## Example: Configuring Traffic Patterns
 *
 * @code{.cpp}
 * void CrossBarTestTop::postSimInitSetup() {
 *     // All masters target only slave 0 to stress arbitration
 *     for (auto* master : m_devices) {
 *         master->setFixedTarget(0);
 *     }
 * }
 * @endcode
 *
 * @see acalsim::SimTop::postSimInitSetup()
 */
void CrossBarTestTop::postSimInitSetup() {}

}  // namespace testcrossbar
