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
 * @file ProjectTemplate.cc
 * @brief Comprehensive template for creating custom ACALSim simulations
 *
 * This file serves as the **primary reference template** for building custom discrete-event
 * simulations using the ACALSim framework. It demonstrates the complete workflow from simulator
 * component design to top-level orchestration and execution.
 *
 * **Three-Step Simulation Creation Workflow:**
 * ```
 * Step 1: Define Simulator Components (SimBase subclasses)
 *   │
 *   ├─ Create custom simulator classes inheriting from SimBase
 *   ├─ Implement step() for hardware modeling logic
 *   ├─ Define MasterPorts and SlavePorts for communication
 *   └─ Register event handlers and initialize state
 *
 * Step 2: Define Top-Level Orchestrator (SimTop subclass)
 *   │
 *   ├─ Override registerConfigs() to register custom SimConfig objects
 *   ├─ Override registerCLIArguments() to add command-line options
 *   └─ Override registerSimulators() to instantiate and connect components
 *
 * Step 3: Write main() Function
 *   │
 *   ├─ Instantiate your SimTop subclass
 *   ├─ Call init() to configure and initialize framework
 *   ├─ Call run() to execute simulation main loop
 *   └─ Call finish() to cleanup and generate reports
 * ```
 *
 * **Complete Simulation Architecture Example:**
 * ```
 * User Project Structure:
 *   ├─ ProjectTemplate.cc (this file)
 *   │    └─ Defines TemplateTop (SimTop subclass) and main()
 *   │
 *   ├─ include/
 *   │    └─ Template.hh
 *   │         └─ Declares Template (SimBase subclass)
 *   │
 *   └─ libs/
 *        └─ Template.cc
 *             └─ Implements Template simulator logic
 *
 * Runtime Execution Flow:
 *   main()
 *     │
 *     ├─ 1. Construct TemplateTop (SimTop subclass)
 *     │
 *     ├─ 2. top->init(argc, argv)
 *     │     ├─ Parse CLI arguments (--threadmanager, --max-tick, etc.)
 *     │     ├─ Call registerConfigs() (user override)
 *     │     ├─ Call registerCLIArguments() (user override)
 *     │     ├─ Call registerSimulators() (user override)
 *     │     ├─ Initialize ThreadManager (parallel execution engine)
 *     │     └─ Initialize all SimBase objects
 *     │
 *     ├─ 3. top->run()
 *     │     └─ Execute two-phase simulation loop until termination
 *     │          ├─ Phase 1: Parallel execution of SimBase::step()
 *     │          └─ Phase 2: Synchronize ports/channels, advance clock
 *     │
 *     └─ 4. top->finish()
 *           ├─ Generate statistics reports
 *           ├─ Output trace files (VCD, Chrome Trace)
 *           └─ Cleanup resources
 * ```
 *
 * **Configuration System Integration:**
 * ```
 * registerConfigs() - Custom Parameter Registration:
 *   void registerConfigs() override {
 *       // 1. Create custom configuration object
 *       auto cpuConfig = new CPUConfig("CPU");
 *
 *       // 2. Register parameters with defaults
 *       cpuConfig->addParameter("cores", 4, ParamType::INT);
 *       cpuConfig->addParameter("frequency", 2.5, ParamType::FLOAT);
 *       cpuConfig->addParameter("arch", "riscv", ParamType::STRING);
 *
 *       // 3. Register config with SimTop
 *       this->addConfig("cpu", cpuConfig);
 *   }
 *
 * Parameter Access in Simulators:
 *   class CPUSimulator : public SimBase {
 *       void init() {
 *           int cores = top->getParameter<int>("cpu", "cores");
 *           float freq = top->getParameter<float>("cpu", "frequency");
 *           // Use parameters in initialization...
 *       }
 *   };
 * ```
 *
 * **CLI Argument Integration:**
 * ```
 * registerCLIArguments() - Two Approaches:
 *
 * Approach 1: Direct CLI11 API (for simple variables)
 *   void registerCLIArguments() override {
 *       this->getCLIApp()
 *           ->add_option("--num-cpus", this->num_cpus, "Number of CPU cores")
 *           ->default_val(4);
 *
 *       this->getCLIApp()
 *           ->add_option("--memory-size", this->memory_size, "Memory size in MB")
 *           ->default_val(8192);
 *   }
 *
 * Approach 2: SimConfig Integration (for framework parameters)
 *   void registerCLIArguments() override {
 *       // Updates SimConfig parameter from CLI
 *       this->addCLIOption<int>(
 *           "--cores",                    // CLI flag
 *           "Number of CPU cores",        // Help text
 *           "cpu",                        // Config name
 *           "cores",                      // Parameter name
 *           ParamType::INT                // Type
 *       );
 *   }
 *
 * Usage:
 *   ./ProjectTemplate --cores 8 --memory-size 16384 --threadmanager V3
 * ```
 *
 * **Simulator Registration and Connection:**
 * ```
 * registerSimulators() - Component Instantiation:
 *   void registerSimulators() override {
 *       // 1. Instantiate simulator components
 *       auto cpu = new CPUSimulator("CPU");
 *       auto memory = new MemoryController("Memory");
 *       auto cache = new CacheSimulator("Cache");
 *
 *       // 2. Register with SimTop (adds to global simulator list)
 *       this->addSimulator(cpu);
 *       this->addSimulator(memory);
 *       this->addSimulator(cache);
 *
 *       // 3. Connect via Ports (for packet-based communication)
 *       SimPortManager::ConnectPort(
 *           cpu,                  // Master simulator
 *           cache,                // Slave simulator
 *           "req",                // MasterPort name in cpu
 *           "cpu_req"             // SlavePort name in cache
 *       );
 *
 *       SimPortManager::ConnectPort(
 *           cache,                // Master simulator
 *           memory,               // Slave simulator
 *           "mem_req",            // MasterPort name in cache
 *           "req"                 // SlavePort name in memory
 *       );
 *
 *       // 4. Connect via Channels (for request-based communication)
 *       ChannelPortManager::ConnectChannelPort(
 *           cpu,                  // Sender
 *           memory,               // Receiver
 *           "to_memory",          // MasterChannelPort name in cpu
 *           "from_cpu",           // SlaveChannelPort name in memory
 *           10                    // Latency in simulation ticks
 *       );
 *   }
 * ```
 *
 * **Port Naming Convention:**
 * ACALSim uses directional prefixes for port names:
 *
 * | Prefix | Direction      | Example                  | Usage                               |
 * |--------|----------------|--------------------------|-------------------------------------|
 * | US     | Upstream       | "USMemory"               | Receiving from upstream component   |
 * | DS     | Downstream     | "DSCPU"                  | Sending to downstream component     |
 * | (none) | Generic        | "req", "resp", "data"    | Internal port, direction implicit   |
 *
 * Example hierarchical connection:
 * ```
 * TrafficGenerator (upstream)
 *   │
 *   ├─ DS port: "DSNOC" (send to NOC)
 *   │
 *   ▼
 * NOC (middle layer)
 *   │
 *   ├─ US port: "USTrafficGenerator" (receive from traffic generator)
 *   ├─ DS port: "DSCache" (send to cache)
 *   │
 *   ▼
 * Cache (downstream)
 *   │
 *   └─ US port: "USNOC" (receive from NOC)
 * ```
 *
 * **Main Function Template:**
 * ```cpp
 * int main(int argc, char** argv) {
 *     // 1. Instantiate top-level simulator
 *     top = std::make_shared<YourSimTop>();
 *
 *     // 2. Filter arguments (removes gtest args if present)
 *     std::vector<char*> acalsim_args = acalsim::getACALSimArguments(argc, argv);
 *
 *     // 3. Initialize framework
 *     top->init(acalsim_args.size(), acalsim_args.data());
 *
 *     // 4. Run simulation loop
 *     top->run();
 *
 *     // 5. Cleanup and report
 *     top->finish();
 *
 *     return 0;
 * }
 * ```
 *
 * **Built-in Framework CLI Options:**
 * ACALSim provides standard options automatically:
 *
 * | Option               | Type    | Default      | Description                              |
 * |----------------------|---------|--------------|------------------------------------------|
 * | --threadmanager      | string  | "V3"         | ThreadManager version (V1-V8)            |
 * | --num-threads        | int     | 8            | Worker threads in thread pool            |
 * | --max-tick           | uint64  | UINT64_MAX   | Maximum simulation ticks                 |
 * | --verbose            | bool    | false        | Enable verbose logging                   |
 * | --config             | string  | ""           | JSON config file path                    |
 * | --statistics         | bool    | false        | Enable statistics collection             |
 *
 * **Common Simulation Patterns:**
 *
 * 1. **Multi-Core CPU System:**
 *    - Multiple CPUSimulator instances
 *    - Shared cache hierarchy
 *    - Memory controller arbiter
 *
 * 2. **Accelerator Co-Design:**
 *    - HostCPU with ChannelPorts
 *    - Accelerator with bidirectional channels
 *    - Shared memory via port-based protocol
 *
 * 3. **Network-on-Chip (NOC):**
 *    - TrafficGenerator for stimulus
 *    - NOC router with multi-master arbitration
 *    - Cache/memory endpoints
 *
 * 4. **RISC-V ISA Simulation:**
 *    - RISCVCore inheriting SimBase
 *    - Instruction fetch via MasterPort
 *    - Data memory via separate MasterPort
 *
 * **Debugging and Tracing:**
 * ```cpp
 * TemplateTop() : SimTop() {
 *     // Configure VCD trace output
 *     this->traceCntr.run(0, &SimTraceContainer::setFilePath,
 *                         "trace", "output/vcd_trace");
 *
 *     // Configure Chrome Trace (for visualization)
 *     this->traceCntr.run(1, &SimTraceContainer::setFilePath,
 *                         "chrome-trace", "output/chrome_trace");
 * }
 * ```
 *
 * **Best Practices:**
 * - Keep registerSimulators() clean: Extract complex logic to helper methods
 * - Use meaningful port names: "cpu_req", "mem_resp", not "port1", "port2"
 * - Document custom CLI options with clear help text
 * - Validate configuration parameters in SimBase::init()
 * - Use events for timed actions, not busy-wait loops in step()
 * - Recycle packets/events via RecycleContainer (avoid new/delete)
 *
 * **Next Steps After Copying This Template:**
 * 1. Rename TemplateTop to YourProjectTop
 * 2. Create your SimBase subclasses in include/ and libs/
 * 3. Implement registerConfigs() for custom parameters
 * 4. Implement registerSimulators() to build your system
 * 5. Build and test: `cmake --build build --target YourProject`
 * 6. Run with options: `./YourProject --verbose --max-tick 10000`
 *
 * @see SimTop For top-level simulation orchestrator interface
 * @see SimBase For simulator component base class
 * @see SimConfig For configuration parameter management
 * @see SimPortManager For port-based communication setup
 * @see ChannelPortManager For channel-based communication setup
 * @see CLIManager For command-line argument integration
 */

#include "ACALSim.hh"
using namespace acalsim;

#include <map>
#include <string>

// Step 1 include header files of the simulator classes
#include "Template.hh"

/**
 * @brief Represents the top-level simulation class.
 * @note Step 2. Inherit SimTop to create your own top-level simulation class
 */
class TemplateTop : public SimTop {
public:
	TemplateTop() : SimTop() {}

	~TemplateTop() {}

	/**
	 * @brief (optional) Register user-defined configurations.
	 *
	 * @note This API will execute automatically when executing `init` in SimTop
	 *
	 * @details This virtual function can be implemented by derived classes to
	 * 			register user-defined configurations. By default, it does nothing.
	 */
	void registerConfigs() override {}

	/**
	 * @brief (optional) Adds command-line arguments to the CLI app.
	 *
	 * @note This method is intended to be overridden by derived classes to add specific command-line arguments relevant
	 * 		 to their functionality.
	 */
	void registerCLIArguments() override {
		// 1. Use CLI framework provided method.
		this->getCLIApp()
		    ->add_option("--template", this->template_idx, "A Template Argument in ProjectTemplate")
		    ->default_val(0);

		// 2. Use CLIManager::addCLIOption() to update the SimConfig via Command Line Argument
		// --> Example: this->addCLIOption<int>("--test", "This is test CLI for test", "ExampleConfig", "test",
		// ParamType::INT);
	}

	/**
	 * @brief (required) Register simulators and establish connections.
	 *
	 * @note This API will execute automatically when executing `init` in SimTop
	 */
	void registerSimulators() override {
		// 1. Instantiate the simulators.
		// --> Example: `SimBase* template = (SimBase*)new Template("Template");`

		// 2. Register the simulators to the top level using the `addSimulator` API of SimTop.
		// --> Example: `this->addSimulator(template);`

		// 3. Establish connections between simulators based on port names using
		//		the `addUpStream` and `addDownStream` APIs of SimBase.
		// 3.1 To connect to an upstream simulator, name it as `US{PORT_NAME}`.
		// --> Example: templateDOWN->addUpStream(template, "USTemplate")

		// 3.2 To connect to a downstream simulator, name it as DS{PORT_NAME}.
		// --> Example: templateUP->addDownStream(template, "DSTemplate")
	}

protected:
	int template_idx;
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TemplateTop>();
	// Step 4. get ACALSimArguments (remove gTestConfiguration)
	std::vector<char*> acalsim_args = acalsim::getACALSimArguments(argc, argv);
	top->init(acalsim_args.size(), acalsim_args.data());
	top->run();
	top->finish();
	return 0;
}
