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
 * @file TestSimPortTop.cc
 * @brief Top-level orchestrator for the testSimPort CPU-Bus-Memory demonstration
 *
 * This file implements the SimTop subclass that configures and orchestrates the complete
 * testSimPort example system. It demonstrates **system-level integration** of multiple
 * hardware components through port-based communication.
 *
 * **System Architecture Overview:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────────────┐
 * │                          TestSimPortTop (SimTop)                                │
 * │                                                                                 │
 * │  ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐ │
 * │  │    CPUCore       │         │    CrossBar      │         │     Memory       │ │
 * │  │                  │         │      (Bus)       │         │                  │ │
 * │  │  - Generate      │         │                  │         │  - Fixed         │ │
 * │  │    requests      │         │  - Arbitrate     │         │    latency       │ │
 * │  │  - Track         │         │    requests      │         │  - Generate      │ │
 * │  │    outstanding   │         │  - Forward       │         │    responses     │ │
 * │  │    requests      │         │    packets       │         │                  │ │
 * │  │                  │         │  - Add bus       │         │                  │ │
 * │  │  MasterPort ─────┼────────►│    latency       │────────►│  SlavePort       │ │
 * │  │  SlavePort  ◄────┼─────────│                  │◄────────│  MasterPort      │ │
 * │  │                  │         │                  │         │                  │ │
 * │  └──────────────────┘         └──────────────────┘         └──────────────────┘ │
 * │                                                                                 │
 * │  Command-Line Configuration:                                                   │
 * │    --cpu-outstanding-requests : Max in-flight requests (default: 5)            │
 * │    --cpu-total-requests       : Total requests to generate (default: 100)      │
 * │    --cpu-latency              : CPU internal latency (default: 1)              │
 * │    --bus-latency              : Bus traversal latency (default: 2)             │
 * │    --mem-latency              : Memory access latency (default: 5)             │
 * │    --queue-size               : Internal queue sizes (default: 2)              │
 * └─────────────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Port Connection Topology:**
 * ```
 * CPUCore                  CrossBar                 Memory
 *   │                         │                        │
 *   │  bus-m (MasterPort)     │                        │
 *   ├────────────────────────►│ cpu-s (SlavePort)      │
 *   │                         │                        │
 *   │  bus-s (SlavePort)      │                        │
 *   │◄────────────────────────┤ cpu-m (MasterPort)     │
 *   │                         │                        │
 *   │                         │  mem-m (MasterPort)    │
 *   │                         ├───────────────────────►│ bus-s (SlavePort)
 *   │                         │                        │
 *   │                         │  mem-s (SlavePort)     │
 *   │                         │◄───────────────────────┤ bus-m (MasterPort)
 *   │                         │                        │
 *
 * Port Connection Legend:
 *   ──►  : Request path (master → slave)
 *   ◄──  : Response path (master → slave, but carrying responses)
 * ```
 *
 * **Configuration Parameters and Their Effects:**
 *
 * | Parameter                    | Default | Effect on System                                  |
 * |------------------------------|---------|---------------------------------------------------|
 * | cpu_outstanding_requests     | 5       | Max concurrent requests in flight                 |
 * | cpu_total_requests           | 100     | Total requests before CPU stops generating        |
 * | cpu_latency                  | 1       | CPU internal processing delay (ticks)             |
 * | bus_latency                  | 2       | Bus traversal time (adds to req+rsp paths)        |
 * | mem_latency                  | 5       | Memory access time (from receive to respond)      |
 * | queue_size                   | 2       | Size of internal queues in each component         |
 *
 * **Total Request Latency Calculation:**
 * ```
 * Total Latency = cpu_latency           // CPU internal queue
 *               + bus_latency           // CPU → Memory via bus
 *               + mem_latency           // Memory processing
 *               + bus_latency           // Memory → CPU via bus
 *               + cpu_latency           // CPU internal queue
 *
 * With defaults: 1 + 2 + 5 + 2 + 1 = 11 ticks per request
 * ```
 *
 * **System Throughput Analysis:**
 * ```
 * Theoretical Max Throughput:
 *   - Limited by smallest queue_size (default: 2)
 *   - With cpu_outstanding_requests=5, up to 5 requests in flight
 *   - Steady-state: ~1 request every cpu_latency ticks (if no backpressure)
 *
 * Backpressure Scenarios:
 *   1. CPU queue full → CPU pauses request generation
 *   2. Bus queue full → CPU retry callback triggered
 *   3. Memory queue full → Bus retry callback triggered
 * ```
 *
 * **SimTop Lifecycle in testSimPort:**
 * ```
 * main() {
 *   1. top = new TestSimPortTop()
 *      └─ Constructor initializes default parameters
 *
 *   2. top->init(argc, argv)
 *      ├─ Parse CLI arguments
 *      ├─ registerCLIArguments()    ◄─── User-defined CLI options
 *      ├─ Update parameters from CLI
 *      ├─ registerSimulators()      ◄─── Instantiate and connect components
 *      │   ├─ new CPUCore(...)
 *      │   ├─ new CrossBar(...)
 *      │   ├─ new Memory(...)
 *      │   ├─ addSimulator(cpu)
 *      │   ├─ addSimulator(bus)
 *      │   ├─ addSimulator(mem)
 *      │   └─ SimPortManager::ConnectPort(...) x4
 *      ├─ preSimInitSetup()         ◄─── (empty in this example)
 *      ├─ Initialize all simulators
 *      └─ postSimInitSetup()        ◄─── (empty in this example)
 *
 *   3. top->run()
 *      └─ Execute simulation loop until completion
 *
 *   4. top->finish()
 *      └─ Cleanup and report statistics
 * }
 * ```
 *
 * **Usage Examples:**
 *
 * 1. **Run with default settings:**
 *    ```bash
 *    ./testSimPort
 *    # Uses: 5 outstanding, 100 total requests
 *    # Latencies: CPU=1, Bus=2, Memory=5
 *    # Queue size: 2
 *    ```
 *
 * 2. **Increase concurrency and throughput:**
 *    ```bash
 *    ./testSimPort --cpu-outstanding-requests 16 --queue-size 8
 *    # More requests in flight, larger queues reduce backpressure
 *    ```
 *
 * 3. **Stress test with many requests:**
 *    ```bash
 *    ./testSimPort --cpu-total-requests 10000 --max-tick 200000
 *    # Run 10k requests with extended simulation time
 *    ```
 *
 * 4. **Simulate slow memory:**
 *    ```bash
 *    ./testSimPort --mem-latency 100 --cpu-outstanding-requests 32
 *    # High memory latency requires more outstanding requests
 *    ```
 *
 * 5. **Minimal latency configuration:**
 *    ```bash
 *    ./testSimPort --cpu-latency 0 --bus-latency 1 --mem-latency 1
 *    # Fastest possible configuration for testing
 *    ```
 *
 * 6. **Combine with framework options:**
 *    ```bash
 *    ./testSimPort --cpu-outstanding-requests 8 \
 *                  --threadmanager V3 \
 *                  --num-threads 4 \
 *                  --verbose
 *    # Custom system with specific ThreadManager and logging
 *    ```
 *
 * **Performance Tuning Guidelines:**
 *
 * 1. **Maximizing Throughput:**
 *    - Increase `cpu_outstanding_requests` to keep pipeline full
 *    - Increase `queue_size` to prevent backpressure stalls
 *    - Balance: More outstanding requests = more memory usage
 *
 * 2. **Minimizing Latency:**
 *    - Reduce `cpu_latency`, `bus_latency`, `mem_latency`
 *    - Note: Lower latencies stress-test synchronization logic
 *
 * 3. **Testing Backpressure Handling:**
 *    - Set `queue_size=1` to force frequent backpressure
 *    - Set `cpu_outstanding_requests=2` with `queue_size=1`
 *    - Observe retry callback behavior in logs
 *
 * **Port Connection Details:**
 *
 * The system uses 4 bidirectional port connections:
 *
 * | Connection | Master Port    | Slave Port  | Direction        | Packet Type  |
 * |------------|----------------|-------------|------------------|--------------|
 * | CPU→Bus    | cpu.bus-m      | bus.cpu-s   | Request path     | BaseReqPacket|
 * | Bus→CPU    | bus.cpu-m      | cpu.bus-s   | Response path    | BaseRspPacket|
 * | Bus→Mem    | bus.mem-m      | mem.bus-s   | Request path     | BaseReqPacket|
 * | Mem→Bus    | mem.bus-m      | bus.mem-s   | Response path    | BaseRspPacket|
 *
 * **Design Patterns Demonstrated:**
 *
 * 1. **Three-Step Setup Pattern:**
 *    - Step 1: Instantiate simulators with configuration
 *    - Step 2: Register simulators with SimTop
 *    - Step 3: Connect ports using SimPortManager
 *
 * 2. **Configuration via CLI:**
 *    - Default values in member variables
 *    - Override via command-line arguments
 *    - Pass to simulator constructors
 *
 * 3. **Separation of Concerns:**
 *    - SimTop: System orchestration and configuration
 *    - SimBase components: Hardware modeling logic
 *    - SimPortManager: Port connection management
 *
 * **Extending This Example:**
 *
 * 1. **Add Multiple CPUs:**
 *    ```cpp
 *    for (int i = 0; i < num_cpus; i++) {
 *        auto cpu = new CPUCore("cpu" + std::to_string(i), ...);
 *        this->addSimulator(cpu);
 *        SimPortManager::ConnectPort(cpu, bus, "bus-m", "cpu" + std::to_string(i) + "-s");
 *        SimPortManager::ConnectPort(bus, cpu, "cpu" + std::to_string(i) + "-m", "bus-s");
 *    }
 *    ```
 *
 * 2. **Add Cache Layer:**
 *    ```cpp
 *    auto cache = new Cache("cache", cache_size, cache_latency);
 *    // Connect: CPU → Cache → Bus → Memory
 *    SimPortManager::ConnectPort(cpu, cache, "bus-m", "cpu-s");
 *    SimPortManager::ConnectPort(cache, bus, "bus-m", "cache-s");
 *    ```
 *
 * 3. **Add Configuration System:**
 *    ```cpp
 *    void registerConfigs() override {
 *        auto config = new SimConfig("system");
 *        config->addParameter("cpu_count", 4, ParamType::INT);
 *        this->addConfig("system", config);
 *    }
 *    ```
 *
 * **Common Issues and Solutions:**
 *
 * | Issue                        | Symptom                          | Solution                              |
 * |------------------------------|----------------------------------|---------------------------------------|
 * | Deadlock                     | Simulation hangs                 | Increase queue_size or outstanding    |
 * | Backpressure storms          | Retry callbacks spam             | Increase queue_size                   |
 * | Slow simulation              | Low throughput                   | Increase cpu_outstanding_requests     |
 * | Memory usage high            | Large heap allocation            | Decrease outstanding_requests         |
 *
 * @see CPUCore For request generator and outstanding request tracker
 * @see CrossBar For multi-master bus arbiter implementation
 * @see Memory For fixed-latency memory model
 * @see BasePacket For packet structure and visitor pattern
 * @see main.cc For detailed system architecture documentation
 */

#include "TestSimPortTop.hh"

namespace test_port {

/**
 * @brief Instantiates and connects all simulator components in the testSimPort system.
 *
 * This method follows the **three-step setup pattern** for building a complete system:
 * 1. Instantiate simulator objects with configuration parameters
 * 2. Register simulators with SimTop
 * 3. Connect ports to establish communication paths
 *
 * **Setup Step 1: Instantiate Simulators**
 *
 * Each simulator is created with parameters that were either:
 * - Initialized to default values in TestSimPortTop constructor
 * - Overridden by command-line arguments in registerCLIArguments()
 *
 * Component Instantiation:
 * ```cpp
 * CPUCore(name, max_outstanding, total_requests, latency, queue_size)
 *   - Generates memory requests with flow control
 *   - Tracks outstanding requests (max 'cpu_outstanding_requests' in-flight)
 *   - Internal queue delay of 'cpu_latency' ticks
 *
 * CrossBar(name, latency, queue_size)
 *   - Arbitrates between CPU and Memory requests/responses
 *   - Adds 'bus_latency' ticks to all packets
 *   - Internal queues of size 'queue_size'
 *
 * Memory(name, latency, queue_size)
 *   - Processes requests with fixed 'mem_latency' delay
 *   - Generates response packets
 *   - Internal queue of size 'queue_size'
 * ```
 *
 * **Setup Step 2: Register Simulators**
 *
 * Registration adds simulators to the global list managed by SimTop:
 * ```cpp
 * addSimulator(cpu) → Framework will call cpu->init() and cpu->step()
 * addSimulator(bus) → Framework will call bus->init() and bus->step()
 * addSimulator(mem) → Framework will call mem->init() and mem->step()
 * ```
 *
 * **Setup Step 3: Connect Ports**
 *
 * Port connections establish bidirectional communication channels:
 *
 * ```
 * Connection 1 (CPU→Bus Request Path):
 *   SimPortManager::ConnectPort(cpu, bus, "bus-m", "cpu-s")
 *     ├─ cpu's MasterPort "bus-m" → bus's SlavePort "cpu-s"
 *     ├─ CPU generates requests and pushes to its MasterPort
 *     └─ Bus's SlavePort receives and arbitrates requests
 *
 * Connection 2 (Bus→CPU Response Path):
 *   SimPortManager::ConnectPort(bus, cpu, "cpu-m", "bus-s")
 *     ├─ bus's MasterPort "cpu-m" → cpu's SlavePort "bus-s"
 *     ├─ Bus forwards responses to CPU
 *     └─ CPU matches responses with outstanding requests
 *
 * Connection 3 (Bus→Memory Request Path):
 *   SimPortManager::ConnectPort(bus, mem, "mem-m", "bus-s")
 *     ├─ bus's MasterPort "mem-m" → mem's SlavePort "bus-s"
 *     ├─ Bus forwards CPU requests to Memory
 *     └─ Memory receives and processes requests
 *
 * Connection 4 (Memory→Bus Response Path):
 *   SimPortManager::ConnectPort(mem, bus, "bus-m", "mem-s")
 *     ├─ mem's MasterPort "bus-m" → bus's SlavePort "mem-s"
 *     ├─ Memory generates responses
 *     └─ Bus forwards responses back to CPU
 * ```
 *
 * **Port Connection Rules:**
 * - ConnectPort(A, B, "port_a", "port_b") connects A's MasterPort to B's SlavePort
 * - MasterPort: Used for sending packets (push operation)
 * - SlavePort: Used for receiving packets (pop operation, with arbitration)
 * - Port names must match what was registered in simulator's init() or constructor
 *
 * **Complete Data Flow:**
 * ```
 * Request Path:
 *   CPU.bus-m → Bus.cpu-s → [bus latency] → Bus.mem-m → Mem.bus-s
 *
 * Response Path:
 *   Mem.bus-m → Bus.mem-s → [bus latency] → Bus.cpu-m → CPU.bus-s
 * ```
 *
 * **Why This Pattern:**
 * 1. Separation of concerns: Instantiation, registration, connection are distinct
 * 2. Flexibility: Easy to add/remove components or change connections
 * 3. Clarity: Each step has a clear purpose and is self-documenting
 * 4. Maintainability: Modifications are localized to specific steps
 *
 * @note This method is called automatically by SimTop::init() after CLI parsing
 * @note Port connections are validated at runtime; mismatched names will cause errors
 * @note The order of addSimulator() calls doesn't matter; all simulators run in Phase 1
 */
void TestSimPortTop::registerSimulators() {
	// Setup Step (1) - Generate Simulator
	this->cpu = new CPUCore("cpu", this->cpu_outstanding_requests, this->cpu_total_requests, this->cpu_latency,
	                        this->queue_size);
	this->bus = new CrossBar("bus", this->bus_latency, this->queue_size);
	this->mem = new Memory("mem", this->mem_latency, this->queue_size);

	// Setup Step (2) - Register Simulator
	this->addSimulator(this->cpu);
	this->addSimulator(this->bus);
	this->addSimulator(this->mem);

	// Setup Step (3) - Connect MasterPort and SlavePort
	// SimPortManager::ConnectPort(master, slave, mport_name, sport_name)
	acalsim::SimPortManager::ConnectPort(this->cpu, this->bus, "bus-m", "cpu-s");
	acalsim::SimPortManager::ConnectPort(this->bus, this->cpu, "cpu-m", "bus-s");
	acalsim::SimPortManager::ConnectPort(this->bus, this->mem, "mem-m", "bus-s");
	acalsim::SimPortManager::ConnectPort(this->mem, this->bus, "bus-m", "mem-s");
}

/**
 * @brief Pre-simulation initialization hook (currently unused in testSimPort).
 *
 * This method is called after registerSimulators() but before simulator init() calls.
 * It can be used for:
 * - Additional system-level setup that requires all simulators to be registered
 * - Cross-simulator configuration validation
 * - Loading external configuration files
 * - Setting up shared resources
 *
 * **When to Use:**
 * ```cpp
 * void preSimInitSetup() override {
 *     // Example: Validate configuration consistency
 *     if (cpu_outstanding_requests > queue_size * 3) {
 *         throw std::runtime_error("Outstanding requests too high for queue size");
 *     }
 *
 *     // Example: Load trace files
 *     trace_loader = new TraceLoader("input.trace");
 *     cpu->setTraceSource(trace_loader);
 * }
 * ```
 *
 * @note Called in main thread before parallel execution begins
 */
void TestSimPortTop::preSimInitSetup() {}

/**
 * @brief Post-simulation initialization hook (currently unused in testSimPort).
 *
 * This method is called after all simulators have completed their init() methods.
 * It can be used for:
 * - Final system consistency checks
 * - Scheduling initial global events
 * - Printing configuration summary
 *
 * **When to Use:**
 * ```cpp
 * void postSimInitSetup() override {
 *     // Example: Print system configuration
 *     std::cout << "System Configuration:\n"
 *               << "  CPU Outstanding: " << cpu_outstanding_requests << "\n"
 *               << "  Total Latency: " << (cpu_latency * 2 + bus_latency * 2 + mem_latency) << "\n";
 *
 *     // Example: Verify port connections
 *     if (!cpu->isMasterPortConnected("bus-m")) {
 *         throw std::runtime_error("CPU bus-m port not connected!");
 *     }
 * }
 * ```
 *
 * @note Called in main thread after all simulator init() calls complete
 */
void TestSimPortTop::postSimInitSetup() {}

/**
 * @brief Registers command-line arguments for system configuration.
 *
 * This method defines all user-configurable parameters for the testSimPort example.
 * Arguments registered here can override the default values defined in TestSimPortTop's
 * member variables.
 *
 * **CLI Argument Categories:**
 *
 * 1. **CPU Configuration:**
 *    ```bash
 *    --cpu-outstanding-requests <N>  # Max concurrent in-flight requests (default: 5)
 *    --cpu-total-requests <N>        # Total requests to generate (default: 100)
 *    --cpu-latency <N>               # CPU internal processing delay (default: 1)
 *    ```
 *
 * 2. **Bus Configuration:**
 *    ```bash
 *    --bus-latency <N>               # Bus traversal time (default: 2)
 *    ```
 *
 * 3. **Memory Configuration:**
 *    ```bash
 *    --mem-latency <N>               # Memory access latency (default: 5)
 *    ```
 *
 * 4. **System Configuration:**
 *    ```bash
 *    --queue-size <N>                # Internal queue sizes for all components (default: 2)
 *    ```
 *
 * **Argument Implementation Pattern:**
 *
 * Each argument follows this pattern:
 * ```cpp
 * this->getCLIApp()                           // Get CLI11 application object
 *     ->add_option(                           // Add a new option
 *         "--flag-name",                      // Command-line flag
 *         member_variable,                    // Member variable to update
 *         "Description text"                  // Help text
 *     )
 *     ->default_str(std::to_string(member_variable));  // Show default in help
 * ```
 *
 * **Parameter Effects on System Behavior:**
 *
 * | Parameter                 | Effect                                              | Tuning Advice |
 * |---------------------------|-----------------------------------------------------|------------------------------------|
 * | cpu_outstanding_requests  | Controls pipeline depth and throughput              | Higher = more throughput | |
 * cpu_total_requests        | Determines simulation workload                      | Higher = longer simulation | |
 * cpu_latency               | Adds delay to req/rsp paths                         | Lower = faster testing | |
 * bus_latency               | Models interconnect delay (2x per request)          | Realistic: 1-10 cycles | |
 * mem_latency               | Models memory access time                           | Realistic: 50-200 cycles | |
 * queue_size                | Affects backpressure frequency                      | Larger = less backpressure |
 *
 * **Usage Examples:**
 *
 * 1. Show all available options:
 *    ```bash
 *    ./testSimPort --help
 *    ```
 *
 * 2. Override single parameter:
 *    ```bash
 *    ./testSimPort --cpu-outstanding-requests 16
 *    ```
 *
 * 3. Override multiple parameters:
 *    ```bash
 *    ./testSimPort --cpu-latency 2 --bus-latency 5 --mem-latency 100
 *    ```
 *
 * 4. Stress test configuration:
 *    ```bash
 *    ./testSimPort --cpu-outstanding-requests 32 \
 *                  --cpu-total-requests 10000 \
 *                  --queue-size 16
 *    ```
 *
 * **Why Use CLI Arguments:**
 * - No recompilation needed to test different configurations
 * - Easy parameter sweeps for performance analysis
 * - Scripting and automation friendly
 * - Self-documenting via --help output
 *
 * **Best Practices:**
 * - Use descriptive flag names (--cpu-latency, not --cl)
 * - Provide clear, concise help text
 * - Set reasonable defaults that demonstrate key features
 * - Use default_str() to show defaults in help text
 * - Group related parameters together in code
 *
 * @note This method is called automatically during SimTop::init()
 * @note CLI arguments are parsed before registerSimulators() executes
 * @note Framework arguments (--threadmanager, --max-tick, etc.) are registered automatically
 */
void TestSimPortTop::registerCLIArguments() {
	this->getCLIApp()
	    ->add_option("--cpu-outstanding-requests", cpu_outstanding_requests,
	                 "Set the number of outstanding requests for the CPU.")
	    ->default_str(std::to_string(cpu_outstanding_requests));

	this->getCLIApp()
	    ->add_option("--cpu-latency", cpu_latency, "Set the CPU latency in cycles.")
	    ->default_str(std::to_string(cpu_latency));

	this->getCLIApp()
	    ->add_option("--bus-latency", bus_latency, "Set the bus latency in cycles.")
	    ->default_str(std::to_string(bus_latency));

	this->getCLIApp()
	    ->add_option("--mem-latency", mem_latency, "Set the memory latency in cycles.")
	    ->default_str(std::to_string(mem_latency));

	this->getCLIApp()
	    ->add_option("--queue-size", queue_size, "Set the Internal Queue Sizw.")
	    ->default_str(std::to_string(queue_size));

	this->getCLIApp()
	    ->add_option("--cpu-total-requests", cpu_total_requests, "Set the total number of requests for the CPU.")
	    ->default_str(std::to_string(cpu_total_requests));
}

}  // namespace test_port
