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
 * @file SimTop.hh
 * @brief Top-level simulation orchestration and system coordination
 *
 * SimTop provides the main entry point and orchestration layer for ACALSim simulations.
 * It manages the complete simulation lifecycle: initialization, execution, and cleanup,
 * coordinating multiple simulators, thread pools, configuration, and tracing systems.
 *
 * **System Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                         SimTop                              │
 * │  (Top-Level Orchestrator & Control Thread)                  │
 * ├─────────────────────────────────────────────────────────────┤
 * │  • Configuration Management (CLI + JSON files)              │
 * │  • Device & Channel Port Management                         │
 * │  • Global Clock & Synchronization                           │
 * │  • Tracing & Profiling Infrastructure                       │
 * └────────────┬───────────────────────────────┬────────────────┘
 *              │                               │
 *    ┌─────────▼────────┐           ┌─────────▼────────┐
 *    │  ThreadManager   │           │   TaskManager    │
 *    │  (Thread Pool)   │◄─────────►│  (Task Queues)   │
 *    └─────────┬────────┘           └──────────────────┘
 *              │
 *    ┌─────────┴─────────────────────────────┐
 *    │         │         │         │         │
 * ┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
 * │Sim 0│   │Sim 1│   │Sim 2│   │Sim 3│   │Sim N│
 * │(CPU)│   │(GPU)│   │(NoC)│   │(Mem)│   │ ... │
 * └─────┘   └─────┘   └─────┘   └─────┘   └─────┘
 * ```
 *
 * **Simulation Lifecycle:**
 * ```
 * User Code                    SimTop                           Simulators
 *    |                            |                                 |
 *    |-- init(argc, argv) ------->|                                 |
 *    |                            |-- initConfig() ---------------->|
 *    |                            |-- registerSimulators() -------->|
 *    |                            |-- preSimInitSetup() ----------->|
 *    |                            |-- threadManager->init() ------->|
 *    |                            |-- postSimInitSetup() ---------->|
 *    |                            |                                 |
 *    |-- run() ------------------>|                                 |
 *    |                            |-- startSimThreads() ----------->|
 *    |                            |                                 |
 *    |                            |== Main Simulation Loop =======  |
 *    |                            |  while (!allSimsDone) {         |
 *    |                            |    startPhase1() ------------->|
 *    |                            |    control_thread_step()        |
 *    |                            |    finishPhase1() <-----------|
 *    |                            |    startPhase2() ------------->|
 *    |                            |    runInterIterationUpdate()    |
 *    |                            |    finishPhase2() <-----------|
 *    |                            |    globalTick++ ────────────────►
 *    |                            |  }                              |
 *    |                            |                                 |
 *    |-- finish() --------------->|                                 |
 *    |                            |-- terminateAllThreads() ------->|
 *    |                            |-- postSimCleanup() ------------>|
 *    |                            |-- dumpTraces() ---------------->|
 *    |<---------------------------| (JSON/Chrome trace files)       |
 * ```
 *
 * **Key Features:**
 * - **Unified Orchestration**: Single entry point managing all simulation components
 * - **Multi-Simulator Coordination**: Manages heterogeneous simulators (CPUs, GPUs, NoCs, memory)
 * - **Thread Pool Management**: Delegates parallel execution to ThreadManager
 * - **Global Time Management**: Maintains a global clock synchronized across all simulators
 * - **Flexible Configuration**: Supports JSON config files and command-line arguments
 * - **Comprehensive Tracing**: Chrome Trace format and custom trace records
 * - **Device Management**: Register and route to devices by name or ID
 * - **Channel Communication**: Thread-safe inter-simulator message passing
 * - **Testing Support**: GTest bitmask utilities for validation
 *
 * **Use Cases:**
 *
 * | Use Case                      | Description                                             | Example |
 * |-------------------------------|---------------------------------------------------------|--------------------------------------|
 * | Multi-Core CPU Simulation     | Simulate CPU cores with shared memory                   | 4-core RISC-V processor
 * system      | | Heterogeneous System          | Combine CPU, GPU, NoC, and memory simulators            | SoC with
 * ARM CPU + NPU accelerator  | | NoC Architecture Evaluation   | Simulate network-on-chip with multiple routers |
 * Mesh/torus NoC performance analysis | | Memory Hierarchy Modeling     | Coordinate cache, memory controller, and DRAM
 * simulators| Multi-level cache coherence study   | | Performance Profiling         | Trace execution with Chrome Trace
 * viewer integration    | Identify bottlenecks in parallel sim|
 *
 * **Thread Safety:**
 * - SimTop runs in a dedicated control thread
 * - ThreadManager coordinates worker threads
 * - Channel ports provide thread-safe communication
 * - Atomic globalTick ensures consistent time view
 *
 * **Performance Characteristics:**
 *
 * | Operation                     | Complexity       | Notes                                  |
 * |-------------------------------|------------------|----------------------------------------|
 * | addSimulator()                | O(1)             | Register new simulator                 |
 * | getSimulator()                | O(1)             | HashMap lookup by name                 |
 * | setPendingEventBitMask()      | O(1)             | Atomic bit set operation               |
 * | addTraceRecord()              | O(1) amortized   | Queue append (lock-free)               |
 * | run() - per iteration         | O(N)             | N = number of simulators               |
 *
 * @see ThreadManager For thread pool and parallel execution details
 * @see SimBase For individual simulator implementation
 * @see ChannelPortManager For inter-simulator communication
 * @see DeviceManager For device registration and routing
 *
 * @example
 * @code
 * // Example 1: Basic multi-core CPU simulation
 * class MultiCoreCPUSimulation : public acalsim::SimTop {
 * public:
 *     MultiCoreCPUSimulation(const std::string& configFile)
 *         : SimTop(configFile) {}
 *
 *     void registerSimulators() override {
 *         // Create 4 CPU core simulators
 *         for (int i = 0; i < 4; i++) {
 *             auto* cpu = new CPUSimulator("CPU" + std::to_string(i), this);
 *             addSimulator(cpu);
 *         }
 *         // Create shared L3 cache
 *         auto* l3cache = new CacheSimulator("L3Cache", this);
 *         addSimulator(l3cache);
 *     }
 *
 *     void preSimInitSetup() override {
 *         // Configure interconnect before simulation starts
 *         connectSimulatorsInMesh();
 *     }
 *
 *     void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) override {
 *         threadManager = new ThreadManager("ThreadMgr", hw_nthreads, true);
 *         taskManager = new FIFOTaskManager("TaskMgr");
 *         threadManager->linkTaskManager(taskManager);
 *         taskManager->linkThreadManager(threadManager);
 *     }
 *
 *     void control_thread_step() override {
 *         // Dump statistics every 10K cycles
 *         if (getGlobalTick() % 10000 == 0) {
 *             dumpStatistics();
 *         }
 *     }
 * };
 *
 * int main(int argc, char** argv) {
 *     MultiCoreCPUSimulation sim("config.json");
 *     sim.init(argc, argv);
 *     sim.run();
 *     sim.finish();
 *     return 0;
 * }
 * @endcode
 *
 * @code
 * // Example 2: Heterogeneous SoC simulation with tracing
 * class SoCSimulation : public acalsim::SimTop {
 * public:
 *     SoCSimulation() : SimTop("soc_config.json", "soc_trace") {}
 *
 *     void registerSimulators() override {
 *         // ARM CPU cluster
 *         auto* cpuCluster = new ARMClusterSimulator("ARMCluster", this);
 *         addSimulator(cpuCluster);
 *         registerDevice("ARMCluster");
 *
 *         // NPU accelerator
 *         auto* npu = new NPUSimulator("NPU", this);
 *         addSimulator(npu);
 *         registerDevice("NPU");
 *
 *         // Network-on-Chip
 *         auto* noc = new NoCSimulator("NoC", this);
 *         addSimulator(noc);
 *
 *         // Memory controller
 *         auto* memCtrl = new MemoryController("MemCtrl", this);
 *         addSimulator(memCtrl);
 *         registerDevice("MemCtrl");
 *     }
 *
 *     void preSimInitSetup() override {
 *         // Setup address map for routing
 *         setupAddressMap("address_map.json");
 *     }
 *
 *     void control_thread_step() override {
 *         // Record parallelism degree for analysis
 *         size_t activeSimulators = countActiveSimulators();
 *         auto record = std::make_shared<ParallelismRecord>(getGlobalTick(), activeSimulators);
 *         addTraceRecord(record, "parallelism");
 *
 *         // Add Chrome trace for visualization
 *         auto chromeTrace = std::make_shared<ChromeTraceRecord>(
 *             "SoC", "iteration", getGlobalTick(), 1
 *         );
 *         addChromeTraceRecord(chromeTrace);
 *     }
 *
 *     void postSimInitSetup() override {
 *         // Warm up caches
 *         LOG(INFO) << "Running cache warm-up...";
 *     }
 *
 *     void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) override {
 *         threadManager = new ThreadManager("SoCThreadMgr", hw_nthreads, true);
 *         taskManager = new PriorityTaskManager("SoCTaskMgr");
 *         threadManager->linkTaskManager(taskManager);
 *         taskManager->linkThreadManager(threadManager);
 *     }
 * };
 *
 * int main(int argc, char** argv) {
 *     SoCSimulation sim;
 *     sim.init(argc, argv);
 *     sim.run();
 *     sim.finish();
 *     // Open soc_trace.json in Chrome trace viewer (chrome://tracing)
 *     return 0;
 * }
 * @endcode
 *
 * @code
 * // Example 3: Testing with GTest bitmasks
 * class TestableSimulation : public acalsim::SimTop {
 * public:
 *     TestableSimulation() : SimTop() {}
 *
 *     void registerSimulators() override {
 *         auto* cpu = new CPUSimulator("CPU0", this);
 *         addSimulator(cpu);
 *     }
 *
 *     void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) override {
 *         threadManager = new ThreadManager("TestThreadMgr", 1, false);
 *         taskManager = new FIFOTaskManager("TestTaskMgr");
 *         threadManager->linkTaskManager(taskManager);
 *         taskManager->linkThreadManager(threadManager);
 *     }
 *
 *     void control_thread_step() override {
 *         // Test checkpoint: check if CPU reached expected state
 *         if (getGlobalTick() == 1000) {
 *             // Set bit 0 to indicate checkpoint reached
 *             setGTestBitMask(0, 0);
 *         }
 *     }
 *
 *     void checkTestCondition() {
 *         // Verify bit was set correctly
 *         assert(checkGTestBitMask(0, 0x1) && "Checkpoint not reached!");
 *     }
 * };
 *
 * TEST(SimulationTest, CheckpointReached) {
 *     TestableSimulation sim;
 *     char* argv[] = {(char*)"test"};
 *     sim.init(1, argv);
 *     sim.run();
 *     sim.checkTestCondition();
 *     sim.finish();
 * }
 * @endcode
 *
 * @code
 * // Example 4: Dynamic configuration and device management
 * class ConfigurableSimulation : public acalsim::SimTop {
 * public:
 *     ConfigurableSimulation(const std::vector<std::string>& configs)
 *         : SimTop(configs, "configurable_trace") {}
 *
 *     void registerSimulators() override {
 *         // Read number of cores from configuration
 *         int numCores = getConfigInt("system.num_cores");
 *
 *         for (int i = 0; i < numCores; i++) {
 *             auto* cpu = new CPUSimulator("CPU" + std::to_string(i), this);
 *             addSimulator(cpu);
 *
 *             // Register each CPU as a device for routing
 *             int deviceId = registerDevice("CPU" + std::to_string(i));
 *             LOG(INFO) << "Registered CPU" << i << " with device ID " << deviceId;
 *         }
 *
 *         // Create memory controllers based on config
 *         int numMemCtrl = getConfigInt("memory.num_controllers");
 *         for (int i = 0; i < numMemCtrl; i++) {
 *             auto* mem = new MemoryController("MemCtrl" + std::to_string(i), this);
 *             addSimulator(mem);
 *             registerDevice("MemCtrl" + std::to_string(i));
 *         }
 *     }
 *
 *     void preSimInitSetup() override {
 *         // Load address map from configuration
 *         std::string addressMapFile = getConfigString("system.address_map");
 *         loadAddressMap(addressMapFile);
 *     }
 *
 *     void control_thread_step() override {
 *         // Monitor for termination condition
 *         if (getGlobalTick() > getConfigInt("simulation.max_ticks")) {
 *             LOG(WARNING) << "Maximum simulation time reached, terminating...";
 *             // Force termination by clearing all pending events
 *             for (size_t i = 0; i < getNumSimulators(); i++) {
 *                 clearPendingEventBitMask(i);
 *             }
 *         }
 *     }
 *
 *     void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) override {
 *         unsigned int threads = getConfigInt("threading.num_threads", hw_nthreads);
 *         threadManager = new ThreadManager("ConfigThreadMgr", threads, true);
 *
 *         std::string taskPolicy = getConfigString("threading.task_policy", "fifo");
 *         if (taskPolicy == "priority") {
 *             taskManager = new PriorityTaskManager("ConfigTaskMgr");
 *         } else {
 *             taskManager = new FIFOTaskManager("ConfigTaskMgr");
 *         }
 *
 *         threadManager->linkTaskManager(taskManager);
 *         taskManager->linkThreadManager(threadManager);
 *     }
 *
 *     void postSimInitSetup() override {
 *         LOG(INFO) << "Simulation configured with:";
 *         LOG(INFO) << "  - " << getNumSimulators() << " simulators";
 *         LOG(INFO) << "  - " << getNumThreads() << " worker threads";
 *         LOG(INFO) << "  - " << getNumDevices() << " registered devices";
 *     }
 *
 * private:
 *     int getConfigInt(const std::string& key, int defaultValue = 0) {
 *         // Retrieve integer from configContainer
 *         return configContainer->getInt(key, defaultValue);
 *     }
 *
 *     std::string getConfigString(const std::string& key, const std::string& defaultValue = "") {
 *         return configContainer->getString(key, defaultValue);
 *     }
 * };
 *
 * int main(int argc, char** argv) {
 *     // Load multiple configuration files
 *     std::vector<std::string> configs = {
 *         "base_config.json",
 *         "system_config.json",
 *         "memory_config.json"
 *     };
 *
 *     ConfigurableSimulation sim(configs);
 *     sim.init(argc, argv);
 *     sim.run();
 *     sim.finish();
 *     return 0;
 * }
 * @endcode
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// ACALSim
#include "channel/ChannelPortManager.hh"
#include "common/BitVector.hh"
#include "config/ACALSimConfig.hh"
#include "config/CLIManager.hh"
#include "container/ChromeTraceRecord.hh"
#include "container/JsonContainer.hh"
#include "container/SharedContainer.hh"
#include "container/SimTraceContainer.hh"
#include "sim/DeviceManager.hh"
#include "sim/PipeRegisterManager.hh"
#include "sim/ThreadManager.hh"
#include "utils/Logging.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

class RecycleContainer;
class SimBase;
class SimConfig;
class SimPacket;

#ifdef ACALSIM_STATISTICS
/**
 * @class ParallelismRecord
 * @brief Trace record for tracking parallel simulation activity
 *
 * Records the degree of parallelism at each simulation tick, useful for
 * analyzing how many simulators are actively processing events concurrently.
 *
 * @see SimTopBase::addTraceRecord()
 * @see SimTraceRecord
 */
class ParallelismRecord : public SimTraceRecord {
public:
	/**
	 * @brief Construct a parallelism trace record
	 * @param _tick Simulation time when parallelism is measured
	 * @param _degree Number of active simulators executing in parallel
	 */
	ParallelismRecord(Tick _tick, size_t _degree) : SimTraceRecord(), tick_(_tick), degree_(_degree) {}

	/**
	 * @brief Convert trace record to JSON format
	 * @return nlohmann::json JSON object with tick and parallel-degree fields
	 */
	nlohmann::json toJson() const override {
		nlohmann::json j     = nlohmann::json::object();
		j["tick"]            = this->tick_;
		j["parallel-degree"] = this->degree_;
		return j;
	}

private:
	Tick   tick_;    ///< Simulation tick when recorded
	size_t degree_;  ///< Number of simulators active in parallel
};
#endif  // ACALSIM_STATISTICS

/**
 * @class SimTopBase
 * @brief Abstract base class for top-level simulation orchestration
 *
 * SimTopBase provides the complete infrastructure for managing multi-simulator
 * parallel discrete-event simulations. It inherits functionality from CLIManager
 * (command-line parsing), DeviceManager (device registration), and ChannelPortManager
 * (inter-simulator communication).
 *
 * **Responsibilities:**
 * - Lifecycle management: initialization → execution → cleanup
 * - Thread pool coordination via ThreadManager
 * - Configuration loading from JSON and command-line arguments
 * - Global time management with atomic clock
 * - Trace collection (custom format and Chrome Trace)
 * - Device and channel port registration
 *
 * **Design Pattern:**
 * Template Method Pattern - defines simulation flow with customization hooks:
 * - registerSimulators(): Create simulator instances
 * - preSimInitSetup(): Configure before simulator init
 * - postSimInitSetup(): Configure after simulator init
 * - control_thread_step(): Execute per-iteration control logic
 * - connectTopChannelPorts(): Setup inter-simulator channels
 *
 * @see SimTop For concrete implementation with default behavior
 * @see ThreadManager For parallel execution management
 * @see SimBase For individual simulator base class
 */
class SimTopBase : public CLIManager, public DeviceManager, public ChannelPortManager, virtual public HashableType {
public:
	/**
	 * @brief Construct top-level simulation orchestrator
	 *
	 * @param _configFilePaths Vector of JSON configuration file paths (loaded in order)
	 * @param _tracingJsonFileName Base name for trace output files (e.g., "trace" → "trace.json")
	 *
	 * @note Configuration files are merged in order, later files override earlier values
	 */
	SimTopBase(const std::vector<std::string>& _configFilePaths, const std::string& _tracingJsonFileName = "trace");

	/**
	 * @brief Destructor - releases all resources and simulators
	 */
	virtual ~SimTopBase();

	/**
	 * @brief Initialize simulation with configuration and command-line arguments
	 *
	 * Performs complete pre-simulation setup:
	 * 1. Parse command-line arguments (initConfig)
	 * 2. Register simulators (registerSimulators)
	 * 3. User pre-init setup (preSimInitSetup)
	 * 4. Initialize thread manager and simulators
	 * 5. User post-init setup (postSimInitSetup)
	 *
	 * @param argc Number of command-line arguments
	 * @param argv Array of command-line argument strings
	 *
	 * @note Must be called before run()
	 * @see run(), finish()
	 */
	void init(int argc, char** argv);

	/**
	 * @brief Execute main simulation loop until all simulators complete
	 *
	 * Main iteration loop coordinates worker threads through two-phase execution:
	 * ```
	 * while (!allSimulatorsDone) {
	 *     startPhase1()             // Parallel event processing
	 *     control_thread_step()     // User control logic
	 *     finishPhase1()            // Synchronization barrier
	 *     startPhase2()             // Inter-simulator communication
	 *     runInterIterationUpdate() // Update shared state
	 *     finishPhase2()            // Synchronization barrier
	 *     globalTick++              // Advance time
	 * }
	 * ```
	 *
	 * @note Blocking call - returns only when simulation completes
	 * @see init(), finish(), control_thread_step()
	 */
	void run();

	/**
	 * @brief Clean up and finalize simulation
	 *
	 * Performs shutdown sequence:
	 * 1. Terminate all worker threads
	 * 2. Cleanup simulators (postSimCleanup)
	 * 3. Dump trace files (JSON and Chrome Trace formats)
	 * 4. Release resources
	 *
	 * @note Must be called after run() to ensure proper cleanup
	 * @see init(), run()
	 */
	void finish();

	/**
	 * @brief Fast-forward global clock to specified tick
	 *
	 * Directly sets global clock without executing intervening events.
	 * Useful for skipping initialization periods or fast-forwarding to
	 * regions of interest.
	 *
	 * @param t Target tick value to set
	 *
	 * @warning Use with caution - may cause time inconsistencies if
	 *          simulators have pending events before target tick
	 */
	void fastForwardGlobalTick(Tick t) { globalTick = t; }

	/**
	 * @brief Get current global simulation time
	 *
	 * @return Tick Current global clock value (atomic read)
	 *
	 * @note Thread-safe - can be called from any simulator
	 */
	Tick getGlobalTick() const { return globalTick; }

	/**
	 * @brief Retrieve simulator instance by name
	 *
	 * @param name Unique simulator name (assigned during registration)
	 * @return SimBase* Pointer to simulator, or nullptr if not found
	 *
	 * @note Thread-safe lookup via ThreadManager
	 * @see addSimulator()
	 */
	SimBase* getSimulator(std::string name) const { return this->threadManager->getSimulator(name); }

	/**
	 * @brief Set bit in GTest validation bitmask
	 *
	 * Used in testing to track simulation checkpoints or state transitions.
	 * Supports multiple independent bitmasks for different test conditions.
	 *
	 * @param which Bitmask index (0-N for multiple independent masks)
	 * @param id Bit position to set (0-63)
	 *
	 * @note Thread-safe via SharedContainer
	 * @see checkGTestBitMask(), clearGTestBitMask()
	 */
	void setGTestBitMask(int which, size_t id) { this->pGTestBitMask->run(which, &BitVector::setBit, id, true); }

	/**
	 * @brief Clear bit in GTest validation bitmask
	 *
	 * @param which Bitmask index (0-N for multiple independent masks)
	 * @param id Bit position to clear (0-63)
	 *
	 * @note Thread-safe via SharedContainer
	 * @see setGTestBitMask(), checkGTestBitMask()
	 */
	void clearGTestBitMask(int which, size_t id) { this->pGTestBitMask->run(0, &BitVector::setBit, id, false); }

	/**
	 * @brief Verify GTest bitmask matches expected value
	 *
	 * @param which Bitmask index to check
	 * @param value Expected 64-bit value to compare against
	 * @return bool True if bitmask equals value, false otherwise
	 *
	 * @note Useful for asserting simulation reached expected state in tests
	 */
	bool checkGTestBitMask(int which, uint64_t value) const {
		return this->pGTestBitMask->run(which, &BitVector::gTestBitMaskEqual, value);
	}

	/**
	 * @brief Record message packet trace event (Begin/End format)
	 *
	 * Adds trace event for message passing between simulators using
	 * Chrome Trace Event Format phase indicators ('B' for begin, 'E' for end).
	 *
	 * @param _source Source simulator or module name
	 * @param _dest Destination simulator or module name
	 * @param _tensorId Tensor identifier for ML workloads (0 if N/A)
	 * @param _packetId Unique packet identifier
	 * @param _name Event name (e.g., "MemoryRequest", "ComputeTask")
	 * @param _ph Phase indicator: 'B' (begin) or 'E' (end)
	 * @param _ts Timestamp when event occurred
	 *
	 * @note Thread-safe - queued for dump at simulation end
	 * @see addMsgPktJsonEvent() for complete event format
	 */
	void addMsgPktJsonEvent(std::string _source, std::string _dest, uint64_t _tensorId, uint64_t _packetId,
	                        std::string _name, std::string _ph, Tick _ts) {
		this->pMsgPktJsonContainer->add(_source, _dest, _tensorId, _packetId, _name, _ph, _ts);
	}

	/**
	 * @brief Record message packet trace event (Complete format)
	 *
	 * Adds trace event for message passing using Chrome Trace 'X' (complete)
	 * format, which includes both start time and duration in single event.
	 *
	 * @param _source Source simulator or module name
	 * @param _dest Destination simulator or module name
	 * @param _tensorId Tensor identifier for ML workloads (0 if N/A)
	 * @param _packetId Unique packet identifier
	 * @param _name Event name (e.g., "DMATransfer")
	 * @param _ts Timestamp when transfer started
	 * @param _dur Duration of the transfer (in ticks)
	 *
	 * @note More efficient than separate 'B'/'E' events for visualization
	 */
	void addMsgPktJsonEvent(std::string _source, std::string _dest, uint64_t _tensorId, uint64_t _packetId,
	                        std::string _name, Tick _ts, Tick _dur) {
		this->pMsgPktJsonContainer->add(_source, _dest, _tensorId, _packetId, _name, _ts, _dur);
	}

	/**
	 * @brief Check if worker threads are active
	 *
	 * @return bool True if simulation threads are running, false otherwise
	 *
	 * @note Useful for verifying simulation state before issuing commands
	 */
	bool isRunning() const { return this->threadManager->isRunning(); }

	/**
	 * @brief Check if all simulators have no pending events
	 *
	 * @return bool True if no simulators have pending events, false otherwise
	 *
	 * @note Used internally to detect simulation completion
	 * @see isPendingEventBitMaskSet()
	 */
	bool isPendingEventBitMaskZero() const { return this->threadManager->isPendingEventBitMaskZero(); }

	/**
	 * @brief Check if specific simulator has pending events
	 *
	 * @param id Simulator index (0 to N-1)
	 * @return bool True if simulator has events in queue or channels, false otherwise
	 *
	 * @note Each simulator sets its bit when events are queued
	 * @see setPendingEventBitMask(), clearPendingEventBitMask()
	 */
	bool isPendingEventBitMaskSet(size_t id) const { return this->threadManager->isPendingEventBitMaskSet(id); }

	/**
	 * @brief Mark simulator as having pending events
	 *
	 * @param id Simulator index (0 to N-1)
	 *
	 * @note Simulators call this when new events arrive
	 * @see clearPendingEventBitMask()
	 */
	void setPendingEventBitMask(size_t id) { this->threadManager->setPendingEventBitMask(id); }

	/**
	 * @brief Mark simulator as having no pending events
	 *
	 * @param id Simulator index (0 to N-1)
	 *
	 * @note Simulators call this when event queue becomes empty
	 * @see setPendingEventBitMask()
	 */
	void clearPendingEventBitMask(size_t id) { this->threadManager->clearPendingEventBitMask(id); }

	/**
	 * @brief Check if simulation can terminate
	 *
	 * Returns true when all simulators have finished processing and no
	 * events remain in queues or channels.
	 *
	 * @return bool True if safe to terminate, false otherwise
	 *
	 * @note run() loop exits when this returns true
	 */
	bool isReadyToTerminate() const { return this->readyToTerminate; }

	/**
	 * @brief Get GTest bitmask value as 64-bit integer
	 *
	 * @param which Bitmask index (0-N for multiple independent masks)
	 * @return uint64_t Current bitmask value (0x0000000000000000 to 0xFFFFFFFFFFFFFFFF)
	 *
	 * @note Useful for logging or batch verification in tests
	 * @see setGTestBitMask(), checkGTestBitMask()
	 */
	uint64_t getGTestBitMask(int which) const { return this->pGTestBitMask->run(which, &BitVector::getGTestBitMask); }

	/**
	 * @brief Get shared recycling container for event objects
	 *
	 * @return std::shared_ptr<RecycleContainer> Shared pointer to recycle container
	 *
	 * @note Used by simulators to recycle event objects and reduce allocations
	 */
	std::shared_ptr<RecycleContainer> getRecycleContainer() const { return this->recycleContainer; }

	/**
	 * @brief Register simulator with thread manager
	 *
	 * Adds simulator to the managed pool and assigns it to a worker thread.
	 * Must be called during registerSimulators() override.
	 *
	 * @param sim Pointer to heap-allocated simulator (ownership transferred)
	 *
	 * @note Simulator name must be unique
	 * @see registerSimulators(), getSimulator()
	 */
	void addSimulator(SimBase* sim);

	/**
	 * @brief Configure bidirectional channel ports to/from top-level
	 *
	 * Sets up thread-safe communication channels between SimTop control thread
	 * and individual simulators running on worker threads.
	 *
	 * @param _name Channel identifier (unique name)
	 * @param _toTopChannelPort Slave port for receiving data from simulators
	 * @param _fromTopChannelPort Master port for sending data to simulators
	 *
	 * @see connectTopChannelPorts()
	 */
	void setTopChannelPort(const std::string& _name, SlaveChannelPort::SharedPtr _toTopChannelPort,
	                       MasterChannelPort::SharedPtr _fromTopChannelPort);

	/**
	 * @brief Handle inbound notification from channels
	 *
	 * SimTop does not process inbound channel packets directly - simulators
	 * handle their own inbound notifications.
	 *
	 * @note This override throws an error if called
	 * @see ChannelPortManager::handleInboundNotification()
	 */
	void handleInboundNotification() override {
		CLASS_ERROR
		    << "SimTop::handleInboundNotification() is not implemented for updating its status upon receiving inbound "
		       "packets.";
	}

	/**
	 * @brief Add custom trace record with automatic timestamp
	 *
	 * Queues trace record using current global tick. Traces are dumped to
	 * JSON file at simulation end, organized by category.
	 *
	 * @param _trace Shared pointer to custom trace record (derived from SimTraceRecord)
	 * @param _category Category label for organizing traces (e.g., "cache", "memory", "cpu")
	 *
	 * @note Thread-safe - uses lock-free queue internally
	 * @see addTraceRecord(const std::shared_ptr<SimTraceRecord>&, const std::string&, const Tick&)
	 */
	void addTraceRecord(const std::shared_ptr<SimTraceRecord>& _trace, const std::string& _category);

	/**
	 * @brief Add custom trace record with explicit timestamp
	 *
	 * Queues trace record with specified tick for ordering. Useful when logging
	 * events that occurred at specific times different from current global tick.
	 *
	 * @param _trace Shared pointer to custom trace record (derived from SimTraceRecord)
	 * @param _category Category label for organizing traces
	 * @param _tick Explicit simulation tick for chronological ordering
	 *
	 * @note All traces are sorted by tick before dumping to file
	 * @see SimTraceRecord::toJson()
	 */
	void addTraceRecord(const std::shared_ptr<SimTraceRecord>& _trace, const std::string& _category, const Tick& _tick);

	/**
	 * @brief Add Chrome Trace event for visualization
	 *
	 * Queues trace event compatible with Chrome's trace viewer (chrome://tracing).
	 * Supports 'B' (begin), 'E' (end), 'X' (complete), and other trace event phases.
	 *
	 * @param _trace Shared pointer to ChromeTraceRecord with timing and metadata
	 *
	 * @note Generates JSON file viewable in chrome://tracing for visual analysis
	 * @see ChromeTraceRecord, addMsgPktJsonEvent()
	 */
	void addChromeTraceRecord(const std::shared_ptr<ChromeTraceRecord>& _trace);

	/**
	 * @brief Control thread iteration hook (pure virtual)
	 *
	 * Called once per simulation iteration while simulators execute Phase 1.
	 * Override to implement custom control logic such as:
	 * - Periodic statistics dumping
	 * - Progress monitoring
	 * - Dynamic configuration updates
	 * - Checkpoint creation
	 * - Early termination conditions
	 *
	 * @note Executes in control thread, not simulator threads
	 * @note Do not perform heavy computation here - affects simulation throughput
	 */
	virtual void control_thread_step() = 0;

	/**
	 * @brief Create and register all simulator instances (pure virtual)
	 *
	 * Derived classes must implement this to instantiate simulators and
	 * register them using addSimulator(). Called during init() sequence.
	 *
	 * @note Simulators should be heap-allocated (ownership transferred to SimTop)
	 * @see addSimulator()
	 */
	virtual void registerSimulators() = 0;

	/**
	 * @brief Setup channel ports between SimTop and simulators (pure virtual)
	 *
	 * Create and connect bidirectional channel ports for thread-safe communication
	 * between control thread and worker threads.
	 *
	 * @note Called during init() after registerSimulators()
	 * @see setTopChannelPort()
	 */
	virtual void connectTopChannelPorts() = 0;

	/**
	 * @brief Initialize thread manager with specific version (pure virtual)
	 *
	 * Derived classes must create ThreadManager and TaskManager instances,
	 * configure them, and link them together.
	 *
	 * @param version ThreadManager implementation version (V1-V8)
	 * @param hw_nthreads Number of hardware threads available
	 *
	 * @note ThreadManagerV8 recommended for best performance
	 * @see ThreadManager, TaskManager
	 */
	virtual void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) = 0;

	/**
	 * @brief Synchronize all simulator ports
	 *
	 * Forces synchronization of all port communications across simulators.
	 * Called automatically during Phase 2 of each iteration.
	 */
	void runSyncSimPort();

	/**
	 * @brief Register pipeline registers for top-level coordination
	 *
	 * Override to create custom PipeRegisterManager for managing
	 * pipeline register state across simulators.
	 *
	 * @note Default creates basic PipeRegisterManager
	 */
	virtual void registerPipeRegisters() {
		this->setPipeRegisterManager(new PipeRegisterManager("Top-Level Pipe Register Manager"));
	}

	/**
	 * @brief Set pipeline register manager
	 * @param manager Pointer to PipeRegisterManager (ownership transferred)
	 */
	void setPipeRegisterManager(PipeRegisterManagerBase* manager) { pipeRegisterManager = manager; }

	/**
	 * @brief Get pipeline register manager
	 * @return PipeRegisterManagerBase* Pointer to current manager
	 */
	PipeRegisterManagerBase* getPipeRegisterManager() { return pipeRegisterManager; }

protected:
	/**
	 * @brief Parse configuration files and command-line arguments
	 *
	 * Internal initialization method that:
	 * 1. Creates framework and user configuration objects
	 * 2. Parses command-line arguments (via CLIManager)
	 * 3. Loads JSON configuration files
	 * 4. Merges configurations (CLI args override file settings)
	 *
	 * @param argc Number of command-line arguments
	 * @param argv Array of command-line argument strings
	 *
	 * @note Called automatically during init()
	 */
	void initConfig(int argc, char** argv);

	/**
	 * @brief Get number of registered simulators
	 * @return size_t Total simulator count
	 */
	size_t getNumSimulators() const { return threadManager->getNumSimulators(); }

	/**
	 * @brief Get number of worker threads
	 * @return size_t Thread pool size
	 */
	size_t getNumThreads() const { return threadManager->getNumThreads(); }

	/// @brief Centralized thread pool coordinator
	ThreadManagerBase* threadManager = nullptr;

	/// @brief Centralized task queue manager
	TaskManager* taskManager = nullptr;

	/// @brief Top-level pipeline register state manager
	PipeRegisterManagerBase* pipeRegisterManager = nullptr;

	/**
	 * @brief Global simulation clock (atomic for thread-safe reads)
	 *
	 * Read-only for simulators, writable only by SimTop control thread.
	 * Incremented after each iteration in run() loop.
	 */
	std::atomic<Tick> globalTick = 0;

	/// @brief Bit vector for GTest validation and checkpoint tracking
	std::shared_ptr<SharedContainer<BitVector>> pGTestBitMask = nullptr;

	/// @brief Thread-safe container for message packet trace events
	std::shared_ptr<JsonContainer<MessagePacketJsonEvent>> pMsgPktJsonContainer = nullptr;

	/// @brief Shared configuration container accessible to all simulators
	std::shared_ptr<SimConfig> configContainer = nullptr;

	/// @brief Shared object recycling container to reduce allocations
	std::shared_ptr<RecycleContainer> recycleContainer = nullptr;

	/// @brief Thread-safe container for custom trace records
	SharedContainer<SimTraceContainer> traceCntr;

	/// @brief Termination flag - true when all simulators finished
	bool readyToTerminate;

	/// @brief Slave channel ports for receiving from simulators (indexed by name)
	std::unordered_map<std::string, SlaveChannelPort::SharedPtr> toTopChannelPorts;

	/// @brief Master channel ports for sending to simulators (indexed by name)
	std::unordered_map<std::string, MasterChannelPort::SharedPtr> fromTopChannelPorts;

	/**
	 * @brief Pre-initialization setup hook (pure virtual)
	 *
	 * Called after registerSimulators() but before simulator init().
	 * Override to perform system-wide configuration:
	 * - Load additional configuration files
	 * - Configure address maps
	 * - Setup interconnect topology
	 * - Allocate shared resources
	 *
	 * @note Executes before simulators are initialized
	 * @see postSimInitSetup()
	 */
	virtual void preSimInitSetup() = 0;

	/**
	 * @brief Post-initialization setup hook (pure virtual)
	 *
	 * Called after all simulators have initialized but before run() starts.
	 * Override to perform final configuration:
	 * - Verify simulator connections
	 * - Warm up caches
	 * - Load initial state
	 * - Log configuration summary
	 *
	 * @note Executes after simulators are initialized
	 * @see preSimInitSetup()
	 */
	virtual void postSimInitSetup() = 0;

#ifdef ACALSIM_STATISTICS
private:
	/// @brief Cumulative time spent in Phase 1 (parallel execution) in microseconds
	double timer_phase1_us = 0;

	/// @brief Cumulative time spent in Phase 2 (communication) in microseconds
	double timer_phase2_us = 0;

	/// @brief Cumulative time spent synchronizing pipeline registers in microseconds
	double simpipereg_cost_us = 0;

	/// @brief Cumulative time spent synchronizing ports in microseconds
	double simport_cost_us = 0;

	/// @brief Cumulative time spent in inter-iteration updates in microseconds
	double inter_iter_update_cost_us = 0;

	/// @brief Cumulative time spent computing next tick in microseconds
	double get_next_tick_cost = 0;

	/// @brief Statistics for task execution time distribution across workers
	CategorizedStatistics<size_t, double, StatisticsMode::AccumulatorWithSize, true> tasks_time_dist_statistics;
#endif  // ACALSIM_STATISTICS
};

/**
 * @class SimTop
 * @brief Concrete top-level simulation orchestrator with default implementations
 *
 * SimTop provides a concrete implementation of SimTopBase with sensible defaults
 * for all pure virtual methods. Derived classes can override only the methods
 * they need to customize.
 *
 * **Default Behaviors:**
 * - registerSimulators(): No-op (derived class must override to add simulators)
 * - preSimInitSetup(): No-op (optional customization)
 * - postSimInitSetup(): No-op (optional customization)
 * - control_thread_step(): No-op (optional customization)
 * - connectTopChannelPorts(): Empty implementation (override if channels needed)
 *
 * **Typical Usage Pattern:**
 * ```cpp
 * class MySimulation : public acalsim::SimTop {
 * public:
 *     MySimulation() : SimTop("config.json") {}
 *
 *     void registerSimulators() override {
 *         // Only need to override methods actually used
 *         auto* cpu = new CPUSimulator("CPU0", this);
 *         addSimulator(cpu);
 *     }
 *
 *     void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) override {
 *         threadManager = new ThreadManager("TM", hw_nthreads, true);
 *         taskManager = new FIFOTaskManager("FM");
 *         threadManager->linkTaskManager(taskManager);
 *         taskManager->linkThreadManager(threadManager);
 *     }
 * };
 * ```
 *
 * @see SimTopBase For full API documentation
 */
class SimTop : public SimTopBase {
public:
	/**
	 * @brief Construct SimTop with single configuration file
	 *
	 * @param _configFilePath Path to JSON configuration file (empty string = no config)
	 * @param _tracingFileName Base name for trace output files
	 *
	 * @note Convenience constructor for single config file
	 * @see SimTop(const std::vector<std::string>&, const std::string&)
	 */
	SimTop(const std::string& _configFilePath = "", const std::string& _tracingFileName = "trace")
	    : SimTopBase(_configFilePath == "" ? std::vector<std::string>({}) : std::vector<std::string>({_configFilePath}),
	                 _tracingFileName) {}

	/**
	 * @brief Construct SimTop with multiple configuration files
	 *
	 * Configuration files are loaded and merged in order. Later files override
	 * values from earlier files, enabling layered configuration.
	 *
	 * @param _configFilePaths Vector of JSON configuration file paths
	 * @param _tracingFileName Base name for trace output files
	 *
	 * @example
	 * @code
	 * std::vector<std::string> configs = {
	 *     "base_config.json",      // Base system parameters
	 *     "cpu_config.json",       // CPU-specific overrides
	 *     "experiment_config.json" // Experiment-specific settings
	 * };
	 * SimTop sim(configs, "experiment_trace");
	 * @endcode
	 */
	SimTop(const std::vector<std::string>& _configFilePaths, const std::string& _tracingFileName = "trace")
	    : SimTopBase(_configFilePaths, _tracingFileName) {}

	/**
	 * @brief Destructor - default behavior (base class handles cleanup)
	 */
	virtual ~SimTop() = default;

	/**
	 * @brief Connect top-level channel ports (final implementation)
	 *
	 * Default implementation is empty. Override in derived class if
	 * SimTop needs to communicate with simulators via channels.
	 *
	 * @note Marked final - cannot be overridden further
	 */
	void connectTopChannelPorts() final;

	/**
	 * @brief Control thread iteration hook (default: no-op)
	 *
	 * Override to add per-iteration control logic. Default does nothing.
	 *
	 * @note Override only if needed - empty implementation has minimal overhead
	 */
	void control_thread_step() override {}

	/**
	 * @brief Register simulators (default: no-op)
	 *
	 * **Must be overridden** to add simulators via addSimulator().
	 * Default implementation registers no simulators.
	 *
	 * @warning Simulation will fail if no simulators are registered
	 */
	void registerSimulators() override {}

	/**
	 * @brief Pre-initialization setup (default: no-op)
	 *
	 * Override to perform setup before simulator initialization.
	 * Default does nothing.
	 */
	void preSimInitSetup() override {}

	/**
	 * @brief Post-initialization setup (default: no-op)
	 *
	 * Override to perform setup after simulator initialization.
	 * Default does nothing.
	 */
	void postSimInitSetup() override {}

	/**
	 * @brief Initialize thread and task managers (final implementation)
	 *
	 * Creates default ThreadManager and TaskManager configuration.
	 * Marked final to ensure consistent thread management setup.
	 *
	 * @param version ThreadManager version to use (V1-V8)
	 * @param hw_nthreads Number of hardware threads available
	 *
	 * @note Override in derived class if custom thread management needed
	 */
	void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) final;
};

/**
 * @brief Global shared pointer to top-level simulation instance
 *
 * Provides global access to SimTop for simulators and modules.
 * Initialized by SimTop constructor.
 *
 * @note Use with caution - prefer passing SimTop pointer explicitly
 */
extern std::shared_ptr<SimTopBase> top;

}  // end of namespace acalsim
