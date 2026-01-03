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

#include "common/LinkManager.hh"
#include "event/SimEvent.hh"
#include "packet/SimPacket.hh"
#include "port/SimPortManager.hh"
#include "utils/Logging.hh"

/**
 * @file SimModule.hh
 * @brief Modular building blocks for component-based simulation architecture
 *
 * @details
 * SimModule provides the foundation for creating reusable, composable simulation
 * components in ACALSim. Inspired by gem5's SimObject design, it enables modular
 * construction of complex systems from independent, interconnected components.
 *
 * **Component-Based Architecture:**
 * ```
 * SimBase (Simulator)
 *    |
 *    |--- SimModule (CPU)
 *    |       |--- MasterPort → SlavePort → SimModule (L1 Cache)
 *    |       |--- SimEvent scheduling
 *    |       |--- step() for each cycle
 *    |
 *    |--- SimModule (L1 Cache)
 *    |       |--- MasterPort → SlavePort → SimModule (L2 Cache)
 *    |       |--- Packet handling via accept()
 *    |
 *    |--- SimModule (L2 Cache)
 *            |--- MasterPort → SlavePort → SimModule (Memory)
 * ```
 *
 * **Key Features:**
 *
 * - **Modular Design**: Self-contained components with clear interfaces
 * - **Port Management**: Inherits SimPortManager for communication
 * - **Link Management**: Inherits LinkManager for module connections
 * - **Event Scheduling**: Wrapper methods for event-driven simulation
 * - **Packet Handling**: Visitor pattern for type-safe packet processing
 * - **Hierarchical ID**: Tracks both simulator ID and module ID
 * - **Lifecycle Hooks**: init(), step(), cleanup patterns
 *
 * **Use Cases:**
 *
 * | Component Type | Description | Example |
 * |----------------|-------------|---------|
 * | **Processing Units** | Computational components | CPU core, GPU shader, NPU |
 * | **Memory Hierarchy** | Storage and caching | L1/L2 cache, DRAM controller |
 * | **Interconnect** | Communication infrastructure | Router, crossbar, bus |
 * | **Peripherals** | I/O and external devices | DMA, disk controller, NIC |
 * | **Custom Logic** | Domain-specific components | Crypto accelerator, DSP |
 *
 * **Inheritance and Composition:**
 * ```
 * SimPortManager (port communication)
 *    ↑
 *    |
 * LinkManager<SimModule*> (module connections)
 *    ↑
 *    |
 * SimModule (base component)
 *    ↑
 *    |
 * ├── CPUModule (processor implementation)
 * ├── CacheModule (memory hierarchy)
 * ├── RouterModule (NoC routing)
 * └── Custom modules...
 * ```
 *
 * **Module Lifecycle:**
 * ```
 * 1. Construction:  new MyModule("ModuleName")
 * 2. Registration:  simulator->addModule(module)
 * 3. ID Assignment: setSimID(), setID()
 * 4. Initialization: init() → Setup ports, connections
 * 5. Simulation:    step() called each cycle/event
 * 6. Packet Flow:   accept() → Process incoming packets
 * 7. Events:        scheduleEvent() → Async operations
 * ```
 *
 * **Communication Patterns:**
 *
 * | Pattern | Mechanism | Use Case |
 * |---------|-----------|----------|
 * | **Synchronous** | step() invocation | Clock-driven logic |
 * | **Asynchronous** | Event scheduling | Delayed responses |
 * | **Message Passing** | Packet accept() | Data transfers |
 * | **Port Connection** | Master/Slave ports | Point-to-point links |
 * | **Module Linking** | LinkManager | Arbitrary connections |
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | Construction | O(1) | ID assignment, name copy |
 * | scheduleEvent() | O(log n) | Event queue insertion |
 * | accept() | O(1) | Visitor pattern dispatch |
 * | step() | Varies | User-defined logic |
 *
 * @code{.cpp}
 * // Example: Simple cache module
 * class SimpleCacheModule : public SimModule {
 * public:
 *     SimpleCacheModule(const std::string& name, size_t size)
 *         : SimModule(name), cacheSize(size), hits(0), misses(0) {}
 *
 *     void init() override {
 *         LOG_INFO << getName() << ": Initializing cache of size " << cacheSize;
 *
 *         // Create ports for communication
 *         cpuPort = createSlavePort("cpu_side");
 *         memPort = createMasterPort("mem_side");
 *
 *         // Initialize cache data structures
 *         cache.resize(cacheSize);
 *     }
 *
 *     void accept(Tick when, SimPacket& pkt) override {
 *         // Handle incoming memory requests
 *         if (auto* memReq = dynamic_cast<MemoryRequest*>(&pkt)) {
 *             handleMemoryRequest(when, memReq);
 *         }
 *     }
 *
 *     void step() override {
 *         // Process pending requests each cycle
 *         if (!requestQueue.empty()) {
 *             processNextRequest();
 *         }
 *     }
 *
 * private:
 *     void handleMemoryRequest(Tick when, MemoryRequest* req) {
 *         uint64_t addr = req->getAddress();
 *
 *         if (cache.contains(addr)) {
 *             // Cache hit - respond immediately
 *             hits++;
 *             auto* response = new MemoryResponse(req);
 *             scheduleEvent(response, when + hitLatency);
 *         } else {
 *             // Cache miss - forward to memory
 *             misses++;
 *             memPort->send(req);
 *         }
 *     }
 *
 *     size_t cacheSize;
 *     std::unordered_map<uint64_t, CacheLine> cache;
 *     SlavePort* cpuPort;
 *     MasterPort* memPort;
 *     uint64_t hits, misses;
 * };
 *
 * // Example: CPU module with event-driven execution
 * class CPUModule : public SimModule {
 * public:
 *     CPUModule(const std::string& name)
 *         : SimModule(name), pc(0), cycles(0) {}
 *
 *     void init() override {
 *         LOG_INFO << "CPU " << getName() << " initialized";
 *
 *         // Create memory port
 *         memPort = createMasterPort("mem");
 *
 *         // Schedule first instruction fetch
 *         auto* fetchEvent = new FetchEvent(this);
 *         scheduleEvent(fetchEvent, currentTick() + 1);
 *     }
 *
 *     void step() override {
 *         cycles++;
 *
 *         // Execute pipeline stages
 *         if (executeReady) {
 *             executeStage();
 *         }
 *         if (decodeReady) {
 *             decodeStage();
 *         }
 *     }
 *
 *     void accept(Tick when, SimPacket& pkt) override {
 *         // Handle memory responses
 *         if (auto* resp = dynamic_cast<MemoryResponse*>(&pkt)) {
 *             handleMemoryResponse(when, resp);
 *         }
 *     }
 *
 * private:
 *     uint64_t pc;
 *     uint64_t cycles;
 *     MasterPort* memPort;
 *     bool executeReady = false;
 *     bool decodeReady = false;
 * };
 *
 * // Example: Building a system with modules
 * class SimpleSystem {
 * public:
 *     SimpleSystem(SimBase* sim) {
 *         // Create modules
 *         cpu = new CPUModule("CPU0");
 *         l1Cache = new SimpleCacheModule("L1Cache", 32 * 1024);
 *         l2Cache = new SimpleCacheModule("L2Cache", 256 * 1024);
 *         memory = new MemoryModule("DRAM");
 *
 *         // Register modules with simulator
 *         sim->addModule(cpu);
 *         sim->addModule(l1Cache);
 *         sim->addModule(l2Cache);
 *         sim->addModule(memory);
 *
 *         // Initialize modules
 *         cpu->init();
 *         l1Cache->init();
 *         l2Cache->init();
 *         memory->init();
 *
 *         // Connect modules via ports
 *         cpu->getMasterPort("mem")->connect(l1Cache->getSlavePort("cpu_side"));
 *         l1Cache->getMasterPort("mem_side")->connect(l2Cache->getSlavePort("cpu_side"));
 *         l2Cache->getMasterPort("mem_side")->connect(memory->getSlavePort("requests"));
 *     }
 *
 * private:
 *     CPUModule* cpu;
 *     SimpleCacheModule* l1Cache;
 *     SimpleCacheModule* l2Cache;
 *     MemoryModule* memory;
 * };
 * @endcode
 *
 * @note SimModule is an abstract base class - derive to implement components
 * @note Module name is immutable after construction
 * @note IDs are assigned by simulator during registration
 *
 * @warning Do not manually delete modules - simulator manages lifetime
 * @warning Ensure init() is called before simulation starts
 * @warning Port connections must be established during initialization
 *
 * @see SimPortManager for port communication interface
 * @see LinkManager for module connection management
 * @see SimPacket for packet-based communication
 * @since ACALSim 0.1.0
 */

/**
 * @defgroup PortManagement
 * @brief Functions for managing ports.
 */

namespace acalsim {

// Forward declarations for circular dependencies
class SimModule;
class SimBase;

/**
 * @class SimModule
 * @brief Base class for modular simulation components
 *
 * @details
 * SimModule extends SimPortManager and LinkManager to provide a complete
 * infrastructure for building simulation components. Inspired by gem5's
 * SimObject, it enables hierarchical, modular system construction.
 *
 * **Design Principles:**
 * - **Single Responsibility**: Each module handles one logical component
 * - **Clear Interfaces**: Communication via ports and events
 * - **Composability**: Modules combine to form complex systems
 * - **Reusability**: Generic modules work across different simulations
 *
 * @note Modules are managed by SimBase - do not delete manually
 * @note Name is const - cannot be changed after construction
 */
class SimModule : public SimPortManager, public LinkManager<SimModule*> {
	/** @brief ID of the SimBase simulator this module belongs to */
	int simID;

	/** @brief Unique module ID within the simulator */
	int id;

	/** @brief Immutable module name for identification and logging */
	const std::string name;

	/** @brief Pointer to parent simulator (not owned) */
	SimBase* simulator = nullptr;

public:
	/**
	 * @brief Construct a simulation module with a name
	 *
	 * @param _name Module name (default: "anonymous")
	 *
	 * @note Name is immutable after construction
	 * @note Logs verbose construction message
	 * @note Does not assign IDs - done by simulator during registration
	 *
	 * @code{.cpp}
	 * class MyCPU : public SimModule {
	 * public:
	 *     MyCPU() : SimModule("CPU0") {}
	 * };
	 *
	 * auto* cpu = new MyCPU();  // Logs: "Constructing SimModule CPU0"
	 * @endcode
	 */
	SimModule(std::string _name = "anonymous") : LinkManager<SimModule*>(), SimPortManager(_name), name(_name) {
		VERBOSE_CLASS_INFO << "Constructing SimModule " + _name;
	}

	/**
	 * @brief Virtual destructor for safe polymorphic deletion
	 *
	 * @note Typically not called directly - simulator manages lifetime
	 */
	virtual ~SimModule() {}

	/**
	 * @brief Set the ID of the simulator this module belongs to
	 *
	 * @param id Simulator ID
	 *
	 * @note Called by simulator during module registration
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * // Typically called by simulator:
	 * module->setSimID(simulator->getID());
	 * @endcode
	 */
	void setSimID(int id) { simID = id; }

	/**
	 * @brief Get the ID of the simulator this module belongs to
	 *
	 * @return int Simulator ID
	 *
	 * @note Used for tracking which simulator owns this module
	 * @note Useful in multi-simulator scenarios
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * int simID = module->getSimID();
	 * LOG_DEBUG << "Module " << module->getName()
	 *          << " belongs to simulator " << simID;
	 * @endcode
	 */
	int getSimID() const { return simID; }

	/**
	 * @brief Get the module name
	 *
	 * @return std::string Module name
	 *
	 * @note Name is immutable - set at construction
	 * @note Used for logging, debugging, and visualization
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * LOG_INFO << module->getName() << " processing request";
	 * @endcode
	 */
	std::string getName() const { return this->name; }

	/**
	 * @brief Set the unique module ID
	 *
	 * @param i Module ID (unique within simulator)
	 *
	 * @note Called by simulator during registration
	 * @note IDs are typically assigned sequentially
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * // Simulator assigns sequential IDs:
	 * module->setID(nextModuleID++);
	 * @endcode
	 */
	void setID(int i) { this->id = i; }

	/**
	 * @brief Get the unique module ID
	 *
	 * @return int Module ID (unique within simulator)
	 *
	 * @note Used for module identification and indexing
	 * @note Useful for tracking in arrays or maps
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * int moduleID = module->getID();
	 * moduleStats[moduleID].recordAccess();
	 * @endcode
	 */
	int getID() const { return this->id; }

	/**
	 * @brief Set the parent simulator pointer
	 *
	 * @param _simlator Pointer to parent SimBase
	 *
	 * @note Called by simulator during registration
	 * @note Enables module to access simulator services
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * // Simulator sets itself as parent:
	 * module->setSimulator(this);
	 * @endcode
	 */
	void setSimulator(SimBase* _simlator) { this->simulator = _simlator; }

	/**
	 * @brief Get the parent simulator pointer
	 *
	 * @return SimBase* Pointer to parent simulator
	 *
	 * @note Returns nullptr if not yet registered
	 * @note Used to access simulator global state
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * SimBase* sim = module->getSimulator();
	 * Tick now = sim->currentTick();
	 * @endcode
	 */
	SimBase* getSimulator() const { return this->simulator; }

	/**
	 * @brief Schedule an event for future execution
	 *
	 * @param event Event to schedule
	 * @param when Absolute tick when event should fire
	 *
	 * @note Wrapper that delegates to parent simulator
	 * @note Event will be executed at specified tick
	 * @note Complexity: O(log n) - event queue insertion
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * class MyModule : public SimModule {
	 *     void handleRequest() {
	 *         // Schedule response 10 cycles later
	 *         auto* response = new ResponseEvent();
	 *         scheduleEvent(response, currentTick() + 10);
	 *     }
	 * };
	 * @endcode
	 */
	void scheduleEvent(SimEvent* event, Tick when);

	/**
	 * @brief Module-level initialization hook
	 *
	 * @details
	 * Called once before simulation starts. Override to implement
	 * module-specific initialization: create ports, establish
	 * connections, allocate resources, etc.
	 *
	 * @note Default implementation does nothing
	 * @note Called after setSimulator(), setID(), setSimID()
	 *
	 * @code{.cpp}
	 * class CacheModule : public SimModule {
	 *     void init() override {
	 *         // Create ports
	 *         cpuPort = createSlavePort("cpu");
	 *         memPort = createMasterPort("mem");
	 *
	 *         // Initialize data structures
	 *         cache.resize(cacheSize);
	 *
	 *         LOG_INFO << getName() << " initialized";
	 *     }
	 * };
	 * @endcode
	 */
	virtual void init() { ; }

	/**
	 * @brief Accept incoming packet using visitor pattern
	 *
	 * @param when Tick when packet arrives
	 * @param pkt Reference to incoming packet
	 *
	 * @details
	 * Uses visitor pattern to dispatch packet to type-specific handler.
	 * Default implementation calls pkt.visit(when, *this), allowing
	 * packet to invoke appropriate overloaded method on this module.
	 *
	 * @note Override for custom packet handling
	 * @note Complexity: O(1) - virtual dispatch
	 *
	 * @code{.cpp}
	 * class MemoryModule : public SimModule {
	 *     void accept(Tick when, SimPacket& pkt) override {
	 *         if (auto* req = dynamic_cast<MemoryRequest*>(&pkt)) {
	 *             handleMemoryRequest(when, req);
	 *         } else if (auto* resp = dynamic_cast<MemoryResponse*>(&pkt)) {
	 *             handleMemoryResponse(when, resp);
	 *         } else {
	 *             SimModule::accept(when, pkt);  // Default visitor
	 *         }
	 *     }
	 * };
	 * @endcode
	 */
	virtual void accept(Tick when, SimPacket& pkt) { pkt.visit(when, *this); }

	/**
	 * @brief Execute one simulation step
	 *
	 * @details
	 * Called each simulation cycle/iteration. Override to implement
	 * module's per-cycle behavior: pipeline stages, state updates,
	 * request processing, etc.
	 *
	 * @note Default implementation does nothing
	 * @note Called repeatedly during simulation loop
	 * @note Complexity varies by implementation
	 *
	 * @code{.cpp}
	 * class PipelinedCPU : public SimModule {
	 *     void step() override {
	 *         // Execute pipeline stages each cycle
	 *         writebackStage();
	 *         memoryStage();
	 *         executeStage();
	 *         decodeStage();
	 *         fetchStage();
	 *     }
	 * };
	 * @endcode
	 */
	virtual void step() {}

	/**
	 * @brief Callback when master port wins arbitration
	 *
	 * @param port Pointer to master port that won
	 *
	 * @details
	 * Called when a master port successfully arbitrates for access.
	 * Override to retry failed transactions or send queued requests.
	 *
	 * @note Default implementation does nothing
	 * @note Inherits from SimPortManager
	 *
	 * @code{.cpp}
	 * class RequestingModule : public SimModule {
	 *     void masterPortRetry(MasterPort* port) override {
	 *         // Retry previously failed request
	 *         if (!pendingRequests.empty()) {
	 *             auto* req = pendingRequests.front();
	 *             if (port->send(req)) {
	 *                 pendingRequests.pop();
	 *             }
	 *         }
	 *     }
	 *
	 * private:
	 *     std::queue<Request*> pendingRequests;
	 * };
	 * @endcode
	 */
	void masterPortRetry(MasterPort* port) override {}
};

}  // end of namespace acalsim
