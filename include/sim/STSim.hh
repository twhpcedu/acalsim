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

#pragma once

#include "SimBase.hh"
#include "SimTop.hh"

namespace acalsim {

/**
 * @file STSim.hh
 * @brief Single-threaded simulation infrastructure with simplified API
 *
 * @details
 * STSim provides a streamlined interface for building single-threaded (ST)
 * simulations without the complexity of multi-threaded synchronization. It's
 * designed for simpler models, prototyping, and cases where parallelism isn't
 * needed or beneficial.
 *
 * **Single-Threaded Simulation Model:**
 * ```
 * Simulation Loop (Single Thread)
 *     |
 *     |--- init() → registerModules() → simInit()
 *     |
 *     |--- Loop until termination:
 *     |       |
 *     |       |--- step() → Process events, update state
 *     |       |--- Advance simulation time
 *     |       |
 *     |--- cleanup() → Finalize, dump stats
 *     |
 * ```
 *
 * **Key Features:**
 *
 * - **Simplified API**: Fewer abstractions than multi-threaded simulation
 * - **Template-based**: Type-safe simulator access via STSim<YourSimulator>
 * - **Single Simulator**: One simulator instance per STSim (no parallel execution)
 * - **Module Support**: Integrate SimModule-based components
 * - **Configuration**: Inherits SimTop configuration capabilities
 * - **No Thread Overhead**: Sequential execution without synchronization costs
 *
 * **Comparison: STSim vs Multi-Threaded Simulation:**
 *
 * | Feature | STSim | Multi-Threaded (SimTop) |
 * |---------|-------|-------------------------|
 * | **Execution** | Sequential | Parallel workers |
 * | **Complexity** | Low | High (synchronization) |
 * | **Use Case** | Simple models, prototypes | Complex systems, performance |
 * | **Overhead** | Minimal | Thread management |
 * | **Debugging** | Easy (single thread) | Challenging (race conditions) |
 * | **Scalability** | 1 core | Multi-core |
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Benefits |
 * |----------|-------------|----------|
 * | **Prototyping** | Quick model development | Simple API, fast iteration |
 * | **Educational** | Teaching simulation concepts | Easier to understand |
 * | **Small Models** | Systems with few components | No parallelism overhead |
 * | **Debugging** | Isolate complex simulation issues | Deterministic execution |
 * | **Sequential Logic** | Models with strict dependencies | Natural execution model |
 *
 * **Inheritance Hierarchy:**
 * ```
 * SimTop (multi-simulator management)
 *    ↑
 *    |
 * STSim<T> (single-threaded wrapper)
 *    |
 *    |--- creates and manages one instance of T
 *    |
 *    ↓
 * STSimBase (base class for user simulators)
 *    ↑
 *    |
 * CPPSimBase (C++ simulation base)
 * ```
 *
 * **Lifecycle:**
 * ```
 * 1. Construction:  STSim<MySimulator> sim("MySim", "config.json")
 * 2. Registration:  registerSimulators() → Creates MySimulator instance
 * 3. Initialization: init() → registerModules() → simInit()
 * 4. Simulation:    run() → Repeatedly calls step() until termination
 * 5. Cleanup:       cleanup() → Finalize and dump statistics
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | init() | Varies | User-defined module registration |
 * | step() | Varies | User-defined simulation logic |
 * | getSimulator() | O(1) | Type-safe cast |
 * | addModule() | O(1) | Delegates to SimBase |
 *
 * @code{.cpp}
 * // Example: Simple CPU simulator with STSim
 * class SimpleCPU : public STSimBase {
 * public:
 *     SimpleCPU(const std::string& name = "SimpleCPU")
 *         : STSimBase(name), pc(0), cycles(0) {}
 *
 *     void simInit() override {
 *         LOG_INFO << "SimpleCPU initialized";
 *         pc = 0;
 *         cycles = 0;
 *     }
 *
 *     void registerModules() override {
 *         // Register any modules needed
 *         // (In this simple example, we have none)
 *     }
 *
 *     void step() override {
 *         // Execute one instruction per step
 *         executeInstruction(memory[pc]);
 *         pc++;
 *         cycles++;
 *
 *         // Termination condition
 *         if (pc >= programSize) {
 *             scheduleExit(currentTick());
 *         }
 *     }
 *
 *     void cleanup() override {
 *         LOG_INFO << "Simulation complete";
 *         LOG_INFO << "Total cycles: " << cycles;
 *         LOG_INFO << "IPC: " << (double)instructions / cycles;
 *     }
 *
 * private:
 *     uint64_t pc;
 *     uint64_t cycles;
 *     uint64_t instructions;
 * };
 *
 * // Usage
 * int main() {
 *     // Create single-threaded simulation
 *     STSim<SimpleCPU> sim("CPUSimulation", "config.json");
 *
 *     // Get typed simulator reference
 *     SimpleCPU* cpu = sim.getSimulator();
 *
 *     // Run simulation
 *     sim.run();
 *
 *     return 0;
 * }
 *
 * // Example: Cache simulator with modules
 * class CacheSimulator : public STSimBase {
 * public:
 *     CacheSimulator(const std::string& name = "CacheSimulator")
 *         : STSimBase(name) {}
 *
 *     void registerModules() override {
 *         // Create and register L1 cache module
 *         l1Cache = new L1CacheModule("L1Cache");
 *         this->addModule(l1Cache);
 *
 *         // Create and register L2 cache module
 *         l2Cache = new L2CacheModule("L2Cache");
 *         this->addModule(l2Cache);
 *     }
 *
 *     void simInit() override {
 *         LOG_INFO << "Cache hierarchy initialized";
 *         // Connect L1 to L2
 *         l1Cache->connectToL2(l2Cache);
 *     }
 *
 *     void step() override {
 *         // Process memory requests
 *         if (!requestQueue.empty()) {
 *             MemoryRequest* req = requestQueue.front();
 *             l1Cache->handleRequest(req);
 *             requestQueue.pop();
 *         }
 *     }
 *
 * private:
 *     L1CacheModule* l1Cache;
 *     L2CacheModule* l2Cache;
 *     std::queue<MemoryRequest*> requestQueue;
 * };
 *
 * // Example: Using STSim addModule() convenience method
 * int main() {
 *     STSim<CacheSimulator> sim("CacheSim");
 *
 *     // Can add modules directly through STSim
 *     sim.addModule(new TraceCollectorModule("Tracer"));
 *     sim.addModule(new StatisticsModule("Stats"));
 *
 *     sim.run();
 *     return 0;
 * }
 * @endcode
 *
 * @note STSim manages exactly one simulator instance
 * @note For multi-simulator or parallel execution, use SimTop directly
 * @note Template parameter must derive from STSimBase
 *
 * @warning Single-threaded execution only - no parallelism
 * @warning getSimulator() performs unchecked cast - ensure type correctness
 *
 * @see STSimBase for base class to derive your simulator from
 * @see SimTop for multi-threaded simulation support
 * @see CPPSimBase for C++ simulation base class
 * @since ACALSim 0.1.0
 */

/**
 * @class STSimBase
 * @brief Base class for single-threaded simulation implementations
 *
 * @details
 * STSimBase extends CPPSimBase to provide a simplified interface for
 * single-threaded simulators. Users derive from this class and implement
 * the virtual methods to define their simulation behavior.
 *
 * **Initialization Sequence:**
 * ```
 * init()
 *   ├─→ registerModules()    [User implements: Register SimModules]
 *   └─→ simInit()            [User implements: Custom initialization]
 * ```
 *
 * **Simulation Loop:**
 * ```
 * while (!terminated) {
 *     step();    [User implements: One simulation iteration]
 * }
 * cleanup();     [User implements: Finalization]
 * ```
 *
 * @note All methods are virtual - override as needed
 * @note Default implementations do nothing (no-op)
 */
class STSimBase : public CPPSimBase {
public:
	/**
	 * @brief Construct a single-threaded simulation base
	 *
	 * @param name Simulator name for identification
	 *
	 * @note Delegates to CPPSimBase constructor
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * class MySimulator : public STSimBase {
	 * public:
	 *     MySimulator() : STSimBase("MySimulator") {}
	 * };
	 * @endcode
	 */
	STSimBase(const std::string& name = "STSimBase") : CPPSimBase(name) {}

	/**
	 * @brief Virtual destructor for safe polymorphic deletion
	 */
	virtual ~STSimBase() {}

	/**
	 * @brief Initialize the simulator (DO NOT OVERRIDE)
	 *
	 * @details
	 * This wrapper method orchestrates initialization by calling:
	 * 1. registerModules() - Register simulation modules
	 * 2. simInit() - Custom initialization logic
	 *
	 * @note Override simInit() for custom initialization, not this method
	 * @note Called automatically by simulation framework
	 *
	 * @code{.cpp}
	 * // Do NOT override init() - it's the initialization orchestrator
	 * // Instead, override simInit():
	 *
	 * class MySimulator : public STSimBase {
	 *     void simInit() override {
	 *         LOG_INFO << "Custom initialization";
	 *     }
	 * };
	 * @endcode
	 */
	void init() override {
		this->registerModules();
		simInit();
	}

	/**
	 * @brief Custom initialization hook (override this)
	 *
	 * @details
	 * Called after registerModules() during init() sequence.
	 * Implement simulator-specific initialization here.
	 *
	 * @note Default implementation does nothing
	 * @note Called exactly once before simulation starts
	 *
	 * @code{.cpp}
	 * class CPUSimulator : public STSimBase {
	 *     void simInit() override {
	 *         pc = 0x1000;           // Reset program counter
	 *         registers.fill(0);     // Clear registers
	 *         loadProgram("app.bin");// Load program
	 *     }
	 * };
	 * @endcode
	 */
	virtual void simInit() {}

	/**
	 * @brief Register simulation modules (override this)
	 *
	 * @details
	 * Called first in init() sequence. Use this to create and register
	 * SimModule instances that are part of your simulation.
	 *
	 * @note Default implementation does nothing
	 * @note Use addModule() to register modules with the simulator
	 *
	 * @code{.cpp}
	 * class SystemSimulator : public STSimBase {
	 *     void registerModules() override {
	 *         cpu = new CPUModule("CPU");
	 *         this->addModule(cpu);
	 *
	 *         memory = new MemoryModule("Memory");
	 *         this->addModule(memory);
	 *     }
	 *
	 * private:
	 *     CPUModule* cpu;
	 *     MemoryModule* memory;
	 * };
	 * @endcode
	 */
	virtual void registerModules() override {}

	/**
	 * @brief Execute one simulation step (override this)
	 *
	 * @details
	 * Called repeatedly by the simulation loop. Implement the logic
	 * for one simulation iteration (e.g., one cycle, one event).
	 *
	 * @note Default implementation does nothing
	 * @note Called in a loop until termination condition
	 *
	 * @code{.cpp}
	 * class InstructionSimulator : public STSimBase {
	 *     void step() override {
	 *         // Fetch instruction
	 *         Instruction inst = memory[pc];
	 *
	 *         // Decode and execute
	 *         execute(inst);
	 *
	 *         // Update PC
	 *         pc += inst.size;
	 *
	 *         // Check termination
	 *         if (inst.isHalt()) {
	 *             scheduleExit(currentTick());
	 *         }
	 *     }
	 * };
	 * @endcode
	 */
	virtual void step() override {}

	/**
	 * @brief Post-simulation cleanup (override this)
	 *
	 * @details
	 * Called after simulation completes. Use this to finalize state,
	 * dump statistics, close files, etc.
	 *
	 * @note Default implementation does nothing
	 * @note Called exactly once after simulation terminates
	 *
	 * @code{.cpp}
	 * class StatefulSimulator : public STSimBase {
	 *     void cleanup() override {
	 *         LOG_INFO << "Total cycles: " << cycles;
	 *         LOG_INFO << "Instructions: " << instructions;
	 *
	 *         // Dump statistics
	 *         stats.dump("simulation_stats.json");
	 *
	 *         // Close trace file
	 *         traceFile.close();
	 *     }
	 * };
	 * @endcode
	 */
	virtual void cleanup() override {}
};

/**
 * @class STSim
 * @brief Template wrapper for single-threaded simulation with type-safe access
 *
 * @tparam STSimBaseType User's simulator class (must derive from STSimBase)
 *
 * @details
 * STSim is a template class that simplifies single-threaded simulation setup.
 * It automatically creates and manages one instance of your simulator class,
 * providing type-safe access without manual casting.
 *
 * **Template Constraints:**
 * - STSimBaseType must derive from STSimBase
 * - One simulator instance per STSim
 *
 * **Advantages over SimTop:**
 * - Type-safe getSimulator() returns STSimBaseType*
 * - Simplified API for single-simulator case
 * - Automatic simulator instantiation
 * - Convenience methods like addModule()
 *
 * @code{.cpp}
 * // Define your simulator
 * class MySimulator : public STSimBase {
 *     // ... simulator implementation
 * };
 *
 * // Use STSim template
 * STSim<MySimulator> sim("MySim", "config.json");
 *
 * // Type-safe access - no casting needed
 * MySimulator* mySim = sim.getSimulator();
 * mySim->myCustomMethod();
 *
 * // Run simulation
 * sim.run();
 * @endcode
 *
 * @note STSim<T> inherits from SimTop but uses only single-simulator features
 * @note For multiple simulators or parallel execution, use SimTop directly
 */
template <class STSimBaseType>
class STSim : public SimTop {
	/** @brief Simulator name for identification */
	std::string name;

public:
	/**
	 * @brief Construct a single-threaded simulation wrapper
	 *
	 * @param _name Simulator name (default: "STSim")
	 * @param _configFile Path to configuration file (default: "")
	 *
	 * @note Inherits SimTop's configuration loading capabilities
	 * @note Simulator instance not created until registerSimulators() called
	 *
	 * @code{.cpp}
	 * // Create with name only
	 * STSim<MyCPU> sim1("CPUSim");
	 *
	 * // Create with name and config
	 * STSim<MyCPU> sim2("CPUSim", "cpu_config.json");
	 * @endcode
	 */
	STSim(std::string _name = "STSim", std::string _configFile = "") : SimTop(_configFile), name(_name) {}

	/**
	 * @brief Get type-safe simulator pointer
	 *
	 * @param _name Simulator name (default: "STSim")
	 * @return STSimBaseType* Pointer to simulator instance
	 *
	 * @note Returns typed pointer - no manual casting required
	 * @note Unchecked cast - ensure simulator exists and type matches
	 *
	 * @code{.cpp}
	 * STSim<MySimulator> sim("Sim");
	 * MySimulator* mySim = sim.getSimulator();
	 * mySim->customMethod();  // Type-safe access
	 * @endcode
	 */
	STSimBaseType* getSimulator(std::string _name = "STSim") { return (STSimBaseType*)SimTop::getSimulator(_name); }

	/**
	 * @brief Create and register simulator instance (called by framework)
	 *
	 * @details
	 * Automatically called during initialization. Creates one instance
	 * of STSimBaseType and registers it with SimTop.
	 *
	 * @note Do not call manually - framework invokes during setup
	 * @note Creates exactly one simulator instance
	 *
	 * @code{.cpp}
	 * // Framework automatically calls this:
	 * // 1. Creates: new STSimBaseType(name)
	 * // 2. Registers: addSimulator(instance)
	 * @endcode
	 */
	void registerSimulators() override {
		// Create simulator
		SimBase* stSimBase = (SimBase*)new STSimBaseType(name);

		// register Simulators
		this->addSimulator(stSimBase);
	}

	/**
	 * @brief Convenience method to add module to simulator
	 *
	 * @param module Pointer to SimModule to add
	 *
	 * @note Asserts that simulator is registered before adding module
	 * @note Simplifies module addition - no need to get simulator first
	 *
	 * @code{.cpp}
	 * STSim<MySimulator> sim("Sim");
	 *
	 * // Instead of:
	 * // sim.getSimulator()->addModule(module);
	 *
	 * // Just do:
	 * sim.addModule(new MyModule("Module"));
	 * @endcode
	 */
	void addModule(SimModule* module) {
		SimBase* sim = this->getSimulator();
		CLASS_ASSERT_MSG(sim, "The Simulator is not registered in STSim yet!!");
		sim->addModule(module);
	}

	/**
	 * @brief Hook for registering configuration options
	 *
	 * @note Default implementation does nothing
	 * @note Override to add configuration parameters
	 *
	 * @code{.cpp}
	 * class MySTSim : public STSim<MySimulator> {
	 *     void registerConfigs() override {
	 *         configMgr.addConfig("cache_size", 64 * 1024);
	 *         configMgr.addConfig("num_cores", 4);
	 *     }
	 * };
	 * @endcode
	 */
	virtual void registerConfigs() {}
};

}  // end of namespace acalsim
