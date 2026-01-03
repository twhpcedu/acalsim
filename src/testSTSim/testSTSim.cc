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
 * @file testSTSim.cc
 * @brief Single-threaded simulation template demonstrating pure C++ simulator creation with ACALSim
 *
 * @details
 * This file provides a minimal template for creating single-threaded discrete-event simulations
 * using the ACALSim framework without SystemC dependencies. It demonstrates the fundamental
 * three-step pattern for building custom simulators using pure C++ event-driven simulation.
 *
 * ## Overview
 *
 * Single-threaded simulators (STSim) provide a lightweight alternative to SystemC-based
 * simulations when hardware-level timing accuracy is not required. This approach is ideal for:
 * - High-level architectural modeling
 * - Software performance simulation
 * - Algorithm validation
 * - Rapid prototyping without SystemC overhead
 *
 * ## Architecture
 *
 * The simulation framework follows a hierarchical architecture:
 *
 * ```
 * STSim<MySimulator>           (Top-level simulation manager)
 *     |
 *     +-- MySimulator          (Custom simulator, inherits STSimBase)
 *         |
 *         +-- DummyModule      (Custom module, inherits SimModule)
 *         +-- [Other modules]
 * ```
 *
 * ## STSimBase vs. SimBase
 *
 * **STSimBase** (Single-Threaded Simulation Base):
 * - Pure C++ event-driven simulation
 * - No SystemC dependencies
 * - Discrete event scheduling using ACALSim's event queue
 * - Lightweight timing model based on abstract "ticks"
 * - Suitable for high-level modeling
 * - Single execution thread
 *
 * **SimBase** (General Simulation Base):
 * - Base class for all simulator types
 * - Can be extended for SystemC integration (SCSimBase)
 * - Supports multi-threaded simulation
 * - Provides port-based communication infrastructure
 *
 * ## Three-Step Implementation Pattern
 *
 * ### Step 1: Create Custom Module Classes
 *
 * Modules represent individual simulation components (e.g., CPU cores, caches, memory).
 * Each module inherits from SimModule and implements required lifecycle methods:
 *
 * ```cpp
 * class DummyModule : public SimModule {
 * public:
 *     DummyModule(std::string name) : SimModule(name) {}
 *     void init() override {
 *         // Initialize module state
 *     }
 * };
 * ```
 *
 * ### Step 2: Create Custom Simulator Class
 *
 * The simulator class inherits from STSimBase and orchestrates module execution:
 *
 * ```cpp
 * class MySimulator : public STSimBase {
 * public:
 *     MySimulator(std::string name) : STSimBase(name) {}
 *
 *     void registerModules() override {
 *         // Create and register all simulation modules
 *         this->addModule(new DummyModule("module1"));
 *     }
 *
 *     void simInit() override {
 *         // Pre-simulation initialization
 *     }
 *
 *     void step() override {
 *         // Execute one simulation cycle
 *     }
 *
 *     void cleanup() override {
 *         // Post-simulation cleanup
 *     }
 * };
 * ```
 *
 * ### Step 3: Instantiate and Run Simulation
 *
 * The main function creates the top-level simulation object and executes the standard
 * three-phase simulation lifecycle:
 *
 * ```cpp
 * int main(int argc, char** argv) {
 *     // Create top-level simulation manager
 *     top = std::make_shared<STSim<MySimulator>>();
 *
 *     // Phase 1: Initialize (parse args, build modules, connect components)
 *     top->init(argc, argv);
 *
 *     // Phase 2: Run (execute main simulation loop)
 *     top->run();
 *
 *     // Phase 3: Finalize (collect statistics, cleanup resources)
 *     top->finish();
 *
 *     return 0;
 * }
 * ```
 *
 * ## Simulation Lifecycle
 *
 * The simulation progresses through three distinct phases:
 *
 * **1. Initialization Phase (init())**:
 * - Parse command-line arguments
 * - Call registerModules() to create simulation hierarchy
 * - Connect modules via ports (if using port-based communication)
 * - Call simInit() and module init() methods
 * - Prepare event queue and initial state
 *
 * **2. Execution Phase (run())**:
 * - Execute main simulation loop
 * - Repeatedly call step() method
 * - Process events in chronological order
 * - Continue until termination condition met
 * - Update global simulation time (tick counter)
 *
 * **3. Finalization Phase (finish())**:
 * - Call cleanup() method
 * - Generate statistics and reports
 * - Close trace files
 * - Free allocated resources
 *
 * ## When to Use STSim vs. SystemC
 *
 * **Use STSim (this template) when**:
 * - Modeling software or high-level architecture
 * - Cycle-accurate timing not required
 * - Avoiding SystemC compilation overhead
 * - Simple event-driven behavior sufficient
 * - Rapid prototyping needed
 *
 * **Use SystemC (testSTSystemC) when**:
 * - Modeling hardware components (RTL-level)
 * - Cycle-accurate or pin-accurate simulation needed
 * - Integrating existing SystemC IP blocks
 * - Hardware/software co-simulation required
 * - Clock-driven synchronous logic modeling
 *
 * ## Code Example: Building a Simple Cache Simulator
 *
 * ```cpp
 * class CacheModule : public SimModule {
 * private:
 *     int hits = 0;
 *     int misses = 0;
 *     std::unordered_map<uint64_t, int> cache;
 *
 * public:
 *     CacheModule(std::string name) : SimModule(name) {}
 *
 *     void init() override {
 *         INFO << "Initializing cache";
 *     }
 *
 *     bool access(uint64_t addr) {
 *         if (cache.count(addr)) {
 *             hits++;
 *             return true;
 *         }
 *         misses++;
 *         cache[addr] = 1;
 *         return false;
 *     }
 *
 *     void printStats() {
 *         INFO << "Hit Rate: " << (float)hits / (hits + misses);
 *     }
 * };
 *
 * class CacheSimulator : public STSimBase {
 * private:
 *     CacheModule* cache;
 *
 * public:
 *     CacheSimulator(std::string name) : STSimBase(name) {}
 *
 *     void registerModules() override {
 *         cache = new CacheModule("L1Cache");
 *         this->addModule(cache);
 *     }
 *
 *     void simInit() override {
 *         INFO << "Starting cache simulation";
 *     }
 *
 *     void step() override {
 *         // Generate random memory access
 *         uint64_t addr = rand() % 1000;
 *         cache->access(addr);
 *     }
 *
 *     void cleanup() override {
 *         cache->printStats();
 *     }
 * };
 * ```
 *
 * ## Event-Driven Simulation Model
 *
 * STSim uses a discrete event simulation model where:
 * - Time advances in discrete steps (ticks)
 * - Events are scheduled at specific tick values
 * - The step() method processes events for current tick
 * - Events can schedule future events
 *
 * This differs from SystemC's process-based simulation model which supports:
 * - Multiple concurrent processes
 * - Wait states and sensitivities
 * - Delta cycle evaluation
 * - Hierarchical module instantiation
 *
 * ## Key Classes and Methods
 *
 * **SimModule**:
 * - Base class for all simulation modules
 * - Provides init() lifecycle hook
 * - Supports hierarchical naming
 * - Logging and debug infrastructure
 *
 * **STSimBase**:
 * - Base class for single-threaded simulators
 * - Manages module registry
 * - Provides simulation lifecycle hooks:
 *   - registerModules(): Module instantiation
 *   - simInit(): Pre-simulation setup
 *   - step(): Per-cycle execution
 *   - cleanup(): Post-simulation teardown
 *
 * **STSim<T>**:
 * - Template class parameterized by simulator type
 * - Implements top-level simulation control flow
 * - Provides init(), run(), finish() interface
 * - Manages global simulation state
 *
 * ## Global Variables
 *
 * **top**: Shared pointer to the top-level simulation object
 * - Accessible from any module or simulator
 * - Provides access to global tick counter
 * - Used for event scheduling and synchronization
 *
 * ## Extending This Template
 *
 * To build a custom simulator:
 *
 * 1. Define module classes inheriting from SimModule
 * 2. Define simulator class inheriting from STSimBase
 * 3. Implement registerModules() to create module hierarchy
 * 4. Implement step() for per-cycle behavior
 * 5. Add custom configuration via command-line arguments
 * 6. Add statistics collection in cleanup()
 *
 * ## Related Files
 *
 * For SystemC-based simulation examples, see:
 * - testSTSystemC.cc: SystemC integration example
 * - MacSim.cc: SystemC MAC layer simulation
 * - TGSim.cc: Traffic generator with SystemC
 *
 * @see SimModule
 * @see STSimBase
 * @see STSim
 * @see testSTSystemC.cc for SystemC integration
 * @see SCSimBase for SystemC-based simulators
 */

/* --------------------------------------------------------------------------------------
 *  A test template to demonstrate how to create your own single-thread simulator using this framework
 *  Step 1. Inherit SimModule to create your own module class
 *  Step 2. Inherit STSimBase to create your own simulator class
 *  Step 3. instantiate a top-level simulation instance and call the following APIs in turn
 *          1) STSim::init(); //Pre-Simulation Initialization
 *          2) STSim::run();  //Simulation main loop
 *          3) STSim::finish(); // Post-Simulation cleanup
 * --------------------------------------------------------------------------------------*/

#include "ACALSim.hh"
using namespace acalsim;

// Step 1 include header files of the module classes or declare your own module class here
class DummyModule : public SimModule {
public:
	DummyModule(std::string name) : SimModule(name) { CLASS_INFO << "Constructing DummyModule " + name; }
	~DummyModule() {}

	void init() override { ; }
};

// Step 2. Inherit STSimBase to create your own simulator class.
// Override virtual functions for the simulator implementation

class MySimulator : public STSimBase {
public:
	MySimulator(std::string name = "My simulator") : STSimBase(name) {}
	void registerModules() override {
		DummyModule* module = new DummyModule("TestModule");
		this->addModule(module);
	}

	void simInit() override { CLASS_INFO << "MySimulator::simInit() " + name; }

	void step() override { CLASS_INFO << "MySimulator::simStep() " + name; }

	void cleanup() override { CLASS_INFO << "MySimulator::cleanup() " + name; }
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	// Remember 1) to cast the top-level instance to the SimTop* type and set it to the global variable top
	// 2) Pass your own simulator class type to the STSim class template
	top = std::make_shared<STSim<MySimulator>>();
	top->init(argc, argv);
	top->run();
	top->finish();

	return 0;
}
