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
 * @file PETile.cc
 * @brief Processing Element (PE) tile simulator with configuration management and module orchestration
 *
 * This file implements the PETile class, which serves as the main simulator orchestrator for
 * a single PE tile architecture. It demonstrates configuration parameter handling, module
 * instantiation, and interconnection topology setup for tile-based computing systems.
 *
 * **PETile Role in Tile-Based Architecture:**
 * ```
 * ┌───────────────────────────────────────────────────────────────────┐
 * │                      PETile Simulator                             │
 * │                    (Orchestration Layer)                          │
 * │                                                                   │
 * │  Configuration Management:                                        │
 * │  ┌─────────────────────────────────────────────────────────────┐ │
 * │  │ PETile Constructor                                          │ │
 * │  │ ├─ Load configuration from JSON                             │ │
 * │  │ ├─ Parse primitive types (int, float, string, Tick)         │ │
 * │  │ ├─ Parse complex types (CacheStruct, BusStruct)             │ │
 * │  │ └─ Log all configuration parameters                         │ │
 * │  └─────────────────────────────────────────────────────────────┘ │
 * │                                                                   │
 * │  Module Registration & Interconnection:                           │
 * │  ┌─────────────────────────────────────────────────────────────┐ │
 * │  │ registerModules()                                           │ │
 * │  │ ├─ Create CPUTraffic (Processing Element)                   │ │
 * │  │ ├─ Create AXI Bus (Interconnect)                            │ │
 * │  │ ├─ Create SRAM (Local Memory)                               │ │
 * │  │ ├─ Register modules with simulator                          │ │
 * │  │ └─ Connect modules (upstream/downstream ports)              │ │
 * │  └─────────────────────────────────────────────────────────────┘ │
 * │                                                                   │
 * │  Simulation Lifecycle:                                            │
 * │  ┌─────────────────────────────────────────────────────────────┐ │
 * │  │ simInit()  → Initialize all child modules                   │ │
 * │  │ step()     → Per-tick simulation step (currently no-op)     │ │
 * │  │ cleanup()  → Post-simulation cleanup                        │ │
 * │  └─────────────────────────────────────────────────────────────┘ │
 * └───────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **PE Tile Internal Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────┐
 * │                     Single PE Tile                         │
 * │                                                            │
 * │  ┌──────────────┐                                         │
 * │  │  CPUTraffic  │ ← pcu (Processing/Computation Unit)     │
 * │  │    (PCU)     │                                         │
 * │  └──────┬───────┘                                         │
 * │         │ Req/Resp                                        │
 * │         ▼                                                 │
 * │  ┌──────────────┐                                         │
 * │  │   AXI Bus    │ ← bus (Interconnect Network)           │
 * │  │ (Interconnect│                                         │
 * │  └──────┬───────┘                                         │
 * │         │ Req/Resp                                        │
 * │         ▼                                                 │
 * │  ┌──────────────┐                                         │
 * │  │     SRAM     │ ← pcuMem (Local Private Memory)        │
 * │  │ (Local Mem)  │                                         │
 * │  └──────────────┘                                         │
 * │                                                            │
 * │  Port Connections:                                         │
 * │    PCU ──["Bus"]──> AXI Bus ──["PCUMem"]──> SRAM          │
 * │    PCU <──["PCU"]── AXI Bus <──["Bus"]──── SRAM           │
 * └────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Configuration Parameter Hierarchy:**
 * ```
 * PETile Configuration (from configs.json):
 * ═══════════════════════════════════════════════════════════
 *
 * Primitive Parameters:
 * ┌─────────────────────────────────────────────────────────┐
 * │ mem_width (int)         : Memory data bus width         │
 * │ test_for_float (float)  : Floating-point test parameter │
 * │ bus_protocol (string)   : Bus protocol specification    │
 * │ sram_req_delay (Tick)   : SRAM access latency           │
 * │ bus_req_delay (Tick)    : Bus request latency           │
 * │ bus_resp_delay (Tick)   : Bus response latency          │
 * └─────────────────────────────────────────────────────────┘
 *
 * Complex Structures:
 * ┌─────────────────────────────────────────────────────────┐
 * │ cache_struct (CacheStruct):                             │
 * │   ├─ associativity (int)      : Cache set associativity │
 * │   ├─ mem_size (int)            : Cache capacity (bytes) │
 * │   ├─ replacement_policy (enum) : LRU, FIFO, Random      │
 * │   └─ write_policy (string)     : WriteBack, WriteThrough│
 * │                                                          │
 * │ bus_struct (BusStruct):                                 │
 * │   ├─ bus_width (int)           : Data bus width (bits)  │
 * │   ├─ max_outstanding_request   : Outstanding txn limit  │
 * │   └─ architecture (string)     : Crossbar, Mesh, etc.   │
 * └─────────────────────────────────────────────────────────┘
 * ```
 *
 * **Module Interconnection Topology:**
 * ```
 * Bidirectional Connections:
 * ══════════════════════════════════════════════════════════
 *
 * Forward Path (Request Flow):
 * ────────────────────────────
 *   CPUTraffic::addDownStream(bus, "Bus")
 *     │
 *     └─> AXI Bus receives requests from CPU
 *
 *   AXI Bus::addDownStream(pcuMem, "PCUMem")
 *     │
 *     └─> SRAM receives requests from Bus
 *
 * Backward Path (Response Flow):
 * ────────────────────────────
 *   SRAM::addUpStream(bus, "Bus")
 *     │
 *     └─> AXI Bus receives responses from SRAM
 *
 *   AXI Bus::addUpStream(pcu, "PCU")
 *     │
 *     └─> CPUTraffic receives responses from Bus
 *
 * Port Name Semantics:
 * ────────────────────────────
 *   "Bus"    : Connection to AXI interconnect bus
 *   "PCUMem" : Connection to PCU's private/local memory
 *   "PCU"    : Connection to Processing/Computation Unit
 * ```
 *
 * **Constructor Execution Flow:**
 * ```
 * PETile::PETile(name) invoked
 *   │
 *   ├─ Call STSimBase(name) parent constructor
 *   │
 *   ├─ Log primitive parameters:
 *   │  ├─ mem_width → int from JSON
 *   │  ├─ test_for_float → float from JSON
 *   │  ├─ bus_protocol → string from JSON
 *   │  └─ sram_req_delay → Tick from JSON
 *   │
 *   ├─ Log cache_struct complex parameter:
 *   │  ├─ Extract CacheStruct from JSON
 *   │  ├─ Log c.associativity
 *   │  ├─ Log c.mem_size
 *   │  ├─ Log c.replacement_policy (convert enum to string)
 *   │  └─ Log c.write_policy
 *   │
 *   └─ Log bus_struct complex parameter:
 *      ├─ Extract BusStruct from JSON
 *      ├─ Log b.bus_width
 *      ├─ Log b.max_outstanding_request
 *      └─ Log b.architecture
 * ```
 *
 * **Module Registration Flow:**
 * ```
 * registerModules() invoked
 *   │
 *   ├─ Phase 1: Module Creation
 *   │  ├─ pcu    = new CPUTraffic("CPU Traffic Generator")
 *   │  ├─ bus    = new AXIBus("AXI Bus")
 *   │  └─ pcuMem = new SRAM("PCU private memory")
 *   │
 *   ├─ Phase 2: Module Registration
 *   │  ├─ addModule(pcu)    → Register with simulator
 *   │  ├─ addModule(bus)    → Register with simulator
 *   │  └─ addModule(pcuMem) → Register with simulator
 *   │
 *   └─ Phase 3: Module Interconnection
 *      ├─ pcu->addDownStream(bus, "Bus")         → PCU sends to Bus
 *      ├─ bus->addDownStream(pcuMem, "PCUMem")   → Bus sends to SRAM
 *      ├─ pcuMem->addUpStream(bus, "Bus")        → SRAM responds to Bus
 *      └─ bus->addUpStream(pcu, "PCU")           → Bus responds to PCU
 * ```
 *
 * **Configuration Access Pattern:**
 * ```cpp
 * // Example: Accessing configuration parameters from any module
 *
 * // Primitive types:
 * int width = top->getParameter<int>("PETile", "mem_width");
 * float val = top->getParameter<float>("PETile", "test_for_float");
 * std::string proto = top->getParameter<std::string>("PETile", "bus_protocol");
 * Tick delay = top->getParameter<Tick>("PETile", "sram_req_delay");
 *
 * // Complex structures:
 * auto cache = top->getParameter<CacheStruct>("PETile", "cache_struct");
 * int assoc = cache.associativity;
 * std::string policy = enumToString(cache.replacement_policy);
 *
 * auto busConfig = top->getParameter<BusStruct>("PETile", "bus_struct");
 * int busWidth = busConfig.bus_width;
 * ```
 *
 * **Scalability to Multi-Tile Arrays:**
 * ```
 * Current: Single Tile
 * ═══════════════════════════════════════
 * ┌──────────────┐
 * │   PETile     │
 * │ PCU+Bus+SRAM │
 * └──────────────┘
 *
 * Future: 2x2 Mesh Array
 * ═══════════════════════════════════════
 * ┌──────────┬──────────┐
 * │Tile(0,0) │Tile(0,1) │
 * ├──────────┼──────────┤  ← Inter-tile mesh interconnect
 * │Tile(1,0) │Tile(1,1) │
 * └──────────┴──────────┘
 *
 * Extension Approach:
 *   1. Create TileArray class managing multiple PETile instances
 *   2. Add inter-tile routing logic in AXI Bus
 *   3. Implement mesh routing protocols (XY, dimension-ordered)
 *   4. Add network interface controllers (NICs) at tile boundaries
 *   5. Configure tile coordinates (x, y) for routing decisions
 * ```
 *
 * **Simulation Lifecycle Methods:**
 * ```
 * simInit() - Pre-simulation initialization
 *   │
 *   ├─ Called once before simulation starts
 *   ├─ Iterates through all registered modules
 *   ├─ Calls module->init() for each module
 *   └─ Modules inject initial events (e.g., CPUTraffic injects requests)
 *
 * step() - Per-tick simulation step
 *   │
 *   ├─ Called every simulation tick
 *   ├─ Currently no-op (event-driven simulation)
 *   └─ Could be used for cycle-accurate modeling
 *
 * cleanup() - Post-simulation cleanup
 *   │
 *   ├─ Called after simulation completes
 *   ├─ Release resources
 *   ├─ Finalize statistics
 *   └─ Close trace files
 * ```
 *
 * **Design Patterns Demonstrated:**
 *
 * 1. **Configuration Abstraction:**
 *    - Centralized configuration management via top->getParameter<T>()
 *    - Type-safe parameter access with template methods
 *    - Support for both primitive and user-defined types
 *    - Single source of truth (JSON file)
 *
 * 2. **Module Registration Pattern:**
 *    - Explicit module lifecycle: create → register → connect
 *    - Named port connections for clarity
 *    - Bidirectional connection establishment
 *    - Loose coupling via port names
 *
 * 3. **Hierarchical Composition:**
 *    - PETile owns child modules (pcu, bus, pcuMem)
 *    - Parent simulator orchestrates module interactions
 *    - Modules communicate via standardized packet interfaces
 *    - Clean separation of concerns
 *
 * **Configuration Example (configs.json):**
 * ```json
 * {
 *   "PETile": {
 *     "mem_width": 64,
 *     "test_for_float": 3.14159,
 *     "bus_protocol": "AXI4",
 *     "sram_req_delay": 10,
 *     "bus_req_delay": 2,
 *     "bus_resp_delay": 3,
 *     "cache_struct": {
 *       "associativity": 4,
 *       "mem_size": 32768,
 *       "replacement_policy": "LRU",
 *       "write_policy": "WriteBack"
 *     },
 *     "bus_struct": {
 *       "bus_width": 128,
 *       "max_outstanding_request": 8,
 *       "architecture": "Crossbar"
 *     }
 *   }
 * }
 * ```
 *
 * **Module Access Methods:**
 * ```cpp
 * // PETile provides accessor methods for child modules:
 * SimModule* pcu = getPCU();       // Returns CPUTraffic pointer
 * SimModule* bus = getBUS();       // Returns AXI Bus pointer
 * SimModule* mem = getPCUMem();    // Returns SRAM pointer
 *
 * // Use case: External monitoring or debugging
 * if (auto traffic = dynamic_cast<CPUTraffic*>(pcu)) {
 *     bool received = traffic->receivedOrNot();
 * }
 * ```
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/PETile.hh
 * - Config: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/PETileConfig.hh
 * - Top-level: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/PETileTop.hh
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - SRAM: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/SRAM.cc
 * - CPU Traffic: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUTraffic.cc
 * - Main: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/testPETile.cc
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "PETile.hh"

#include "PETileConfig.hh"
using namespace petile_config;

/**
 * @brief Constructs a PETile simulator and loads configuration parameters
 *
 * The constructor demonstrates how to access various configuration parameter types
 * from the centralized configuration system. It logs all parameters for verification
 * and debugging purposes.
 *
 * **Parameter Access Demonstration:**
 * - Primitive types: int, float, string, Tick
 * - Complex types: CacheStruct, BusStruct (user-defined)
 * - Enum conversion: ReplacementPolicy → string
 *
 * **Execution Flow:**
 * ```
 * PETile constructor
 *   ├─ Initialize STSimBase parent
 *   ├─ Access and log mem_width (int)
 *   ├─ Access and log test_for_float (float)
 *   ├─ Access and log bus_protocol (string)
 *   ├─ Access and log sram_req_delay (Tick)
 *   ├─ Access cache_struct (CacheStruct)
 *   │  └─ Log all cache structure fields
 *   └─ Access bus_struct (BusStruct)
 *      └─ Log all bus structure fields
 * ```
 *
 * @param name Simulator instance name (e.g., "PETile Simulator")
 *
 * @note All configuration parameters are accessed via top->getParameter<T>(category, key)
 * @note Category is "PETile" for all parameters in this example
 * @note Configuration must be registered in PETileTop::registerConfigs() before use
 *
 * @see PETileTop::registerConfigs() Configuration registration
 * @see PETileConfig Definition of configuration structure
 */
PETile::PETile(std::string name) : STSimBase(name) {
	// [TEST] busWidth (int)
	CLASS_INFO << "mem_width : " << top->getParameter<int>("PETile", "mem_width");
	// [TEST]  test_for_float (float)
	CLASS_INFO << "test_for_float : " << top->getParameter<float>("PETile", "test_for_float");
	// [TEST]  bus_protocol (string)
	CLASS_INFO << "bus_protocol : " + top->getParameter<std::string>("PETile", "bus_protocol");
	// [Test] sram_req_delay (Tick)
	CLASS_INFO << "sram_req_delay : " << top->getParameter<Tick>("PETile", "sram_req_delay");
	// [Test] cache_struct (USER_DEFINE : CacheStruct)
	auto c = top->getParameter<CacheStruct>("PETile", "cache_struct");
	CLASS_INFO << "cache associativity : " << c.associativity;
	CLASS_INFO << "cache mem_size : " << c.mem_size;
	CLASS_INFO << "cache replacement_policy : " + enumToString(c.replacement_policy);
	CLASS_INFO << "cache write_policy : " + c.write_policy;
	// [Test] bus_struct (USER_DEFINE : BusStruct)
	auto b = top->getParameter<BusStruct>("PETile", "bus_struct");
	CLASS_INFO << "bus bus_width : " << b.bus_width;
	CLASS_INFO << "bus max_outstanding_request : " << b.max_outstanding_request;
	CLASS_INFO << "bus architecture : " + b.architecture;
}

/**
 * @brief Registers and interconnects all PE tile modules
 *
 * This method implements the three-phase module setup pattern:
 * 1. Module Creation - Instantiate all child modules
 * 2. Module Registration - Register modules with simulator framework
 * 3. Module Interconnection - Establish upstream/downstream connections
 *
 * **Module Creation Phase:**
 * ```
 * CPUTraffic ("CPU Traffic Generator")
 *   - Generates memory read/write requests
 *   - Acts as processing element (PE/PCU)
 *   - Manages transaction IDs
 *   - Collects response statistics
 *
 * AXI Bus ("AXI Bus")
 *   - Routes requests from PCU to memory
 *   - Routes responses from memory to PCU
 *   - Adds protocol latencies
 *   - Manages outstanding transactions
 *
 * SRAM ("PCU private memory")
 *   - Services memory read/write requests
 *   - Models access latencies
 *   - Generates response packets
 * ```
 *
 * **Interconnection Topology:**
 * ```
 * Request Flow (Downstream):
 * ═══════════════════════════════════
 *   CPUTraffic ──["Bus"]──> AXI Bus ──["PCUMem"]──> SRAM
 *
 * Response Flow (Upstream):
 * ═══════════════════════════════════
 *   CPUTraffic <──["PCU"]── AXI Bus <──["Bus"]──── SRAM
 *
 * Port Naming Convention:
 *   "Bus"    : Connection to bus/interconnect
 *   "PCUMem" : PCU's local/private memory
 *   "PCU"    : Processing/Computation Unit
 * ```
 *
 * **Connection Semantics:**
 * - addDownStream(module, portName): Establishes master→slave connection
 * - addUpStream(module, portName): Establishes slave←master connection
 * - Port names enable dynamic module discovery via getDownStream()/getUpStream()
 *
 * @note This method is called automatically during simulator initialization
 * @note Modules must be registered before interconnection
 * @note Both downstream and upstream connections are required for bidirectional flow
 *
 * @see CPUTraffic Memory request traffic generator
 * @see AXIBus AXI protocol bus interconnect
 * @see SRAM Static RAM memory model
 */
void PETile::registerModules() {
	// create modules
	pcu    = new CPUTraffic("CPU Traffic Generator");
	bus    = new AXIBus("AXI Bus");
	pcuMem = new SRAM("PCU private memory");

	// register modules
	this->addModule(pcu);
	this->addModule(bus);
	this->addModule(pcuMem);

	// connect modules (connected_module, master port name, slave port name)
	pcu->addDownStream(bus, "Bus");
	bus->addDownStream(pcuMem, "PCUMem");
	pcuMem->addUpStream(bus, "Bus");
	bus->addUpStream(pcu, "PCU");
}
