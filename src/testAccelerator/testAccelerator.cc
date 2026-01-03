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
 * @file testAccelerator.cc
 * @brief Host-accelerator co-design pattern demonstrating CPU-PE offloading via NoC
 *
 * This example showcases a complete **accelerator architecture** where a host CPU (MCPU)
 * offloads computational tasks to hardware accelerators (PE tiles) through a Network-on-Chip
 * (NoC) interconnect, with shared cache memory for data access. This pattern is fundamental
 * to modern heterogeneous computing systems.
 *
 * **System Architecture Overview:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                      Accelerator Co-Design System                       │
 * │                                                                         │
 * │  ┌──────────┐                                                           │
 * │  │   MCPU   │  (Host CPU - Traffic Generator & Task Coordinator)        │
 * │  │  (Host)  │  - Generates 20 tasks (IDs 0-19)                          │
 * │  └────┬─────┘  - 16 tasks → PE tiles (compute offload)                  │
 * │       │        - 4 tasks → Cache tiles (memory access)                  │
 * │       │ DSNOC                                                            │
 * │       ↓                                                                  │
 * │  ┌──────────────────────────────────────────────┐                       │
 * │  │              NoC (Network-on-Chip)           │                       │
 * │  │  - Central routing fabric                    │                       │
 * │  │  - Unicast/Multicast/Broadcast routing       │                       │
 * │  │  - Bidirectional channels to all endpoints   │                       │
 * │  └──┬───────────────┬───────────────────────┬───┘                       │
 * │     │ DSPEx         │ DSCACHEx              │ USMCPU                     │
 * │     ↓               ↓                       ↓                            │
 * │  ┌─────────┐    ┌─────────┐           ┌─────────┐                       │
 * │  │ PE Tile │... │ PE Tile │           │  Cache  │...                    │
 * │  │   #0    │    │   #15   │           │  Tile#0 │                       │
 * │  └─────────┘    └─────────┘           └─────────┘                       │
 * │  (16 Processing Elements)              (4 Cache Tiles)                  │
 * │  - Receive tasks from MCPU             - Handle memory requests         │
 * │  - Process computation                 - Respond to MCPU                │
 * │  - Return results to MCPU              - Test mode blackhole            │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Communication Pattern - Request/Response Flow:**
 * ```
 * Tick 0: MCPU Initialization
 *   └─ injectTraffic() → Sends first task (ID 0)
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ Task Offload Flow (MCPU → PE → MCPU)                               │
 * └─────────────────────────────────────────────────────────────────────┘
 * Tick 5: MCPU → NoC (Task Request)
 *   ├─ MCPUSim::genTraffic(destRId=0, destDMType=PE)
 *   ├─ Create: DLLRoutingInfo(UNICAST, destRId=0, PE)
 *   ├─ Create: RNocPacket with routing info
 *   ├─ Create: RNocEvent scheduled for Tick 5
 *   └─ Push to channel: DSNOC → NocSim
 *
 * Tick 5: NoC → PE (Task Routing)
 *   ├─ NocSim::RNocPacketHandler() receives packet
 *   ├─ execUnicast() routes to DSPE0
 *   ├─ Create: PERNocPacket + PERNocEvent
 *   └─ Push to channel: DSPE0 → PESim#0
 *
 * Tick 5: PE → NoC (Result Return)
 *   ├─ PESim#0::RNocPacketHandler() processes task
 *   ├─ Create response: RNocPacket(UNICAST, destRId=0, TRAFFIC_GENERATOR)
 *   ├─ Schedule: Tick 5 + 10 = 15
 *   └─ Push to channel: USNOC_PE0 → NocSim
 *
 * Tick 15: NoC → MCPU (Result Delivery)
 *   ├─ NocSim routes response to USMCPU
 *   ├─ Create: MCPUPacket scheduled for Tick 15 + 10 = 25
 *   └─ Push to channel: USMCPU → MCPUSim
 *
 * Tick 25: MCPU Receives Response
 *   ├─ MCPUSim::accept() invokes catchResponse()
 *   ├─ catchResponse() increments transactionID
 *   └─ injectTraffic() sends next task (ID 1)
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ Memory Access Flow (MCPU → Cache → MCPU)                           │
 * └─────────────────────────────────────────────────────────────────────┘
 * Similar flow but:
 *   - destDMType = CACHE
 *   - Routes to CacheSim instead of PESim
 *   - Cache may perform memory operations
 * ```
 *
 * **Channel Port Topology:**
 * ```
 * MCPU Connections:
 *   MasterChannelPort "DSNOC" → NocSim::SlaveChannelPort "USMCPU"
 *   SlaveChannelPort "DSNOC" ← NocSim::MasterChannelPort "USMCPU"
 *
 * NoC ↔ PE Connections (for each PE #i):
 *   NocSim::MasterChannelPort "DSPEi" → PESim#i::SlaveChannelPort "DSPEi"
 *   NocSim::SlaveChannelPort "USPE_NOCi" ← PESim#i::MasterChannelPort "USNOC_PEi"
 *   NocSim::MasterChannelPort "USPE_NOCi" → PESim#i::SlaveChannelPort "USPE_NOC"
 *   NocSim::SlaveChannelPort "DSNOCi" ← PESim#i::MasterChannelPort "DSNOC"
 *
 * NoC ↔ Cache Connections (for each Cache #j):
 *   NocSim::MasterChannelPort "DSCACHEj" → CacheSim::SlaveChannelPort "DSCACHEj"
 *   NocSim::SlaveChannelPort "USNOC_CACHEj" ← CacheSim::MasterChannelPort "USNOC_CACHEj"
 * ```
 *
 * **Test Modes (--test flag):**
 * ```
 * Test 0: Basic Round-Trip Test (Default)
 *   - MCPU sends 20 sequential tasks (16 to PEs, 4 to Cache)
 *   - Each task follows: MCPU → NoC → PE/Cache → NoC → MCPU
 *   - Validates basic request-response protocol
 *   - Total transactions: 20
 *
 * Test 1: PE Traffic Generation (Stress Test #1)
 *   - MCPU sends initial 16 tasks to PEs
 *   - Each PE generates 100 additional BLACKHOLE tasks
 *   - BLACKHOLE tasks → Cache (absorbed without response)
 *   - Tests: High-throughput PE→Cache traffic
 *   - Total transactions: 16 (initial) + 16×100 (PE-generated) = 1616
 *
 * Test 2: PE Processing Delay (Stress Test #2)
 *   - Same as Test 0 but PEs inject random delays
 *   - Each PE: sleepus(rand() % 1000) microseconds
 *   - Tests: Asynchronous timing, out-of-order completion
 *   - Validates: Event ordering under timing variability
 * ```
 *
 * **CLI Arguments:**
 * ```bash
 * --pe <count>      Number of PE tiles (default: 16, minimum: 16)
 * --cache <count>   Number of Cache tiles (default: 4, minimum: 4)
 * --test <0|1|2>    Test mode (0=basic, 1=stress, 2=async)
 *
 * Example Usage:
 *   ./testAccelerator --pe 32 --cache 8 --test 1
 *   # Runs stress test with 32 PEs and 8 caches
 * ```
 *
 * **Key Implementation Classes:**
 *
 * 1. **TestAccTop (SimTop):**
 *    - Top-level simulation coordinator
 *    - Registers all simulators and channels
 *    - Configures CLI parameters
 *    - Manages simulation lifecycle
 *
 * 2. **MCPUSim (CPPSimBase):**
 *    - Host CPU simulator / traffic generator
 *    - Generates 20 tasks sequentially
 *    - Catches responses and triggers next task
 *    - Source: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testAccelerator/libs/mcpu/MCPUSim.cc
 *
 * 3. **PESim (CPPSimBase):**
 *    - Processing Element (hardware accelerator)
 *    - Receives tasks from MCPU via NoC
 *    - Processes and returns results
 *    - Test 1: Generates additional traffic
 *    - Source: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testAccelerator/libs/peTile/PESim.cc
 *
 * 4. **NocSim (CPPSimBase):**
 *    - Network-on-Chip router
 *    - Routes packets between MCPU, PEs, and Caches
 *    - Supports UNICAST/MULTICAST/BROADCAST
 *    - Central communication fabric
 *    - Source: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testAccelerator/libs/noc/NocSim.cc
 *
 * 5. **CacheSim (CPPSimBase):**
 *    - Shared cache memory
 *    - Handles memory access requests
 *    - Returns data to MCPU
 *    - Test 1: Acts as blackhole for stress traffic
 *    - Source: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testAccelerator/libs/cache/CacheSim.cc
 *
 * **Accelerator Design Pattern Benefits:**
 *
 * 1. **Task Offloading:**
 *    - MCPU offloads compute-intensive tasks to specialized PEs
 *    - Frees CPU for coordination and control tasks
 *
 * 2. **Scalability:**
 *    - Configurable number of PEs (--pe flag)
 *    - NoC enables scalable interconnect
 *    - Multiple PEs can process tasks in parallel
 *
 * 3. **Shared Resources:**
 *    - Cache tiles shared among all PEs
 *    - NoC provides unified communication fabric
 *
 * 4. **Heterogeneous Computing:**
 *    - Different endpoint types (MCPU, PE, Cache)
 *    - Each optimized for specific functions
 *
 * 5. **Request-Response Protocol:**
 *    - Clean separation of concerns
 *    - Asynchronous communication
 *    - Event-driven execution
 *
 * **Real-World Applications:**
 * - GPU-style compute accelerators
 * - AI/ML inference accelerators
 * - DSP array processors
 * - FPGA-based accelerators
 * - Cryptographic co-processors
 *
 * **Performance Metrics:**
 * ```
 * Test 0 (20 tasks):
 *   - Expected duration: ~500 ticks
 *   - Round-trip latency: ~20 ticks per task
 *
 * Test 1 (1616 tasks):
 *   - Expected duration: Much higher
 *   - Tests NoC saturation and queueing
 *   - Validates high-throughput scenarios
 *
 * Test 2 (20 tasks + random delays):
 *   - Variable duration
 *   - Tests timing robustness
 *   - Validates out-of-order handling
 * ```
 *
 * **Execution Timeline Example (Test 0, first 3 tasks):**
 * ```
 * Tick 0:   MCPUSim::init() → injectTraffic()
 * Tick 5:   Task 0 → NoC → PE#0
 * Tick 15:  PE#0 → NoC (response)
 * Tick 25:  MCPU receives response → injectTraffic()
 * Tick 30:  Task 1 → NoC → PE#1
 * Tick 40:  PE#1 → NoC (response)
 * Tick 50:  MCPU receives response → injectTraffic()
 * Tick 55:  Task 2 → NoC → PE#2
 * ...
 * ```
 *
 * **Related Files:**
 * - MCPUSim.cc: Host CPU implementation
 * - PESim.cc: Processing Element implementation
 * - NocSim.cc: Network-on-Chip router implementation
 * - CacheSim.cc: Cache memory implementation
 * - PEEvent.cc: PE event processing logic
 * - NocEvent.cc: NoC event processing logic
 * - CacheEvent.cc: Cache event processing logic
 * - NocPacket.cc: NoC packet visitor pattern
 *
 * **Simulation Lifecycle:**
 * ```cpp
 * int main() {
 *     // 1. Create top-level simulation
 *     top = std::make_shared<TestAccTop>();
 *
 *     // 2. Initialize (parse CLI, register simulators, connect channels)
 *     top->init(argc, argv);
 *
 *     // 3. Run simulation (event-driven main loop)
 *     top->run();
 *
 *     // 4. Cleanup and report statistics
 *     top->finish();
 * }
 * ```
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see MCPUSim.cc for host CPU implementation
 * @see PESim.cc for accelerator PE implementation
 * @see NocSim.cc for NoC routing implementation
 * @see CacheSim.cc for cache memory implementation
 */

#include <ACALSim.hh>
#include <chrono>
#include <cmath>
#include <vector>

using namespace acalsim;

// Step 1 include header files of the simulator classes
#include "cache/CacheSim.hh"
#include "mcpu/MCPUSim.hh"
#include "noc/NocSim.hh"
#include "peTile/PESim.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestAccTop : public SimTop {
public:
	TestAccTop() : SimTop() {}

	void registerCLIArguments() override {
		this->getCLIApp()->add_option("--pe", this->peCount, "The number of PE")->default_val(this->peCount);
		this->getCLIApp()
		    ->add_option("--cache", this->cacheCount, "The number of Cache")
		    ->default_val(this->cacheCount);
		this->getCLIApp()->add_option("--test", this->testNum, "Test No")->default_val(this->testNum);
	}

	MCPUSim* getMCPUSim() { return (MCPUSim*)(this->mcpuSim); }

	void connectSimulators() {
		/*
		|	    DSNOC			 DSPEx
		| MCPU --------> NOC <-------------> PE
		|	    USMCPU	  ^	   USNOC_PEx
		|	 			  |
		|	 	DSCACHEx  |	  USNOC_CACHEx
		|				  |
		|				  v
		|			    CACHE
		*/

		// connect simulators (DownStream)
		mcpuSim->addDownStream(nocSim, "DSNOC");
		for (int peID = 0; peID < peCount; ++peID) {
			nocSim->addDownStream(peSims.at(peID), "DSPE" + std::to_string(peID));
			peSims.at(peID)->addDownStream(nocSim, "DSNOC");
		}
		for (int cacheID = 0; cacheID < this->cacheCount; ++cacheID)
			nocSim->addDownStream(cacheSim, "DSCACHE" + std::to_string(cacheID));

		// connect simulators (UpStream)
		nocSim->addUpStream(mcpuSim, "USMCPU");
		for (int peID = 0; peID < peCount; ++peID) {
			peSims.at(peID)->addUpStream(nocSim, "USNOC_PE" + std::to_string(peID));
			nocSim->addUpStream(peSims.at(peID), "USPE_NOC" + std::to_string(peID));
		}
		for (int cacheID = 0; cacheID < this->cacheCount; ++cacheID)
			cacheSim->addUpStream(nocSim, "USNOC_CACHE" + std::to_string(cacheID));
	}

	void registerSimulators() override {
		if (this->peCount < 16) { CLASS_ERROR << "Number of PE should larger than 16 | Current :" << this->peCount; }
		if (this->cacheCount < 4) {
			CLASS_ERROR << "Number of CACHE should larger than 4 | Current :" << this->cacheCount;
		}
		if (this->testNum > 2) { CLASS_ERROR << "Only test {0, 1, 2} are available!\n"; }

		// Create simulators
		mcpuSim  = (SimBase*)new MCPUSim("Master CPU");
		nocSim   = (SimBase*)new NocSim("Noxim");
		cacheSim = (SimBase*)new CacheSim("Cache Simulator");
		for (int peID = 0; peID < peCount; ++peID) {
			peSims.push_back((SimBase*)new PESim("PETile", peID, this->testNum));
		}

		// register Simulators
		this->addSimulator(mcpuSim);
		this->addSimulator(nocSim);
		this->addSimulator(cacheSim);
		for (auto& peSim : this->peSims) this->addSimulator(peSim);

		// Connect channel ports - NoC <> Traffic Generator
		ChannelPortManager::ConnectPort(this->mcpuSim, this->nocSim, "DSNOC", "USMCPU");
		ChannelPortManager::ConnectPort(this->nocSim, this->mcpuSim, "USMCPU", "DSNOC");

		// Connect channel ports - NoC <> PE
		for (int peID = 0; peID < peCount; ++peID) {
			ChannelPortManager::ConnectPort(this->nocSim, this->peSims[peID], "DSPE" + std::to_string(peID),
			                                "DSPE" + std::to_string(peID));
			ChannelPortManager::ConnectPort(this->peSims[peID], this->nocSim, "USNOC_PE" + std::to_string(peID),
			                                "USNOC_PE" + std::to_string(peID));
			ChannelPortManager::ConnectPort(this->peSims[peID], this->nocSim, "DSNOC", "DSNOC" + std::to_string(peID));
			ChannelPortManager::ConnectPort(this->nocSim, this->peSims[peID], "USPE_NOC" + std::to_string(peID),
			                                "USPE_NOC");
		}

		// Connect channel ports - NoC <> Cache
		for (int cacheID = 0; cacheID < this->cacheCount; ++cacheID) {
			ChannelPortManager::ConnectPort(this->nocSim, this->cacheSim, "DSCACHE" + std::to_string(cacheID),
			                                "DSCACHE" + std::to_string(cacheID));
			ChannelPortManager::ConnectPort(this->cacheSim, this->nocSim, "USNOC_CACHE" + std::to_string(cacheID),
			                                "USNOC_CACHE" + std::to_string(cacheID));
		}

		this->connectSimulators();
	}

	int getTestNum() { return testNum; }

private:
	uint32_t peCount    = 16;
	uint32_t cacheCount = 4;
	int      testNum    = 0;
	// simulators
	SimBase*              mcpuSim;
	SimBase*              nocSim;
	SimBase*              cacheSim;
	std::vector<SimBase*> peSims;
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestAccTop>();

	top->init(argc, argv);

	auto start = std::chrono::high_resolution_clock::now();
	top->run();
	auto stop = std::chrono::high_resolution_clock::now();

	auto diff = duration_cast<std::chrono::nanoseconds>(stop - start);

	LogOStream(LoggingSeverity::L_INFO, __FILE__, __LINE__, "Timer")
	    << "Time: " << (double)diff.count() / pow(10, 9) << " seconds.";

	top->finish();
	return 0;
}
