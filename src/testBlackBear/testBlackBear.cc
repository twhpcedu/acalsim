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
 * @file testBlackBear.cc
 * @brief BlackBear AI accelerator architecture simulation with distributed memory hierarchy
 *
 * This file demonstrates a complete AI/ML accelerator architecture simulation featuring:
 * - Multi-core Processing Element (PE) arrays organized in a 2D grid
 * - Distributed cache clusters for intermediate data storage
 * - Dual Network-on-Chip (NOC) for request and data transfer
 * - Global memory system with cache coherence
 * - Master Control Processing Unit (MCPU) for orchestration
 * - PyTorch JIT model execution and tensor management
 *
 * **BlackBear Architecture Overview:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────────────┐
 * │                          BlackBear AI Accelerator                              │
 * │                                                                                │
 * │  ┌──────────┐         ┌─────────────────────────────────────┐                 │
 * │  │  MCPU    │◄───────►│         Network-on-Chip (NOC)       │                 │
 * │  │(Master)  │         │  • Request NOC (RNOC)               │                 │
 * │  └──────────┘         │  • Data NOC (DNOC)                  │                 │
 * │                       └─────────┬───────────────────┬───────┘                 │
 * │                                 │                   │                         │
 * │  ┌──────────────────────────────┼───────────────────┼─────────────────────┐   │
 * │  │         PE Array (gridX × gridY)                 │                     │   │
 * │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐             │                     │   │
 * │  │  │PE_0 │  │PE_1 │  │PE_2 │  │PE_3 │             │                     │   │
 * │  │  └─────┘  └─────┘  └─────┘  └─────┘             │                     │   │
 * │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐             │                     │   │
 * │  │  │PE_4 │  │PE_5 │  │PE_6 │  │PE_7 │             ▼                     │   │
 * │  │  └─────┘  └─────┘  └─────┘  └─────┘      ┌─────────────┐              │   │
 * │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │Cache Cluster│              │   │
 * │  │  │PE_8 │  │PE_9 │  │PE_10│  │PE_11│      │  (gridY)    │              │   │
 * │  │  └─────┘  └─────┘  └─────┘  └─────┘      └──────┬──────┘              │   │
 * │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐             │                     │   │
 * │  │  │PE_12│  │PE_13│  │PE_14│  │PE_15│             │                     │   │
 * │  │  └─────┘  └─────┘  └─────┘  └─────┘             │                     │   │
 * │  │  Each PE has 64KB local memory                   │                     │   │
 * │  └────────────────────────────────────────────────┬─┴─────────────────────┘   │
 * │                                                    │                           │
 * │  ┌─────────────────────────────────────────────────▼─────────────────────┐    │
 * │  │                      Global Memory (16GB)                             │    │
 * │  │  Address Range: 0x10000000000 - 0x103FFFFFFFF                         │    │
 * │  │  • Weight storage                                                     │    │
 * │  │  • Input/Output tensors                                               │    │
 * │  │  • Intermediate results                                               │    │
 * │  └───────────────────────────────────────────────────────────────────────┘    │
 * └────────────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **System Component Hierarchy:**
 * ```
 * TestBlackBearTop (SimTop)
 *   ├─ MCPUSim (Master Control Processing Unit)
 *   │    ├─ Orchestrates workload execution
 *   │    ├─ Parses PyTorch JIT models
 *   │    ├─ Manages tensor distribution
 *   │    └─ Coordinates PE array operations
 *   │
 *   ├─ NocSim (Network-on-Chip)
 *   │    ├─ Request NOC (RNOC) - Control messages
 *   │    ├─ Data NOC (DNOC) - Tensor data transfer
 *   │    ├─ Routing and arbitration
 *   │    └─ Bandwidth management
 *   │
 *   ├─ PESim[0..N] (Processing Elements)
 *   │    ├─ Compute engines (MAC arrays)
 *   │    ├─ Local scratchpad memory (64KB each)
 *   │    ├─ Address range: 0x100000000000 + 0x400000000 * peID
 *   │    └─ Tensor processing units
 *   │
 *   ├─ CacheSim[0..gridY] (Cache Clusters)
 *   │    ├─ Shared cache per row of PEs
 *   │    ├─ Reduces global memory traffic
 *   │    ├─ Cache coherence protocol
 *   │    └─ Write-through/write-back policies
 *   │
 *   └─ MemSim (Global Memory)
 *        ├─ 16GB capacity (0x400000000)
 *        ├─ Base address: 0x10000000000 (1TB offset)
 *        ├─ Stores weights and large tensors
 *        └─ DRAM timing model
 * ```
 *
 * **Data Flow and Communication Patterns:**
 * ```
 * 1. Model Loading Phase:
 *    MCPU → MemSim: Load PyTorch model weights
 *    MCPU → Parse model graph and create execution plan
 *
 * 2. Tensor Distribution Phase:
 *    MCPU → NOC → Cache: Distribute input tensors
 *    Cache → NOC → PEs: Multicast tensor slices to PE array
 *
 * 3. Computation Phase:
 *    PE[i] → NOC → PE[j]: Inter-PE communication
 *    PE → Cache → Mem: Fetch weights/activations
 *    PE: Local computation on scratchpad data
 *
 * 4. Result Collection Phase:
 *    PE → NOC → Cache: Write partial results
 *    Cache → Mem: Writeback final results
 *    MCPU → Verify and export results
 * ```
 *
 * **Connectivity Topology:**
 * ```
 * Node Connectivity (Downstream/Upstream):
 * ═══════════════════════════════════════
 *   MCPU ←→ NOC ←→ {PE[0..N], Cache[0..M]}
 *   Cache[i] ←→ NOC
 *   Cache[i] ←→ Mem
 *
 * Channel Connectivity (Bidirectional):
 * ══════════════════════════════════════
 *   MCPU      ↔ NOC      : MCPU2NOC_M/S, NOC2MCPU_M/S
 *   PE[i]     ↔ NOC      : PEi2NOC_M/S, NOC2PEi_M/S
 *   Cache[i]  ↔ NOC      : CACHEi2NOC_M/S, NOC2CACHEi_M/S
 *   Cache[i]  ↔ Mem      : CACHEi2MEM_M/S, MEM2CACHEi_M/S
 *
 * Hardware Port Connectivity:
 * ═══════════════════════════
 *   MCPU:
 *     Master: MCPU2RNOC_M
 *     Slave:  RNOC2MCPU_S
 *
 *   PE[i] (dual NOC access):
 *     Master: PEi2RNOC_M, PEi2DNOC_M
 *     Slave:  RNOC2PEi_S, DNOC2PEi_S
 *
 *   Cache[i] (triple connectivity):
 *     Master: CACHEi2RNOC_M, CACHEi2DNOC_M, CACHEi2MEM_M
 *     Slave:  RNOC2CACHEi_S, DNOC2CACHEi_S, MEM2CACHEi_S
 *
 *   NOC (central router):
 *     Master: RNOC2MCPU_M, RNOC2PE[i]_M, DNOC2PE[i]_M,
 *             RNOC2CACHE[i]_M, DNOC2CACHE[i]_M
 *     Slave:  MCPU2RNOC_S, PE[i]2RNOC_S, PE[i]2DNOC_S,
 *             CACHE[i]2RNOC_S, CACHE[i]2DNOC_S
 *
 *   Mem:
 *     Master: MEM2CACHE[i]_M
 *     Slave:  CACHE[i]2MEM_S
 * ```
 *
 * **Memory Address Map:**
 * ```
 * ┌──────────────────────────────────────────────────────────┐
 * │ Global Memory Region                                     │
 * │ Base: 0x10000000000 (1TB)                                │
 * │ Size: 0x400000000 (16GB)                                 │
 * │ Range: [0x10000000000, 0x103FFFFFFFF]                    │
 * │ Usage: Model weights, global tensors, I/O buffers        │
 * ├──────────────────────────────────────────────────────────┤
 * │ PE Local Memory Regions (64KB each)                      │
 * │ PE[0]: 0x100000000000 + 0x400000000 * 0 = 0x100000000000 │
 * │ PE[1]: 0x100000000000 + 0x400000000 * 1 = 0x100400000000 │
 * │ ...                                                      │
 * │ PE[N]: 0x100000000000 + 0x400000000 * N                  │
 * │ Each PE: 64KB (0x10000) scratchpad memory                │
 * │ Usage: Local activations, partial sums, temporary data   │
 * └──────────────────────────────────────────────────────────┘
 *
 * Address Translation:
 *   Given address A:
 *     if (A >= 0x100000000000):
 *       peID = (A - 0x100000000000) / 0x400000000
 *       localOffset = (A - 0x100000000000) % 0x400000000
 *       → Route to PE[peID]
 *     elif (A >= 0x10000000000):
 *       → Route to Global Memory
 * ```
 *
 * **Configuration System:**
 * ```cpp
 * // SystemConfig manages all architecture parameters
 * struct TopLevelConfig {
 *     int gridX;  // PE grid width (default: 4)
 *     int gridY;  // PE grid height (default: 4)
 * };
 *
 * // Configuration loaded from config.json
 * {
 *   "System": {
 *     "ModelFileName": "model.pt",     // PyTorch JIT file
 *     "testNo": 0,                     // Test configuration
 *     "top": {
 *       "type": "TopLevelConfig",
 *       "params": {
 *         "gridX": 4,
 *         "gridY": 4
 *       }
 *     }
 *   }
 * }
 * ```
 *
 * **CLI Options:**
 * ```bash
 * ./testBlackBear [options]
 *
 * Options:
 *   --test <N>          Test configuration number (0-2)
 *   --gridx <N>         Number of PE tiles in X dimension (default: 4)
 *   --gridy <N>         Number of PE tiles in Y dimension (default: 4)
 *   --model <file>      PyTorch JIT model filename (default: trace.pt)
 *   --config <file>     JSON configuration file
 *
 * Examples:
 *   # Run with 8x8 PE array
 *   ./testBlackBear --gridx 8 --gridy 8 --model resnet50.pt
 *
 *   # Run test configuration 1 with custom grid
 *   ./testBlackBear --test 1 --gridx 4 --gridy 8
 * ```
 *
 * **Initialization Sequence:**
 * ```
 * 1. main() creates TestBlackBearTop instance
 *    ├─ Load config.json
 *    └─ Parse CLI arguments
 *
 * 2. top->init(argc, argv)
 *    ├─ registerConfigs()
 *    │    └─ Create SystemConfig with parameters
 *    ├─ registerCLIArguments()
 *    │    └─ Map CLI options to config parameters
 *    └─ registerSimulators()
 *         ├─ Validate grid dimensions (peCount >= 16, gridY >= 4)
 *         ├─ Create shared SimTensorManager
 *         ├─ Instantiate simulators:
 *         │    ├─ 1 MCPUSim
 *         │    ├─ 1 NocSim
 *         │    ├─ gridX * gridY PESim instances
 *         │    ├─ gridY CacheSim instances
 *         │    └─ 1 MemSim
 *         ├─ setupNodeConn() - Establish simulator hierarchy
 *         ├─ setupChannelConn() - Create bidirectional channels
 *         ├─ setupHWConn() - Connect hardware ports
 *         └─ registerDeviceAndAddressMap() - Setup address routing
 *
 * 3. top->setupWorkload()
 *    ├─ Create PyTorchJITParser
 *    ├─ Load model file
 *    └─ Parse computation graph
 *
 * 4. top->run()
 *    └─ Execute simulation main loop
 *
 * 5. top->finish()
 *    └─ Cleanup and generate reports
 * ```
 *
 * **Device Registration and Routing:**
 * ```cpp
 * // Register devices for packet routing
 * int mcpuDeviceID = registerDevice("MCPU");
 * int globalMemDeviceID = registerDevice("GlobalMemory");
 *
 * // Register address regions
 * registerAddrRegion("GlobalMemory-16G", globalMemDeviceID,
 *                    0x10000000000, 0x400000000);
 *
 * for (int peID = 0; peID < peCount; peID++) {
 *     int peDeviceID = registerDevice("PE" + std::to_string(peID));
 *     registerAddrRegion("PE" + std::to_string(peID) + "-Memory-64K",
 *                        peDeviceID,
 *                        0x100000000000 + 0x400000000 * peID,
 *                        0x10000);
 * }
 *
 * // Framework uses this to route packets:
 * //   1. Extract destination address from packet
 * //   2. Lookup address region to find device ID
 * //   3. Route packet to corresponding simulator
 * ```
 *
 * **Tensor Management:**
 * ```cpp
 * // Shared tensor manager across all simulators
 * std::shared_ptr<SimTensorManager> pTensorManager =
 *     std::make_shared<SimTensorManager>("GlobalTensorManager");
 *
 * // Each simulator inherits DataMovementManager:
 * //   - aquaireTensor() - Allocate tensor from pool
 * //   - recycleTensor() - Return tensor to pool
 * //   - sendTensorReq() - Send tensor request packet
 * //   - sendTensorData() - Send tensor data packet
 *
 * // Tensor lifecycle:
 * //   1. MCPU acquires tensor from manager
 * //   2. MCPU sends tensor request to PE via NOC
 * //   3. PE processes tensor
 * //   4. PE sends result back via NOC
 * //   5. MCPU recycles tensor to manager
 * ```
 *
 * **Key Implementation Patterns:**
 *
 * 1. **Multi-Level Hierarchy:**
 *    - Top-level orchestrator (MCPU)
 *    - Interconnect layer (NOC)
 *    - Compute layer (PE array)
 *    - Storage layer (Cache + Memory)
 *
 * 2. **Scalable Architecture:**
 *    - Grid dimensions configurable at runtime
 *    - Port naming uses systematic indexing
 *    - Loop-based connection setup
 *    - Supports arbitrary PE counts
 *
 * 3. **Separation of Concerns:**
 *    - Compute (PE) vs. Storage (Cache/Mem)
 *    - Control (RNOC) vs. Data (DNOC) networks
 *    - Simulator logic vs. connectivity setup
 *
 * 4. **Resource Management:**
 *    - Shared tensor pool (avoid allocation overhead)
 *    - Recycle containers for packets/events
 *    - Device registration for routing
 *
 * **Performance Measurement:**
 * ```cpp
 * auto start = std::chrono::high_resolution_clock::now();
 * top->run();
 * auto stop = std::chrono::high_resolution_clock::now();
 * auto diff = duration_cast<std::chrono::nanoseconds>(stop - start);
 * std::cout << "Time: " << (double)diff.count() / pow(10, 9)
 *           << " seconds." << std::endl;
 * ```
 *
 * **Related Files:**
 * - @see SystemConfig.hh - Configuration parameter definitions
 * - @see DataMovementManager.hh - Tensor transfer infrastructure
 * - @see MCPUSim.hh - Master control processing unit
 * - @see NocSim.hh - Network-on-chip router
 * - @see PESim.hh - Processing element implementation
 * - @see CacheSim.hh - Cache cluster simulator
 * - @see MemSim.hh - Global memory simulator
 * - @see PyTorchJITParser.hh - Model parsing utilities
 *
 * @author ACAL/Playlab
 * @version 1.0
 * @date 2023-2025
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <syncstream>
#include <vector>

#include "ACALSim.hh"
using namespace acalsim;

// Step 1 include header files of the simulator classes
#include "SystemConfig.hh"
#include "cachesim/CacheSim.hh"
#include "mcpusim/MCPUSim.hh"
#include "memsim/MemSim.hh"
#include "nocsim/NocSim.hh"
#include "pesim/PESim.hh"
#include "workloads/PyTorchJITParser.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestBlackBearTop : public SimTop {
public:
	TestBlackBearTop(std::string _name = "TestBlackBearTop", std::string _configFile = "") : SimTop(_configFile) {}

	void registerConfigs() override {
		// 1. instantiate "SystemConfig" in constructor of simulator.
		SimConfig* config = new SystemConfig("BlackBear System Configuration");

		// 2. register "SystemConfig" into configContainer
		//   "System" will be the key name in the config.json file
		this->addConfig("System", config);
	}

	void registerCLIArguments() override {
		this->addCLIOption<int>("--test", "Test No Discription", "System", "testNo");
		this->addCLIOption<int, TopLevelConfig>("--gridx", "number of PETile in x-dimension", "System", "top", "gridX");
		this->addCLIOption<int, TopLevelConfig>("--gridy", "number of PETile in y-dimension", "System", "top", "gridY");
		this->addCLIOption<std::string>("--model", "Model Filename", "System", "ModelFileName");
	}

	void registerSimulators() override {
		int peCount = this->gridX * this->gridY;
		if (peCount < 16) { CLASS_ERROR << "Number of PE should larger than 16 | Current :" << peCount; }
		if (this->gridY < 4) { CLASS_ERROR << "Number of CACHE should larger than 4 | Current :" << this->gridY; }
		if (this->testNo > 2) { CLASS_ERROR << "Only test {0, 1, 2} are available!\n"; }

		// Create a shared TensorManager
		std::shared_ptr<SimTensorManager> pTensorManager = std::make_shared<SimTensorManager>("GlobalTensorManager");

		// Create simulators
		mcpuSim = (SimBase*)new MCPUSim("MCPU", pTensorManager);
		nocSim  = (SimBase*)new NocSim("NOC", pTensorManager);
		memSim  = (SimBase*)new MemSim("GlobalMemory", pTensorManager);
		for (int peID = 0; peID < peCount; peID++) {
			peSimVec.push_back((SimBase*)new PESim("PETile_" + std::to_string(peID), pTensorManager, peID));
		}
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			cacheSimVec.push_back(
			    (SimBase*)new CacheSim("CacheCuster_" + std::to_string(cacheID), pTensorManager, cacheID));
		}

		// register Simulators
		this->addSimulator(mcpuSim);
		this->addSimulator(nocSim);
		for (auto& cacheSim : this->cacheSimVec) this->addSimulator(cacheSim);
		for (auto& peSim : this->peSimVec) this->addSimulator(peSim);
		this->addSimulator(memSim);

		this->setupNodeConn();
		this->setupChannelConn();
		this->setupHWConn();
		this->registerDeviceAndAddressMap();
	}

	/* setup connectivities required by the framework */
	void setupNodeConn() {
		int peCount = this->gridX * this->gridY;

		/* connect simulators (DownStream)
		 *   Downstream connectivity (node to node, unidirectional)
		 */

		// 𝑚𝑐𝑝𝑢𝑆𝑖𝑚->𝑛𝑜𝑐𝑆𝑖𝑚
		mcpuSim->addDownStream(nocSim, "DSNOC");

		for (int peID = 0; peID < peCount; peID++) {
			// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌
			nocSim->addDownStream(peSimVec[peID], "DSPE" + std::to_string(peID));
			// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌->𝑛𝑜𝑐𝑆𝑖𝑚
			peSimVec[peID]->addDownStream(nocSim, "DSNOC");
		}

		for (int cacheID = 0; cacheID < this->gridY; ++cacheID) {
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌->𝑛𝑜𝑐𝑆𝑖𝑚
			cacheSimVec[cacheID]->addDownStream(nocSim, "DSNOC");
			// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌
			nocSim->addDownStream(cacheSimVec[cacheID], "DSCACHE" + std::to_string(cacheID));
			// 𝑚𝑒𝑚𝑆𝑖𝑚->𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌
			memSim->addDownStream(cacheSimVec[cacheID], "DSCACHE" + std::to_string(cacheID));
		}
		// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑚𝑐𝑝𝑢𝑆𝑖𝑚
		nocSim->addDownStream(mcpuSim, "DSMCPU");

		/* connect simulators (UpStream)
		 * Upstream connectivity (node to node, unidirectional)
		 */

		// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑚𝑐𝑝𝑢𝑆𝑖𝑚
		mcpuSim->addUpStream(nocSim, "USNOC_MCPU");

		for (int peID = 0; peID < peCount; peID++) {
			// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌
			peSimVec[peID]->addUpStream(nocSim, "USNOC_PE" + std::to_string(peID));
			// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌->𝑛𝑜𝑐𝑆𝑖𝑚
			nocSim->addUpStream(peSimVec[peID], "USPE_NOC" + std::to_string(peID));
		}
		// 𝑚𝑐𝑝𝑢𝑆𝑖𝑚->𝑛𝑜𝑐𝑆𝑖𝑚
		nocSim->addUpStream(mcpuSim, "USMCPU_NOC");

		for (int cacheID = 0; cacheID < this->gridY; ++cacheID) {
			// 𝑛𝑜𝑐𝑆𝑖𝑚->𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌
			cacheSimVec.at(cacheID)->addUpStream(nocSim, "USNOC_CACHE" + std::to_string(cacheID));
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌->𝑛𝑜𝑐𝑆𝑖𝑚
			nocSim->addUpStream(cacheSimVec[cacheID], "USCACHE" + std::to_string(cacheID) + "_NOC");
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌->𝑚𝑒𝑚𝑆𝑖𝑚
			memSim->addUpStream(cacheSimVec[cacheID], "USCACHE" + std::to_string(cacheID) + "_MEM");
		}
	}

	/* Setup Thread Safe Channel Connectivity (node to node, bidirectional) */
	void setupChannelConn() {
		int peCount = this->gridX * this->gridY;

		// 𝑚𝑐𝑝𝑢𝑆𝑖𝑚 <-> 𝑛𝑜𝑐𝑆𝑖𝑚
		ChannelPortManager<void*>::ConnectPort(this->mcpuSim, this->nocSim, "MCPU2NOC_M", "MCPU2NOC_S");
		ChannelPortManager<void*>::ConnectPort(this->nocSim, this->mcpuSim, "NOC2MCPU_M", "NOC2MCPU_S");

		// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌 <-> 𝑛𝑜𝑐𝑆𝑖𝑚
		for (int peID = 0; peID < peCount; peID++) {
			// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌, 𝑛𝑜𝑐𝑆𝑖𝑚, 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝑁𝑂𝐶_𝑀, 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝑁𝑂𝐶_𝑆
			ChannelPortManager<void*>::ConnectPort(this->peSimVec[peID], this->nocSim,
			                                       "PE" + std::to_string(peID) + "2NOC_M",
			                                       "PE" + std::to_string(peID) + "2NOC_S");
			// 𝑛𝑜𝑐𝑆𝑖𝑚, 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌, 𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑀, 𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑆
			ChannelPortManager<void*>::ConnectPort(this->nocSim, this->peSimVec[peID],
			                                       "NOC2PE" + std::to_string(peID) + "_M",
			                                       "NOC2PE" + std::to_string(peID) + "_S");
		}

		for (int cacheID = 0; cacheID < this->gridY; ++cacheID) {
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌 <-> 𝑛𝑜𝑐𝑆𝑖𝑚
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌, 𝑛𝑜𝑐𝑆𝑖𝑚, 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑁𝑂𝐶_𝑀, 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑁𝑂𝐶_𝑆
			ChannelPortManager<void*>::ConnectPort(this->cacheSimVec[cacheID], this->nocSim,
			                                       "CACHE" + std::to_string(cacheID) + "2NOC_M",
			                                       "CACHE" + std::to_string(cacheID) + "2NOC_S");
			// 𝑛𝑜𝑐𝑆𝑖𝑚, 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌, 𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀, 𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑆
			ChannelPortManager<void*>::ConnectPort(this->nocSim, this->cacheSimVec[cacheID],
			                                       "NOC2CACHE" + std::to_string(cacheID) + "_M",
			                                       "NOC2CACHE" + std::to_string(cacheID) + "_S");

			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌 <-> 𝑚𝑒𝑚𝑆𝑖𝑚
			// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌, 𝑚𝑒𝑚𝑆𝑖𝑚, 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑀𝐸𝑀_𝑀, 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑀𝐸𝑀_𝑆
			ChannelPortManager<void*>::ConnectPort(this->cacheSimVec[cacheID], this->memSim,
			                                       "CACHE" + std::to_string(cacheID) + "2MEM_M",
			                                       "CACHE" + std::to_string(cacheID) + "2MEM_S");
			// 𝑚𝑒𝑚𝑆𝑖𝑚, 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌, 𝑀𝐸𝑀2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀, 𝑀𝐸𝑀2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑆
			ChannelPortManager<void*>::ConnectPort(this->memSim, this->cacheSimVec[cacheID],
			                                       "MEM2CACHE" + std::to_string(cacheID) + "_M",
			                                       "MEM2CACHE" + std::to_string(cacheID) + "_S");
		}
	}

	void setupHWConn() {
		// Setup Port Connectivity (port to port, lookup by port name)
		int peCount = this->gridX * this->gridY;

		// 𝑚𝑐𝑝𝑢𝑆𝑖𝑚
		//  master port - 𝑀𝐶𝑃𝑈2𝑅𝑁𝑂𝐶_𝑀
		this->mcpuSim->addMasterPort(/*masterPortName*/ "MCPU2RNOC_M");
		// slave port - 𝑅𝑁𝑂𝐶2𝑀𝐶𝑃𝑈_𝑆
		this->mcpuSim->addSlavePort(/*slavePortName*/ "RNOC2MCPU_S", /*reqQueueSize*/ 1);

		// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌
		for (int peID = 0; peID < peCount; peID++) {
			// master port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑀
			this->peSimVec[peID]->addMasterPort(/*masterPortName*/ "PE" + std::to_string(peID) + "2RNOC_M");
			// master port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑀
			this->peSimVec[peID]->addMasterPort(/*masterPortName*/ "PE" + std::to_string(peID) + "2DNOC_M");
			// slave port - 𝑅𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑆
			this->peSimVec[peID]->addSlavePort(/*slavePortName*/ "RNOC2PE" + std::to_string(peID) + "_S",
			                                   /*reqQueueSize*/ 1);
			// slave port - 𝐷𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑆
			this->peSimVec[peID]->addSlavePort(/*slavePortName*/ "DNOC2PE" + std::to_string(peID) + "_S",
			                                   /*reqQueueSize*/ 1);
		}

		// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌
		for (int cacheID = 0; cacheID < this->gridY; ++cacheID) {
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑀
			this->cacheSimVec[cacheID]->addMasterPort(/*masterPortName*/ "CACHE" + std::to_string(cacheID) + "2RNOC_M");
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑀
			this->cacheSimVec[cacheID]->addMasterPort(/*masterPortName*/ "CACHE" + std::to_string(cacheID) + "2DNOC_M");
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑀𝐸𝑀_𝑀
			this->cacheSimVec[cacheID]->addMasterPort(/*masterPortName*/ "CACHE" + std::to_string(cacheID) + "2MEM_M");
			// slave port - 𝑅𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑆
			this->cacheSimVec[cacheID]->addSlavePort(/*slavePortName*/ "RNOC2CACHE" + std::to_string(cacheID) + "_S",
			                                         /*reqQueueSize*/ 1);
			// slave port - 𝐷𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑆
			this->cacheSimVec[cacheID]->addSlavePort(/*slavePortName*/ "DNOC2CACHE" + std::to_string(cacheID) + "_S",
			                                         /*reqQueueSize*/ 1);
			// slave port - 𝑀𝐸𝑀2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑆
			this->cacheSimVec[cacheID]->addSlavePort(/*slavePortName*/ "MEM2CACHE" + std::to_string(cacheID) + "_S",
			                                         /*reqQueueSize*/ 1);
		}

		// 𝑛𝑜𝑐𝑆𝑖𝑚
		// master port - 𝑅𝑁𝑂𝐶2𝑀𝐶𝑃𝑈_𝑀
		this->nocSim->addMasterPort(/*masterPortName*/ "RNOC2MCPU_M");
		for (int peID = 0; peID < peCount; peID++) {
			// master port - 𝑅𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑀
			this->nocSim->addMasterPort(/*masterPortName*/ "RNOC2PE" + std::to_string(peID) + "_M");
			// master port - 𝐷𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑀
			this->nocSim->addMasterPort(/*masterPortName*/ "DNOC2PE" + std::to_string(peID) + "_M");
		}
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// master port - 𝑅𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			this->nocSim->addMasterPort(/*masterPortName*/ "RNOC2CACHE" + std::to_string(cacheID) + "_M");
			// master port - 𝐷𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			this->nocSim->addMasterPort(/*masterPortName*/ "DNOC2CACHE" + std::to_string(cacheID) + "_M");
		}
		// slave port - 𝑀𝐶𝑃𝑈2𝑅𝑁𝑂𝐶_𝑆
		this->nocSim->addSlavePort(/*slavePortName*/ "MCPU2RNOC_S", /*reqQueueSize*/ 1);
		for (int peID = 0; peID < peCount; peID++) {
			// slave port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑆
			this->nocSim->addSlavePort(/*slavePortName*/ "PE" + std::to_string(peID) + "2RNOC_S", /*reqQueueSize*/ 1);
			// slave port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑆
			this->nocSim->addSlavePort(/*slavePortName*/ "PE" + std::to_string(peID) + "2DNOC_S", /*reqQueueSize*/ 1);
		}
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// slave port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑆
			this->nocSim->addSlavePort(/*slavePortName*/ "CACHE" + std::to_string(cacheID) + "2RNOC_S",
			                           /*reqQueueSize*/ 1);
			// slave port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑆
			this->nocSim->addSlavePort(/*slavePortName*/ "CACHE" + std::to_string(cacheID) + "2DNOC_S",
			                           /*reqQueueSize*/ 1);
		}

		// 𝑚𝑒𝑚𝑆𝑖𝑚
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// master port - 𝑀𝐸𝑀2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			this->memSim->addMasterPort(/*masterPortName*/ "MEM2CACHE" + std::to_string(cacheID) + "_M");
			// slave port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑀𝐸𝑀_𝑆
			this->memSim->addSlavePort(/*slavePortName*/ "CACHE" + std::to_string(cacheID) + "2MEM_S",
			                           /*reqQueueSize*/ 1);
		}

		// Connect Ports for each link
		// 𝑚𝑐𝑝𝑢𝑆𝑖𝑚 master port 𝑀𝐶𝑃𝑈2𝑅𝑁𝑂𝐶_𝑀
		SimPortManager::ConnectPort(this->mcpuSim, this->nocSim, "MCPU2RNOC_M", "MCPU2RNOC_S");

		// 𝑝𝑒𝑆𝑖𝑚_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌
		for (int peID = 0; peID < peCount; peID++) {
			// master port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑀
			SimPortManager::ConnectPort(this->peSimVec[peID], this->nocSim, "PE" + std::to_string(peID) + "2RNOC_M",
			                            "PE" + std::to_string(peID) + "2RNOC_S");
			// master port - 𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑀
			SimPortManager::ConnectPort(this->peSimVec[peID], this->nocSim, "PE" + std::to_string(peID) + "2DNOC_M",
			                            "PE" + std::to_string(peID) + "2DNOC_S");
		}

		// 𝑐𝑎𝑐ℎ𝑒𝑆𝑖𝑚_𝐺𝑟𝑖𝑑𝑌
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑅𝑁𝑂𝐶_𝑀
			SimPortManager::ConnectPort(this->cacheSimVec[cacheID], this->nocSim,
			                            "CACHE" + std::to_string(cacheID) + "2RNOC_M",
			                            "CACHE" + std::to_string(cacheID) + "2RNOC_S");
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝐷𝑁𝑂𝐶_𝑀
			SimPortManager::ConnectPort(this->cacheSimVec[cacheID], this->nocSim,
			                            "CACHE" + std::to_string(cacheID) + "2DNOC_M",
			                            "CACHE" + std::to_string(cacheID) + "2DNOC_S");
			// master port - 𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌2𝑀𝐸𝑀_𝑀
			SimPortManager::ConnectPort(this->cacheSimVec[cacheID], this->memSim,
			                            "CACHE" + std::to_string(cacheID) + "2MEM_M",
			                            "CACHE" + std::to_string(cacheID) + "2MEM_S");
		}

		// 𝑛𝑜𝑐𝑆𝑖𝑚
		// master port - 𝑅𝑁𝑂𝐶2𝑀𝐶𝑃𝑈_𝑀
		SimPortManager::ConnectPort(this->nocSim, this->mcpuSim, "RNOC2MCPU_M", "RNOC2MCPU_S");
		for (int peID = 0; peID < peCount; peID++) {
			// master port - 𝑅𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑀
			SimPortManager::ConnectPort(this->nocSim, this->peSimVec[peID], "RNOC2PE" + std::to_string(peID) + "_M",
			                            "RNOC2PE" + std::to_string(peID) + "_S");
			// master port - 𝐷𝑁𝑂𝐶2𝑃𝐸_𝑋𝐺𝑟𝑖𝑑𝑋_𝑌𝐺𝑟𝑖𝑑𝑌_𝑀
			SimPortManager::ConnectPort(this->nocSim, this->peSimVec[peID], "DNOC2PE" + std::to_string(peID) + "_M",
			                            "DNOC2PE" + std::to_string(peID) + "_S");
		}
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// master port - 𝑅𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			SimPortManager::ConnectPort(this->nocSim, this->cacheSimVec[cacheID],
			                            "RNOC2CACHE" + std::to_string(cacheID) + "_M",
			                            "RNOC2CACHE" + std::to_string(cacheID) + "_S");
			// master port - 𝐷𝑁𝑂𝐶2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			SimPortManager::ConnectPort(this->nocSim, this->cacheSimVec[cacheID],
			                            "DNOC2CACHE" + std::to_string(cacheID) + "_M",
			                            "DNOC2CACHE" + std::to_string(cacheID) + "_S");
		}

		// 𝑚𝑒𝑚𝑆𝑖𝑚
		for (int cacheID = 0; cacheID < this->gridY; cacheID++) {
			// master port - 𝑀𝐸𝑀2𝐶𝐴𝐶𝐻𝐸𝐺𝑟𝑖𝑑𝑌_𝑀
			SimPortManager::ConnectPort(this->memSim, this->cacheSimVec[cacheID],
			                            "MEM2CACHE" + std::to_string(cacheID) + "_M",
			                            "MEM2CACHE" + std::to_string(cacheID) + "_S");
		}
	}

	void registerDeviceAndAddressMap() {
		// register Devices
		// A device needs to be registered if it needs to serve as the
		// source or destination device for a packet transmission

		// register address map regions
		//    {
		//     "Config":{
		//         "GlobalMemory" : "16G",
		//         "PEMemory" : "64K",
		//         "PENum": 16,
		//         "PERegionSize": "0x400000000"
		//     },    //
		//     "GlobalMemory" : { // put GM above 1TB
		//         "Visibility": "Global",
		//         "16G": {
		//             "startAddr": "0x10000000000", //1024GB
		//             "size": "0x400000000"         //16GB
		//         }
		//     },
		//
		//     "PEMemory" : { // start growing from 16TB
		//         "Visibility": "LocalPE"
		//         "64K": {
		//             "startAddr": "0x100000000000", //16TB reserve 16GB per PE, s
		//             "size": "0x10000"              //64KB
		//         }
		//     }
		// }
		int peCount           = this->gridX * this->gridY;
		int mcpuDeviceID      = registerDevice("MCPU");
		int globalMemDeviceID = registerDevice("GlobalMemory");
		registerAddrRegion("GlobalMemory-16G", globalMemDeviceID, /* startAddr */ 0x10000000000,
		                   /* size */ 0x400000000);
		for (int peID = 0; peID < peCount; peID++) {
			int peDeviceID = registerDevice("PE" + std::to_string(peID));
			registerAddrRegion("PE" + std::to_string(peID) + "-Memory-64K", peDeviceID,
			                   /* startAddr */ 0x100000000000 + 0x400000000 * peID, /* size */ 0x10000);
		}
	};

	void setupWorkload() {
		// parse PyTorch JIT file
		pytorchParser       = std::make_shared<PyTorchJITParser>();
		this->modelFileName = this->getParameter<std::string>("System", "ModelFileName");
		pytorchParser->init(this->modelFileName);
	}

	int getTestNo() { return testNo; }

private:
	uint32_t gridX  = 4;
	uint32_t gridY  = 4;
	int      testNo = 0;

	// workloads
	std::string                       modelFileName;
	std::shared_ptr<PyTorchJITParser> pytorchParser;

	// simulators
	SimBase*              mcpuSim;
	SimBase*              nocSim;
	std::vector<SimBase*> cacheSimVec;
	SimBase*              memSim;
	std::vector<SimBase*> peSimVec;
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestBlackBearTop>("testBlackBear", "src/testBlackBear/config.json");

	top->init(argc, argv);

	// parse pytorch JIT trace
	std::dynamic_pointer_cast<TestBlackBearTop>(top)->setupWorkload();

	auto start = std::chrono::high_resolution_clock::now();
	top->run();
	auto stop = std::chrono::high_resolution_clock::now();

	auto diff = duration_cast<std::chrono::nanoseconds>(stop - start);

	std::osyncstream(std::cout) << "Time: " << (double)diff.count() / pow(10, 9) << " seconds." << std::endl;

	top->finish();
	return 0;
}
