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
 * @file testPETile.cc
 * @brief Comprehensive demonstration of Processing Element (PE) tile architecture simulator
 *
 * This example demonstrates the simulation of a **Processing Element tile architecture**
 * with local memory and AXI bus interconnection. It showcases how to build a tile-based
 * computing system where each tile contains a processing unit, local SRAM, and interconnect.
 *
 * **PE Tile System Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                          PE Tile Architecture                          │
 * │                                                                        │
 * │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐  │
 * │  │  CPUTraffic  │  Req    │   AXI Bus    │  Req    │     SRAM     │  │
 * │  │   (PCU/PE)   │────────>│ (Interconnect│────────>│ (Local Mem)  │  │
 * │  │              │         │   Network)   │         │              │  │
 * │  │  - Gen Reqs  │  Resp   │ - Routing    │  Resp   │ - Storage    │  │
 * │  │  - Process   │<────────│ - Buffering  │<────────│ - Latency    │  │
 * │  │    Responses │         │ - Arbitration│         │   Modeling   │  │
 * │  └──────────────┘         └──────────────┘         └──────────────┘  │
 * │         ▲                        ▲                        ▲           │
 * │         │                        │                        │           │
 * │    Transaction                 Bus                     Memory         │
 * │    Management                Protocol                 Management      │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Scalable Multi-Tile Array Vision (Future Extension):**
 * ```
 * Tile Array Layout (2x2 example):
 * ┌──────────┬──────────┐
 * │ Tile(0,0)│ Tile(0,1)│
 * │  PE+Mem  │  PE+Mem  │
 * ├──────────┼──────────┤
 * │ Tile(1,0)│ Tile(1,1)│
 * │  PE+Mem  │  PE+Mem  │
 * └──────────┴──────────┘
 *       │          │
 *    Mesh Interconnect
 *
 * Each tile contains:
 *   - Processing Element (PE/PCU)
 *   - Local SRAM
 *   - AXI Bus interface
 *   - Inter-tile routing capability
 * ```
 *
 * **Data Flow Through PE Tile:**
 * ```
 * Time Flow:
 * ════════════════════════════════════════════════════════════════
 *
 * Tick 1: CPUTraffic generates MemReqPacket
 *   ├─ Type: TENSOR_MEM_READ
 *   ├─ Address: 0x0000
 *   ├─ Size: 0 bytes
 *   └─ Transaction ID: 0
 *
 * Tick 1: CPUReqEvent scheduled
 *   └─ Event sends request to AXI Bus
 *
 * Tick 1+delay: AXI Bus receives request
 *   ├─ Bus latency added (bus_req_delay)
 *   └─ BusReqEvent scheduled to SRAM
 *
 * Tick X: SRAM processes request
 *   ├─ Access latency calculated: sram_req_delay + size/256
 *   ├─ Response packet prepared
 *   └─ SRAMRespEvent scheduled
 *
 * Tick Y: Response propagates back
 *   ├─ BusRespEvent delivers to AXI Bus
 *   ├─ Bus adds response latency (bus_resp_delay + size/32)
 *   └─ CPUReqCallback invoked
 *
 * Tick Z: CPUTraffic receives response
 *   ├─ Transaction completed
 *   └─ Statistics updated
 * ```
 *
 * **Memory Request/Response Packet Flow:**
 * ```
 * Request Packet (MemReqPacket):
 * ┌─────────────────────────────────────┐
 * │ reqType: TENSOR_MEM_READ/WRITE      │
 * │ addr:    Target memory address      │
 * │ size:    Transfer size (bytes)      │
 * │ callback: Response handler function │
 * │ memRespPkt: Embedded response pkt   │
 * └─────────────────────────────────────┘
 *         │
 *         ▼
 * Response Packet (MemRespPacket):
 * ┌─────────────────────────────────────┐
 * │ reqType: Original request type      │
 * │ addr:    Original address           │
 * │ size:    Transfer size              │
 * │ (Delivered via callback chain)      │
 * └─────────────────────────────────────┘
 * ```
 *
 * **Event-Driven Callback Chain:**
 * ```
 * CPUReqEvent::process()
 *   │
 *   ├─ Sets callback: λ(id, resp) { cpuReqCallback(tid, resp); }
 *   │
 *   └─ Sends to AXI Bus via accept()
 *       │
 *       └─ AXIBus::memReqPktHandler()
 *           │
 *           ├─ Extracts caller callback
 *           │
 *           └─ Creates BusReqEvent
 *               │
 *               └─ BusReqEvent::process()
 *                   │
 *                   ├─ Sets callback: λ(id, resp) { busReqCallback(); }
 *                   │
 *                   └─ Sends to SRAM via accept()
 *                       │
 *                       └─ SRAM::memReqPktHandler()
 *                           │
 *                           └─ Creates SRAMRespEvent
 *                               │
 *                               └─ SRAMRespEvent::process()
 *                                   │
 *                                   └─ Invokes callback chain
 *                                       │
 *                                       ├─ BusReqCallback executed
 *                                       │   └─ Creates BusRespEvent
 *                                       │
 *                                       └─ CPUReqCallback executed
 *                                           └─ CPUTraffic::MemRespHandler()
 * ```
 *
 * **Transaction Timing Example:**
 * ```
 * Transaction 0 (addr=0x0000, size=0):
 * ═══════════════════════════════════════════
 * Tick  1: CPUReqEvent scheduled
 * Tick  1: Request sent to AXI Bus
 * Tick  1+bus_req_delay: BusReqEvent scheduled to SRAM
 * Tick  X: SRAM processes, SRAMRespEvent scheduled
 *          (delay = sram_req_delay + (0+1)/256 = sram_req_delay + 0)
 * Tick  Y: BusRespEvent scheduled back to CPU
 *          (delay = bus_resp_delay + (0+1)/32 = bus_resp_delay + 0)
 * Tick  Z: CPUTraffic receives response, stats updated
 *
 * Transaction 1 (addr=0x1000, size=20):
 * ═══════════════════════════════════════════
 * Tick 11: CPUReqEvent scheduled
 * Tick 11: Request sent to AXI Bus
 * Tick 11+bus_req_delay: BusReqEvent to SRAM
 * Tick  X: SRAM delay = sram_req_delay + (20+1)/256
 * Tick  Y: Bus delay = bus_resp_delay + (20+1)/32
 * Tick  Z: Response received
 * ```
 *
 * **Module Interconnection Topology:**
 * ```
 * Module Registration & Connectivity:
 *
 * Downstream Connections (Master → Slave):
 * ───────────────────────────────────────────
 * CPUTraffic ──["Bus"]──> AXI Bus
 * AXI Bus    ──["PCUMem"]──> SRAM
 *
 * Upstream Connections (Slave ← Master):
 * ───────────────────────────────────────────
 * SRAM       ──["Bus"]──> AXI Bus
 * AXI Bus    ──["PCU"]──> CPUTraffic
 *
 * Port Naming Convention:
 *   - "Bus": Connection to AXI interconnect
 *   - "PCUMem": Connection to processing unit's local memory
 *   - "PCU": Connection to processing/computation unit
 * ```
 *
 * **Key Architectural Features:**
 *
 * 1. **Tile-Based Organization:**
 *    - Self-contained processing tile
 *    - Local memory for reduced latency
 *    - Standard AXI bus interface
 *    - Scalable to multi-tile arrays
 *
 * 2. **AXI Bus Interconnect:**
 *    - Protocol-compliant bus interface
 *    - Transaction ID management
 *    - Configurable latencies
 *    - Outstanding request tracking
 *
 * 3. **Callback-Based Communication:**
 *    - Event-driven architecture
 *    - Callback chain for responses
 *    - Automatic latency modeling
 *    - Transaction tracing support
 *
 * 4. **Configuration Management:**
 *    - JSON-based configuration (configs.json)
 *    - Parametric memory models
 *    - Bus protocol parameters
 *    - Cache structure definitions
 *
 * **Simulation Initialization Flow:**
 * ```
 * main()
 *   │
 *   ├─ Create PETileTop simulator
 *   │  ├─ Name: "PESTSim"
 *   │  ├─ Config: "src/testPETile/configs.json"
 *   │  └─ Trace path: "src/testPETile/trace"
 *   │
 *   ├─ top->init(argc, argv)
 *   │  ├─ Parse command-line arguments
 *   │  ├─ Load configuration from JSON
 *   │  ├─ Register PETileConfig
 *   │  ├─ Call PETile::registerModules()
 *   │  │  ├─ Create CPUTraffic instance
 *   │  │  ├─ Create AXI Bus instance
 *   │  │  ├─ Create SRAM instance
 *   │  │  └─ Connect modules (upstream/downstream)
 *   │  └─ Call PETile::simInit()
 *   │     └─ Initialize all child modules
 *   │
 *   ├─ top->run()
 *   │  ├─ CPUTraffic::init()
 *   │  │  └─ Inject memory requests (5 requests)
 *   │  ├─ Event-driven simulation loop
 *   │  │  ├─ Process CPUReqEvents
 *   │  │  ├─ Process BusReqEvents
 *   │  │  ├─ Process SRAMRespEvents
 *   │  │  └─ Process BusRespEvents
 *   │  └─ Continue until event queue empty
 *   │
 *   └─ top->finish()
 *      ├─ Cleanup resources
 *      ├─ Write trace files
 *      └─ Report statistics
 * ```
 *
 * **CLI Usage Examples:**
 * ```bash
 * # Basic execution with default configuration
 * ./testPETile
 *
 * # Run with custom configuration file
 * ./testPETile --config custom_config.json
 *
 * # Enable trace generation
 * ./testPETile --trace-enable
 *
 * # Run in Google Test mode
 * ./testPETile --gtest
 *
 * # Verbose logging for debugging
 * ./testPETile --verbose --log-level DEBUG
 * ```
 *
 * **Configuration Parameters (configs.json):**
 * ```json
 * {
 *   "PETile": {
 *     "mem_width": 64,              // Memory data bus width (bits)
 *     "test_for_float": 3.14,       // Example float parameter
 *     "bus_protocol": "AXI4",       // Bus protocol type
 *     "sram_req_delay": 10,         // SRAM access latency (ticks)
 *     "bus_req_delay": 2,           // Bus request latency (ticks)
 *     "bus_resp_delay": 3,          // Bus response latency (ticks)
 *     "cache_struct": {
 *       "associativity": 4,         // Cache associativity
 *       "mem_size": 32768,          // Cache size (bytes)
 *       "replacement_policy": "LRU",
 *       "write_policy": "WriteBack"
 *     },
 *     "bus_struct": {
 *       "bus_width": 128,           // Bus width (bits)
 *       "max_outstanding_request": 8,
 *       "architecture": "Crossbar"
 *     }
 *   }
 * }
 * ```
 *
 * **Statistics Collected:**
 * - Number of requests generated (CPUTraffic)
 * - Number of responses received (CPUTraffic, AXI Bus)
 * - Transaction completion times
 * - Bus utilization metrics
 * - Memory access patterns
 *
 * **Trace Output:**
 * Simulation generates trace files in src/testPETile/trace/:
 * - CPUReq: CPU request events with transaction IDs, addresses, sizes
 * - Chrome Trace format for visualization in chrome://tracing
 *
 * **Related Files:**
 * - Implementation: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/PETile.cc
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - SRAM: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/SRAM.cc
 * - Traffic Gen: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUTraffic.cc
 * - Events: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUReqEvent.cc
 * - Packets: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/MemReq.cc
 * - Config: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/configs.json
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "ACALSim.hh"
using namespace acalsim;

#include "PETileTop.hh"

/**
 * @brief Main entry point for PE Tile simulator
 *
 * Instantiates and executes the PE tile architecture simulator following the
 * standard ACALSim three-phase workflow: initialization, execution, and cleanup.
 *
 * **Three-Phase Simulation Workflow:**
 * ```
 * Phase 1: Initialization (init)
 * ═══════════════════════════════════════════
 *   - Parse command-line arguments
 *   - Load JSON configuration file
 *   - Register configuration objects
 *   - Create and connect modules
 *   - Initialize module states
 *
 * Phase 2: Execution (run)
 * ═══════════════════════════════════════════
 *   - Inject initial memory requests
 *   - Process events from event queue
 *   - Execute callbacks in time order
 *   - Generate trace records
 *   - Collect statistics
 *
 * Phase 3: Cleanup (finish)
 * ═══════════════════════════════════════════
 *   - Finalize statistics
 *   - Write trace files to disk
 *   - Free allocated resources
 *   - Report simulation results
 * ```
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on successful completion, non-zero on error
 *
 * @see PETileTop Top-level simulator class
 * @see PETile Base simulator implementation
 * @see STSim Single-threaded simulation template
 */
int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	// Remember 1) to cast the top-level instance to the SimTop* type and set it to the global variable top
	// 2) Pass your own simulator class type to the STSim class template
	top = std::make_shared<PETileTop>("PESTSim", "src/testPETile/configs.json");
	top->init(argc, argv);
	top->run();
	top->finish();
	return 0;
}
