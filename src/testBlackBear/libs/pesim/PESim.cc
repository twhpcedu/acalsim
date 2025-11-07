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
 * @file PESim.cc
 * @brief Processing Element (PE) simulator for BlackBear AI accelerator
 *
 * This file implements the PESim component, which represents individual compute tiles
 * in the BlackBear architecture. Each PE is a specialized processing unit designed for
 * tensor operations (matrix multiplication, convolution, activation functions) with
 * local scratchpad memory for high-bandwidth data access.
 *
 * **PESim Role in BlackBear Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                  Processing Element (PESim)                            │
 * │                                                                        │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │                    Compute Engine                                │ │
 * │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │ │
 * │  │  │   MAC Array  │  │  Activation  │  │   Control    │           │ │
 * │  │  │  (Matrix     │  │   Functions  │  │   Logic      │           │ │
 * │  │  │   Multiply)  │  │  (ReLU, etc) │  │              │           │ │
 * │  │  └──────────────┘  └──────────────┘  └──────────────┘           │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │           Local Scratchpad Memory (64KB)                         │ │
 * │  │  • Fast on-chip memory (SRAM)                                    │ │
 * │  │  • Low latency access (1-2 cycles)                               │ │
 * │  │  • Stores activations, partial sums                              │ │
 * │  │  • Address range: 0x100000000000 + 0x400000000 * peID            │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │                   Network Interfaces                             │ │
 * │  │                                                                  │ │
 * │  │  Request NOC (RNOC):          Data NOC (DNOC):                   │ │
 * │  │  ┌────────────────────┐       ┌────────────────────┐            │ │
 * │  │  │ Master: PEi2RNOC_M │       │ Master: PEi2DNOC_M │            │ │
 * │  │  │ Slave:  RNOC2PEi_S │       │ Slave:  DNOC2PEi_S │            │ │
 * │  │  └────────────────────┘       └────────────────────┘            │ │
 * │  │                                                                  │ │
 * │  │  • Send/receive control messages via RNOC                        │ │
 * │  │  • Send/receive tensor data via DNOC                             │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **PE Array Organization (4x4 example):**
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                   PE Array (gridX × gridY)                  │
 * │                                                             │
 * │   Row 0:  PE_0  ─  PE_1  ─  PE_2  ─  PE_3                  │
 * │            │       │       │       │                        │
 * │   Row 1:  PE_4  ─  PE_5  ─  PE_6  ─  PE_7                  │
 * │            │       │       │       │                        │
 * │   Row 2:  PE_8  ─  PE_9  ─ PE_10 ─ PE_11                   │
 * │            │       │       │       │                        │
 * │   Row 3: PE_12 ─ PE_13 ─ PE_14 ─ PE_15                     │
 * │            │       │       │       │                        │
 * │            └───────┴───────┴───────┴────► To NOC           │
 * │                                                             │
 * │  Each PE:                                                   │
 * │    - Unique ID (0 to gridX*gridY-1)                         │
 * │    - 64KB local memory                                      │
 * │    - Dual NOC connectivity (RNOC + DNOC)                    │
 * │    - Tensor processing capability                           │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * **PE Connectivity Pattern:**
 * ```
 * ┌──────────┐
 * │   MCPU   │ Orchestrates workload distribution
 * └────┬─────┘
 *      │
 *      ▼
 * ┌─────────────────────────────────────────┐
 * │         Network-on-Chip (NOC)           │
 * │  • Request NOC (RNOC)                   │
 * │  • Data NOC (DNOC)                      │
 * └──┬──────────────────────────────────┬───┘
 *    │                                  │
 *    ▼                                  ▼
 * ┌────────┐                        ┌────────┐
 * │  PE_i  │ ◄────────────────────► │ Cache  │ Shared cache per row
 * │        │                        │Cluster │
 * │ • MAC  │                        └────┬───┘
 * │ • Mem  │                             │
 * └────────┘                             ▼
 *                                   ┌─────────┐
 *                                   │ Global  │
 *                                   │ Memory  │
 *                                   └─────────┘
 *
 * Data Flow:
 * 1. MCPU → NOC → PE: Distribute input tensors
 * 2. PE → Local Mem: Store activations
 * 3. PE ← Cache ← Mem: Fetch weights
 * 4. PE: Compute (MAC operations)
 * 5. PE → NOC → Cache: Write results
 * ```
 *
 * **PE Memory Address Space:**
 * ```
 * Each PE has dedicated address region:
 *
 * PE[0]: 0x100000000000 + 0x400000000 * 0 = 0x100000000000
 *        Range: [0x100000000000, 0x10000000FFFF] (64KB)
 *
 * PE[1]: 0x100000000000 + 0x400000000 * 1 = 0x100400000000
 *        Range: [0x100400000000, 0x10040000FFFF] (64KB)
 *
 * PE[i]: 0x100000000000 + 0x400000000 * i
 *        Range: [base, base + 0xFFFF] (64KB)
 *
 * Memory Layout (per PE):
 *   0x0000 - 0x3FFF: Input activations (16KB)
 *   0x4000 - 0x7FFF: Weight cache (16KB)
 *   0x8000 - 0xBFFF: Partial sums (16KB)
 *   0xC000 - 0xFFFF: Output activations (16KB)
 * ```
 *
 * **Tensor Processing Operations:**
 * ```
 * Typical PE Workload (Convolution Layer):
 *
 * 1. Receive tensor request from MCPU:
 *    ┌─────────────────────────────────────┐
 *    │ TensorReqPacket                     │
 *    │  - Operation: CONV2D                │
 *    │  - Input tensor: 32x32x3            │
 *    │  - Filter: 3x3x3x64                 │
 *    │  - Stride: 1, Padding: SAME         │
 *    └─────────────────────────────────────┘
 *
 * 2. Fetch input data from Cache/Memory:
 *    PE → RNOC → Cache: Request input[0:1024]
 *    Cache → DNOC → PE: Deliver input data
 *
 * 3. Fetch weights from Cache/Memory:
 *    PE → RNOC → Cache: Request weights[conv1]
 *    Cache → DNOC → PE: Deliver weight data
 *
 * 4. Execute computation:
 *    for each output pixel:
 *        for each filter:
 *            accumulator = 0
 *            for kh in kernel_height:
 *                for kw in kernel_width:
 *                    for ic in input_channels:
 *                        accumulator += input[...] * weight[...]
 *            output[...] = activation(accumulator + bias)
 *
 * 5. Write results back:
 *    PE → DNOC → Cache: Write output activations
 * ```
 *
 * **Port Configuration:**
 * ```
 * Master Ports (PE initiates transactions):
 *   - PEi2RNOC_M: Send requests via Request NOC
 *   - PEi2DNOC_M: Send data via Data NOC
 *
 * Slave Ports (PE receives transactions):
 *   - RNOC2PEi_S: Receive requests from RNOC (queue size: 1)
 *   - DNOC2PEi_S: Receive data from DNOC (queue size: 1)
 *
 * Channel Connectivity:
 *   PE[i] ↔ NOC:
 *     - PEi2NOC_M/S (outgoing channel)
 *     - NOC2PEi_M/S (incoming channel)
 *
 * Example for PE_0:
 *   Master Ports: PE02RNOC_M, PE02DNOC_M
 *   Slave Ports:  RNOC2PE0_S, DNOC2PE0_S
 *   Channels:     PE02NOC_M/S, NOC2PE0_M/S
 * ```
 *
 * **Simulator Lifecycle:**
 * ```
 * 1. Construction:
 *    PESim(name, tensorManager, peID, testNum)
 *      ├─ Initialize CPPSimBase with name
 *      ├─ Initialize DataMovementManager with tensorManager
 *      └─ Store peID for addressing
 *
 * 2. Initialization (init()):
 *    prepareReqList()
 *      └─ Setup request queue (currently placeholder)
 *
 * 3. Simulation Loop (step()):
 *    • Process incoming tensor requests
 *    • Execute compute operations
 *    • Send results to NOC
 *    • Update PE state
 *    (Currently placeholder - template for implementation)
 *
 * 4. Event Handling (accept()):
 *    accept(when, pkt)
 *      ├─ Receive TensorReqPacket: Process computation request
 *      ├─ Receive TensorDataPacket: Store incoming data
 *      └─ Visitor pattern dispatches to appropriate handler
 *    (Currently placeholder)
 *
 * 5. Cleanup (cleanup()):
 *    • Flush pending operations
 *    • Report PE statistics
 *    • Free local resources
 *    (Currently placeholder)
 * ```
 *
 * **Template Implementation Pattern:**
 * ```cpp
 * // This is a template/skeleton implementation
 * // Real PE simulator would include:
 *
 * class PESim : public CPPSimBase, public DataMovementManager {
 * private:
 *     uint32_t peID;                    // PE identifier
 *     std::queue<TensorOp> requestQueue; // Pending operations
 *     LocalMemory scratchpad;           // 64KB SRAM
 *     MACUnit macArray;                 // Compute engine
 *     Statistics stats;                 // Performance counters
 *
 * public:
 *     void step() override {
 *         // Process request queue
 *         if (!requestQueue.empty()) {
 *             TensorOp op = requestQueue.front();
 *             // Execute operation
 *             // Send results
 *         }
 *     }
 *
 *     void accept(Tick when, SimPacket& pkt) override {
 *         // Visitor pattern dispatch
 *         if (auto* reqPkt = dynamic_cast<TensorReqPacket*>(&pkt)) {
 *             handleTensorRequest(reqPkt);
 *         } else if (auto* dataPkt = dynamic_cast<TensorDataPacket*>(&pkt)) {
 *             handleTensorData(dataPkt);
 *         }
 *     }
 *
 *     void handleTensorRequest(TensorReqPacket* pkt) {
 *         // Queue tensor operation
 *         // Fetch required data
 *         // Schedule computation
 *     }
 * };
 * ```
 *
 * **Integration with DataMovementManager:**
 * ```cpp
 * // PESim inherits tensor management capabilities:
 *
 * // 1. Acquire tensor for local computation
 * SimTensor* activationTensor = this->aquaireTensor(
 *     "pe" + std::to_string(peID) + "_activation",
 *     localMemAddr,
 *     width, height,
 *     srcStride, destStride,
 *     SimTensor::TENSORTYPE::ACTIVATION
 * );
 *
 * // 2. Send tensor request to fetch weights
 * this->sendTensorReq(
 *     this,                    // source = this PE
 *     "NOC",                   // downstream
 *     "PE" + std::to_string(peID) + "2RNOC_M",
 *     pushCallback, popCallback,
 *     peDeviceID,              // source
 *     cacheDeviceID,           // destination
 *     seqID++,
 *     REQ_TYPE_READ,
 *     weightAddr,
 *     weightSize,
 *     weightTensor
 * );
 *
 * // 3. Send computed results back
 * this->sendTensorData(
 *     this,
 *     "NOC",
 *     "PE" + std::to_string(peID) + "2DNOC_M",
 *     pushCallback, popCallback,
 *     peDeviceID,
 *     cacheDeviceID,
 *     seqID++,
 *     DATA_TYPE_WRITE,
 *     outputAddr,
 *     outputSize,
 *     activationTensor
 * );
 *
 * // 4. Recycle tensor after use
 * this->recycleTensor(activationTensor);
 * ```
 *
 * **Usage in BlackBear System:**
 * ```cpp
 * // In TestBlackBearTop::registerSimulators()
 * int peCount = gridX * gridY;
 * std::shared_ptr<SimTensorManager> tensorManager = ...;
 *
 * for (int peID = 0; peID < peCount; peID++) {
 *     // Create PE instance
 *     auto peSim = new PESim(
 *         "PETile_" + std::to_string(peID),  // name
 *         tensorManager,                      // shared tensor pool
 *         peID                                // unique ID
 *     );
 *
 *     // Register with simulation framework
 *     this->addSimulator(peSim);
 *
 *     // Setup connectivity (done in setupNodeConn, etc.)
 * }
 * ```
 *
 * **Key Characteristics:**
 *
 * 1. **Template Implementation:**
 *    - Provides skeleton for PE functionality
 *    - Placeholder methods (init, step, accept, cleanup)
 *    - Ready for domain-specific implementation
 *
 * 2. **Dual Inheritance:**
 *    - CPPSimBase: Core simulator lifecycle
 *    - DataMovementManager: Tensor operations
 *
 * 3. **Unique Identification:**
 *    - Each PE has unique peID
 *    - Enables individual addressing
 *    - Supports scalable array configuration
 *
 * 4. **Network Connectivity:**
 *    - Dual NOC access (RNOC + DNOC)
 *    - Bidirectional communication
 *    - Master/Slave port pairs
 *
 * 5. **Extensibility:**
 *    - Add MAC array implementation
 *    - Add local memory model
 *    - Add computation scheduling
 *    - Add performance counters
 *
 * **Related Files:**
 * - @see PESim.hh - PE simulator class declaration
 * - @see DataMovementManager.hh - Tensor movement infrastructure
 * - @see testBlackBear.cc - System integration and PE array setup
 * - @see NocSim.hh - Network-on-chip for inter-PE communication
 * - @see CacheSim.hh - Shared cache accessed by PEs
 *
 * @author ACAL/Playlab
 * @version 1.0
 * @date 2023-2025
 */

#include "pesim/PESim.hh"

void PESim::prepareReqList() {}

void PESim::init() { prepareReqList(); }

void PESim::step() {}

void PESim::cleanup() {}

void PESim::accept(Tick when, SimPacket& pkt) {}
