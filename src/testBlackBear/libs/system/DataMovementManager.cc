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
 * @file DataMovementManager.cc
 * @brief Tensor-based data movement infrastructure for BlackBear architecture
 *
 * This file implements the DataMovementManager, which provides a unified interface for
 * tensor allocation, recycling, and transmission across the BlackBear system. It serves
 * as the foundation for all inter-component data transfers, enabling efficient tensor
 * movement between MCPU, PEs, Caches, and Memory.
 *
 * **DataMovementManager Role in BlackBear:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                      DataMovementManager Layer                         │
 * │                                                                        │
 * │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
 * │  │ Tensor Lifecycle │   │ Packet Creation  │   │ Event Scheduling │  │
 * │  │                  │   │                  │   │                  │  │
 * │  │ • Allocation     │   │ • Request Pkts   │   │ • TensorReqEvent │  │
 * │  │ • Initialization │   │ • Data Packets   │   │ • TensorDataEvent│  │
 * │  │ • Recycling      │   │ • Event Wrapping │   │ • Transaction ID │  │
 * │  └──────────────────┘   └──────────────────┘   └──────────────────┘  │
 * │                                                                        │
 * │  Used by all simulators:                                               │
 * │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │
 * │  │ MCPU   │  │  PE    │  │ Cache  │  │  NOC   │  │  Mem   │          │
 * │  │  Sim   │  │  Sim   │  │  Sim   │  │  Sim   │  │  Sim   │          │
 * │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘          │
 * │      └───────────┴───────────┴───────────┴───────────┘               │
 * │              All inherit DataMovementManager                           │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Core Functionality:**
 *
 * 1. **Tensor Resource Management:**
 *    ```
 *    SimTensorManager (Shared Pool)
 *           │
 *           ├─ aquaireTensor() ──► Allocate from pool
 *           │                      ├─ Set address
 *           │                      ├─ Set dimensions (width, height)
 *           │                      ├─ Set strides (src, dest)
 *           │                      └─ Set type (WEIGHT, ACTIVATION, etc.)
 *           │
 *           └─ recycleTensor() ──► Return to pool
 *                                   └─ Avoid repeated allocation overhead
 *    ```
 *
 * 2. **Tensor Request Transmission:**
 *    ```
 *    sendTensorReq(srcSim, dsSimName, dsPortName, callbacks, ...)
 *           │
 *           ├─ Create TensorReqPacket
 *           │    ├─ srcID, destID (device IDs)
 *           │    ├─ type (request type)
 *           │    ├─ addr, size (memory parameters)
 *           │    └─ pTensor (tensor handle)
 *           │
 *           ├─ Create TensorReqEvent
 *           │    ├─ Generate transaction ID (tid)
 *           │    ├─ Attach packet
 *           │    └─ Set callbacks
 *           │
 *           ├─ Wrap in EventPacket
 *           │    └─ Schedule at globalTick + 1
 *           │
 *           └─ Send via SimPort
 *                └─ Trigger callbacks on send/receive
 *    ```
 *
 * 3. **Tensor Data Transmission:**
 *    ```
 *    sendTensorData(srcSim, dsSimName, dsPortName, callbacks, ...)
 *           │
 *           ├─ Create TensorDataPacket
 *           │    ├─ srcID, destID
 *           │    ├─ type, addr, size
 *           │    └─ pTensor (with actual data)
 *           │
 *           ├─ Create TensorDataEvent
 *           │    ├─ Generate transaction ID
 *           │    └─ Attach data packet
 *           │
 *           ├─ Wrap in EventPacket
 *           │    └─ Schedule at globalTick + 1
 *           │
 *           └─ Send via SimPort
 *                └─ Deliver tensor payload
 *    ```
 *
 * **Transaction ID Generation:**
 * ```
 * Transaction {
 *   upStreamDeviceID:   16 bits
 *   downStreamDeviceID: 16 bits
 *   packetID:           32 bits
 * }
 *
 * tid = (upStreamDeviceID << 48) | (downStreamDeviceID << 32) | packetID
 *
 * Example:
 *   srcID = 0x0001 (MCPU)
 *   destID = 0x0010 (PE_0)
 *   packetID = 0x12345678
 *   tid = 0x0001001012345678
 *
 * Purpose:
 *   - Unique identifier for each request-response pair
 *   - Enables transaction tracking across system
 *   - Supports out-of-order completion
 * ```
 *
 * **Tensor Lifecycle Example:**
 * ```
 * // 1. MCPU acquires tensor for convolution weights
 * SimTensor* weightTensor = dataMovementManager->aquaireTensor(
 *     "conv1_weights",          // name
 *     0x10000000000,            // global memory address
 *     256,                      // width (columns)
 *     256,                      // height (rows)
 *     256,                      // source stride
 *     256,                      // dest stride
 *     SimTensor::TENSORTYPE::WEIGHT
 * );
 *
 * // 2. MCPU sends tensor request to PE via NOC
 * dataMovementManager->sendTensorReq(
 *     mcpuSim,                  // source simulator
 *     "NOC",                    // downstream simulator name
 *     "MCPU2RNOC_M",            // port name
 *     pushCallback,             // callback when pushed to queue
 *     popCallback,              // callback when popped from queue
 *     mcpuDeviceID,             // source device ID
 *     pe0DeviceID,              // destination device ID
 *     seqID,                    // sequence ID
 *     REQ_TYPE_READ,            // request type
 *     0x10000000000,            // address
 *     256*256*4,                // size (256x256 floats)
 *     weightTensor              // tensor handle
 * );
 *
 * // 3. PE processes request and sends data back
 * dataMovementManager->sendTensorData(
 *     peSim,                    // source
 *     "NOC",                    // downstream
 *     "PE02DNOC_M",             // port
 *     pushCallback, popCallback,
 *     pe0DeviceID,              // source
 *     mcpuDeviceID,             // destination
 *     seqID,
 *     RESP_TYPE_DATA,
 *     0x10000000000,
 *     256*256*4,
 *     weightTensor              // tensor with data
 * );
 *
 * // 4. MCPU receives data and recycles tensor
 * dataMovementManager->recycleTensor(weightTensor);
 * ```
 *
 * **Event Packet Structure:**
 * ```
 * EventPacket (scheduled at tick)
 *   │
 *   └─ TensorReqEvent / TensorDataEvent
 *        ├─ tid: Transaction ID (uint64_t)
 *        ├─ name: Event identifier string
 *        ├─ pkt: TensorReqPacket / TensorDataPacket
 *        │    ├─ srcID, destID
 *        │    ├─ type, addr, size
 *        │    └─ pTensor: SimTensor*
 *        ├─ callee: SimPortManager* (caller reference)
 *        └─ callback: std::function<void(MasterPort*)>
 *
 * When event.process() is called:
 *   → Framework invokes packet's visitor pattern
 *   → Packet dispatches to appropriate handler in destination simulator
 *   → Callbacks executed for notification
 * ```
 *
 * **Integration with Simulators:**
 * ```cpp
 * // Each simulator inherits DataMovementManager
 * class MCPUSim : public CPPSimBase, public DataMovementManager {
 * public:
 *     MCPUSim(std::string name, std::shared_ptr<SimTensorManager> mgr)
 *         : CPPSimBase(name),
 *           DataMovementManager("MCPU_DMM", mgr) {}
 *
 *     // Can now use:
 *     //   - this->aquaireTensor(...)
 *     //   - this->sendTensorReq(...)
 *     //   - this->sendTensorData(...)
 *     //   - this->recycleTensor(...)
 * };
 *
 * class PESim : public CPPSimBase, public DataMovementManager {
 * public:
 *     PESim(std::string name, std::shared_ptr<SimTensorManager> mgr, int peID)
 *         : CPPSimBase(name),
 *           DataMovementManager("PEID_" + std::to_string(peID) + "_DMM", mgr),
 *           peID(peID) {}
 * };
 * ```
 *
 * **Callback Mechanism:**
 * ```
 * pushToEntryCallback:
 *   - Invoked when packet is pushed to destination queue
 *   - Use case: Update sender's outstanding request count
 *   - Example: sender->incrementPendingRequests()
 *
 * popFromEntryCallback:
 *   - Invoked when packet is popped from queue for processing
 *   - Use case: Update statistics, log event
 *   - Example: sender->decrementPendingRequests()
 *
 * Flow:
 *   1. sendTensorReq() → packet created
 *   2. sendPacketViaSimPort() → packet transmitted
 *   3. pushToEntryCallback() → packet enters dest queue
 *   4. popFromEntryCallback() → packet removed from queue
 *   5. event.process() → packet processed by destination
 * ```
 *
 * **Recycle Container Integration:**
 * ```cpp
 * // Packets are acquired from recycle container for efficiency
 * auto tensorReqPacket = top->getRecycleContainer()->acquire<TensorReqPacket>();
 * tensorReqPacket->renew(srcID, destID, type, addr, size, pTensor);
 *
 * // Benefits:
 * //   - Avoid repeated allocation/deallocation
 * //   - Reduce memory fragmentation
 * //   - Improve cache locality
 * //   - Lower overhead for high-frequency operations
 *
 * // Events are explicitly allocated (not recycled)
 * auto reqEvent = new TensorReqEvent(tid, name, packet, srcSim, nullptr);
 * auto eventPacket = new EventPacket(reqEvent, scheduleTick);
 * ```
 *
 * **Tensor Types and Usage:**
 * ```cpp
 * enum class TENSORTYPE {
 *     WEIGHT,      // Model weights (read-mostly, shared)
 *     ACTIVATION,  // Layer activations (intermediate results)
 *     INPUT,       // Network input tensors
 *     OUTPUT,      // Network output tensors
 *     GRADIENT,    // Backpropagation gradients
 *     BIAS,        // Bias parameters
 *     BUFFER       // Temporary buffers
 * };
 *
 * // Tensor parameters:
 * //   - addr: Memory address (global or PE-local)
 * //   - width: Number of columns
 * //   - height: Number of rows
 * //   - srcStride: Source memory stride (bytes between rows)
 * //   - destStride: Destination memory stride
 * //   - type: Tensor classification
 * ```
 *
 * **Sequence ID Management:**
 * ```cpp
 * static uint64_t dataMovementSequenceID = 0;
 *
 * // Usage:
 * uint64_t seqID = dataMovementSequenceID++;
 *
 * // Purpose:
 * //   - Track request-response pairs
 * //   - Distinguish between multiple requests to same destination
 * //   - Enable request reordering and out-of-order completion
 * //   - Debug and trace transaction flow
 * ```
 *
 * **Key Design Principles:**
 *
 * 1. **Resource Pooling:**
 *    - Tensors allocated from shared pool
 *    - Packets acquired from recycle container
 *    - Minimizes allocation overhead
 *
 * 2. **Unified Interface:**
 *    - Same API for all simulators
 *    - Consistent tensor handling across system
 *    - Simplified inter-component communication
 *
 * 3. **Transaction Tracking:**
 *    - Unique IDs for each transfer
 *    - Supports complex routing scenarios
 *    - Enables performance analysis
 *
 * 4. **Callback-Driven:**
 *    - Asynchronous notification
 *    - Decoupled sender/receiver
 *    - Flexible event handling
 *
 * 5. **Type Safety:**
 *    - Separate request/data packet types
 *    - Visitor pattern for dispatch
 *    - Compile-time type checking
 *
 * **Related Files:**
 * - @see DataMovementManager.hh - Class declaration and interfaces
 * - @see TensorPacket.hh - Packet type definitions
 * - @see SimTensorManager.hh - Tensor pool management
 * - @see SimTensor.hh - Tensor data structure
 * - @see testBlackBear.cc - System integration example
 *
 * @author ACAL/Playlab
 * @version 1.0
 * @date 2023-2025
 */

#include "system/DataMovementManager.hh"

#include "system/TensorPacket.hh"
#include "workloads/tensor/SimTensorManager.hh"

namespace acalsim {

uint64_t DataMovementManager::dataMovementSequenceID = 0;

SimTensor* DataMovementManager::aquaireTensor(const std::string& name, uint64_t _addr, uint32_t _width,
                                              uint32_t _height, uint32_t _srcStride, uint32_t _destStride,
                                              SimTensor::TENSORTYPE _type) {
	auto pTensor = pTensorManager->allocateTensor(name);
	pTensor->renew(_addr, _width, _height, _srcStride, _destStride, _type);
	return pTensor;
}
void DataMovementManager::recycleTensor(SimTensor* pTensor) { pTensorManager->freeTensor(pTensor); }

bool DataMovementManager::sendTensorReq(SimBase* srcSim, std::string dsSimName, std::string dsPortName,
                                        SimPortEvent::PushToEntryNotifyFnc  pushToEntryCallback,
                                        SimPortEvent::PopFromEntryNotifyFnc popFromEntryCallback, uint32_t srcID,
                                        uint32_t destID,
                                        uint64_t seqID,  // sequence ID, requests issued from the source
                                        int type, uint64_t addr, uint64_t size, SimTensor* pTensor) {
	// get downstream simulator
	// auto downstream = srcSim->getDownStream(dsSimName);

	// create a TensorReqPacket
	auto tensorReqPacket = top->getRecycleContainer()->acquire<TensorReqPacket>();
	tensorReqPacket->renew(srcID, destID, type, addr, size, pTensor);

	// create a TensorReqEvent and stuff the packet in the event
	uint64_t tid  = Transaction(srcID, destID, tensorReqPacket->getID()).getID();
	auto reqEvent = new TensorReqEvent(tid, "ReqEvent-TID-" + std::to_string(tid), tensorReqPacket, srcSim, nullptr);

	// pack the TensorReqEvent into an EventPacket
	auto eventPacket = new EventPacket(reqEvent, top->getGlobalTick() + 1);

	// send the tensor request via SimPort
	return srcSim->sendPacketViaSimPort(dsPortName, 0, eventPacket, pushToEntryCallback, popFromEntryCallback);
}

bool DataMovementManager::sendTensorData(SimBase* srcSim, std::string dsSimName, std::string dsPortName,
                                         SimPortEvent::PushToEntryNotifyFnc  pushToEntryCallback,
                                         SimPortEvent::PopFromEntryNotifyFnc popFromEntryCallback, uint32_t srcID,
                                         uint32_t destID,
                                         uint64_t seqID,  // sequence ID, requests issued from the source
                                         int type, uint64_t addr, uint64_t size, SimTensor* pTensor) {
	// get downstream simulator
	// auto downstream = srcSim->getDownStream(dsSimName);

	// create a TensorDataPacket
	auto tensorDataPacket = top->getRecycleContainer()->acquire<TensorDataPacket>();
	tensorDataPacket->renew(srcID, destID, type, addr, size, pTensor);

	// create a TensorDataEvent and stuff the packet in the event
	uint64_t tid = Transaction(srcID, destID, tensorDataPacket->getID()).getID();
	auto     dataEvent =
	    new TensorDataEvent(tid, "DataEvent-TID-" + std::to_string(tid), tensorDataPacket, srcSim, nullptr);

	// pack the TensorDataEvent into an EventPacket
	auto eventPacket = new EventPacket(dataEvent, top->getGlobalTick() + 1);

	// send the tensor data via SimPort
	return srcSim->sendPacketViaSimPort(dsPortName, 0, eventPacket, pushToEntryCallback, popFromEntryCallback);
}

}  // namespace acalsim
