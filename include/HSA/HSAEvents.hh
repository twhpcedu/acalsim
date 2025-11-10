/*
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _HSA_EVENTS_HH
#define _HSA_EVENTS_HH

#include <sst/core/event.h>
#include <sst/core/sst_types.h>

// HSA Packet Types (from HSA 1.2 spec section 2.8)
#define HSA_PACKET_TYPE_VENDOR_SPECIFIC  0
#define HSA_PACKET_TYPE_INVALID          1
#define HSA_PACKET_TYPE_KERNEL_DISPATCH  2
#define HSA_PACKET_TYPE_BARRIER_AND      3
#define HSA_PACKET_TYPE_AGENT_DISPATCH   4
#define HSA_PACKET_TYPE_BARRIER_OR       5

// HSA Fence Scope
#define HSA_FENCE_SCOPE_NONE    0
#define HSA_FENCE_SCOPE_AGENT   1
#define HSA_FENCE_SCOPE_SYSTEM  2

namespace SST {
namespace ACALSim {
namespace HSA {

/**
 * HSA AQL (Architected Queuing Language) Packet Event
 *
 * Represents a job descriptor sent through HSA queues from host to compute agent.
 * Based on HSA 1.2 specification section 2.8 (Kernel Dispatch Packet).
 */
class HSAAQLPacketEvent : public SST::Event {
public:
    // Packet header (16 bits)
    uint16_t header;               // Packet type (bits 8-15) + setup (bits 0-7)
    uint16_t setup;                // Dimensions and flags

    // Work-group size
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;

    // Grid size
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;

    // Kernel information
    uint64_t kernel_object;        // Address of kernel code
    uint64_t kernarg_address;      // Address of kernel arguments

    // Private and group memory
    uint32_t private_segment_size;
    uint32_t group_segment_size;

    // Completion signal
    uint64_t completion_signal;    // Signal handle for completion notification

    // Metadata (not part of HSA spec, for simulation)
    uint64_t submit_time;          // Simulation time when submitted
    uint32_t dispatch_id;          // Unique dispatch ID

    HSAAQLPacketEvent() : Event(),
        header(0), setup(0),
        workgroup_size_x(1), workgroup_size_y(1), workgroup_size_z(1),
        grid_size_x(1), grid_size_y(1), grid_size_z(1),
        kernel_object(0), kernarg_address(0),
        private_segment_size(0), group_segment_size(0),
        completion_signal(0),
        submit_time(0), dispatch_id(0) {}

    // Get packet type from header
    uint8_t getPacketType() const {
        return (header >> 8) & 0xFF;
    }

    // Set packet type in header
    void setPacketType(uint8_t type) {
        header = (header & 0x00FF) | (type << 8);
    }

    // Get total work-items
    uint64_t getTotalWorkitems() const {
        return static_cast<uint64_t>(grid_size_x) *
               static_cast<uint64_t>(grid_size_y) *
               static_cast<uint64_t>(grid_size_z);
    }

    // Get work-groups count
    uint64_t getTotalWorkgroups() const {
        uint64_t wg_x = (grid_size_x + workgroup_size_x - 1) / workgroup_size_x;
        uint64_t wg_y = (grid_size_y + workgroup_size_y - 1) / workgroup_size_y;
        uint64_t wg_z = (grid_size_z + workgroup_size_z - 1) / workgroup_size_z;
        return wg_x * wg_y * wg_z;
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & header;
        ser & setup;
        ser & workgroup_size_x;
        ser & workgroup_size_y;
        ser & workgroup_size_z;
        ser & grid_size_x;
        ser & grid_size_y;
        ser & grid_size_z;
        ser & kernel_object;
        ser & kernarg_address;
        ser & private_segment_size;
        ser & group_segment_size;
        ser & completion_signal;
        ser & submit_time;
        ser & dispatch_id;
    }

    ImplementSerializable(SST::ACALSim::HSA::HSAAQLPacketEvent);
};

/**
 * HSA Signal Event
 *
 * Represents signal operations for completion notification and synchronization.
 * Based on HSA 1.2 specification section 2.11 (Signals).
 */
class HSASignalEvent : public SST::Event {
public:
    uint64_t signal_handle;        // Signal identifier
    int64_t signal_value;          // Signal value (0 typically means completed)

    // Signal operation type
    enum class Operation {
        WAIT,                      // Wait for signal value
        STORE,                     // Store value to signal
        ADD,                       // Atomic add to signal
        SUB,                       // Atomic subtract from signal
        EXCHANGE,                  // Atomic exchange
        CAS                        // Compare and swap
    } operation;

    // Metadata
    uint64_t timestamp;            // Simulation time
    uint32_t dispatch_id;          // Associated dispatch ID

    HSASignalEvent() : Event(),
        signal_handle(0), signal_value(0),
        operation(Operation::STORE),
        timestamp(0), dispatch_id(0) {}

    HSASignalEvent(uint64_t handle, int64_t value, Operation op = Operation::STORE) : Event(),
        signal_handle(handle), signal_value(value),
        operation(op),
        timestamp(0), dispatch_id(0) {}

    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & signal_handle;
        ser & signal_value;
        ser & operation;
        ser & timestamp;
        ser & dispatch_id;
    }

    ImplementSerializable(SST::ACALSim::HSA::HSASignalEvent);
};

/**
 * HSA Doorbell Event
 *
 * Represents doorbell ring notification from host to compute agent.
 * Signals that new work has been queued.
 */
class HSADoorbellEvent : public SST::Event {
public:
    uint32_t queue_id;             // Queue identifier
    uint64_t doorbell_value;       // New write index value
    uint64_t timestamp;            // Simulation time

    HSADoorbellEvent() : Event(),
        queue_id(0), doorbell_value(0), timestamp(0) {}

    HSADoorbellEvent(uint32_t qid, uint64_t value) : Event(),
        queue_id(qid), doorbell_value(value), timestamp(0) {}

    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & queue_id;
        ser & doorbell_value;
        ser & timestamp;
    }

    ImplementSerializable(SST::ACALSim::HSA::HSADoorbellEvent);
};

/**
 * HSA Memory Event
 *
 * Represents memory operations (kernel argument copying, result retrieval).
 */
class HSAMemoryEvent : public SST::Event {
public:
    enum class Type {
        READ,
        WRITE,
        COPY
    } type;

    uint64_t address;              // Memory address
    uint64_t size;                 // Size in bytes
    uint8_t *data;                 // Data buffer (for simulation)
    bool is_complete;              // Completion status

    HSAMemoryEvent() : Event(),
        type(Type::READ), address(0), size(0), data(nullptr), is_complete(false) {}

    ~HSAMemoryEvent() {
        if (data) delete[] data;
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & type;
        ser & address;
        ser & size;
        ser & is_complete;
        // Note: data buffer not serialized for simplicity
    }

    ImplementSerializable(SST::ACALSim::HSA::HSAMemoryEvent);
};

} // namespace HSA
} // namespace ACALSim
} // namespace SST

#endif // _HSA_EVENTS_HH
