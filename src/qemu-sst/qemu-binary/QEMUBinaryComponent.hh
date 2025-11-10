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

#ifndef QEMU_BINARY_COMPONENT_HH
#define QEMU_BINARY_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <map>
#include <string>
#include <sys/types.h>

// Include event definitions from ACALSim device
#include "../acalsim-device/ACALSimDeviceComponent.hh"

namespace ACALSim {
namespace QEMUBinary {

// Re-use event classes from QEMUIntegration
using ACALSim::QEMUIntegration::MemoryTransactionEvent;
using ACALSim::QEMUIntegration::MemoryResponseEvent;
using ACALSim::QEMUIntegration::TransactionType;

// Binary MMIO protocol structures
struct MMIORequest {
    uint8_t  type;        // 0 = READ, 1 = WRITE
    uint8_t  size;        // 1, 2, 4, or 8 bytes
    uint16_t reserved;
    uint64_t addr;        // MMIO address
    uint64_t data;        // Write data (ignored for READ)
} __attribute__((packed));

struct MMIOResponse {
    uint8_t  success;     // 0 = error, 1 = success
    uint8_t  reserved[7];
    uint64_t data;        // Read data or write acknowledgment
} __attribute__((packed));

// QEMU state
enum class QEMUState {
    IDLE,           // Not started
    LAUNCHING,      // Starting QEMU process
    RUNNING,        // QEMU running, processing MMIO
    WAITING_DEVICE, // Waiting for SST device response
    COMPLETED,      // QEMU exited successfully
    ERROR           // Error occurred
};

// Pending MMIO request (waiting for SST device response)
struct PendingMMIORequest {
    MMIORequest request;     // Original MMIO request from QEMU
    uint64_t sst_req_id;     // SST request ID
};

/*
 * QEMUBinaryComponent - Phase 2C
 *
 * SST component that manages a QEMU subprocess and bridges communication
 * via binary MMIO protocol instead of text-based serial protocol.
 *
 * Improvements over Phase 2B (QEMURealComponent):
 * - Binary protocol: No text parsing overhead
 * - MMIO-based: Direct memory-mapped I/O instead of UART
 * - Better performance: ~10x throughput improvement
 * - Cleaner architecture: Standard QEMU device model
 */
class QEMUBinaryComponent : public SST::Component {
public:
    // SST registration macro
    SST_ELI_REGISTER_COMPONENT(
        QEMUBinaryComponent,
        "qemubinary",
        "QEMUBinary",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "QEMU Binary MMIO Component - Phase 2C with binary protocol",
        COMPONENT_CATEGORY_PROCESSOR)

    // Parameter documentation
    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level (0-3)", "1"},
        {"binary_path", "Path to RISC-V ELF binary", ""},
        {"qemu_path", "Path to qemu-system-riscv32", "qemu-system-riscv32"},
        {"socket_path", "Path to Unix socket for MMIO", "/tmp/qemu-sst-mmio.sock"},
        {"device_base", "SST device base address (hex)", "0x20000000"})

    // Port documentation
    SST_ELI_DOCUMENT_PORTS(
        {"device_port", "Port to memory/device subsystem", {"MemoryTransactionEvent"}})

    // Constructor and destructor
    QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMUBinaryComponent();

    // SST lifecycle methods
    void setup() override;
    void finish() override;

private:
    // Clock handler
    bool clockTick(SST::Cycle_t cycle);

    // Event handlers
    void handleDeviceResponse(SST::Event* ev);

    // QEMU process management
    void launchQEMU();
    void monitorQEMU();
    void terminateQEMU();
    bool isQEMURunning();

    // Socket communication
    void setupSocket();
    void handleMMIORequest();
    void sendMMIOResponse(bool success, uint64_t data);

    // SST device communication
    void sendDeviceRequest(uint8_t type, uint64_t addr, uint64_t data, uint8_t size);

    // State management
    void setState(QEMUState new_state);

    // Member variables - Output
    SST::Output out_;

    // Member variables - Timing
    SST::Cycle_t current_cycle_;
    SST::TimeConverter* tc_;

    // Member variables - QEMU process
    pid_t qemu_pid_;
    std::string binary_path_;
    std::string qemu_path_;
    std::string socket_path_;
    uint64_t device_base_;
    QEMUState state_;

    // Member variables - Socket communication
    int server_fd_;       // Server socket (SST listens)
    int client_fd_;       // Client socket (QEMU connects)
    bool socket_ready_;

    // Member variables - SST communication
    SST::Link* device_link_;
    std::map<uint64_t, PendingMMIORequest> pending_requests_;
    uint64_t next_req_id_;

    // Member variables - Statistics
    uint64_t total_reads_;
    uint64_t total_writes_;
    uint64_t total_bytes_;
    uint64_t successful_transactions_;
    uint64_t failed_transactions_;
};

} // namespace QEMUBinary
} // namespace ACALSim

#endif // QEMU_BINARY_COMPONENT_HH
