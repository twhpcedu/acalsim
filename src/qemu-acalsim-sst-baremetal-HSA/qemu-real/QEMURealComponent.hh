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

#ifndef QEMU_REAL_COMPONENT_HH
#define QEMU_REAL_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <map>
#include <queue>
#include <string>
#include <sys/types.h>

// Include event definitions from ACALSim device
#include "../acalsim-device/ACALSimDeviceComponent.hh"

namespace ACALSim {
namespace QEMUReal {

// Re-use the event classes and transaction types from QEMUIntegration
using ACALSim::QEMUIntegration::MemoryTransactionEvent;
using ACALSim::QEMUIntegration::MemoryResponseEvent;
using ACALSim::QEMUIntegration::TransactionType;

// QEMU state
enum class QEMUState {
    IDLE,           // Not started
    LAUNCHING,      // Starting QEMU process
    RUNNING,        // QEMU running, monitoring serial
    WAITING_DEVICE, // Waiting for SST device response
    COMPLETED,      // QEMU exited successfully
    ERROR           // Error occurred
};

// Pending QEMU request (waiting for SST device response)
struct PendingQEMURequest {
    std::string command;  // Original SST protocol command
    uint64_t sst_req_id;  // SST request ID
    uint64_t address;
    TransactionType type;
};

/*
 * QEMURealComponent
 *
 * SST component that manages a QEMU subprocess and bridges communication
 * between RISC-V programs running in QEMU and SST device components.
 *
 * Architecture:
 *   - Launches QEMU as subprocess with serial over Unix socket
 *   - Monitors QEMU serial output for SST protocol commands
 *   - Parses commands and translates to MemoryTransactionEvent
 *   - Forwards events to SST device component
 *   - Returns responses to QEMU via serial
 */
class QEMURealComponent : public SST::Component {
public:
    // SST registration macro
    SST_ELI_REGISTER_COMPONENT(
        QEMURealComponent,
        "qemureal",
        "QEMUReal",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "QEMU Real Component - Runs actual QEMU process with RISC-V binary",
        COMPONENT_CATEGORY_PROCESSOR)

    // Parameter documentation
    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level (0-3)", "1"},
        {"binary_path", "Path to RISC-V ELF binary", ""},
        {"qemu_path", "Path to qemu-system-riscv32", "qemu-system-riscv32"},
        {"socket_path", "Path to Unix socket for serial", "/tmp/qemu-sst.sock"})

    // Port documentation
    SST_ELI_DOCUMENT_PORTS({"device_port", "Port to memory/device subsystem", {"MemoryTransactionEvent"}})

    // Constructor and destructor
    QEMURealComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMURealComponent();

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

    // Serial communication
    void setupSerial();
    void handleSerialData();
    void sendSerialResponse(const std::string& response);

    // Protocol parsing
    void parseCommand(const std::string& line);
    bool parseSSTCommand(const std::string& cmd, std::string& operation, uint64_t& addr, uint32_t& data);

    // SST device communication
    void sendDeviceRequest(TransactionType type, uint64_t addr, uint32_t data);

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
    QEMUState state_;

    // Member variables - Serial communication
    int serial_fd_;
    std::string serial_buffer_;
    bool serial_ready_;

    // Member variables - SST communication
    SST::Link* device_link_;
    std::map<uint64_t, PendingQEMURequest> pending_requests_;
    uint64_t next_req_id_;

    // Member variables - Statistics
    uint64_t total_commands_;
    uint64_t total_writes_;
    uint64_t total_reads_;
    uint64_t successful_transactions_;
    uint64_t failed_transactions_;
};

} // namespace QEMUReal
} // namespace ACALSim

#endif // QEMU_REAL_COMPONENT_HH
