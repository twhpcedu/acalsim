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

#include "QEMUBinaryComponent.hh"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

namespace ACALSim {
namespace QEMUBinary {

QEMUBinaryComponent::QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
    , out_(getSimulationOutput())
    , current_cycle_(0)
    , qemu_pid_(-1)
    , device_base_(0)
    , state_(QEMUState::IDLE)
    , server_fd_(-1)
    , client_fd_(-1)
    , socket_ready_(false)
    , device_link_(nullptr)
    , next_req_id_(1)
    , total_reads_(0)
    , total_writes_(0)
    , total_bytes_(0)
    , successful_transactions_(0)
    , failed_transactions_(0) {

    // Configure output
    int verbose = params.find<int>("verbose", 1);
    out_.init("QEMUBinary[@p:@l]: ", verbose, 0, SST::Output::STDOUT);

    out_.verbose(CALL_INFO, 1, 0, "Initializing QEMU Binary MMIO Component (Phase 2C)\n");

    // Get parameters
    binary_path_ = params.find<std::string>("binary_path", "");
    qemu_path_ = params.find<std::string>("qemu_path", "qemu-system-riscv32");
    socket_path_ = params.find<std::string>("socket_path", "/tmp/qemu-sst-mmio.sock");

    // Parse hex device_base
    std::string device_base_str = params.find<std::string>("device_base", "0x20000000");
    device_base_ = std::stoull(device_base_str, nullptr, 16);

    if (binary_path_.empty()) {
        out_.fatal(CALL_INFO, -1, "Error: binary_path parameter is required\n");
    }

    out_.verbose(CALL_INFO, 1, 0, "Configuration:\n");
    out_.verbose(CALL_INFO, 1, 0, "  Binary:      %s\n", binary_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  QEMU:        %s\n", qemu_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  Socket:      %s\n", socket_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  Device base: 0x%016" PRIx64 "\n", device_base_);

    // Register clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    tc_ = registerClock(clock_freq, new SST::Clock::Handler<QEMUBinaryComponent>(this, &QEMUBinaryComponent::clockTick));

    // Configure device link
    device_link_ = configureLink("device_port",
                                 new SST::Event::Handler<QEMUBinaryComponent>(this, &QEMUBinaryComponent::handleDeviceResponse));

    if (!device_link_) {
        out_.fatal(CALL_INFO, -1, "Error: Failed to configure device_port link\n");
    }

    out_.verbose(CALL_INFO, 1, 0, "Initialization complete\n");
}

QEMUBinaryComponent::~QEMUBinaryComponent() {
    terminateQEMU();

    if (client_fd_ >= 0) {
        close(client_fd_);
    }
    if (server_fd_ >= 0) {
        close(server_fd_);
    }
}

void QEMUBinaryComponent::setup() {
    out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");
    launchQEMU();
}

void QEMUBinaryComponent::finish() {
    out_.verbose(CALL_INFO, 1, 0, "Finish phase - simulation ending\n");

    // Print statistics
    out_.verbose(CALL_INFO, 1, 0, "=== QEMU Binary Component Statistics ===\n");
    out_.verbose(CALL_INFO, 1, 0, "  Total reads:        %" PRIu64 "\n", total_reads_);
    out_.verbose(CALL_INFO, 1, 0, "  Total writes:       %" PRIu64 "\n", total_writes_);
    out_.verbose(CALL_INFO, 1, 0, "  Total bytes:        %" PRIu64 "\n", total_bytes_);
    out_.verbose(CALL_INFO, 1, 0, "  Successful:         %" PRIu64 "\n", successful_transactions_);
    out_.verbose(CALL_INFO, 1, 0, "  Failed:             %" PRIu64 "\n", failed_transactions_);

    terminateQEMU();
}

bool QEMUBinaryComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    // Check QEMU status
    if (state_ == QEMUState::RUNNING || state_ == QEMUState::WAITING_DEVICE) {
        monitorQEMU();

        // Check for incoming MMIO requests from QEMU
        if (socket_ready_ && client_fd_ >= 0) {
            handleMMIORequest();
        }
    }

    // Continue simulation
    return false;
}

void QEMUBinaryComponent::setState(QEMUState new_state) {
    if (state_ != new_state) {
        out_.verbose(CALL_INFO, 2, 0, "State change: %d -> %d\n", static_cast<int>(state_), static_cast<int>(new_state));
        state_ = new_state;
    }
}

void QEMUBinaryComponent::handleDeviceResponse(SST::Event* ev) {
    auto* resp = dynamic_cast<MemoryResponseEvent*>(ev);
    if (!resp) {
        out_.verbose(CALL_INFO, 1, 0, "Error: Received non-MemoryResponseEvent\n");
        delete ev;
        return;
    }

    uint64_t req_id = resp->req_id;
    out_.verbose(CALL_INFO, 2, 0, "Received device response: req_id=%" PRIu64 " data=0x%" PRIx64 " success=%d\n",
                 req_id, resp->data, resp->success);

    // Find pending request
    auto it = pending_requests_.find(req_id);
    if (it == pending_requests_.end()) {
        out_.verbose(CALL_INFO, 1, 0, "Warning: No pending request for req_id %" PRIu64 "\n", req_id);
        delete ev;
        return;
    }

    PendingMMIORequest& pending = it->second;

    // Send MMIO response back to QEMU
    out_.verbose(CALL_INFO, 2, 0, "Sending MMIO response to QEMU: success=%d data=0x%" PRIx64 "\n",
                 resp->success, resp->data);
    sendMMIOResponse(resp->success, resp->data);

    // Update statistics
    if (resp->success) {
        successful_transactions_++;
    } else {
        failed_transactions_++;
    }

    // Remove from pending
    pending_requests_.erase(it);

    // Return to RUNNING state if no more pending requests
    if (pending_requests_.empty()) {
        setState(QEMUState::RUNNING);
    }

    delete ev;
}

void QEMUBinaryComponent::launchQEMU() {
    out_.verbose(CALL_INFO, 1, 0, "Launching QEMU process...\n");
    setState(QEMUState::LAUNCHING);

    // Remove existing socket
    unlink(socket_path_.c_str());

    // Setup server socket FIRST (before forking)
    // Create server socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        out_.fatal(CALL_INFO, -1, "Error: Failed to create server socket\n");
    }

    // Bind to socket path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_fd_);
        out_.fatal(CALL_INFO, -1, "Error: Failed to bind server socket: %s\n", strerror(errno));
    }

    // Listen for connections
    if (listen(server_fd_, 1) < 0) {
        close(server_fd_);
        out_.fatal(CALL_INFO, -1, "Error: Failed to listen on socket: %s\n", strerror(errno));
    }

    out_.verbose(CALL_INFO, 1, 0, "Server socket listening at %s\n", socket_path_.c_str());

    // Fork QEMU process
    qemu_pid_ = fork();

    if (qemu_pid_ < 0) {
        close(server_fd_);
        out_.fatal(CALL_INFO, -1, "Error: Failed to fork QEMU process\n");
    }

    if (qemu_pid_ == 0) {
        // Child process - exec QEMU as client
        // Close server socket in child
        close(server_fd_);

        // QEMU will connect to SST's server socket via a custom device parameter
        // The device loads at runtime and connects to the socket
        // For now, we launch QEMU and will implement the device connection separately

        const char* args[] = {
            qemu_path_.c_str(),
            "-M", "virt",
            "-bios", "none",
            "-nographic",
            "-kernel", binary_path_.c_str(),
            // TODO: Add -device sst-device,socket=<path> once QEMU device is ready
            NULL
        };

        out_.verbose(CALL_INFO, 1, 0, "Child: Executing QEMU\n");

        execvp(qemu_path_.c_str(), const_cast<char* const*>(args));

        // If exec fails
        fprintf(stderr, "Failed to exec QEMU: %s\n", strerror(errno));
        exit(1);
    }

    // Parent process - accept connection from QEMU
    out_.verbose(CALL_INFO, 1, 0, "QEMU PID: %d\n", qemu_pid_);
    out_.verbose(CALL_INFO, 1, 0, "Waiting for QEMU to connect...\n");

    // Set socket to non-blocking for accept with timeout
    int flags = fcntl(server_fd_, F_GETFL, 0);
    fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

    // Wait for QEMU to connect (with timeout)
    for (int i = 0; i < 50; i++) {
        client_fd_ = accept(server_fd_, NULL, NULL);
        if (client_fd_ >= 0) {
            // Connection accepted!
            close(server_fd_);
            server_fd_ = -1;

            // Set client socket to non-blocking
            flags = fcntl(client_fd_, F_GETFL, 0);
            fcntl(client_fd_, F_SETFL, flags | O_NONBLOCK);

            socket_ready_ = true;
            out_.verbose(CALL_INFO, 1, 0, "QEMU connected to MMIO socket\n");
            setState(QEMUState::RUNNING);
            return;
        }

        // Check if error is EAGAIN/EWOULDBLOCK (no connection yet)
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            close(server_fd_);
            out_.fatal(CALL_INFO, -1, "Error: accept() failed: %s\n", strerror(errno));
        }

        // Wait and retry
        out_.verbose(CALL_INFO, 3, 0, "Waiting for QEMU connection (attempt %d/50)...\n", i + 1);
        usleep(200000);  // 200ms
    }

    // For Phase 2C initial implementation, we'll continue even without connection
    // This allows us to test the component without the QEMU device ready yet
    out_.verbose(CALL_INFO, 1, 0, "Warning: QEMU device connection not established (device not implemented yet)\n");
    out_.verbose(CALL_INFO, 1, 0, "Phase 2C.1: Component framework ready, QEMU device implementation needed\n");
    setState(QEMUState::RUNNING);
}

void QEMUBinaryComponent::setupSocket() {
    // Socket setup is now integrated into launchQEMU()
    out_.verbose(CALL_INFO, 2, 0, "setupSocket() called (integrated into launchQEMU)\n");
}

void QEMUBinaryComponent::monitorQEMU() {
    if (qemu_pid_ <= 0) {
        return;
    }

    // Check QEMU process status
    int status;
    pid_t result = waitpid(qemu_pid_, &status, WNOHANG);

    if (result > 0) {
        // QEMU exited
        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            out_.verbose(CALL_INFO, 1, 0, "QEMU exited with code %d\n", exit_code);

            if (exit_code == 0) {
                setState(QEMUState::COMPLETED);
            } else {
                setState(QEMUState::ERROR);
            }
        } else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            out_.verbose(CALL_INFO, 1, 0, "QEMU terminated by signal %d\n", sig);
            setState(QEMUState::ERROR);
        }

        qemu_pid_ = -1;
    }
}

void QEMUBinaryComponent::handleMMIORequest() {
    // Try to read MMIO request from QEMU
    MMIORequest req;
    ssize_t bytes_read = read(client_fd_, &req, sizeof(req));

    if (bytes_read == 0) {
        // Connection closed
        out_.verbose(CALL_INFO, 2, 0, "QEMU closed MMIO connection\n");
        close(client_fd_);
        client_fd_ = -1;
        socket_ready_ = false;
        return;
    }

    if (bytes_read < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available yet
            return;
        }

        // Read error
        out_.verbose(CALL_INFO, 1, 0, "Error reading MMIO request: %s\n", strerror(errno));
        return;
    }

    if (bytes_read != sizeof(req)) {
        out_.verbose(CALL_INFO, 1, 0, "Warning: Incomplete MMIO request (got %zd bytes, expected %zu)\n",
                     bytes_read, sizeof(req));
        return;
    }

    // Parse MMIO request
    out_.verbose(CALL_INFO, 2, 0, "Received MMIO %s: addr=0x%016" PRIx64 " data=0x%016" PRIx64 " size=%u\n",
                 req.type == 0 ? "READ" : "WRITE", req.addr, req.data, req.size);

    // Update statistics
    if (req.type == 0) {
        total_reads_++;
    } else {
        total_writes_++;
    }
    total_bytes_ += req.size;

    // Send to SST device
    sendDeviceRequest(req.type, req.addr, req.data, req.size);

    setState(QEMUState::WAITING_DEVICE);
}

void QEMUBinaryComponent::sendMMIOResponse(bool success, uint64_t data) {
    MMIOResponse resp;
    resp.success = success ? 1 : 0;
    memset(resp.reserved, 0, sizeof(resp.reserved));
    resp.data = data;

    ssize_t bytes_written = write(client_fd_, &resp, sizeof(resp));

    if (bytes_written != sizeof(resp)) {
        out_.verbose(CALL_INFO, 1, 0, "Error: Failed to send complete MMIO response\n");
    }
}

void QEMUBinaryComponent::sendDeviceRequest(uint8_t type, uint64_t addr, uint64_t data, uint8_t size) {
    // Create MemoryTransactionEvent
    TransactionType tx_type = (type == 0) ? TransactionType::READ : TransactionType::WRITE;
    uint64_t req_id = next_req_id_++;

    auto* event = new MemoryTransactionEvent(tx_type, device_base_ + addr, static_cast<uint32_t>(data), req_id);

    out_.verbose(CALL_INFO, 2, 0, "Sending device request: type=%d addr=0x%016" PRIx64 " data=0x%08x req_id=%" PRIu64 "\n",
                 static_cast<int>(tx_type), device_base_ + addr, static_cast<uint32_t>(data), req_id);

    // Store pending request
    PendingMMIORequest pending;
    pending.request.type = type;
    pending.request.size = size;
    pending.request.addr = addr;
    pending.request.data = data;
    pending.sst_req_id = req_id;

    pending_requests_[req_id] = pending;

    // Send to device
    device_link_->send(event);
}

void QEMUBinaryComponent::terminateQEMU() {
    if (qemu_pid_ > 0) {
        out_.verbose(CALL_INFO, 1, 0, "Terminating QEMU process (PID: %d)\n", qemu_pid_);

        // Send SIGTERM
        kill(qemu_pid_, SIGTERM);

        // Wait for termination (with timeout)
        for (int i = 0; i < 10; i++) {
            int status;
            pid_t result = waitpid(qemu_pid_, &status, WNOHANG);

            if (result > 0) {
                out_.verbose(CALL_INFO, 2, 0, "QEMU terminated\n");
                qemu_pid_ = -1;
                return;
            }

            usleep(100000);  // 100ms
        }

        // Force kill if still running
        out_.verbose(CALL_INFO, 1, 0, "QEMU did not terminate, sending SIGKILL\n");
        kill(qemu_pid_, SIGKILL);
        waitpid(qemu_pid_, NULL, 0);
        qemu_pid_ = -1;
    }
}

bool QEMUBinaryComponent::isQEMURunning() {
    if (qemu_pid_ <= 0) {
        return false;
    }

    int status;
    pid_t result = waitpid(qemu_pid_, &status, WNOHANG);
    return result == 0;  // 0 means process is still running
}

} // namespace QEMUBinary
} // namespace ACALSim
