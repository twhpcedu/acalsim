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
    , use_multi_device_(false)
    , num_devices_(1)
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

    // Check for N-device mode
    num_devices_ = params.find<int>("num_devices", 1);
    use_multi_device_ = (num_devices_ > 1);

    out_.verbose(CALL_INFO, 1, 0, "Configuration:\n");
    out_.verbose(CALL_INFO, 1, 0, "  Binary:      %s\n", binary_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  QEMU:        %s\n", qemu_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  Socket:      %s\n", socket_path_.c_str());

    if (use_multi_device_) {
        out_.verbose(CALL_INFO, 1, 0, "  Mode:        N-device (%d devices)\n", num_devices_);
    } else {
        out_.verbose(CALL_INFO, 1, 0, "  Mode:        Single device (legacy)\n");
        out_.verbose(CALL_INFO, 1, 0, "  Device base: 0x%016" PRIx64 "\n", device_base_);
    }

    // Register clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    tc_ = registerClock(clock_freq, new SST::Clock::Handler<QEMUBinaryComponent>(this, &QEMUBinaryComponent::clockTick));

    if (use_multi_device_) {
        // N-device mode: Configure multiple device ports
        out_.verbose(CALL_INFO, 1, 0, "Configuring %d device ports:\n", num_devices_);

        for (int i = 0; i < num_devices_; i++) {
            DeviceInfo dev;

            // Get device parameters
            std::string base_key = "device" + std::to_string(i) + "_base";
            std::string size_key = "device" + std::to_string(i) + "_size";
            std::string name_key = "device" + std::to_string(i) + "_name";

            std::string base_str = params.find<std::string>(base_key, "0x10000000");
            dev.base_addr = std::stoull(base_str, nullptr, 16);
            dev.size = params.find<uint64_t>(size_key, 4096);
            dev.name = params.find<std::string>(name_key, "device" + std::to_string(i));
            dev.num_requests = 0;

            // Initialize socket fields (will be set up in launchQEMU)
            dev.socket_path = "";
            dev.server_fd = -1;
            dev.client_fd = -1;
            dev.socket_ready = false;

            // Configure port
            std::string port_name = "device_port_" + std::to_string(i);
            dev.link = configureLink(port_name,
                new SST::Event::Handler<QEMUBinaryComponent>(this, &QEMUBinaryComponent::handleDeviceResponse));

            if (!dev.link) {
                out_.fatal(CALL_INFO, -1, "Error: Failed to configure %s\n", port_name.c_str());
            }

            devices_.push_back(dev);

            out_.verbose(CALL_INFO, 1, 0, "  Device %d (%s): [0x%016" PRIx64 ", 0x%016" PRIx64 ")\n",
                        i, dev.name.c_str(), dev.base_addr, dev.base_addr + dev.size);
        }
    } else {
        // Legacy single device mode: Configure single device_port
        device_link_ = configureLink("device_port",
                                     new SST::Event::Handler<QEMUBinaryComponent>(this, &QEMUBinaryComponent::handleDeviceResponse));

        if (!device_link_) {
            out_.fatal(CALL_INFO, -1, "Error: Failed to configure device_port link\n");
        }
    }

    out_.verbose(CALL_INFO, 1, 0, "Initialization complete\n");
}

QEMUBinaryComponent::~QEMUBinaryComponent() {
    terminateQEMU();

    // Close legacy single-device sockets
    if (client_fd_ >= 0) {
        close(client_fd_);
    }
    if (server_fd_ >= 0) {
        close(server_fd_);
    }

    // Close N-device sockets
    for (auto& dev : devices_) {
        if (dev.client_fd >= 0) {
            close(dev.client_fd);
        }
        if (dev.server_fd >= 0) {
            close(dev.server_fd);
        }
        if (!dev.socket_path.empty()) {
            unlink(dev.socket_path.c_str());
        }
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

    // Print per-device statistics in multi-device mode
    if (use_multi_device_) {
        out_.verbose(CALL_INFO, 1, 0, "\n=== Per-Device Statistics ===\n");
        for (size_t i = 0; i < devices_.size(); i++) {
            const DeviceInfo& dev = devices_[i];
            out_.verbose(CALL_INFO, 1, 0, "  Device %zu (%s):\n", i, dev.name.c_str());
            out_.verbose(CALL_INFO, 1, 0, "    Base address:  0x%016" PRIx64 "\n", dev.base_addr);
            out_.verbose(CALL_INFO, 1, 0, "    Size:          %" PRIu64 " bytes\n", dev.size);
            out_.verbose(CALL_INFO, 1, 0, "    Requests:      %" PRIu64 "\n", dev.num_requests);
        }
    }

    terminateQEMU();
}

bool QEMUBinaryComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    // Check QEMU status
    if (state_ == QEMUState::RUNNING || state_ == QEMUState::WAITING_DEVICE) {
        monitorQEMU();

        // Check for incoming MMIO requests from QEMU
        if (use_multi_device_) {
            // N-device mode: poll all device sockets
            pollDeviceSockets();
        } else {
            // Legacy single-device mode
            if (socket_ready_ && client_fd_ >= 0) {
                handleMMIORequest();
            }
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

    uint64_t req_id = resp->getReqId();
    uint32_t data = resp->getData();
    bool success = resp->getSuccess();

    out_.verbose(CALL_INFO, 2, 0, "Received device response: req_id=%" PRIu64 " data=0x%08x success=%d\n",
                 req_id, data, success);

    // Find pending request
    auto it = pending_requests_.find(req_id);
    if (it == pending_requests_.end()) {
        out_.verbose(CALL_INFO, 1, 0, "Warning: No pending request for req_id %" PRIu64 "\n", req_id);
        delete ev;
        return;
    }

    // Send MMIO response back to QEMU
    out_.verbose(CALL_INFO, 2, 0, "Sending MMIO response to QEMU: success=%d data=0x%08x\n",
                 success, data);
    sendMMIOResponse(success, data);

    // Update statistics
    if (success) {
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

    if (use_multi_device_) {
        // N-device mode: Create N socket servers
        out_.verbose(CALL_INFO, 1, 0, "Setting up %zu device sockets...\n", devices_.size());
        for (size_t i = 0; i < devices_.size(); i++) {
            setupDeviceSocket(&devices_[i], i);
        }
    } else {
        // Legacy single-device mode
        unlink(socket_path_.c_str());

        // Setup server socket
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
    }

    // Fork QEMU process
    qemu_pid_ = fork();

    if (qemu_pid_ < 0) {
        close(server_fd_);
        out_.fatal(CALL_INFO, -1, "Error: Failed to fork QEMU process\n");
    }

    if (qemu_pid_ == 0) {
        // Child process - exec QEMU as client
        // Close server sockets in child
        if (use_multi_device_) {
            for (auto& dev : devices_) {
                close(dev.server_fd);
            }
        } else {
            close(server_fd_);
        }

        // Build QEMU command line arguments dynamically
        std::vector<const char*> args;
        args.push_back(qemu_path_.c_str());
        args.push_back("-M");
        args.push_back("virt");
        args.push_back("-bios");
        args.push_back("none");
        args.push_back("-nographic");
        args.push_back("-kernel");
        args.push_back(binary_path_.c_str());

        // Add SST devices
        std::vector<std::string> device_args;
        if (use_multi_device_) {
            // N-device mode: use per-device socket paths
            for (size_t i = 0; i < devices_.size(); i++) {
                char addr_buf[32];
                snprintf(addr_buf, sizeof(addr_buf), "0x%lx", devices_[i].base_addr);
                std::string dev_arg = "sst-device,socket=" + devices_[i].socket_path +
                                     ",base_address=" + std::string(addr_buf);
                device_args.push_back(dev_arg);
                args.push_back("-device");
                args.push_back(device_args.back().c_str());
            }
        } else if (device_link_) {
            // Legacy single device mode
            char addr_buf[32];
            snprintf(addr_buf, sizeof(addr_buf), "0x%lx", device_base_);
            std::string dev_arg = "sst-device,socket=" + socket_path_ +
                                 ",base_address=" + std::string(addr_buf);
            device_args.push_back(dev_arg);
            args.push_back("-device");
            args.push_back(device_args.back().c_str());
        }

        args.push_back(NULL);

        out_.verbose(CALL_INFO, 1, 0, "Child: Executing QEMU with %zu devices\n",
                     use_multi_device_ ? devices_.size() : 1);

        execvp(qemu_path_.c_str(), const_cast<char* const*>(args.data()));

        // If exec fails
        fprintf(stderr, "Failed to exec QEMU: %s\n", strerror(errno));
        exit(1);
    }

    // Parent process
    out_.verbose(CALL_INFO, 1, 0, "QEMU PID: %d\n", qemu_pid_);

    if (use_multi_device_) {
        // N-device mode: Connections accepted asynchronously in clockTick()
        out_.verbose(CALL_INFO, 1, 0, "Waiting for %zu device connections (async)...\n", devices_.size());
        setState(QEMUState::RUNNING);
    } else {
        // Legacy single-device mode: Accept connection synchronously
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
    if (use_multi_device_) {
        // N-device mode: Route to correct device based on address
        DeviceInfo* device = findDeviceForAddress(addr);
        if (device) {
            routeToDevice(device, type, addr, data, size);
        } else {
            // No device found for this address - send error response
            out_.verbose(CALL_INFO, 1, 0, "ERROR: No device mapped at address 0x%016" PRIx64 "\n", addr);
            sendMMIOResponse(false, 0);  // Send error response to QEMU
        }
    } else {
        // Legacy single device mode
        TransactionType tx_type = (type == 0) ? TransactionType::LOAD : TransactionType::STORE;
        uint64_t req_id = next_req_id_++;

        auto* event = new MemoryTransactionEvent(tx_type, device_base_ + addr, static_cast<uint32_t>(data),
                                                 static_cast<uint32_t>(size), req_id);

        out_.verbose(CALL_INFO, 2, 0, "Sending device request: type=%d addr=0x%016" PRIx64 " data=0x%08x size=%u req_id=%" PRIu64 "\n",
                     static_cast<int>(tx_type), device_base_ + addr, static_cast<uint32_t>(data), size, req_id);

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
}

DeviceInfo* QEMUBinaryComponent::findDeviceForAddress(uint64_t address) {
    for (auto& dev : devices_) {
        if (address >= dev.base_addr && address < dev.base_addr + dev.size) {
            return &dev;
        }
    }
    return nullptr;
}

void QEMUBinaryComponent::routeToDevice(DeviceInfo* device, uint8_t type, uint64_t addr, uint64_t data, uint8_t size) {
    TransactionType tx_type = (type == 0) ? TransactionType::LOAD : TransactionType::STORE;
    uint64_t req_id = next_req_id_++;

    auto* event = new MemoryTransactionEvent(tx_type, addr, static_cast<uint32_t>(data),
                                             static_cast<uint32_t>(size), req_id);

    out_.verbose(CALL_INFO, 2, 0, "Routing to %s: type=%d addr=0x%016" PRIx64 " data=0x%08x size=%u req_id=%" PRIu64 "\n",
                 device->name.c_str(), static_cast<int>(tx_type), addr, static_cast<uint32_t>(data), size, req_id);

    // Store pending request
    PendingMMIORequest pending;
    pending.request.type = type;
    pending.request.size = size;
    pending.request.addr = addr;
    pending.request.data = data;
    pending.sst_req_id = req_id;
    pending_requests_[req_id] = pending;

    // Update statistics
    device->num_requests++;
    if (type == 0) {
        total_reads_++;
    } else {
        total_writes_++;
    }
    total_bytes_ += size;

    // Send to device
    device->link->send(event);
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

// =============================================================================
// N-Socket Server Implementation
// =============================================================================

void QEMUBinaryComponent::setupDeviceSocket(DeviceInfo* device, int index) {
    // Generate unique socket path
    device->socket_path = "/tmp/qemu-sst-device" + std::to_string(index) + ".sock";

    // Remove existing socket file
    unlink(device->socket_path.c_str());

    // Create server socket
    device->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (device->server_fd < 0) {
        out_.fatal(CALL_INFO, -1, "Error: Failed to create socket for device %s\n",
                   device->name.c_str());
    }

    // Bind socket
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, device->socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(device->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(device->server_fd);
        out_.fatal(CALL_INFO, -1, "Error: Failed to bind socket %s: %s\n",
                   device->socket_path.c_str(), strerror(errno));
    }

    // Listen
    if (listen(device->server_fd, 1) < 0) {
        close(device->server_fd);
        out_.fatal(CALL_INFO, -1, "Error: Failed to listen on %s: %s\n",
                   device->socket_path.c_str(), strerror(errno));
    }

    // Set non-blocking
    int flags = fcntl(device->server_fd, F_GETFL, 0);
    fcntl(device->server_fd, F_SETFL, flags | O_NONBLOCK);

    out_.verbose(CALL_INFO, 2, 0, "Device %s socket listening at %s\n",
                 device->name.c_str(), device->socket_path.c_str());
}

void QEMUBinaryComponent::acceptDeviceConnection(DeviceInfo* device) {
    if (device->socket_ready || device->server_fd < 0) {
        return;  // Already connected or no server
    }

    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);

    device->client_fd = accept(device->server_fd, (struct sockaddr*)&client_addr, &client_len);

    if (device->client_fd < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            out_.verbose(CALL_INFO, 1, 0, "Error accepting connection for %s: %s\n",
                         device->name.c_str(), strerror(errno));
        }
        return;
    }

    // Set client socket non-blocking
    int flags = fcntl(device->client_fd, F_GETFL, 0);
    fcntl(device->client_fd, F_SETFL, flags | O_NONBLOCK);

    device->socket_ready = true;
    out_.verbose(CALL_INFO, 1, 0, "Device %s connected\n", device->name.c_str());
}

void QEMUBinaryComponent::pollDeviceSockets() {
    for (auto& dev : devices_) {
        // Try to accept connection if not yet connected
        if (!dev.socket_ready) {
            acceptDeviceConnection(&dev);
        }

        // Poll for MMIO requests if connected
        if (dev.socket_ready && dev.client_fd >= 0) {
            handleMMIORequest(&dev);
        }
    }
}

void QEMUBinaryComponent::handleMMIORequest(DeviceInfo* device) {
    if (!device->socket_ready || device->client_fd < 0) {
        return;
    }

    MMIORequest req;
    ssize_t bytes_read = read(device->client_fd, &req, sizeof(req));

    if (bytes_read == 0) {
        // Client disconnected
        out_.verbose(CALL_INFO, 1, 0, "Device %s disconnected\n", device->name.c_str());
        close(device->client_fd);
        device->client_fd = -1;
        device->socket_ready = false;
        return;
    }

    if (bytes_read < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            out_.verbose(CALL_INFO, 1, 0, "Error reading from %s: %s\n",
                         device->name.c_str(), strerror(errno));
        }
        return;
    }

    if (bytes_read != sizeof(req)) {
        out_.verbose(CALL_INFO, 1, 0, "Incomplete request from %s\n", device->name.c_str());
        return;
    }

    // Process request
    out_.verbose(CALL_INFO, 2, 0, "MMIO from %s: type=%d addr=0x%016" PRIx64 " data=0x%08" PRIx64 " size=%u\n",
                 device->name.c_str(), req.type, req.addr, req.data, req.size);

    setState(QEMUState::WAITING_DEVICE);
    sendDeviceRequest(req.type, req.addr, req.data, req.size);
}

void QEMUBinaryComponent::sendMMIOResponse(DeviceInfo* device, bool success, uint64_t data) {
    if (!device->socket_ready || device->client_fd < 0) {
        out_.verbose(CALL_INFO, 1, 0, "Cannot send response to %s: not connected\n",
                     device->name.c_str());
        return;
    }

    MMIOResponse resp;
    resp.success = success ? 1 : 0;
    memset(resp.reserved, 0, sizeof(resp.reserved));
    resp.data = data;

    ssize_t bytes_written = write(device->client_fd, &resp, sizeof(resp));

    if (bytes_written != sizeof(resp)) {
        out_.verbose(CALL_INFO, 1, 0, "Error sending response to %s: %s\n",
                     device->name.c_str(), strerror(errno));
    }
}

} // namespace QEMUBinary
} // namespace ACALSim
