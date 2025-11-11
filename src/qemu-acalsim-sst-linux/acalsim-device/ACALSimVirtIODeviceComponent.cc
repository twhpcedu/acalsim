/**
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

#include "ACALSimVirtIODeviceComponent.hh"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cerrno>

// Include SST protocol definitions
#include "../virtio-device/sst-protocol.h"

using namespace ACALSim::VirtIO;

ACALSimVirtIODeviceComponent::ACALSimVirtIODeviceComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      socket_path_(params.find<std::string>("socket_path", "/tmp/qemu-sst-linux.sock")),
      device_id_(params.find<uint32_t>("device_id", 0)),
      verbose_(params.find<int>("verbose", 1)),
      server_fd_(-1),
      client_fd_(-1),
      client_connected_(false),
      current_cycle_(0),
      total_requests_(0),
      noop_requests_(0),
      echo_requests_(0),
      compute_requests_(0) {

    // Initialize output
    out_.init("VirtIODevice[@f:@l:@p] ", verbose_, 0, SST::Output::STDOUT);

    out_.verbose(CALL_INFO, 1, 0, "Initializing VirtIO Device Component\n");
    out_.verbose(CALL_INFO, 1, 0, "Configuration:\n");
    out_.verbose(CALL_INFO, 1, 0, "  Socket path: %s\n", socket_path_.c_str());
    out_.verbose(CALL_INFO, 1, 0, "  Device ID: %u\n", device_id_);

    // Initialize socket
    initSocket();

    // Configure clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    clock_handler_ = new SST::Clock::Handler<ACALSimVirtIODeviceComponent>(
        this, &ACALSimVirtIODeviceComponent::clockTick);
    registerClock(clock_freq, clock_handler_);

    // Register as primary component and keep simulation running
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();  // Keep ticking indefinitely

    out_.verbose(CALL_INFO, 1, 0, "VirtIO Device initialized successfully\n");
}

ACALSimVirtIODeviceComponent::~ACALSimVirtIODeviceComponent() {
    cleanupSocket();
}

void ACALSimVirtIODeviceComponent::setup() {
    out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");
}

void ACALSimVirtIODeviceComponent::finish() {
    out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");
    out_.verbose(CALL_INFO, 1, 0, "Statistics:\n");
    out_.verbose(CALL_INFO, 1, 0, "  Total requests: %lu\n", total_requests_);
    out_.verbose(CALL_INFO, 1, 0, "  NOOP requests: %lu\n", noop_requests_);
    out_.verbose(CALL_INFO, 1, 0, "  ECHO requests: %lu\n", echo_requests_);
    out_.verbose(CALL_INFO, 1, 0, "  COMPUTE requests: %lu\n", compute_requests_);
}

bool ACALSimVirtIODeviceComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    // Check for new connections
    if (!client_connected_) {
        checkForConnections();
    }

    // Process incoming data
    if (client_connected_) {
        handleSocketData();
    }

    return false;  // Continue ticking
}

void ACALSimVirtIODeviceComponent::initSocket() {
    // Remove existing socket file
    unlink(socket_path_.c_str());

    // Create socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        out_.fatal(CALL_INFO, -1, "Failed to create socket: %s\n", strerror(errno));
    }

    // Set non-blocking
    int flags = fcntl(server_fd_, F_GETFL, 0);
    fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

    // Bind socket
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        out_.fatal(CALL_INFO, -1, "Failed to bind socket: %s\n", strerror(errno));
    }

    // Listen for connections
    if (listen(server_fd_, 1) < 0) {
        out_.fatal(CALL_INFO, -1, "Failed to listen on socket: %s\n", strerror(errno));
    }

    out_.verbose(CALL_INFO, 1, 0, "Socket ready: %s\n", socket_path_.c_str());
}

void ACALSimVirtIODeviceComponent::cleanupSocket() {
    if (client_fd_ >= 0) {
        close(client_fd_);
        client_fd_ = -1;
    }

    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    unlink(socket_path_.c_str());
}

void ACALSimVirtIODeviceComponent::checkForConnections() {
    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);

    client_fd_ = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd_ >= 0) {
        // Set non-blocking
        int flags = fcntl(client_fd_, F_GETFL, 0);
        fcntl(client_fd_, F_SETFL, flags | O_NONBLOCK);

        client_connected_ = true;
        out_.verbose(CALL_INFO, 1, 0, "Client connected\n");
    }
}

void ACALSimVirtIODeviceComponent::handleSocketData() {
    uint8_t buffer[sizeof(struct SSTRequest)];  // Must match SSTRequest size (4104 bytes)
    ssize_t n = recv(client_fd_, buffer, sizeof(buffer), 0);

    if (n > 0) {
        processRequest(buffer, n);
    } else if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
        // Connection closed or error
        out_.verbose(CALL_INFO, 1, 0, "Client disconnected\n");
        close(client_fd_);
        client_fd_ = -1;
        client_connected_ = false;
    }
}

void ACALSimVirtIODeviceComponent::processRequest(const uint8_t* data, size_t len) {
    if (len < sizeof(struct SSTRequest)) {
        out_.verbose(CALL_INFO, 2, 0, "Invalid request size: %zu\n", len);
        return;
    }

    const struct SSTRequest* req = (const struct SSTRequest*)data;
    struct SSTResponse resp;
    memset(&resp, 0, sizeof(resp));

    // Echo request metadata
    resp.request_id = req->request_id;
    resp.user_data = req->user_data;

    total_requests_++;

    out_.verbose(CALL_INFO, 2, 0, "Processing request type %u\n", req->type);

    switch (req->type) {
        case SST_REQ_NOOP:
            noop_requests_++;
            resp.status = SST_STATUS_OK;
            resp.result = 0;
            out_.verbose(CALL_INFO, 2, 0, "NOOP request\n");
            break;

        case SST_REQ_ECHO:
            echo_requests_++;
            resp.status = SST_STATUS_OK;
            // Echo back the data
            memcpy(resp.payload.data, req->payload.data, SST_MAX_DATA_SIZE);
            resp.result = SST_MAX_DATA_SIZE;
            out_.verbose(CALL_INFO, 2, 0, "ECHO request\n");
            break;

        case SST_REQ_COMPUTE:
            compute_requests_++;
            resp.status = SST_STATUS_OK;
            // Simulate compute: 100 cycles per compute unit
            resp.payload.compute.cycles = req->payload.compute.compute_units * 100;
            resp.payload.compute.timestamp = current_cycle_;
            resp.result = resp.payload.compute.cycles;
            out_.verbose(CALL_INFO, 2, 0, "COMPUTE request (units=%lu, cycles=%lu)\n",
                        req->payload.compute.compute_units, resp.payload.compute.cycles);
            break;

        default:
            out_.verbose(CALL_INFO, 1, 0, "Unknown request type: %u\n", req->type);
            resp.status = SST_STATUS_INVALID;
            resp.result = 0;
            break;
    }

    // Send response
    sendResponse((const uint8_t*)&resp, sizeof(resp));
}

void ACALSimVirtIODeviceComponent::sendResponse(const uint8_t* data, size_t len) {
    if (!client_connected_ || client_fd_ < 0) {
        return;
    }

    ssize_t n = send(client_fd_, data, len, 0);
    if (n < 0) {
        out_.verbose(CALL_INFO, 1, 0, "Failed to send response: %s\n", strerror(errno));
        close(client_fd_);
        client_fd_ = -1;
        client_connected_ = false;
    }
}
