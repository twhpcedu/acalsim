# N-Socket Server Implementation Guide

**Date**: 2025-11-10
**Status**: 🚧 **In Progress - DeviceInfo struct updated, implementation pending**

---

## Overview

This document describes how to complete the N-socket server implementation for full N-device QEMU-SST integration. The DeviceInfo struct has been updated with socket fields; the remaining work is to implement the socket server logic.

---

## What's Already Done

### ✅ DeviceInfo Struct Updated

**File**: `QEMUBinaryComponent.hh:73-85`

```cpp
struct DeviceInfo {
    uint64_t base_addr;      // Device base address
    uint64_t size;           // Device memory size
    SST::Link* link;         // Link to device
    std::string name;        // Device name (for debugging)
    uint64_t num_requests;   // Statistics: requests routed to this device

    // N-socket support: per-device socket connections
    std::string socket_path; // Unix socket path (e.g., /tmp/qemu-sst-device0.sock)
    int server_fd;           // Server socket file descriptor
    int client_fd;           // Client connection file descriptor
    bool socket_ready;       // True when client connected
};
```

### ✅ Method Signatures Added

**File**: `QEMUBinaryComponent.hh:157-165`

```cpp
void handleMMIORequest();  // Legacy single-device mode
void handleMMIORequest(DeviceInfo* device);  // N-device mode
void sendMMIOResponse(bool success, uint64_t data);  // Legacy
void sendMMIOResponse(DeviceInfo* device, bool success, uint64_t data);  // N-device

// Per-device socket management (N-device mode)
void setupDeviceSocket(DeviceInfo* device, int index);
void acceptDeviceConnection(DeviceInfo* device);
void pollDeviceSockets();
```

### ✅ Constructor Updated

**File**: `QEMUBinaryComponent.cc:109-113`

```cpp
// Initialize socket fields (will be set up in launchQEMU)
dev.socket_path = "";
dev.server_fd = -1;
dev.client_fd = -1;
dev.socket_ready = false;
```

### ✅ Destructor Updated

**File**: `QEMUBinaryComponent.cc:154-164`

```cpp
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
```

---

## What Needs To Be Implemented

### 1. setupDeviceSocket() Method

**Purpose**: Create and bind a Unix socket for one device.

**Implementation** (add at end of file ~line 580):

```cpp
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
```

---

### 2. acceptDeviceConnection() Method

**Purpose**: Accept a connection on a device's socket.

**Implementation**:

```cpp
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
```

---

### 3. pollDeviceSockets() Method

**Purpose**: Poll all device sockets for incoming MMIO requests.

**Implementation**:

```cpp
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
```

---

### 4. handleMMIORequest(DeviceInfo*) Overload

**Purpose**: Handle MMIO request from a specific device.

**Implementation**:

```cpp
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
```

---

### 5. sendMMIOResponse(DeviceInfo*, bool, uint64_t) Overload

**Purpose**: Send MMIO response to a specific device.

**Implementation**:

```cpp
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
```

---

### 6. Refactor launchQEMU()

**Current Code** (lines 272-299): Single socket server

**New Implementation**:

```cpp
void QEMUBinaryComponent::launchQEMU() {
    out_.verbose(CALL_INFO, 1, 0, "Launching QEMU process...\n");
    setState(QEMUState::LAUNCHING);

    if (use_multi_device_) {
        // N-device mode: Create N socket servers
        for (size_t i = 0; i < devices_.size(); i++) {
            setupDeviceSocket(&devices_[i], i);
        }
    } else {
        // Legacy single-device mode (unchanged)
        unlink(socket_path_.c_str());
        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        // ... (existing code)
    }

    // Fork QEMU process
    qemu_pid_ = fork();
    if (qemu_pid_ < 0) {
        out_.fatal(CALL_INFO, -1, "Error: Failed to fork QEMU process\n");
    }

    if (qemu_pid_ == 0) {
        // Child process - close server sockets
        if (use_multi_device_) {
            for (auto& dev : devices_) {
                close(dev.server_fd);
            }
        } else {
            close(server_fd_);
        }

        // Build QEMU command line
        std::vector<const char*> args;
        args.push_back(qemu_path_.c_str());
        args.push_back("-M"); args.push_back("virt");
        args.push_back("-bios"); args.push_back("none");
        args.push_back("-nographic");
        args.push_back("-kernel"); args.push_back(binary_path_.c_str());

        std::vector<std::string> device_args;
        if (use_multi_device_) {
            // Use per-device socket paths
            for (size_t i = 0; i < devices_.size(); i++) {
                char addr_buf[32];
                snprintf(addr_buf, sizeof(addr_buf), "0x%lx", devices_[i].base_addr);
                std::string dev_arg = "sst-device,socket=" + devices_[i].socket_path +
                                     ",base_address=" + std::string(addr_buf);
                device_args.push_back(dev_arg);
                args.push_back("-device");
                args.push_back(device_args.back().c_str());
            }
        }
        // ... (rest unchanged)
        execvp(qemu_path_.c_str(), const_cast<char* const*>(args.data()));
        exit(1);
    }

    // Parent process - wait for connections
    if (use_multi_device_) {
        out_.verbose(CALL_INFO, 1, 0, "Waiting for %zu device connections...\n", devices_.size());
        // Connections will be accepted in clockTick via pollDeviceSockets()
    } else {
        // Legacy mode (unchanged)
        // ... existing accept logic
    }

    setState(QEMUState::RUNNING);
}
```

---

### 7. Update clockTick()

**Current Code** (lines 177-191): Calls handleMMIORequest() for single device

**New Implementation**:

```cpp
bool QEMUBinaryComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    if (state_ == QEMUState::RUNNING) {
        if (use_multi_device_) {
            // N-device mode: poll all device sockets
            pollDeviceSockets();
        } else {
            // Legacy single-device mode
            handleMMIORequest();
        }

        monitorQEMU();
    }

    return false;  // Continue simulation
}
```

---

## Testing Steps

After implementation:

### 1. Build and Install

```bash
cd qemu-binary
make clean && make && make install
```

### 2. Run 2-Device Test

```bash
sst test_2device_integration.py
```

**Expected Output**:
```
Device echo_device socket listening at /tmp/qemu-sst-device0.sock
Device compute_device socket listening at /tmp/qemu-sst-device1.sock
QEMU PID: xxxxx
Waiting for 2 device connections...
Device echo_device connected
Device compute_device connected
MMIO from echo_device: type=1 addr=0x10200000 ...
MMIO from compute_device: type=1 addr=0x10300000 ...
```

### 3. Run 4-Device Test

```bash
sst qemu_4device_test.py
```

---

## Benefits of N-Socket Implementation

1. **True Multi-Device Support**: Each QEMU device has its own socket connection
2. **Independent Communication**: Devices don't block each other
3. **Scalability**: Can handle N devices without contention
4. **Debugging**: Easy to trace which device is communicating
5. **Isolation**: Device failures don't affect other devices

---

## Estimated Effort

- **Implementation**: 2-3 hours
- **Testing**: 1-2 hours
- **Debugging**: 1 hour
- **Total**: 4-6 hours

---

## Current Status

**Completed**:
- ✅ DeviceInfo struct with socket fields
- ✅ Method signatures in header
- ✅ Constructor initialization
- ✅ Destructor cleanup

**Remaining**:
- ⏳ Implement setupDeviceSocket()
- ⏳ Implement acceptDeviceConnection()
- ⏳ Implement pollDeviceSockets()
- ⏳ Implement handleMMIORequest(DeviceInfo*)
- ⏳ Implement sendMMIOResponse(DeviceInfo*, bool, uint64_t)
- ⏳ Refactor launchQEMU() for N sockets
- ⏳ Update clockTick() to poll all devices

---

## Next Developer Tasks

1. Copy the implementation code from this guide
2. Add methods to QEMUBinaryComponent.cc (around line 580)
3. Update launchQEMU() (lines 268-400)
4. Update clockTick() (lines 177-191)
5. Build and test with 2 devices
6. Scale to 4+ devices
7. Performance benchmarking

---

**Document Date**: 2025-11-10
**Status**: 🚧 Implementation Guide Complete, Code Pending
