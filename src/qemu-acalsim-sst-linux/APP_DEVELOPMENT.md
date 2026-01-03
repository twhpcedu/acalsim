<!--
Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Application Development Guide

**Writing Linux Applications for ACALSim Hardware Models**

Copyright 2023-2026 Playlab/ACAL

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The `/dev/sst*` Interface](#the-devsst-interface)
3. [Request/Response API](#requestresponse-api)
4. [Complete Examples](#complete-examples)
5. [Best Practices](#best-practices)
6. [Advanced Patterns](#advanced-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example (10 Lines)

```c
#include <fcntl.h>
#include <unistd.h>
#include "sst-protocol.h"

int main() {
    int fd = open("/dev/sst0", O_RDWR);
    struct SSTRequest req = { .type = SST_REQ_NOOP };
    struct SSTResponse resp;
    write(fd, &req, sizeof(req));
    read(fd, &resp, sizeof(resp));
    close(fd);
    return resp.status == SST_STATUS_OK ? 0 : 1;
}
```

**Compile** (on host):
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
riscv64-linux-gnu-gcc -static -o myapp myapp.c -I../../virtio-device
```

**Deploy** (add to initramfs):
```bash
cp myapp /home/user/rootfs/apps/
cd /home/user/rootfs
find . | cpio -o -H newc | gzip > /home/user/initramfs.cpio.gz
```

**Run** (in QEMU):
```bash
/apps/myapp
```

---

## The `/dev/sst*` Interface

### Device Discovery

The kernel driver creates character devices for each SST device:

```bash
# In QEMU Linux
ls -l /dev/sst*
# Output:
# crw-rw-rw- 1 root root 240, 0 Jan  1 00:00 /dev/sst0
# crw-rw-rw- 1 root root 240, 1 Jan  1 00:00 /dev/sst1
```

**Device numbering**:
- `/dev/sst0` → device_id 0
- `/dev/sst1` → device_id 1
- etc.

### Basic Operations

All operations use standard POSIX I/O:

```c
#include <fcntl.h>
#include <unistd.h>

// 1. OPEN - Establish connection
int fd = open("/dev/sst0", O_RDWR);
if (fd < 0) {
    perror("Failed to open device");
    return -1;
}

// 2. WRITE - Send request
ssize_t written = write(fd, &request, sizeof(request));
if (written != sizeof(request)) {
    perror("Failed to write request");
}

// 3. READ - Receive response
ssize_t received = read(fd, &response, sizeof(response));
if (received != sizeof(response)) {
    perror("Failed to read response");
}

// 4. CLOSE - Release resources
close(fd);
```

**Important**: Currently, **one request at a time per file descriptor**. For concurrent requests, open multiple file descriptors.

---

## Request/Response API

### Data Structures

```c
// Request structure (4104 bytes total)
struct SSTRequest {
    uint32_t type;              // Request type (enum SSTRequestType)
    uint32_t flags;             // Request flags (currently unused)
    uint64_t request_id;        // User-defined ID (for tracking)
    uint64_t user_data;         // User-defined context (opaque to driver)

    union {
        // COMPUTE request
        struct {
            uint64_t compute_units;     // Amount of work
            uint32_t latency_model;     // Which model to use
            uint32_t reserved;
        } compute;

        // MEMORY request (READ/WRITE)
        struct {
            uint64_t addr;              // Address in device memory
            uint32_t size;              // Transfer size
            uint32_t reserved;
        } memory;

        // CONFIGURE request
        struct {
            uint32_t param_id;          // Which parameter
            uint32_t reserved;
            uint64_t value;             // New value
        } config;

        // Generic data (for ECHO, custom data transfer)
        uint8_t data[4080];
    } payload;
} __attribute__((packed));

// Response structure (4104 bytes total)
struct SSTResponse {
    uint32_t status;            // Status code (enum SSTStatus)
    uint32_t reserved;
    uint64_t request_id;        // Echoed from request
    uint64_t user_data;         // Echoed from request
    uint64_t result;            // Generic result value

    union {
        // COMPUTE response
        struct {
            uint64_t cycles;            // Simulated cycles
            uint64_t timestamp;         // Simulation time
        } compute;

        // GET_INFO response
        struct {
            uint32_t version;           // Protocol version
            uint32_t capabilities;      // Feature flags
            uint64_t max_compute_units; // Maximum work size
            uint64_t memory_size;       // Device memory size
        } info;

        // Generic data (for READ, ECHO)
        uint8_t data[4080];
    } payload;
} __attribute__((packed));
```

### Request Types

```c
enum SSTRequestType {
    SST_REQ_NOOP        = 0,    // No-op (connectivity test)
    SST_REQ_ECHO        = 1,    // Echo data back
    SST_REQ_COMPUTE     = 2,    // Perform computation
    SST_REQ_READ        = 3,    // Read device memory
    SST_REQ_WRITE       = 4,    // Write device memory
    SST_REQ_RESET       = 5,    // Reset device
    SST_REQ_GET_INFO    = 6,    // Query device info
    SST_REQ_CONFIGURE   = 7,    // Configure parameters
};
```

### Status Codes

```c
enum SSTStatus {
    SST_STATUS_OK           = 0,    // Success
    SST_STATUS_ERROR        = 1,    // Generic error
    SST_STATUS_BUSY         = 2,    // Device busy, retry
    SST_STATUS_INVALID      = 3,    // Invalid request
    SST_STATUS_TIMEOUT      = 4,    // Request timed out
    SST_STATUS_NO_DEVICE    = 5,    // SST not connected
};
```

---

## Complete Examples

### Example 1: Device Information Query

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "sst-protocol.h"

int main() {
    // Open device
    int fd = open("/dev/sst0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Prepare GET_INFO request
    struct SSTRequest req;
    memset(&req, 0, sizeof(req));
    req.type = SST_REQ_GET_INFO;
    req.request_id = 1;

    // Send request
    if (write(fd, &req, sizeof(req)) != sizeof(req)) {
        perror("write");
        close(fd);
        return 1;
    }

    // Receive response
    struct SSTResponse resp;
    if (read(fd, &resp, sizeof(resp)) != sizeof(resp)) {
        perror("read");
        close(fd);
        return 1;
    }

    // Check status
    if (resp.status != SST_STATUS_OK) {
        fprintf(stderr, "Request failed: status=%u\n", resp.status);
        close(fd);
        return 1;
    }

    // Print device info
    printf("Device Information:\n");
    printf("  Version: 0x%08x\n", resp.payload.info.version);
    printf("  Capabilities: 0x%08x\n", resp.payload.info.capabilities);
    printf("  Max compute units: %lu\n", resp.payload.info.max_compute_units);
    printf("  Memory size: %lu bytes\n", resp.payload.info.memory_size);

    close(fd);
    return 0;
}
```

### Example 2: Computation Offload

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "sst-protocol.h"

// Offload computation to SST simulator
uint64_t compute_on_sst(int fd, uint64_t work_size) {
    struct SSTRequest req;
    struct SSTResponse resp;

    memset(&req, 0, sizeof(req));
    req.type = SST_REQ_COMPUTE;
    req.request_id = (uint64_t)time(NULL);  // Use timestamp as ID
    req.payload.compute.compute_units = work_size;

    // Send and receive
    if (write(fd, &req, sizeof(req)) != sizeof(req)) {
        perror("write");
        return 0;
    }

    if (read(fd, &resp, sizeof(resp)) != sizeof(resp)) {
        perror("read");
        return 0;
    }

    if (resp.status != SST_STATUS_OK) {
        fprintf(stderr, "Compute failed: %u\n", resp.status);
        return 0;
    }

    return resp.payload.compute.cycles;
}

int main() {
    int fd = open("/dev/sst0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    printf("Offloading computation to accelerator...\n");

    // Compute different workload sizes
    uint64_t work_sizes[] = {100, 1000, 10000, 100000};
    for (int i = 0; i < 4; i++) {
        uint64_t cycles = compute_on_sst(fd, work_sizes[i]);
        printf("Work size %lu → %lu cycles\n", work_sizes[i], cycles);
    }

    close(fd);
    return 0;
}
```

### Example 3: Data Transfer (ECHO)

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "sst-protocol.h"

int main() {
    int fd = open("/dev/sst0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct SSTRequest req;
    struct SSTResponse resp;

    memset(&req, 0, sizeof(req));
    req.type = SST_REQ_ECHO;
    req.request_id = 42;

    // Fill payload with test data
    const char *msg = "Hello from Linux!";
    strncpy((char*)req.payload.data, msg, sizeof(req.payload.data));

    // Send request
    if (write(fd, &req, sizeof(req)) != sizeof(req)) {
        perror("write");
        close(fd);
        return 1;
    }

    // Receive echo response
    if (read(fd, &resp, sizeof(resp)) != sizeof(resp)) {
        perror("read");
        close(fd);
        return 1;
    }

    // Verify echo
    if (resp.status == SST_STATUS_OK &&
        strcmp((char*)resp.payload.data, msg) == 0) {
        printf("ECHO test PASSED\n");
    } else {
        printf("ECHO test FAILED\n");
    }

    close(fd);
    return 0;
}
```

### Example 4: Error Handling

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include "sst-protocol.h"

// Helper: send request with error checking
int sst_request(int fd, struct SSTRequest *req, struct SSTResponse *resp) {
    // Write request
    ssize_t written = write(fd, req, sizeof(*req));
    if (written < 0) {
        fprintf(stderr, "write() failed: %s\n", strerror(errno));
        return -1;
    }
    if (written != sizeof(*req)) {
        fprintf(stderr, "write() incomplete: %zd/%zu bytes\n",
                written, sizeof(*req));
        return -1;
    }

    // Read response
    ssize_t received = read(fd, resp, sizeof(*resp));
    if (received < 0) {
        fprintf(stderr, "read() failed: %s\n", strerror(errno));
        return -1;
    }
    if (received != sizeof(*resp)) {
        fprintf(stderr, "read() incomplete: %zd/%zu bytes\n",
                received, sizeof(*resp));
        return -1;
    }

    // Check response status
    if (resp->status != SST_STATUS_OK) {
        fprintf(stderr, "SST error: %s\n", sst_status_str(resp->status));
        return -1;
    }

    // Verify request ID matches
    if (resp->request_id != req->request_id) {
        fprintf(stderr, "Request ID mismatch: sent %lu, got %lu\n",
                req->request_id, resp->request_id);
        return -1;
    }

    return 0;
}

int main() {
    int fd = open("/dev/sst0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/sst0");
        return 1;
    }

    struct SSTRequest req;
    struct SSTResponse resp;

    memset(&req, 0, sizeof(req));
    req.type = SST_REQ_COMPUTE;
    req.request_id = 1;
    req.payload.compute.compute_units = 5000;

    if (sst_request(fd, &req, &resp) == 0) {
        printf("Computation completed: %lu cycles\n",
               resp.payload.compute.cycles);
    }

    close(fd);
    return 0;
}
```

---

## Best Practices

### 1. Always Check Return Values

```c
// BAD:
write(fd, &req, sizeof(req));
read(fd, &resp, sizeof(resp));

// GOOD:
if (write(fd, &req, sizeof(req)) != sizeof(req)) {
    perror("write");
    return -1;
}
if (read(fd, &resp, sizeof(resp)) != sizeof(resp)) {
    perror("read");
    return -1;
}
```

### 2. Initialize Structures

```c
// Prevents undefined behavior from padding bytes
struct SSTRequest req;
memset(&req, 0, sizeof(req));

// Or use designated initializers
struct SSTRequest req = {
    .type = SST_REQ_COMPUTE,
    .request_id = 1,
    .payload.compute.compute_units = 1000
};
```

### 3. Use Request IDs for Tracking

```c
static uint64_t next_request_id = 1;

struct SSTRequest req = {
    .type = SST_REQ_COMPUTE,
    .request_id = next_request_id++,
    .user_data = (uint64_t)&my_context  // Store context pointer
};
```

### 4. Handle All Status Codes

```c
switch (resp.status) {
case SST_STATUS_OK:
    // Process result
    break;
case SST_STATUS_BUSY:
    // Retry after delay
    usleep(1000);
    retry_request();
    break;
case SST_STATUS_TIMEOUT:
    // Increase timeout, retry
    break;
case SST_STATUS_NO_DEVICE:
    // SST simulator not connected
    fprintf(stderr, "SST simulator offline\n");
    return -1;
default:
    fprintf(stderr, "Unknown status: %u\n", resp.status);
    return -1;
}
```

### 5. Resource Management

```c
int fd = -1;

// Always use cleanup pattern
fd = open("/dev/sst0", O_RDWR);
if (fd < 0) {
    goto cleanup;
}

// ... use fd ...

cleanup:
    if (fd >= 0) {
        close(fd);
    }
    return ret;
```

### 6. Static Linking for Embedded Systems

Since the minimal rootfs has no dynamic linker, always compile with `-static`:

```bash
riscv64-linux-gnu-gcc -static -o myapp myapp.c -I../../virtio-device
```

---

## Advanced Patterns

### Pattern 1: Concurrent Requests

For throughput, use multiple file descriptors:

```c
#include <pthread.h>

#define NUM_WORKERS 4

typedef struct {
    int worker_id;
    int fd;
} worker_context_t;

void *worker_thread(void *arg) {
    worker_context_t *ctx = (worker_context_t*)arg;
    struct SSTRequest req;
    struct SSTResponse resp;

    for (int i = 0; i < 100; i++) {
        memset(&req, 0, sizeof(req));
        req.type = SST_REQ_COMPUTE;
        req.request_id = ctx->worker_id * 1000 + i;
        req.payload.compute.compute_units = 1000;

        write(ctx->fd, &req, sizeof(req));
        read(ctx->fd, &resp, sizeof(resp));

        printf("Worker %d: request %d completed (%lu cycles)\n",
               ctx->worker_id, i, resp.payload.compute.cycles);
    }

    return NULL;
}

int main() {
    pthread_t threads[NUM_WORKERS];
    worker_context_t contexts[NUM_WORKERS];

    // Open one fd per worker
    for (int i = 0; i < NUM_WORKERS; i++) {
        contexts[i].worker_id = i;
        contexts[i].fd = open("/dev/sst0", O_RDWR);
        if (contexts[i].fd < 0) {
            perror("open");
            return 1;
        }
        pthread_create(&threads[i], NULL, worker_thread, &contexts[i]);
    }

    // Wait for completion
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(threads[i], NULL);
        close(contexts[i].fd);
    }

    return 0;
}
```

### Pattern 2: Application-Level Batching

```c
// Send multiple requests, then harvest responses
int batch_compute(int fd, uint64_t *work_sizes, int count) {
    // Note: Current driver processes one at a time, but this prepares
    // for future async API support

    for (int i = 0; i < count; i++) {
        struct SSTRequest req = {
            .type = SST_REQ_COMPUTE,
            .request_id = i,
            .payload.compute.compute_units = work_sizes[i]
        };
        write(fd, &req, sizeof(req));  // Would be non-blocking in future
    }

    // Harvest responses (in future: poll/select)
    for (int i = 0; i < count; i++) {
        struct SSTResponse resp;
        read(fd, &resp, sizeof(resp));
        printf("Request %lu: %lu cycles\n", resp.request_id,
               resp.payload.compute.cycles);
    }

    return 0;
}
```

### Pattern 3: High-Level API Wrapper

```c
// sst_api.h - Clean wrapper for applications
#ifndef SST_API_H
#define SST_API_H

#include "sst-protocol.h"

typedef struct {
    int fd;
    uint64_t next_req_id;
} sst_device_t;

// Initialize device
int sst_init(sst_device_t *dev, const char *device_path);

// Cleanup
void sst_cleanup(sst_device_t *dev);

// High-level operations
int sst_compute(sst_device_t *dev, uint64_t work_size, uint64_t *cycles_out);
int sst_get_info(sst_device_t *dev, uint32_t *version, uint64_t *mem_size);
int sst_echo(sst_device_t *dev, const void *data, size_t len, void *out);

#endif

// sst_api.c - Implementation
#include "sst_api.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int sst_init(sst_device_t *dev, const char *device_path) {
    dev->fd = open(device_path, O_RDWR);
    if (dev->fd < 0) {
        return -1;
    }
    dev->next_req_id = 1;
    return 0;
}

void sst_cleanup(sst_device_t *dev) {
    if (dev->fd >= 0) {
        close(dev->fd);
        dev->fd = -1;
    }
}

int sst_compute(sst_device_t *dev, uint64_t work_size, uint64_t *cycles_out) {
    struct SSTRequest req;
    struct SSTResponse resp;

    memset(&req, 0, sizeof(req));
    req.type = SST_REQ_COMPUTE;
    req.request_id = dev->next_req_id++;
    req.payload.compute.compute_units = work_size;

    if (write(dev->fd, &req, sizeof(req)) != sizeof(req) ||
        read(dev->fd, &resp, sizeof(resp)) != sizeof(resp)) {
        return -1;
    }

    if (resp.status != SST_STATUS_OK) {
        return -1;
    }

    *cycles_out = resp.payload.compute.cycles;
    return 0;
}

// Usage:
int main() {
    sst_device_t dev;
    if (sst_init(&dev, "/dev/sst0") < 0) {
        return 1;
    }

    uint64_t cycles;
    if (sst_compute(&dev, 5000, &cycles) == 0) {
        printf("Computed in %lu cycles\n", cycles);
    }

    sst_cleanup(&dev);
    return 0;
}
```

---

## Troubleshooting

### Device Not Found

**Problem**: `/dev/sst0` doesn't exist

**Solutions**:
```bash
# 1. Check if driver loaded
lsmod | grep virtio_sst

# 2. Load driver manually
insmod /virtio-sst.ko

# 3. Check kernel messages
dmesg | grep virtio-sst

# 4. Verify VirtIO device exists
ls /sys/bus/virtio/devices/
```

### Permission Denied

**Problem**: `open("/dev/sst0")` fails with `EACCES`

**Solutions**:
```bash
# 1. Check permissions
ls -l /dev/sst0

# 2. Fix permissions (as root in init script)
chmod 666 /dev/sst0

# 3. Or run as root
sudo /apps/myapp
```

### Request Timeout

**Problem**: `read()` blocks forever

**Solutions**:
```c
// 1. Use non-blocking I/O with timeout
#include <poll.h>

struct pollfd pfd = {
    .fd = fd,
    .events = POLLIN
};

// Write request
write(fd, &req, sizeof(req));

// Wait up to 5 seconds for response
if (poll(&pfd, 1, 5000) <= 0) {
    fprintf(stderr, "Request timed out\n");
    return -1;
}

read(fd, &resp, sizeof(resp));
```

```bash
# 2. Check SST is running and connected
# In SST terminal, should see "Client connected"
```

### Partial Read/Write

**Problem**: `read()` or `write()` returns less than expected

**Solutions**:
```c
// Wrap in loop to handle partial I/O
ssize_t write_all(int fd, const void *buf, size_t count) {
    size_t written = 0;
    while (written < count) {
        ssize_t n = write(fd, (char*)buf + written, count - written);
        if (n < 0) {
            if (errno == EINTR) continue;  // Interrupted, retry
            return -1;
        }
        if (n == 0) break;  // EOF
        written += n;
    }
    return written;
}

// Use:
if (write_all(fd, &req, sizeof(req)) != sizeof(req)) {
    perror("write_all");
    return -1;
}
```

---

## Next Steps

- **Explore examples/**: Working code for common patterns
- **Read ARCHITECTURE.md**: Understand how it works underneath
- **Read DEPLOYMENT.md**: Learn about distributed deployment
- **Check out HSA demo**: Multi-accelerator example

**Questions?** Open an issue on GitHub or consult the FAQ.
