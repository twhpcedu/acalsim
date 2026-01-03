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

# ACALSim Linux Integration Architecture

**A Comprehensive Tutorial: From Zero to Hero**

Copyright 2023-2026 Playlab/ACAL

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Big Picture](#the-big-picture)
3. [Component Deep Dive](#component-deep-dive)
4. [Communication Flow](#communication-flow)
5. [Protocol Design](#protocol-design)
6. [Building From Scratch](#building-from-scratch)
7. [Advanced Topics](#advanced-topics)

---

## Introduction

### What Are We Building?

We're building a system that allows **Linux applications** running in a **RISC-V virtual machine (QEMU)** to communicate with **hardware simulators (SST)** that model custom accelerators, AI chips, or other specialized hardware.

### Why Is This Useful?

Traditional hardware development cycle:
1. Design hardware in HDL (Verilog/VHDL)
2. Synthesize and fabricate chip ($$$ expensive, months/years)
3. Write software for it
4. Discover bugs → go back to step 1

Our approach:
1. **Model** hardware in SST (fast, free)
2. **Simulate** complete Linux OS + software in QEMU
3. **Co-simulate** hardware and software together
4. Fix bugs quickly, iterate rapidly
5. Only fabricate when confident

### What Makes This Different from Bare-Metal?

We already have `qemu-acalsim-sst-baremetal` which runs bare-metal code. The **Linux integration** adds:

- **Full operating system**: Process management, memory protection, file systems
- **Standard I/O interface**: Applications use `/dev/sst*` devices (like any Linux device)
- **Realistic overhead**: Models actual OS context switches, system calls, kernel drivers
- **Ecosystem compatibility**: Can run real Linux applications, use standard libraries
- **Multi-user/multi-process**: Multiple apps can share accelerators

---

## The Big Picture

### 30,000 Foot View

```
┌─────────────────────────────────────────────┐
│          Linux Application                   │  ← Your code
│  (C/C++/Python running in QEMU-RISC-V)      │
└──────────────────┬──────────────────────────┘
                   │ read()/write()/ioctl()
                   ↓
┌─────────────────────────────────────────────┐
│        Kernel Driver (virtio-sst.ko)         │  ← In Linux kernel
│        Manages /dev/sst* devices             │
└──────────────────┬──────────────────────────┘
                   │ VirtQueue operations
                   ↓
┌─────────────────────────────────────────────┐
│    QEMU VirtIO SST Device (virtio-sst.c)    │  ← In QEMU
│    Emulates VirtIO hardware                  │
└──────────────────┬──────────────────────────┘
                   │ Unix Domain Socket
                   ↓
┌─────────────────────────────────────────────┐
│  SST VirtIO Component (ACALSimVirtIODevice)  │  ← In SST simulator
│  Processes requests, models hardware         │
└──────────────────┬──────────────────────────┘
                   │ SST Links (optional)
                   ↓
┌─────────────────────────────────────────────┐
│   Custom Accelerator Models (HSA, etc.)      │  ← Your hardware models
│   AI chips, GPUs, custom processors          │
└─────────────────────────────────────────────┘
```

### Key Insight: Four Different Environments

1. **Linux Userspace** (inside QEMU): Where your application runs
2. **Linux Kernel** (inside QEMU): Device driver land
3. **QEMU Process** (on host): Hypervisor emulating hardware
4. **SST Process** (on host): Simulator modeling your custom hardware

---

## Component Deep Dive

### Component 1: Linux Application (`rootfs/apps/`)

**Location**: Runs inside RISC-V Linux (in QEMU)

**What it does**: Normal C/C++ application that talks to custom hardware

**How it works**:
```c
// 1. Open device (like opening a file)
int fd = open("/dev/sst0", O_RDWR);

// 2. Prepare request structure
struct SSTRequest req = {
    .type = SST_REQ_COMPUTE,
    .request_id = 1,
    .payload.compute.compute_units = 1000
};

// 3. Send request to hardware (kernel driver handles this)
write(fd, &req, sizeof(req));

// 4. Receive response from hardware
struct SSTResponse resp;
read(fd, &resp, sizeof(resp));

// 5. Use results
printf("Simulated %lu cycles\n", resp.payload.compute.cycles);

close(fd);
```

**Key points**:
- Must be **statically linked** (`-static`) because minimal BusyBox rootfs has no dynamic linker
- Cross-compiled for RISC-V: `riscv64-linux-gnu-gcc`
- Uses standard POSIX I/O (open/read/write/close)
- Completely unaware of QEMU or SST underneath

---

### Component 2: Kernel Driver (`drivers/sst-virtio.c`)

**Location**: Runs in Linux kernel (inside QEMU)

**What it does**: Creates `/dev/sst*` character devices and translates syscalls to VirtIO operations

**Architecture**:

```
User syscalls          Kernel Driver                VirtIO Layer
─────────────          ─────────────                ────────────
open("/dev/sst0")  →  sst_open()
                       ├→ Allocate context
                       └→ Return file descriptor

write(fd, req, sz)  →  sst_write()
                       ├→ Allocate virtqueue element
                       ├→ Setup scatter-gather lists:
                       │   - OUT: request buffer
                       │   - IN: response buffer
                       ├→ Add to request virtqueue   →  virtqueue_add_sgs()
                       ├→ Kick virtqueue to notify QEMU  →  virtqueue_kick()
                       └→ Sleep waiting for response  →  wait_for_completion()

[QEMU processes request and sends response]

<interrupt from QEMU>  ←  VirtIO interrupt
                       →  sst_response_done()
                           ├→ Extract response from virtqueue
                           ├→ Wake up sleeping write()  →  complete()
                           └→ Return

read(fd, resp, sz)  →  sst_read()
                       ├→ Copy response to userspace
                       └→ Return

close(fd)           →  sst_release()
                       └→ Free resources
```

**Critical Design: VirtQueue Management**

The driver uses **ONE virtqueue** (`req_vq`) with **bidirectional buffers**:

```c
// Setup scatter-gather lists for BOTH directions
struct scatterlist sg_out, sg_in;
struct scatterlist *sgs[2];

sg_init_one(&sg_out, &req, sizeof(req));   // OUT: request to device
sg_init_one(&sg_in, &resp, sizeof(resp));  // IN: response from device
sgs[0] = &sg_out;
sgs[1] = &sg_in;

// Add BOTH to the same queue
virtqueue_add_sgs(req_vq, sgs, 1, 1, ctx, GFP_ATOMIC);
                                   ↑  ↑
                              1 OUT, 1 IN buffer
```

**Bug we fixed**: Initially had callbacks on wrong queue:
```c
// WRONG (BUG):
callbacks[SST_VQ_REQUEST] = NULL;
callbacks[SST_VQ_RESPONSE] = sst_response_done;

// CORRECT (FIXED):
callbacks[SST_VQ_REQUEST] = sst_response_done;  // Callback fires here!
callbacks[SST_VQ_RESPONSE] = NULL;
```

---

### Component 3: QEMU VirtIO Device (`virtio-device/virtio-sst.c`)

**Location**: Runs in QEMU process on host

**What it does**: Emulates VirtIO hardware that Linux kernel can talk to

**VirtIO in 60 Seconds**:

VirtIO is a standardized interface for virtual I/O devices. Key concepts:

1. **VirtQueues**: Circular buffers in guest memory for DMA-like transfers
2. **Descriptors**: Point to guest memory buffers (requests/responses)
3. **Available Ring**: Guest → Host notification ("I put something in queue")
4. **Used Ring**: Host → Guest notification ("I processed your request")

```
Guest (Linux)                   Host (QEMU)
─────────────                   ───────────
1. Prepare request in memory
2. Put descriptor in virtqueue
3. Kick queue (write to MMIO)  →  Trap to QEMU
                                  4. QEMU reads descriptor
                                  5. Copies guest memory
                                  6. Processes request
                                  7. Writes response to guest memory
                                  8. Updates "used ring"
9. Interrupt!                  ←  9. Injects interrupt
10. Read response from memory
11. Return to application
```

**Our Implementation**:

```c
void virtio_sst_handle_request(VirtIODevice *vdev, VirtQueue *vq)
{
    VirtIOSST *s = VIRTIO_SST(vdev);
    VirtQueueElement *elem;

    // Process all available requests
    while ((elem = virtqueue_pop(vq, sizeof(VirtQueueElement)))) {
        // elem->out_sg[] = request from guest
        // elem->in_sg[] = response buffer in guest

        virtio_sst_process_request(s, vq, elem);
    }
}

void virtio_sst_process_request(VirtIOSST *s, VirtQueue *vq,
                                 VirtQueueElement *elem)
{
    struct SSTRequest *req = g_malloc(sizeof(*req));
    struct SSTResponse *resp = g_malloc(sizeof(*resp));

    // Copy request from guest memory
    iov_to_buf(elem->out_sg, elem->out_num, 0, req, sizeof(*req));

    switch (req->type) {
    case SST_REQ_NOOP:
        // Handle locally
        resp->status = SST_STATUS_OK;
        break;

    case SST_REQ_COMPUTE:
        // Forward to SST via Unix socket
        virtio_sst_send_request(s, req, resp);
        break;
    }

    // Copy response back to guest memory
    iov_from_buf(elem->in_sg, elem->in_num, 0, resp, sizeof(*resp));

    // Return buffer to guest (triggers interrupt)
    virtqueue_push(vq, elem, sizeof(*resp));
    virtio_notify(vdev, vq);
}
```

**Critical**: Must ALWAYS push element back to virtqueue, even on error!

---

### Component 4: SST VirtIO Component (`acalsim-device/ACALSimVirtIODeviceComponent`)

**Location**: Runs in SST simulator process on host

**What it does**: Receives requests via Unix socket, models hardware behavior

**SST Component Lifecycle**:

```
SST Framework
    │
    ├→ Constructor: Create component, parse parameters
    │
    ├→ setup(): Initialize (called once before simulation)
    │
    ├→ registerClock(): Register tick function
    │
    ├→ clockTick(): Called every cycle
    │   └→ checkForConnections()
    │   └→ handleSocketData()
    │       └→ processRequest()
    │           └→ sendResponse()
    │
    └→ finish(): Cleanup, print statistics
```

**Implementation**:

```cpp
class ACALSimVirtIODeviceComponent : public SST::Component {
    // Constructor: Setup socket listener
    ACALSimVirtIODeviceComponent(ComponentId_t id, Params& params) {
        // Parse config
        socket_path_ = params.find<std::string>("socket_path");

        // Create Unix domain socket
        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        bind(server_fd_, &addr, sizeof(addr));
        listen(server_fd_, 1);

        // Register clock (CRITICAL for receiving data!)
        clock_handler_ = new Clock::Handler<>(this, &clockTick);
        registerClock("1GHz", clock_handler_);

        // CRITICAL: Keep simulation running!
        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();  // Without this, simulation ends immediately!
    }

    // Called every cycle at 1GHz
    bool clockTick(Cycle_t cycle) {
        if (!client_connected_) {
            checkForConnections();  // Accept QEMU connection
        }

        if (client_connected_) {
            handleSocketData();  // Process incoming requests
        }

        return false;  // Keep ticking
    }

    void handleSocketData() {
        uint8_t buffer[sizeof(SSTRequest)];  // MUST be large enough!

        ssize_t n = recv(client_fd_, buffer, sizeof(buffer), 0);
        if (n > 0) {
            processRequest(buffer, n);
        }
    }

    void processRequest(const uint8_t* data, size_t len) {
        const SSTRequest* req = (const SSTRequest*)data;
        SSTResponse resp;

        switch (req->type) {
        case SST_REQ_COMPUTE:
            // Model hardware behavior
            resp.payload.compute.cycles = req->payload.compute.compute_units * 100;
            resp.status = SST_STATUS_OK;
            break;
        }

        sendResponse((uint8_t*)&resp, sizeof(resp));
    }
};
```

**Bug we fixed**: Buffer was only 1024 bytes, but `sizeof(SSTRequest)` = 4104 bytes!

---

## Communication Flow

### Complete Request Flow: User → SST → User

Let's trace a COMPUTE request step-by-step:

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: Application prepares request (Linux userspace)      │
└──────────────────────────────────────────────────────────────┘

app.c:50:  struct SSTRequest req = {
             .type = SST_REQ_COMPUTE,
             .request_id = 42,
             .payload.compute.compute_units = 1000
           };
           write(fd, &req, sizeof(req));

                         ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Syscall enters kernel (Linux kernel)                │
└──────────────────────────────────────────────────────────────┘

sst-virtio.c:250:  sst_write(filp, buf, count, off)
                   ├→ copy_from_user(&req, buf, sizeof(req))
                   ├→ Setup virtqueue element:
                   │   sg_out ← &req  (4104 bytes)
                   │   sg_in ← &resp (4104 bytes)
                   ├→ virtqueue_add_sgs(req_vq, {&sg_out, &sg_in})
                   ├→ virtqueue_kick(req_vq)  // Notify QEMU!
                   └→ wait_for_completion(&ctx->completion)  // Sleep

                         ↓ (QEMU gets MMIO write trap)
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: QEMU VirtIO device handles notification             │
└──────────────────────────────────────────────────────────────┘

virtio-sst.c:237:  virtio_sst_handle_request(vdev, vq)
                   ├→ elem = virtqueue_pop(vq)
                   ├→ iov_to_buf(elem->out_sg) → Read request from guest RAM
                   ├→ Forward to SST:
                   │   write(socket_fd, &req, sizeof(req))
                   │   read(socket_fd, &resp, sizeof(resp))
                   ├→ iov_from_buf(elem->in_sg) → Write response to guest RAM
                   ├→ virtqueue_push(vq, elem)
                   └→ virtio_notify(vdev, vq)  // Inject interrupt to guest!

                         ↓ (Unix socket carries data)
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: SST simulator processes request                     │
└──────────────────────────────────────────────────────────────┘

ACALSimVirtIODeviceComponent.cc:163:
                   clockTick() is called every cycle
                   ├→ handleSocketData()
                   │   ├→ recv(client_fd, buffer, 4104)  // Receive full request!
                   │   └→ processRequest(buffer)
                   │       ├→ switch (req->type)
                   │       │   case SST_REQ_COMPUTE:
                   │       │     cycles = compute_units * 100  // Model!
                   │       └→ sendResponse(&resp)
                   │           └→ send(client_fd, &resp, 4104)

                         ↓ (Response travels back)
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: QEMU receives response and returns it               │
└──────────────────────────────────────────────────────────────┘

(Response already written to guest RAM in step 3)

                         ↓ (Interrupt arrives)
┌──────────────────────────────────────────────────────────────┐
│  STEP 6: Kernel driver handles interrupt                     │
└──────────────────────────────────────────────────────────────┘

sst-virtio.c:180:  sst_response_done(vq)  // Interrupt handler
                   ├→ elem = virtqueue_get_buf(vq)
                   ├→ ctx = elem->context
                   ├→ complete(&ctx->completion)  // Wake up write()!
                   └→ return

sst-virtio.c:265:  // write() wakes up from wait_for_completion()
                   └→ return count  // Success!

                         ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 7: Application receives response                       │
└──────────────────────────────────────────────────────────────┘

app.c:52:  read(fd, &resp, sizeof(resp));
sst-virtio.c:300:  sst_read()
                   ├→ copy_to_user(buf, &ctx->resp, sizeof(resp))
                   └→ return sizeof(resp)

app.c:53:  printf("Cycles: %lu\n", resp.payload.compute.cycles);
           // Output: Cycles: 100000

```

**Timeline** (approximate):
- Step 1-2: ~1 microsecond (syscall overhead)
- Step 3: ~10 microseconds (QEMU VirtIO processing)
- Step 4: Instant to minutes (depends on SST model complexity)
- Step 5-7: ~10 microseconds (return path)

---

## Protocol Design

### Why a Binary Protocol?

We use binary structs instead of JSON/XML because:
1. **Performance**: No parsing overhead
2. **Simplicity**: Direct memory copies
3. **Type safety**: Compiler checks sizes
4. **Compatibility**: Works in kernel, userspace, and C++

### Protocol Structure (`virtio-device/sst-protocol.h`)

```c
// Request: Linux → SST
struct SSTRequest {
    uint32_t type;              // What operation?
    uint32_t flags;             // Reserved
    uint64_t request_id;        // Track this request
    uint64_t user_data;         // Opaque context

    union {
        struct { uint64_t compute_units; } compute;
        struct { uint64_t addr; uint32_t size; } memory;
        uint8_t data[4080];     // Large buffer for data
    } payload;
} __attribute__((packed));      // No padding!

// Response: SST → Linux
struct SSTResponse {
    uint32_t status;            // Success/error code
    uint32_t reserved;
    uint64_t request_id;        // Matches request
    uint64_t user_data;         // Echo back
    uint64_t result;            // Generic result value

    union {
        struct { uint64_t cycles; uint64_t timestamp; } compute;
        struct { uint32_t version; uint64_t memory_size; } info;
        uint8_t data[4080];
    } payload;
} __attribute__((packed));
```

**Size**: Both structures are exactly **4104 bytes** (24-byte header + 4080-byte payload)

**Design Rationale**:
- 4KB page-aligned for efficient DMA
- Large payload supports bulk data transfer
- Union keeps size constant regardless of request type
- `__attribute__((packed))` ensures binary compatibility across architectures

### Request Types

```c
enum SSTRequestType {
    SST_REQ_NOOP     = 0,  // Connectivity test
    SST_REQ_ECHO     = 1,  // Data loopback test
    SST_REQ_COMPUTE  = 2,  // Computation request → SST
    SST_REQ_READ     = 3,  // Read device memory
    SST_REQ_WRITE    = 4,  // Write device memory
    SST_REQ_RESET    = 5,  // Reset device
    SST_REQ_GET_INFO = 6,  // Query capabilities
    SST_REQ_CONFIGURE= 7,  // Set parameters
};
```

**NOOP vs COMPUTE**:
- `NOOP`: Handled entirely in QEMU, doesn't reach SST (fast, tests driver/QEMU path)
- `COMPUTE`: Forwarded to SST, models actual hardware (slow, tests full stack)

---

## Building From Scratch

### What Gets Built Where?

```
Component                      Build Environment    Output
─────────────────────────────  ──────────────────   ──────────────────
1. Linux Kernel                Docker container     arch/riscv/boot/Image
2. Kernel Driver               Docker container     virtio-sst.ko
3. Test Application            Docker container     sst-test (RISC-V binary)
4. BusyBox/Rootfs              Docker container     initramfs.cpio.gz
5. QEMU + VirtIO Device        Docker container     qemu-system-riscv64
6. SST Component               Docker container     libacalsim.so
```

### Dependency Graph

```
                  Linux Kernel (v6.1)
                        │
                        ├─→ Kernel Driver (needs kernel headers)
                        │
                  BusyBox + Apps
                        │
                        └─→ initramfs (contains driver + apps)
                              │
                              ↓
                        QEMU (starts with kernel + initramfs)
                              │
                              ↓ (Unix socket)
                        SST Component (waits for connection)
```

### Build Order (Critical!)

```bash
# 1. Build kernel FIRST (generates headers for driver)
cd /home/user/linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)

# 2. Build driver (needs kernel headers from step 1)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=/home/user/linux

# 3. Build apps (standalone, needs sst-protocol.h)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
make CROSS_COMPILE=riscv64-linux-gnu-

# 4. Build QEMU (integrates VirtIO device)
cd /home/user/qemu-build/qemu
./setup-qemu-virtio-sst.sh
cd build && ninja

# 5. Build initramfs (packages driver + apps)
cd /home/user/rootfs
find . | cpio -o -H newc | gzip > /home/user/initramfs.cpio.gz

# 6. Build SST component (independent)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/acalsim-device
make && make install
```

---

## Advanced Topics

### Debugging Techniques

**1. Trace VirtIO Operations**

Add prints to driver:
```c
// drivers/sst-virtio.c
pr_info("virtio-sst: Adding buffer to queue, req_id=%llu\n", req->request_id);
```

Rebuild driver, update initramfs, reboot QEMU.

**2. Monitor Unix Socket Traffic**

```bash
# In SST component
out_.verbose(CALL_INFO, 2, 0, "Received %zd bytes\n", n);
```

Set `verbose: "2"` in SST config.

**3. Inspect VirtQueue State**

```c
// In kernel driver
pr_info("virtio-sst: Queue has %u entries\n",
        virtqueue_get_vring_size(sdev->req_vq));
```

**4. Use QEMU Monitor**

```bash
# In QEMU terminal, press Ctrl-A then C
(qemu) info virtio
(qemu) info virtio-queue sst_device_0 0
```

### Performance Optimization

**1. Batching Requests**

Instead of:
```c
for (int i = 0; i < 1000; i++) {
    write(fd, &req, sizeof(req));
    read(fd, &resp, sizeof(resp));  // Round-trip each time!
}
```

Do:
```c
// TODO: Implement batch API in driver
write_batch(fd, reqs, 1000);
read_batch(fd, resps, 1000);
```

**2. Asynchronous I/O**

Use `io_uring` or `aio` for non-blocking operations:
```c
struct io_uring ring;
io_uring_queue_init(32, &ring, 0);

// Submit multiple requests without blocking
for (int i = 0; i < 10; i++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_write(sqe, fd, &reqs[i], sizeof(req), 0);
}
io_uring_submit(&ring);

// Harvest completions
for (int i = 0; i < 10; i++) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    // Process response
    io_uring_cqe_seen(&ring, cqe);
}
```

**3. Zero-Copy with mmap**

Map device memory directly (requires driver support):
```c
void *mem = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// Direct access to device memory, no copies!
```

### Security Considerations

**1. Input Validation**

Always validate in kernel driver:
```c
if (req->type >= SST_REQ_MAX) {
    pr_warn("virtio-sst: Invalid request type %u\n", req->type);
    return -EINVAL;
}

if (req->payload.memory.size > MAX_TRANSFER_SIZE) {
    return -EINVAL;
}
```

**2. Resource Limits**

Prevent DoS attacks:
```c
// Limit concurrent requests per user
if (atomic_read(&user->active_requests) > MAX_USER_REQUESTS) {
    return -EBUSY;
}
```

**3. Capability-Based Access**

Require privileges for sensitive operations:
```c
if (req->type == SST_REQ_RESET && !capable(CAP_SYS_ADMIN)) {
    return -EPERM;
}
```

### Multi-Queue Support

For high-performance workloads, use multiple virtqueues:

```c
// Create per-CPU queues
for (int cpu = 0; cpu < num_cpus; cpu++) {
    sdev->req_vqs[cpu] = virtio_find_single_vq(vdev,
                                               callbacks[cpu],
                                               names[cpu]);
}

// In write(), use CPU-local queue
int cpu = get_cpu();
vq = sdev->req_vqs[cpu];
virtqueue_add_sgs(vq, ...);
put_cpu();
```

Benefits:
- Eliminate queue contention
- Better cache locality
- Linear scaling with cores

---

## Next Steps

1. **Read GETTING_STARTED.md**: Build and run the basic example
2. **Read APP_DEVELOPMENT.md**: Learn to write your own applications
3. **Read DEPLOYMENT.md**: Understand single/multi-server deployment
4. **Study examples/**: Working code for common patterns
5. **Join the community**: Ask questions, contribute improvements

---

**Questions?** See `docs/FAQ.md` or open an issue on GitHub.

**Want to contribute?** See `CONTRIBUTING.md` for guidelines.
