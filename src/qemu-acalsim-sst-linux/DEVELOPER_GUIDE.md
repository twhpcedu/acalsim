<!--
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
-->

# Developer Guide - Linux SST Integration

This guide provides detailed architecture and implementation information for developers working with or extending the Linux SST integration.

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Linux Guest (RISC-V)                                       │
│                                                              │
│  ┌────────────────┐        ┌──────────────────┐            │
│  │  Application   │◄──────►│  /dev/sst0       │            │
│  │  (user-space)  │        │  character device│            │
│  └────────────────┘        └────────┬─────────┘            │
│                                      │                      │
│  ┌───────────────────────────────────▼────────────────┐    │
│  │  virtio-sst.ko (Kernel Driver)                     │    │
│  │  - Character device interface                       │    │
│  │  - Request/response management                      │    │
│  │  - VirtQueue operations                             │    │
│  └───────────────────────────┬────────────────────────┘    │
│                               │ VirtIO Protocol             │
└───────────────────────────────┼─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│  QEMU (Host)                                                │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  virtio-sst-device.c (VirtIO Device)                │   │
│  │  - VirtQueue management (req/resp/event)            │   │
│  │  - Feature negotiation                              │   │
│  │  - Configuration space                              │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │ Unix Socket                     │
└──────────────────────────┼─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  SST Simulator                                             │
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │  ACALSimDeviceComponent                          │     │
│  │  - Socket server                                  │     │
│  │  - Request dispatching                            │     │
│  └──────────┬───────────────────────────────────────┘     │
│             │                                              │
│  ┌──────────▼───────────────────────────────────────┐     │
│  │  ACALSimComputeDeviceComponent (optional)        │     │
│  │  - Cycle-accurate simulation                      │     │
│  │  - Latency modeling                               │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Data Flow

**Request Path** (Application → SST):
1. User application writes `SSTRequest` to `/dev/sst0`
2. Kernel driver copies request to kernel memory
3. Driver adds request to VirtQueue (scatter-gather list)
4. Driver kicks VirtQueue (MMIO write)
5. QEMU VirtIO device receives notification
6. Device copies request from guest memory
7. Device sends request to SST via Unix socket
8. SST component processes request

**Response Path** (SST → Application):
9. SST component sends `SSTResponse` via socket
10. QEMU device receives response
11. Device copies response to guest memory (VirtQueue buffer)
12. Device injects interrupt to guest
13. Guest kernel handles interrupt
14. Driver processes response from VirtQueue
15. Driver wakes up waiting application
16. Application reads response from kernel

## Component Details

### VirtIO Protocol (sst-protocol.h)

#### Request Structure

```c
struct SSTRequest {
    uint32_t type;          // Request type (SST_REQ_*)
    uint32_t flags;         // Reserved for future use
    uint64_t request_id;    // Unique ID (assigned by driver)
    uint64_t user_data;     // Application-specific data

    union {
        struct {
            uint64_t compute_units;
            uint32_t latency_model;
        } compute;

        struct {
            uint64_t addr;
            uint32_t size;
        } memory;

        uint8_t data[SST_MAX_DATA_SIZE];  // 4080 bytes
    } payload;
} __attribute__((packed));
```

**Total size**: 4096 bytes (one page)

#### Response Structure

```c
struct SSTResponse {
    uint32_t status;        // SST_STATUS_*
    uint32_t reserved;
    uint64_t request_id;    // Matches SSTRequest.request_id
    uint64_t user_data;     // Echo of SSTRequest.user_data
    uint64_t result;        // Operation-specific result

    union {
        struct {
            uint64_t cycles;
            uint64_t timestamp;
        } compute;

        struct {
            uint32_t version;
            uint32_t capabilities;
            uint64_t max_compute_units;
            uint64_t memory_size;
        } info;

        uint8_t data[SST_MAX_DATA_SIZE];
    } payload;
} __attribute__((packed));
```

#### Request Types

| Type | Value | Description | Payload |
|------|-------|-------------|---------|
| `SST_REQ_NOOP` | 0 | No operation | None |
| `SST_REQ_ECHO` | 1 | Echo data back | `data[]` |
| `SST_REQ_COMPUTE` | 2 | Submit computation | `compute` |
| `SST_REQ_READ` | 3 | Read device memory | `memory` |
| `SST_REQ_WRITE` | 4 | Write device memory | `memory` + `data[]` |
| `SST_REQ_RESET` | 5 | Reset device | None |
| `SST_REQ_GET_INFO` | 6 | Get device info | None |
| `SST_REQ_CONFIGURE` | 7 | Configure device | `config` |

#### Status Codes

| Status | Value | Meaning |
|--------|-------|---------|
| `SST_STATUS_OK` | 0 | Success |
| `SST_STATUS_ERROR` | 1 | Generic error |
| `SST_STATUS_BUSY` | 2 | Device busy |
| `SST_STATUS_INVALID` | 3 | Invalid request |
| `SST_STATUS_TIMEOUT` | 4 | Operation timeout |
| `SST_STATUS_NO_DEVICE` | 5 | SST not connected |

### VirtIO Device (virtio-sst.c)

#### Device State

```c
struct VirtIOSST {
    VirtIODevice parent_obj;

    /* VirtQueues */
    VirtQueue *req_vq;      // Request queue (driver → device)
    VirtQueue *resp_vq;     // Response queue (device → driver)
    VirtQueue *event_vq;    // Event queue (device → driver, async)

    /* Configuration */
    struct SSTConfig config;
    uint64_t features;

    /* SST Connection */
    char *socket_path;
    int socket_fd;
    bool connected;

    /* Statistics */
    uint64_t total_requests;
    uint64_t total_responses;
    uint64_t total_events;
    uint64_t total_errors;
};
```

#### Key Functions

**Device Initialization** (`virtio-sst.c:322`):
```c
void virtio_sst_realize(DeviceState *dev, Error **errp)
```
- Initializes VirtIO device
- Creates 3 VirtQueues
- Connects to SST via Unix socket

**Request Processing** (`virtio-sst.c:140`):
```c
void virtio_sst_process_request(VirtIOSST *s, VirtQueueElement *elem)
```
- Extracts request from VirtQueue buffer
- Handles local requests (NOOP, ECHO, GET_INFO)
- Forwards compute requests to SST
- Prepares and sends response

**VirtQueue Handler** (`virtio-sst.c:188`):
```c
void virtio_sst_handle_request(VirtIODevice *vdev, VirtQueue *vq)
```
- Called when driver kicks VirtQueue
- Pops requests from queue
- Processes each request
- Notifies guest of completion

### Kernel Driver (sst-virtio.c)

#### Driver State

```c
struct sst_virt_device {
    struct virtio_device *vdev;

    /* VirtQueues */
    struct virtqueue *req_vq;
    struct virtqueue *resp_vq;
    struct virtqueue *event_vq;

    /* Character Device */
    struct cdev cdev;
    dev_t devt;
    int minor;

    /* Request Management */
    spinlock_t req_lock;
    struct list_head pending_requests;
    struct list_head completed_requests;
    u64 next_request_id;

    /* Statistics */
    atomic64_t total_requests;
    atomic64_t total_responses;
    atomic64_t total_events;
    atomic64_t total_errors;

    bool ready;
};
```

#### Request Context

```c
struct sst_request_ctx {
    struct SSTRequest req;
    struct SSTResponse resp;
    enum sst_request_state state;
    wait_queue_head_t wait;
    struct list_head list;
};
```

#### Key Functions

**Request Submission** (`sst-virtio.c:91`):
```c
int sst_submit_request(struct sst_virt_device *sdev,
                      struct SSTRequest *req,
                      struct SSTResponse *resp)
```
- Allocates request context
- Assigns unique request ID
- Adds to VirtQueue with scatter-gather lists
- Kicks VirtQueue
- Waits for response (5-second timeout)

**Response Callback** (`sst-virtio.c:33`):
```c
void sst_response_done(struct virtqueue *vq)
```
- Called on interrupt from QEMU
- Retrieves responses from VirtQueue
- Marks requests as completed
- Wakes up waiting processes

**File Operations** (`sst-virtio.c:166-243`):
```c
ssize_t sst_write(struct file *filp, const char __user *buf,
                 size_t count, loff_t *pos)
```
- Copies `SSTRequest` from user space
- Submits to VirtQueue
- Stores response for subsequent read

```c
ssize_t sst_read(struct file *filp, char __user *buf,
                size_t count, loff_t *pos)
```
- Returns cached `SSTResponse` to user
- Clears cached response

## Development Workflows

### Adding New Request Type

1. **Update Protocol** (`sst-protocol.h`):
```c
enum SSTRequestType {
    ...
    SST_REQ_MY_NEW_REQUEST = 8,
};

// Add payload structure if needed
struct SSTRequest {
    ...
    union {
        ...
        struct {
            uint32_t my_param1;
            uint64_t my_param2;
        } my_request;
    } payload;
};
```

2. **Handle in VirtIO Device** (`virtio-sst.c:140`):
```c
void virtio_sst_process_request(VirtIOSST *s, VirtQueueElement *elem)
{
    ...
    switch (req->type) {
    ...
    case SST_REQ_MY_NEW_REQUEST:
        // Handle locally or forward to SST
        success = virtio_sst_send_request(s, req, resp);
        break;
    }
}
```

3. **Update SST Component** (if forwarding to SST):
Implement handling in `ACALSimDeviceComponent` to process new request type.

4. **Add Application Example**:
```c
struct SSTRequest req = {
    .type = SST_REQ_MY_NEW_REQUEST,
    .payload.my_request = {
        .my_param1 = 123,
        .my_param2 = 456
    }
};

write(fd, &req, sizeof(req));
read(fd, &resp, sizeof(resp));
```

### Implementing Async Events

1. **Define Event Structure** (`sst-protocol.h`):
```c
enum SSTEventType {
    SST_EVENT_COMPLETION = 0,
    SST_EVENT_ERROR = 1,
    SST_EVENT_NOTIFICATION = 2,
};

struct SSTEvent {
    uint32_t type;
    uint32_t reserved;
    uint64_t timestamp;
    uint64_t data;
} __attribute__((packed));
```

2. **Inject from VirtIO Device** (`virtio-sst.c:220`):
```c
void virtio_sst_inject_event(VirtIOSST *s, struct SSTEvent *event)
{
    // Already implemented - pops buffer from event queue
    // Copies event to guest memory
    // Triggers interrupt
}
```

3. **Handle in Driver** (`sst-virtio.c:66`):
```c
void sst_event_done(struct virtqueue *vq)
{
    // Receive event from VirtQueue
    // Notify user space (e.g., via poll/select)
}
```

4. **User Space Reception**:
```c
// Register event queue buffer
ioctl(fd, SST_IOCTL_REGISTER_EVENT_BUF, &event_buf);

// Poll for events
struct pollfd pfd = {.fd = fd, .events = POLLIN};
poll(&pfd, 1, -1);

// Read event
struct SSTEvent event;
read(fd, &event, sizeof(event));
```

### Debugging Techniques

#### QEMU Debugging

```bash
# Enable VirtIO logging
qemu-system-riscv64 ... \
    -d guest_errors,unimp \
    -D qemu-debug.log

# View VirtIO SST messages
grep "VirtIO SST" qemu-debug.log
```

#### Kernel Driver Debugging

```bash
# Enable dynamic debug
echo 'module virtio_sst +p' > /sys/kernel/debug/dynamic_debug/control

# View kernel messages
dmesg -w | grep virtio-sst

# Check VirtQueue status
cat /sys/devices/virtio*/virtio*/status
```

#### SST Debugging

```python
# Add verbose logging in SST configuration
sst_device.addParams({
    "socket_path": SOCKET_PATH,
    "verbose": "1",  # Enable verbose output
    "debug": "1"     # Enable debug logging
})
```

#### Protocol Debugging

Add tracing to requests/responses:

```c
// In application
printf("Sending request: type=%s, id=%lu\n",
       sst_request_type_str(req.type), req.request_id);

// In driver
pr_debug("Request: type=%u, id=%llu, user_data=%llx\n",
         req->type, req->request_id, req->user_data);

// In QEMU device
qemu_log("Processing request: type=%s, id=%lu\n",
         sst_request_type_str(req->type), req->request_id);
```

### Performance Optimization

#### Request Batching

Modify driver to batch multiple requests:

```c
// Submit multiple requests before waiting
for (i = 0; i < batch_size; i++) {
    virtqueue_add_sgs(sdev->req_vq, sgs[i], 1, 1, ctx[i], GFP_ATOMIC);
}
virtqueue_kick(sdev->req_vq);

// Wait for all completions
for (i = 0; i < batch_size; i++) {
    wait_event(ctx[i]->wait, ctx[i]->state == COMPLETED);
}
```

#### Zero-Copy with mmap

Implement shared memory for large transfers:

```c
// Driver: expose device memory
static int sst_mmap(struct file *filp, struct vm_area_struct *vma)
{
    // Map VirtQueue buffers to user space
    // Avoid copy_to_user/copy_from_user overhead
}
```

#### Multi-Queue Support

For parallel operations:

1. Add multiple VirtQueue pairs in QEMU device
2. Update driver to manage multiple queues
3. Distribute requests across queues
4. Use per-CPU queues for scalability

## Testing

### Unit Testing

**Test VirtIO Device** (QEMU):
```c
// Test request processing
struct SSTRequest req = {.type = SST_REQ_NOOP};
struct SSTResponse resp;
virtio_sst_process_request(sdev, &req, &resp);
assert(resp.status == SST_STATUS_OK);
```

**Test Kernel Driver**:
```bash
# Load module with test mode
insmod virtio-sst.ko test_mode=1

# Run internal self-tests
echo 1 > /sys/module/virtio_sst/parameters/run_tests
```

### Integration Testing

**End-to-End Test**:
```bash
# Terminal 1: Start SST
sst linux_basic.py

# Terminal 2: Start QEMU
./run-linux.sh

# Terminal 3: Run test suite
/apps/sst-test
```

### Stress Testing

**High-Volume Requests**:
```c
for (i = 0; i < 100000; i++) {
    write(fd, &req, sizeof(req));
    read(fd, &resp, sizeof(resp));
}
```

**Concurrent Access**:
```bash
# Multiple processes accessing device
for i in {1..10}; do
    /apps/sst-test &
done
wait
```

## Common Pitfalls

1. **VirtQueue Buffer Alignment**: Ensure buffers are properly aligned for DMA
2. **Request ID Overflow**: Handle 64-bit request ID wraparound
3. **Timeout Handling**: Always set reasonable timeouts to avoid deadlocks
4. **Memory Barriers**: Use proper barriers when sharing data between CPU and device
5. **Error Handling**: Always check return values and handle errors gracefully

## Future Enhancements

- **Async I/O**: Support for io_uring
- **Hot-plug**: Dynamic device addition/removal
- **Live Migration**: Save/restore device state
- **Performance Counters**: Expose performance metrics via sysfs
- **Multi-queue**: Parallel request processing
- **DMA**: Direct memory access for large transfers

## References

- [VirtIO Specification v1.1](https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html)
- [Linux VirtIO Driver API](https://www.kernel.org/doc/html/latest/driver-api/virtio/virtio.html)
- [QEMU VirtIO Implementation](https://www.qemu.org/docs/master/devel/virtio-backends.html)
- [SST Documentation](http://sst-simulator.org/SSTPages/SSTMainDocumentation/)

---

For questions or contributions, please refer to the project repository and documentation.
