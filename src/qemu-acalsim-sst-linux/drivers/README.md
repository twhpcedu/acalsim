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

# VirtIO SST Linux Kernel Driver

This directory contains the Linux kernel driver for the VirtIO SST device, enabling user-space applications to communicate with SST simulator through standard `/dev/sst*` character devices.

## Files

- **sst-device.h** - Driver header with data structures and prototypes
- **sst-virtio.c** - Main driver implementation
- **Makefile** - Build system for kernel module

## Architecture

```
┌─────────────────────────────────────┐
│  User Space                         │
│  ┌──────────────┐                  │
│  │  Application │                  │
│  └──────┬───────┘                  │
│         │ read/write/ioctl         │
│  ┌──────▼───────┐                  │
│  │  /dev/sst0   │                  │
│  └──────┬───────┘                  │
└─────────┼───────────────────────────┘
          │ system calls
┌─────────▼───────────────────────────┐
│  Kernel Space                       │
│  ┌──────────────┐                  │
│  │ sst-virtio.c │  Character device│
│  └──────┬───────┘  driver          │
│         │                           │
│  ┌──────▼───────┐                  │
│  │  VirtIO Core │  Linux VirtIO    │
│  └──────┬───────┘  subsystem       │
└─────────┼───────────────────────────┘
          │ VirtIO Protocol
┌─────────▼───────────────────────────┐
│  QEMU virtio-sst Device             │
└─────────────────────────────────────┘
```

## Building

### Prerequisites

- Linux kernel headers (matching your kernel version)
- GCC compiler
- Make

Install on Debian/Ubuntu:
```bash
sudo apt-get install linux-headers-$(uname -r) build-essential
```

Install on RISC-V:
```bash
# For cross-compilation
sudo apt-get install gcc-riscv64-linux-gnu
```

### Build Module

```bash
# Build for current kernel
make

# Build for specific kernel
make KDIR=/path/to/kernel/build
```

This produces `virtio-sst.ko` kernel module.

### Install Module

```bash
# Install to system
sudo make install

# Load module
sudo modprobe virtio-sst

# Or load directly
sudo insmod virtio-sst.ko
```

### Verify Module

```bash
# Check if module is loaded
lsmod | grep virtio_sst

# View module info
modinfo virtio-sst.ko

# Check kernel messages
dmesg | grep virtio-sst

# Check device nodes
ls -l /dev/sst*
```

## Usage

### Device Interface

The driver creates character devices at `/dev/sst0`, `/dev/sst1`, etc., one for each VirtIO SST device discovered.

### Programming Interface

Applications interact with the device using standard system calls:

```c
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "../virtio-device/sst-protocol.h"

// Open device
int fd = open("/dev/sst0", O_RDWR);

// Prepare request
struct SSTRequest req = {
    .type = SST_REQ_COMPUTE,
    .request_id = 0,  // Will be assigned by driver
    .user_data = 123,
    .payload.compute = {
        .compute_units = 1000,
        .latency_model = 0
    }
};

// Send request
write(fd, &req, sizeof(req));

// Receive response
struct SSTResponse resp;
read(fd, &resp, sizeof(resp));

// Check response
if (resp.status == SST_STATUS_OK) {
    printf("Compute completed: %llu cycles\n",
           resp.payload.compute.cycles);
}

close(fd);
```

### Request Flow

1. **Open device**: `open("/dev/sst0", O_RDWR)`
2. **Prepare request**: Fill `SSTRequest` structure
3. **Submit request**: `write(fd, &req, sizeof(req))`
4. **Wait for completion**: Request automatically submitted to VirtQueue
5. **Read response**: `read(fd, &resp, sizeof(resp))`
6. **Process result**: Check `resp.status` and extract data

### Supported Request Types

- **SST_REQ_NOOP**: No operation (connectivity test)
- **SST_REQ_ECHO**: Echo data back
- **SST_REQ_COMPUTE**: Submit compute request to SST
- **SST_REQ_READ**: Read from SST device memory
- **SST_REQ_WRITE**: Write to SST device memory
- **SST_REQ_RESET**: Reset device state
- **SST_REQ_GET_INFO**: Get device capabilities
- **SST_REQ_CONFIGURE**: Configure device parameters

## Implementation Details

### Driver Components

#### Character Device
- Creates `/dev/sst*` nodes for user-space access
- Implements file operations: open, release, read, write, ioctl, poll
- Manages per-file context with active requests

#### VirtIO Integration
- Implements VirtIO driver interface (probe, remove, validate)
- Manages three VirtQueues: request, response, event
- Handles VirtQueue callbacks for async notifications

#### Request Management
- Tracks pending and completed requests
- Assigns unique request IDs
- Implements wait queues for synchronous I/O
- Timeout handling (5 second default)

### Synchronization

- **Spinlock** (`req_lock`): Protects request lists and VirtQueue operations
- **Wait queues**: Block read operations until response available
- **Atomic counters**: Track statistics without locking

### Memory Management

- Request contexts allocated per operation
- Scatter-gather lists for VirtQueue buffers
- Proper cleanup on errors and device removal

## Debugging

### Enable Debug Output

```bash
# Load module with debug enabled (if compiled with DEBUG)
sudo insmod virtio-sst.ko debug=1

# View kernel messages
dmesg -w | grep virtio-sst

# Or view full kernel log
sudo journalctl -k -f | grep virtio-sst
```

### Common Issues

#### Device Not Found
```
Error: No such device
```
**Solution**: Ensure VirtIO SST device is present in QEMU configuration:
```bash
qemu-system-riscv64 ... -device virtio-sst-device,socket=/tmp/qemu-sst.sock
```

#### Module Load Failure
```
Error: Could not insert module: Unknown symbol in module
```
**Solution**: Rebuild module against correct kernel headers:
```bash
make clean
make KDIR=/lib/modules/$(uname -r)/build
```

#### Request Timeout
```
virtio-sst: Request timed out
```
**Solution**: Check SST simulator is running and socket is accessible.

## Statistics

The driver tracks per-device statistics:

```bash
# View in kernel log
dmesg | grep "Statistics"

# Or read from sysfs (if implemented)
cat /sys/class/virtio-sst/sst0/stats
```

Statistics include:
- Total requests submitted
- Total responses received
- Total events received
- Total errors encountered

## Module Parameters

Currently no module parameters. Future enhancements may include:
- `request_timeout`: Request timeout in milliseconds
- `max_devices`: Maximum number of devices to support
- `debug`: Enable debug output

## Future Enhancements

- **Async I/O**: Support for `io_uring` and `AIO`
- **mmap support**: Zero-copy data transfer
- **Multiple readers**: Support concurrent access from multiple processes
- **Event notifications**: Async event delivery to user space
- **Sysfs interface**: Runtime configuration and statistics
- **Netlink interface**: For management and monitoring

## Related Files

- **../virtio-device/** - QEMU VirtIO device implementation
- **../rootfs/apps/** - Example user-space applications
- **../kernel/device-tree/** - Device tree configuration

## References

- [Linux Device Drivers (LDD3)](https://lwn.net/Kernel/LDD3/)
- [Linux VirtIO API](https://www.kernel.org/doc/html/latest/driver-api/virtio/virtio.html)
- [VirtIO Specification](https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html)
- [Linux Kernel Module Programming Guide](https://tldp.org/LDP/lkmpg/2.6/html/)
