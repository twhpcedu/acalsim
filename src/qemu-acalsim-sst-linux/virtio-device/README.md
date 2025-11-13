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

# VirtIO SST Device for QEMU

This directory contains the VirtIO SST device implementation for QEMU, enabling Linux guests to communicate with SST simulator components through a standard VirtIO interface.

## Files

- **sst-protocol.h** - Protocol definitions shared between QEMU device and Linux driver
- **virtio-sst.h** - VirtIO SST device header
- **virtio-sst.c** - VirtIO SST device implementation

## Architecture

```
┌─────────────────────────────────────┐
│  Linux Guest                        │
│  ┌──────────────┐                  │
│  │  /dev/sst0   │  Character device │
│  └──────┬───────┘                  │
│         │                           │
│  ┌──────▼───────┐                  │
│  │ virtio-sst.ko│  Kernel driver   │
│  └──────┬───────┘                  │
│         │ VirtIO Protocol          │
└─────────┼───────────────────────────┘
          │
┌─────────▼───────────────────────────┐
│  QEMU                               │
│  ┌──────────────┐                  │
│  │ virtio-sst.c │  VirtIO device   │
│  └──────┬───────┘                  │
│         │ Unix socket              │
└─────────┼───────────────────────────┘
          │
┌─────────▼───────────────────────────┐
│  SST Simulator                      │
│  ┌────────────────────────────┐    │
│  │  SST Component             │    │
│  │  (Cycle-accurate model)    │    │
│  └────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Building

### Integration with QEMU

To integrate this device into QEMU, you need to:

1. **Copy files to QEMU source tree**:
   ```bash
   cp sst-protocol.h $QEMU_SRC/include/hw/virtio/
   cp virtio-sst.h $QEMU_SRC/include/hw/virtio/
   cp virtio-sst.c $QEMU_SRC/hw/virtio/
   ```

2. **Update QEMU build files**:

   Add to `$QEMU_SRC/hw/virtio/meson.build`:
   ```meson
   virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))
   ```

   Add to `$QEMU_SRC/hw/virtio/Kconfig`:
   ```kconfig
   config VIRTIO_SST
       bool
       default y
       depends on VIRTIO
   ```

3. **Rebuild QEMU**:
   ```bash
   cd $QEMU_BUILD
   ninja
   ```

## Usage

### Starting QEMU with VirtIO SST Device

```bash
qemu-system-riscv64 \
    -machine virt \
    -m 2G \
    -kernel vmlinux \
    -append "root=/dev/vda ro" \
    -device virtio-sst-device,socket=/tmp/qemu-sst.sock,device-id=0 \
    -serial mon:stdio
```

### Device Parameters

- **socket**: Unix socket path for SST communication (required)
- **device-id**: SST device ID for multi-device scenarios (default: 0)

### Multiple Devices

For multi-device simulation:
```bash
qemu-system-riscv64 \
    ... \
    -device virtio-sst-device,socket=/tmp/sst-dev0.sock,device-id=0 \
    -device virtio-sst-device,socket=/tmp/sst-dev1.sock,device-id=1 \
    ...
```

## Protocol

### VirtQueues

The device uses three VirtQueues:

1. **Request Queue** (driver → device): Guest sends SSTRequest structures
2. **Response Queue** (device → driver): Device returns SSTResponse structures
3. **Event Queue** (device → driver): Async event notifications

### Request Flow

1. Driver prepares `SSTRequest` in guest memory
2. Driver adds buffer to request virtqueue
3. Driver kicks virtqueue (MMIO write to queue notifier)
4. QEMU device receives notification
5. Device processes request:
   - Local handling (NOOP, ECHO, GET_INFO)
   - Forward to SST via socket (COMPUTE, READ, WRITE)
6. Device prepares `SSTResponse`
7. Device adds response to response virtqueue
8. Device injects interrupt to guest
9. Driver handles interrupt and processes response

### Request Types

- **NOOP**: Connectivity test
- **ECHO**: Echo data back
- **COMPUTE**: SST computation request
- **READ**: Read from SST device memory
- **WRITE**: Write to SST device memory
- **RESET**: Reset device state
- **GET_INFO**: Get device capabilities
- **CONFIGURE**: Configure device parameters

## Features

Device advertises these feature bits:

- **SST_FEATURE_ECHO** (bit 0): Echo requests supported
- **SST_FEATURE_COMPUTE** (bit 1): Compute operations supported
- **SST_FEATURE_MEMORY** (bit 2): Memory read/write supported
- **SST_FEATURE_EVENTS** (bit 3): Async events supported
- **SST_FEATURE_MULTI_QUEUE** (bit 4): Multiple queue pairs supported
- **SST_FEATURE_RESET** (bit 5): Device reset supported

## SST Connection

The device connects to SST via Unix domain socket. The SST component must:

1. Create Unix socket server at specified path
2. Accept connection from QEMU
3. Read `SSTRequest` structures (binary protocol)
4. Process requests with cycle-accurate simulation
5. Return `SSTResponse` structures

## Debugging

Enable QEMU logging to see device activity:
```bash
qemu-system-riscv64 ... -d guest_errors,unimp -D qemu.log
```

View device messages in QEMU console:
```
VirtIO SST: Initializing device (socket=/tmp/qemu-sst.sock, id=0)
VirtIO SST: Connected to SST at /tmp/qemu-sst.sock
VirtIO SST: Processing request type=COMPUTE id=1
VirtIO SST: Device initialized successfully
```

## Implementation Notes

### Thread Safety
- VirtQueue operations are protected by QEMU's BQL (Big QEMU Lock)
- Socket I/O is synchronous and blocking
- No locking needed for single-threaded SST communication

### Memory Management
- Request/response buffers allocated per operation
- VirtQueueElement freed after completion
- No persistent request tracking (stateless)

### Error Handling
- Socket errors logged and counted in statistics
- Failed requests return SST_STATUS_ERROR or SST_STATUS_NO_DEVICE
- Device remains functional even without SST connection

### Future Enhancements
- Async socket I/O with GSource/QIOChannel
- Request pipelining for better throughput
- Multi-queue support for parallel operations
- Hot-plug/unplug support
- Live migration support

## Related Files

- **../drivers/sst-virtio.c** - Linux kernel driver
- **../rootfs/apps/** - User-space test applications
- **../sst-config/** - SST configuration files

## References

- [VirtIO Specification v1.1](https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html)
- [QEMU VirtIO Documentation](https://www.qemu.org/docs/master/system/devices/virtio-net.html)
- [Linux VirtIO Driver API](https://www.kernel.org/doc/html/latest/driver-api/virtio/virtio.html)
