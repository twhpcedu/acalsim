# How to Add N SST Devices Systematically

This document describes systematic approaches for adding N devices to the QEMU-SST integration framework.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current Architecture Limitations](#current-architecture-limitations)
3. [Approach 1: Device Router Component](#approach-1-device-router-component)
4. [Approach 2: Multi-Port QEMUBinary Component](#approach-2-multi-port-qemubinary-component)
5. [Approach 3: Hierarchical Device Networks](#approach-3-hierarchical-device-networks)
6. [Approach 4: Python Configuration Generator](#approach-4-python-configuration-generator)
7. [Comparison and Recommendations](#comparison-and-recommendations)
8. [Implementation Guide](#implementation-guide)

---

## Problem Statement

### Current Status (N=2 devices)

The current implementation supports 2 devices with limitations:
- Device 1 (Echo) @ 0x10200000
- Device 2 (Compute) @ 0x10300000
- All transactions route through QEMUBinaryComponent's single `device_port`
- No systematic way to add Device 3, 4, ..., N

### Requirements for N Devices

1. **Scalability**: Support arbitrary number of devices (N = 2, 4, 8, 16, ...)
2. **Address-Based Routing**: Route MMIO transactions to correct device based on address
3. **Configuration Simplicity**: Easy to add new devices without modifying core components
4. **Performance**: Minimal routing overhead
5. **Maintainability**: Clean architecture, easy to debug

---

## Current Architecture Limitations

### Single Device Port

`QEMUBinaryComponent` currently has one `device_port`:

```cpp
class QEMUBinaryComponent : public SST::Component {
    SST_ELI_DOCUMENT_PORTS(
        {"device_port", "Port to device component", {}}
    )

    // Problem: How to route to multiple devices?
    SST::Link* device_port;  // Only ONE port!
};
```

### No Address-Based Routing

When MMIO request arrives, component cannot distinguish which device:

```cpp
void handleMMIORequest() {
    MMIORequest req = receive_from_qemu();

    // Problem: Which device should receive this?
    // Address 0x10200000 → Device 1 (Echo)
    // Address 0x10300000 → Device 2 (Compute)
    // Address 0x10400000 → Device 3 (???)

    device_port->send(event);  // Always sends to same port!
}
```

---

## Approach 1: Device Router Component

**Concept**: Create a separate router component that distributes transactions based on address.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        SST Simulation                          │
│                                                                 │
│  ┌─────────────┐      ┌──────────────┐     ┌──────────────┐  │
│  │   QEMU      │      │    Device    │     │   Device 1   │  │
│  │  Component  ├─────►│    Router    ├────►│  (Echo)      │  │
│  └─────────────┘      │  Component   │     └──────────────┘  │
│                        │              │                        │
│                        │              │     ┌──────────────┐  │
│                        │              ├────►│   Device 2   │  │
│                        │              │     │  (Compute)   │  │
│                        │              │     └──────────────┘  │
│                        │              │                        │
│                        │              │     ┌──────────────┐  │
│                        │              ├────►│   Device N   │  │
│                        └──────────────┘     └──────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**1. Define Device Router Component**

```cpp
// DeviceRouterComponent.hh
#include <sst/core/component.h>
#include <vector>

namespace ACALSim {

struct DeviceRoute {
    uint64_t base_addr;   // Device base address
    uint64_t size;        // Device memory size
    SST::Link* port;      // Link to device
    std::string name;     // Device name (for debugging)
};

class DeviceRouterComponent : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        DeviceRouterComponent,
        "acalsim",
        "DeviceRouter",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Routes MMIO transactions to multiple devices based on address",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"num_devices", "Number of devices to support", "1"},
        {"device%d_base", "Base address for device %d", "0x10000000"},
        {"device%d_size", "Memory size for device %d", "4096"},
        {"verbose", "Verbosity level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port from CPU (QEMU)", {"MemoryTransaction"}},
        {"device_port_%d", "Port to device %d", {"MemoryTransaction"}}
    )

    DeviceRouterComponent(SST::ComponentId_t id, SST::Params& params);
    ~DeviceRouterComponent() override;

    void handleCPURequest(SST::Event* ev);
    void handleDeviceResponse(SST::Event* ev);

private:
    SST::Link* cpu_link_;
    std::vector<DeviceRoute> routes_;
    SST::Output out_;

    // Find route for address
    DeviceRoute* findRoute(uint64_t address);
};

}  // namespace ACALSim
```

**2. Implement Router Logic**

```cpp
// DeviceRouterComponent.cc
#include "DeviceRouterComponent.hh"
#include "ACALSimDeviceComponent.hh"

using namespace ACALSim;

DeviceRouterComponent::DeviceRouterComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id) {

    int verbose = params.find<int>("verbose", 0);
    out_.init("DeviceRouter[@f:@l:@p] ", verbose, 0, SST::Output::STDOUT);

    // Configure CPU link
    cpu_link_ = configureLink("cpu_port",
        new SST::Event::Handler<DeviceRouterComponent>(
            this, &DeviceRouterComponent::handleCPURequest));

    // Get number of devices
    int num_devices = params.find<int>("num_devices", 1);
    out_.verbose(CALL_INFO, 1, 0, "Configuring router for %d devices\n", num_devices);

    // Configure each device port
    for (int i = 0; i < num_devices; i++) {
        DeviceRoute route;

        // Get device parameters
        std::string base_key = "device" + std::to_string(i) + "_base";
        std::string size_key = "device" + std::to_string(i) + "_size";

        route.base_addr = params.find<uint64_t>(base_key, 0x10000000 + (i * 0x100000));
        route.size = params.find<uint64_t>(size_key, 4096);
        route.name = "device" + std::to_string(i);

        // Configure port
        std::string port_name = "device_port_" + std::to_string(i);
        route.port = configureLink(port_name,
            new SST::Event::Handler<DeviceRouterComponent>(
                this, &DeviceRouterComponent::handleDeviceResponse));

        if (!route.port) {
            out_.fatal(CALL_INFO, -1, "Failed to configure %s\n", port_name.c_str());
        }

        routes_.push_back(route);

        out_.verbose(CALL_INFO, 1, 0, "Device %d: [0x%lx, 0x%lx) → %s\n",
                     i, route.base_addr, route.base_addr + route.size, port_name.c_str());
    }
}

DeviceRouterComponent::~DeviceRouterComponent() {
    // Nothing to do
}

void DeviceRouterComponent::handleCPURequest(SST::Event* ev) {
    auto* trans = dynamic_cast<QEMUIntegration::MemoryTransactionEvent*>(ev);
    if (!trans) {
        out_.fatal(CALL_INFO, -1, "Invalid event type from CPU\n");
    }

    uint64_t address = trans->getAddress();

    // Find matching route
    DeviceRoute* route = findRoute(address);
    if (!route) {
        out_.verbose(CALL_INFO, 1, 0, "ERROR: No device mapped at address 0x%lx\n", address);

        // Send error response
        auto* resp = new QEMUIntegration::MemoryResponseEvent(
            trans->getReqId(), 0, false);
        cpu_link_->send(resp);

        delete ev;
        return;
    }

    // Forward to device
    out_.verbose(CALL_INFO, 2, 0, "Routing 0x%lx to %s\n", address, route->name.c_str());
    route->port->send(trans);
}

void DeviceRouterComponent::handleDeviceResponse(SST::Event* ev) {
    // Forward response back to CPU
    cpu_link_->send(ev);
}

DeviceRoute* DeviceRouterComponent::findRoute(uint64_t address) {
    for (auto& route : routes_) {
        if (address >= route.base_addr &&
            address < route.base_addr + route.size) {
            return &route;
        }
    }
    return nullptr;
}
```

**3. SST Configuration**

```python
#!/usr/bin/env python3
import sst

# Device configurations
devices = [
    {"name": "echo",    "base": 0x10200000, "component": "acalsim.QEMUDevice"},
    {"name": "compute", "base": 0x10300000, "component": "acalsim.ComputeDevice"},
    {"name": "memory",  "base": 0x10400000, "component": "acalsim.MemoryDevice"},
    {"name": "dma",     "base": 0x10500000, "component": "acalsim.DMADevice"},
]

# QEMU component
qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "binary_path": "test.elf",
})

# Device Router
router = sst.Component("router0", "acalsim.DeviceRouter")
router.addParams({
    "num_devices": str(len(devices)),
})

# Configure router parameters for each device
for i, dev in enumerate(devices):
    router.addParams({
        f"device{i}_base": str(dev["base"]),
        f"device{i}_size": "4096",
    })

# Link QEMU to Router
qemu_router_link = sst.Link("qemu_router_link")
qemu_router_link.connect(
    (qemu, "device_port", "1ns"),
    (router, "cpu_port", "1ns")
)

# Create devices and link to router
for i, dev in enumerate(devices):
    # Create device component
    device = sst.Component(dev["name"], dev["component"])
    device.addParams({
        "base_addr": str(dev["base"]),
        "size": "4096",
    })

    # Link router to device
    link = sst.Link(f"router_device{i}_link")
    link.connect(
        (router, f"device_port_{i}", "1ns"),
        (device, "cpu_port", "1ns")
    )

print(f"Configuration complete: {len(devices)} devices configured")
```

### Advantages

✅ **Clean separation**: Router logic isolated from QEMUBinary and devices
✅ **Easy to extend**: Add devices by modifying SST config only
✅ **No core component changes**: QEMUBinary and devices unchanged
✅ **Flexible routing**: Can implement complex routing logic

### Disadvantages

❌ **Extra component**: Additional hop adds latency
❌ **More links**: 2N+1 links instead of N+1 (QEMU→Router, Router→N devices)
❌ **Configuration complexity**: More parameters to configure

---

## Approach 2: Multi-Port QEMUBinary Component

**Concept**: Extend QEMUBinaryComponent to support multiple device ports with built-in routing.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        SST Simulation                          │
│                                                                 │
│  ┌─────────────────────────────────┐                          │
│  │   QEMUBinaryComponent           │                          │
│  │                                 │                          │
│  │   ┌─────────────────────────┐   │    ┌──────────────┐    │
│  │   │  Address Router         │   │    │   Device 1   │    │
│  │   │  0x10200000 → Port 0    ├───┼───►│  (Echo)      │    │
│  │   │  0x10300000 → Port 1    │   │    └──────────────┘    │
│  │   │  0x10400000 → Port 2    ├───┼──┐                      │
│  │   │  ...                    │   │  │ ┌──────────────┐    │
│  │   └─────────────────────────┘   │  │ │   Device 2   │    │
│  │                                 │  └►│  (Compute)   │    │
│  └─────────────────────────────────┘    └──────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Implementation

**1. Update QEMUBinaryComponent Header**

```cpp
// QEMUBinaryComponent.hh
class QEMUBinaryComponent : public SST::Component {
public:
    SST_ELI_DOCUMENT_PARAMS(
        // ... existing params ...
        {"num_devices", "Number of devices", "1"},
        {"device%d_base", "Base address for device %d", "0x10000000"},
        {"device%d_size", "Size for device %d", "4096"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"device_port_%d", "Port to device %d", {}}
    )

private:
    struct DeviceInfo {
        uint64_t base_addr;
        uint64_t size;
        SST::Link* link;
    };

    std::vector<DeviceInfo> devices_;

    // Route MMIO request to correct device
    void routeMMIORequest(const MMIORequest& req);

    // Find device for address
    DeviceInfo* findDevice(uint64_t address);
};
```

**2. Update Constructor**

```cpp
QEMUBinaryComponent::QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id) {

    // ... existing initialization ...

    // Configure multiple devices
    int num_devices = params.find<int>("num_devices", 1);
    for (int i = 0; i < num_devices; i++) {
        DeviceInfo dev;

        // Get device parameters
        std::string base_key = "device" + std::to_string(i) + "_base";
        std::string size_key = "device" + std::to_string(i) + "_size";

        dev.base_addr = params.find<uint64_t>(base_key, 0x10000000);
        dev.size = params.find<uint64_t>(size_key, 4096);

        // Configure link
        std::string port_name = "device_port_" + std::to_string(i);
        dev.link = configureLink(port_name,
            new SST::Event::Handler<QEMUBinaryComponent>(
                this, &QEMUBinaryComponent::handleDeviceResponse));

        devices_.push_back(dev);

        out_.verbose(CALL_INFO, 1, 0, "Device %d: [0x%lx, 0x%lx)\n",
                     i, dev.base_addr, dev.base_addr + dev.size);
    }
}
```

**3. Implement Routing**

```cpp
void QEMUBinaryComponent::routeMMIORequest(const MMIORequest& req) {
    // Find device for this address
    DeviceInfo* dev = findDevice(req.address);

    if (!dev) {
        // No device at this address - send error
        MMIOResponse error_resp = {
            .magic = MMIO_MAGIC_RESPONSE,
            .status = 1,  // ERROR
            .data = 0,
            .latency = 0
        };
        send(client_fd, &error_resp, sizeof(error_resp), 0);
        return;
    }

    // Create SST event
    auto* event = new MemoryTransactionEvent(
        (req.type == MMIO_TYPE_READ) ? TransactionType::LOAD : TransactionType::STORE,
        req.address,
        req.data,
        req.size,
        next_req_id++
    );

    // Send to appropriate device
    dev->link->send(event);
}

DeviceInfo* QEMUBinaryComponent::findDevice(uint64_t address) {
    for (auto& dev : devices_) {
        if (address >= dev.base_addr &&
            address < dev.base_addr + dev.size) {
            return &dev;
        }
    }
    return nullptr;
}
```

**4. SST Configuration**

```python
import sst

devices = [
    {"name": "echo",    "base": 0x10200000, "comp": "acalsim.QEMUDevice"},
    {"name": "compute", "base": 0x10300000, "comp": "acalsim.ComputeDevice"},
    {"name": "memory",  "base": 0x10400000, "comp": "acalsim.MemoryDevice"},
]

# QEMU with routing config
qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "binary_path": "test.elf",
    "num_devices": str(len(devices)),
})

# Add device address ranges
for i, dev in enumerate(devices):
    qemu.addParams({
        f"device{i}_base": f"0x{dev['base']:08X}",
        f"device{i}_size": "4096",
    })

# Create devices and link
for i, dev in enumerate(devices):
    device = sst.Component(dev["name"], dev["comp"])
    device.addParams({"base_addr": str(dev["base"])})

    link = sst.Link(f"qemu_device{i}_link")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (device, "cpu_port", "1ns")
    )
```

### Advantages

✅ **Performance**: No extra hop, direct routing
✅ **Fewer links**: N+1 total links
✅ **Centralized routing**: All logic in one component

### Disadvantages

❌ **Core component modification**: Requires changing QEMUBinaryComponent
❌ **Recompilation**: Need to rebuild component for changes
❌ **Less flexible**: Routing logic fixed in C++ code

---

## Approach 3: Hierarchical Device Networks

**Concept**: Organize devices in a hierarchy with multiple routers.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SST Simulation                        │
│                                                               │
│  ┌──────────┐      ┌──────────────┐                         │
│  │  QEMU    ├─────►│ Main Router  │                         │
│  └──────────┘      └──────┬───────┘                         │
│                           │                                   │
│                ┌──────────┴──────────┐                       │
│                │                     │                       │
│         ┌──────▼────────┐     ┌─────▼────────┐             │
│         │ Peripheral    │     │ Memory       │             │
│         │ Router        │     │ Router       │             │
│         └──┬─────┬──────┘     └──┬────┬──────┘             │
│            │     │               │    │                     │
│      ┌─────▼┐  ┌─▼────┐    ┌────▼┐  ┌▼────┐               │
│      │Echo  │  │Compute│    │DRAM │  │Flash│               │
│      └──────┘  └───────┘    └─────┘  └─────┘               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Use Case

Good for complex systems with device categories:
- 0x1000_0000 - 0x1FFF_FFFF: Peripherals (echo, compute, UART, etc.)
- 0x2000_0000 - 0x2FFF_FFFF: Memory devices (DRAM, Flash, etc.)
- 0x3000_0000 - 0x3FFF_FFFF: Accelerators (GPU, NPU, etc.)

### Implementation

Similar to Approach 1, but with multiple router instances.

---

## Approach 4: Python Configuration Generator

**Concept**: Generate SST configuration programmatically from device specification.

### Device Specification (YAML)

```yaml
# devices.yaml
devices:
  - name: echo_dev
    component: acalsim.QEMUDevice
    base_addr: 0x10200000
    size: 4096
    params:
      echo_latency: 10

  - name: compute_dev
    component: acalsim.ComputeDevice
    base_addr: 0x10300000
    size: 4096
    params:
      compute_latency: 100

  - name: memory_dev
    component: acalsim.MemoryDevice
    base_addr: 0x10400000
    size: 1048576  # 1MB
    params:
      read_latency: 50
      write_latency: 100

  - name: dma_dev
    component: acalsim.DMADevice
    base_addr: 0x10500000
    size: 4096
    params:
      transfer_rate: 1000  # MB/s
```

### Configuration Generator

```python
#!/usr/bin/env python3
# generate_config.py
import sst
import yaml

def generate_config(device_file):
    # Load device specification
    with open(device_file, 'r') as f:
        spec = yaml.safe_load(f)

    devices = spec['devices']

    # Create QEMU component
    qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
    qemu.addParams({
        "binary_path": "test.elf",
        "num_devices": str(len(devices)),
    })

    # Add device routing info to QEMU
    for i, dev in enumerate(devices):
        qemu.addParams({
            f"device{i}_base": f"0x{dev['base_addr']:08X}",
            f"device{i}_size": str(dev['size']),
        })

    # Create each device
    for i, dev_spec in enumerate(devices):
        device = sst.Component(dev_spec['name'], dev_spec['component'])

        # Add base parameters
        device.addParams({
            "base_addr": str(dev_spec['base_addr']),
            "size": str(dev_spec['size']),
        })

        # Add device-specific parameters
        if 'params' in dev_spec:
            for key, value in dev_spec['params'].items():
                device.addParams({key: str(value)})

        # Link to QEMU
        link = sst.Link(f"qemu_device{i}_link")
        link.connect(
            (qemu, f"device_port_{i}", "1ns"),
            (device, "cpu_port", "1ns")
        )

        print(f"Configured {dev_spec['name']} @ 0x{dev_spec['base_addr']:08X}")

    print(f"\nTotal devices: {len(devices)}")

# Usage
generate_config("devices.yaml")
```

### Advantages

✅ **Declarative**: Devices defined in data, not code
✅ **Easy to maintain**: YAML file easy to edit
✅ **Version control**: Device configurations can be tracked
✅ **Validation**: Can validate device specs before generation

### Disadvantages

❌ **Extra dependency**: Requires YAML parser
❌ **Two-step process**: Edit YAML → Generate config
❌ **Less flexible**: Limited to predefined device types

---

## Comparison and Recommendations

| Aspect | Router Component | Multi-Port QEMU | Hierarchical | Config Generator |
|--------|-----------------|-----------------|--------------|------------------|
| **Complexity** | Medium | Medium | High | Low |
| **Performance** | -1 hop | Direct | -2+ hops | Depends |
| **Flexibility** | High | Medium | Very High | Medium |
| **Maintainability** | Good | Fair | Fair | Excellent |
| **Core Changes** | None | Moderate | None | Depends |
| **Best For** | 2-16 devices | 2-8 devices | Complex SoCs | Any N |

### Recommendations

**For 2-8 Devices**: Use **Approach 2 (Multi-Port QEMU)** combined with **Approach 4 (Config Generator)**
- Modify QEMUBinaryComponent once
- Use Python/YAML for configuration
- Best performance and maintainability

**For 8-32 Devices**: Use **Approach 1 (Router Component)** + **Approach 4 (Config Generator)**
- Separate router component
- Clean separation of concerns
- Slightly higher latency acceptable

**For Complex SoCs (32+ devices)**: Use **Approach 3 (Hierarchical)** + **Approach 4 (Config Generator)**
- Multiple routers organized hierarchically
- Models real hardware organization
- Best for large-scale simulations

---

## Implementation Guide

### Step-by-Step: Multi-Port QEMU + Config Generator

**Step 1: Modify QEMUBinaryComponent**

```bash
cd qemu-binary
# Edit QEMUBinaryComponent.hh and .cc as shown in Approach 2
make clean && make && make install
```

**Step 2: Create Device Specification**

```yaml
# Save as devices.yaml
devices:
  - name: echo
    component: acalsim.QEMUDevice
    base_addr: 0x10200000
    size: 4096
    params:
      echo_latency: 10

  - name: compute
    component: acalsim.ComputeDevice
    base_addr: 0x10300000
    size: 4096
    params:
      compute_latency: 100

  # Add more devices as needed
```

**Step 3: Create Configuration Generator**

```python
# Save as generate_sst_config.py
#!/usr/bin/env python3
import sst
import yaml
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_sst_config.py <device_file.yaml>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        spec = yaml.safe_load(f)

    # Generate configuration (as shown above)
    # ...

if __name__ == "__main__":
    main()
```

**Step 4: Generate and Run**

```bash
# Generate SST config
python3 generate_sst_config.py devices.yaml > auto_config.py

# Run simulation
sst auto_config.py
```

**Step 5: Add New Device**

```yaml
# Add to devices.yaml
  - name: uart
    component: acalsim.UARTDevice
    base_addr: 0x10400000
    size: 256
    params:
      baud_rate: 115200
```

```bash
# Regenerate and run
python3 generate_sst_config.py devices.yaml > auto_config.py
sst auto_config.py
```

### Adding Device #N Checklist

✅ **1. Define Device Component** (if new type)
```bash
cd acalsim-device
# Create DeviceN.hh and DeviceN.cc
make clean && make && make install
```

✅ **2. Add to Device Spec**
```yaml
devices:
  - name: device_n
    component: acalsim.DeviceN
    base_addr: 0x10N00000  # Unique address
    size: 4096
```

✅ **3. Update RISC-V Test Program**
```c
#define DEVICE_N_BASE  0x10N00000
#define DEVICE_N_REG1  (*(volatile uint32_t *)(DEVICE_N_BASE + 0x00))
// ... register definitions ...
```

✅ **4. Rebuild and Test**
```bash
# Rebuild firmware
cd riscv-programs
make clean && make

# Regenerate SST config
python3 generate_sst_config.py devices.yaml > auto_config.py

# Run test
sst auto_config.py
```

---

## Summary

Adding N SST devices systematically requires:

1. **Architecture Decision**: Choose between router, multi-port, or hierarchical
2. **Core Implementation**: Modify QEMUBinaryComponent or create router component
3. **Configuration Management**: Use Python/YAML for device specifications
4. **Testing Framework**: Test each device individually and together

**Recommended Path Forward**:
1. Start with **Approach 2** (Multi-Port) for immediate needs (2-8 devices)
2. Add **Approach 4** (Config Generator) for maintainability
3. Migrate to **Approach 1** (Router) if scaling beyond 8 devices
4. Consider **Approach 3** (Hierarchical) for complex SoC modeling

This systematic approach allows starting simple and scaling as needed without major rewrites.

---

**Last Updated**: 2025-11-10
**Status**: Design Document
**Next Steps**: Implement chosen approach and create working examples
