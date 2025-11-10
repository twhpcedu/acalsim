# QEMU-AcalSim-SST Baremetal Developer Guide

**Target Audience**: Developers who want to understand the internal architecture, extend the system, or build similar simulators.

**Prerequisites**:
- Understanding of SST simulator framework
- C++ development experience
- Familiarity with QEMU internals
- Knowledge of RISC-V architecture
- Unix socket programming experience

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Building from Scratch](#building-from-scratch)
3. [Component Design Patterns](#component-design-patterns)
4. [N-Device Architecture](#n-device-architecture)
5. [SST Python Configuration Customization](#sst-python-configuration-customization)
6. [Device Development Guide](#device-development-guide)
7. [QEMU Integration Details](#qemu-integration-details)
8. [Debugging and Profiling](#debugging-and-profiling)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Topics](#advanced-topics)

---

## System Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: RISC-V Baremetal Application                      │
│  - Compiled ELF binary                                       │
│  - Runs inside QEMU virtual machine                         │
│  - Accesses devices via MMIO                                │
└───────────────┬─────────────────────────────────────────────┘
                │ MMIO Requests (global addresses)
                ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: QEMU Virtual Machine                              │
│  - QEMU RISC-V system emulator                              │
│  - sst-device: Custom QEMU device (SysBusDevice)            │
│  - Unix socket communication with SST                       │
│  - N-device support via environment variables               │
└───────────────┬─────────────────────────────────────────────┘
                │ Binary Protocol (MMIORequest/MMIOResponse)
                ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: SST Simulation Framework                          │
│  - QEMUBinaryComponent: Socket server + QEMU launcher       │
│  - Device Components: Simulation models                     │
│  - SST Links: Inter-component communication (MPI-capable)   │
└─────────────────────────────────────────────────────────────┘
```

### N-Device Communication Flow

**Complete Transaction Example (2-device system)**:
```
RISC-V Program          QEMU VM             QEMUBinaryComponent    Device 0    Device 1
     │                     │                         │                │            │
     │ Write 0x10200008   │                         │                │            │
     ├────────────────────►│                         │                │            │
     │                     │ MMIOReq(addr=0x10200008)│                │            │
     │                     ├────socket0──────────────►│                │            │
     │                     │                         │ Route by addr  │            │
     │                     │                         ├───SST Event───►│            │
     │                     │                         │                │ Process    │
     │                     │                         │ ◄──Response────┤            │
     │                     │ MMIOResp                │                │            │
     │                     │◄────socket0─────────────┤                │            │
     │ Continue execution │                         │                │            │
     │◄────────────────────┤                         │                │            │
     │                     │                         │                │            │
     │ Read 0x10300004    │                         │                │            │
     ├────────────────────►│                         │                │            │
     │                     │ MMIOReq(addr=0x10300004)│                │            │
     │                     ├────socket1──────────────►│                │            │
     │                     │                         │ Route by addr  │            │
     │                     │                         ├────SST Event──────────────►│
     │                     │                         │                │            │ Process
     │                     │                         │ ◄───Response───────────────┤
     │                     │ MMIOResp                │                │            │
     │                     │◄────socket1─────────────┤                │            │
     │ data=0x...         │                         │                │            │
     │◄────────────────────┤                         │                │            │
```

### Key Design Decisions

#### 1. **Why Per-Device Unix Sockets?**
- **Independent channels**: No head-of-line blocking between devices
- **Isolated failures**: One device crash doesn't affect others
- **Scalability**: Linear scaling to N devices
- **Debugging**: Easy to trace per-device traffic with `strace`

**Alternative Considered**: Single multiplexed socket with device IDs
- Rejected due to routing complexity, contention, and debugging difficulty

#### 2. **Why Environment Variables for QEMU Configuration?**
- **SysBusDevice constraint**: Cannot use `-device` command line
- **QEMU machine code**: virt.c must create devices during initialization
- **Clean interface**: No QEMU source modification per configuration
- **Scalability**: Easy to extend to 16+ devices

**Alternative Considered**: QEMU config file or command-line args
- Rejected as virt machine doesn't support device config files, and SysBusDevices can't be instantiated via `-device`

#### 3. **Why Binary Protocol?**
- **Performance**: 10x faster than text-based serial protocol
- **Simplicity**: Fixed-size structures, no parsing overhead
- **Reliability**: Type-safe, no string conversion errors
- **Efficiency**: Direct memory copy, minimal CPU usage

**Protocol Structures** (qemu-sst-device/sst-device.c:15-28):
```c
struct MMIORequest {
    uint8_t type;      // 0=read, 1=write
    uint64_t addr;     // Global physical address
    uint64_t data;     // Write data (ignored for reads)
} __attribute__((packed));

struct MMIOResponse {
    uint8_t success;   // 0=error, 1=success
    uint64_t data;     // Read data (0 for writes)
} __attribute__((packed));
```

---

## Building from Scratch

This section walks through building the entire system from source on a clean Ubuntu/Debian system.

### Prerequisites Installation

#### 1. SST Core Installation

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    automake \
    autoconf \
    libtool \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

# Clone and build SST Core
cd ~/projects
git clone https://github.com/sstsimulator/sst-core.git
cd sst-core
./autogen.sh
./configure --prefix=$HOME/local/sst-core
make -j$(nproc)
make install

# Set environment variables (add to ~/.bashrc)
export SST_CORE_HOME=$HOME/local/sst-core
export PATH=$SST_CORE_HOME/bin:$PATH
export PKG_CONFIG_PATH=$SST_CORE_HOME/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
```

#### 2. RISC-V Toolchain Installation

```bash
# Download prebuilt toolchain (recommended)
cd ~/projects
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.09.03/riscv32-elf-ubuntu-22.04-gcc-nightly-2024.09.03-nightly.tar.gz
tar xzf riscv32-elf-ubuntu-22.04-gcc-nightly-2024.09.03-nightly.tar.gz
mv riscv ~/local/riscv32-toolchain

# Or build from source (slow, ~2 hours)
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=$HOME/local/riscv32-toolchain --with-arch=rv32gc --with-abi=ilp32d
make -j$(nproc)

# Set environment
export RISCV=$HOME/local/riscv32-toolchain
export PATH=$RISCV/bin:$PATH
```

#### 3. QEMU Build with sst-device Integration

**Critical**: The sst-device must be integrated into QEMU source and built as part of QEMU.

```bash
# Clone QEMU
cd ~/projects
git clone https://gitlab.com/qemu-project/qemu.git qemu-sst
cd qemu-sst
git checkout v8.1.0  # Tested version

# Create sst-device directory
mkdir -p hw/misc/qemu-sst-device
```

**Step 3a: Copy sst-device.c**

From your project: `cp qemu-sst-device/sst-device.c hw/misc/qemu-sst-device/`

**Step 3b: Create hw/misc/qemu-sst-device/meson.build**:
```meson
softmmu_ss.add(when: 'CONFIG_SST_DEVICE', if_true: files('sst-device.c'))
```

**Step 3c: Update hw/misc/Kconfig** - Add to the end:
```kconfig
config SST_DEVICE
    bool
    depends on RISCV
    default y
```

**Step 3d: Update hw/riscv/Kconfig** - Add to RISCV_VIRT section (around line 40):
```kconfig
config RISCV_VIRT
    bool
    select RISCV_NUMA
    select GOLDFISH_RTC
    select MSI_NONBROKEN
    select PCI
    select PCI_EXPRESS
    select PCI_EXPRESS_GENERIC_BRIDGE
    select PFLASH_CFI01
    select SERIAL
    select RISCV_ACLINT
    select RISCV_APLIC
    select RISCV_IMSIC
    select SIFIVE_PLIC
    select SIFIVE_TEST
    select VIRTIO_MMIO
    select FW_CFG_DMA
    select PLATFORM_BUS
    select ACPI
    select ACPI_PCI
    select SST_DEVICE    # ADD THIS LINE
```

**Step 3e: Modify hw/riscv/virt.c** - Add N-device support (around line 948):

Replace the single device instantiation with:
```c
    /* SST integration device(s) - support N devices via environment variables */
    const char *num_devices_str = getenv("SST_NUM_DEVICES");
    int num_sst_devices = num_devices_str ? atoi(num_devices_str) : 1;

    /* Clamp to valid range */
    if (num_sst_devices < 1) num_sst_devices = 1;
    if (num_sst_devices > 16) num_sst_devices = 16;

    for (int dev_idx = 0; dev_idx < num_sst_devices; dev_idx++) {
        char env_socket[64], env_base[64];
        snprintf(env_socket, sizeof(env_socket), "SST_DEVICE%d_SOCKET", dev_idx);
        snprintf(env_base, sizeof(env_base), "SST_DEVICE%d_BASE", dev_idx);

        const char *socket_path = getenv(env_socket);
        const char *base_str = getenv(env_base);

        /* Default values if not specified */
        char default_socket[128];
        if (!socket_path) {
            snprintf(default_socket, sizeof(default_socket),
                     "/tmp/qemu-sst-device%d.sock", dev_idx);
            socket_path = default_socket;
        }

        uint64_t base_addr = base_str ? strtoul(base_str, NULL, 0) :
                             (0x10200000 + dev_idx * 0x100000);

        /* Create and configure device */
        DeviceState *sst_dev = qdev_new("sst-device");
        qdev_prop_set_string(sst_dev, "socket", socket_path);
        qdev_prop_set_uint64(sst_dev, "base_address", base_addr);

        /* Realize and map to system bus */
        sysbus_realize_and_unref(SYS_BUS_DEVICE(sst_dev), &error_fatal);
        sysbus_mmio_map(SYS_BUS_DEVICE(sst_dev), 0, base_addr);

        printf("SST Device %d: socket=%s, base=0x%lx\n",
               dev_idx, socket_path, base_addr);
    }
```

**Step 3f: Configure and build QEMU**:
```bash
cd ~/projects/qemu-sst
mkdir build
cd build
../configure \
    --target-list=riscv32-softmmu \
    --enable-debug \
    --prefix=$HOME/local/qemu-sst

ninja
ninja install

# Verify sst-device compiled
grep -r "sst-device" build.ninja
```

**Step 3g: Verify QEMU Installation**:
```bash
$HOME/local/qemu-sst/bin/qemu-system-riscv32 --version
# Should show: QEMU emulator version 8.1.0
```

#### 4. Build QEMUBinaryComponent

```bash
cd src/qemu-acalsim-sst-baremetal/qemu-binary

# The Makefile uses environment variables
export SST_CORE_HOME=$HOME/local/sst-core
export QEMU_PATH=$HOME/local/qemu-sst/bin/qemu-system-riscv32

# Build
make clean
make -j$(nproc)
make install

# Verify installation
sst-info qemubinary
# Should show QEMUBinary component with parameters
```

**Makefile Key Variables** (qemu-binary/Makefile):
```makefile
SST_CONFIG = $(SST_CORE_HOME)/bin/sst-config
CXX = $(shell $(SST_CONFIG) --CXX)
CXXFLAGS = $(shell $(SST_CONFIG) --ELEMENT_CXXFLAGS) -std=c++14 -g -O2
LDFLAGS = $(shell $(SST_CONFIG) --ELEMENT_LDFLAGS)

# Component source
SOURCES = QEMUBinaryComponent.cc
TARGET = libqemubinary.so

# Installation
INSTALL_DIR = $(shell $(SST_CONFIG) --prefix)/lib/sstcore

$(TARGET): $(SOURCES:.cc=.o)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

install: $(TARGET)
	install -D $(TARGET) $(INSTALL_DIR)/$(TARGET)
```

---

## Component Design Patterns

### QEMUBinaryComponent Architecture

**File**: qemu-binary/QEMUBinaryComponent.hh

```cpp
class QEMUBinaryComponent : public SST::Component {
public:
    // SST Component interface
    QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMUBinaryComponent();

    void setup() override;
    void finish() override;

    // Clock handler (called each simulation cycle)
    bool clockTick(SST::Cycle_t cycle);

    // Event handlers (called when device sends response)
    void handleDeviceEvent(SST::Event* event, int device_id);

    // SST ELI registration macros
    SST_ELI_REGISTER_COMPONENT(
        QEMUBinaryComponent,
        "qemubinary",
        "QEMUBinary",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "QEMU RISC-V binary executor with N-device integration",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"num_devices", "Number of devices", "1"},
        {"device%d_base", "Base address for device %d", "0x10000000"},
        {"device%d_size", "Memory size for device %d", "4096"},
        {"device%d_name", "Name for device %d", "device_%d"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"device_port_%d", "Port to device %d", {}}
    )

private:
    // Device management
    struct DeviceInfo {
        uint64_t base_addr;        // Device base address
        uint64_t size;             // Device memory size
        SST::Link* link;           // Link to device component
        std::string name;          // Device name
        uint64_t num_requests;     // Statistics

        // N-socket support fields
        std::string socket_path;   // Unix socket path
        int server_fd;             // Server socket file descriptor
        int client_fd;             // Client connection file descriptor
        bool socket_ready;         // Connection status
    };

    std::vector<DeviceInfo> devices_;
    bool use_multi_device_;

    // QEMU process management
    pid_t qemu_pid_;
    std::string binary_path_;
    std::string qemu_path_;

    // N-socket methods
    void setupDeviceSocket(DeviceInfo* dev, int dev_id);
    void acceptDeviceConnection(DeviceInfo* dev);
    void pollDeviceSockets();
    void handleMMIORequest(DeviceInfo* dev);
    void sendMMIOResponse(DeviceInfo* dev, bool success, uint64_t data);

    // QEMU lifecycle
    void launchQEMU();
    void shutdownQEMU();

    // Address routing
    DeviceInfo* findDeviceByAddress(uint64_t addr);

    // Logging
    SST::Output output_;
    SST::TimeConverter* clock_tc_;
};
```

### Event Design Pattern

**Why Event Serialization is Critical**:

SST events can cross MPI boundaries in multi-server deployments. Serialization allows events to be transmitted over network between simulation ranks.

```cpp
// Example custom event for device communication
class DeviceEvent : public SST::Event {
public:
    enum Type { READ, WRITE };

    DeviceEvent(Type t, uint64_t addr, uint64_t data = 0)
        : type(t), address(addr), data(data) {}

    Type getType() const { return type; }
    uint64_t getAddress() const { return address; }
    uint64_t getData() const { return data; }

    // CRITICAL: Serialize ALL member variables for MPI transmission
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        ser& type;
        ser& address;
        ser& data;
    }

    ImplementSerializable(DeviceEvent);

private:
    Type type;
    uint64_t address;
    uint64_t data;
};
```

### Link Handler Pattern with Device ID Context

**File**: qemu-binary/QEMUBinaryComponent.cc (Constructor)

```cpp
// In QEMUBinaryComponent constructor
for (size_t i = 0; i < num_devices; i++) {
    std::string port_name = "device_port_" + std::to_string(i);

    // Create handler with device ID as context parameter
    SST::Link* link = configureLink(
        port_name,
        new SST::Event::Handler<QEMUBinaryComponent>(
            this,
            &QEMUBinaryComponent::handleDeviceEvent,
            i  // Pass device ID as handler context
        )
    );

    if (!link) {
        output_.fatal(CALL_INFO, -1, "Failed to configure %s\n",
                      port_name.c_str());
    }

    devices_[i].link = link;
}

// Handler implementation receives device_id directly
void QEMUBinaryComponent::handleDeviceEvent(SST::Event* ev, int device_id) {
    DeviceEvent* dev_ev = dynamic_cast<DeviceEvent*>(ev);
    if (!dev_ev) {
        output_.fatal(CALL_INFO, -1, "Invalid event type\n");
    }

    // Access device directly by ID (no map lookup needed)
    DeviceInfo& dev = devices_[device_id];

    // Process device response
    bool success = (dev_ev->getType() == DeviceEvent::READ);
    uint64_t data = dev_ev->getData();

    sendMMIOResponse(&dev, success, data);

    delete ev;
}
```

**Pattern Benefits**:
- Type-safe event handling
- O(1) device lookup (no map needed)
- Clean separation of concerns
- Handler context eliminates global state

---

## N-Device Architecture

### Socket Lifecycle

**File**: qemu-binary/QEMUBinaryComponent.cc:624-656

```cpp
void QEMUBinaryComponent::setupDeviceSocket(DeviceInfo* dev, int dev_id) {
    // 1. Create unique socket path per device
    dev->socket_path = "/tmp/qemu-sst-device" + std::to_string(dev_id) + ".sock";

    // 2. Remove stale socket file from previous runs
    unlink(dev->socket_path.c_str());

    // 3. Create Unix domain socket
    dev->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (dev->server_fd < 0) {
        output_.fatal(CALL_INFO, -1, "Failed to create socket for device %d: %s\n",
                      dev_id, strerror(errno));
    }

    // 4. Set non-blocking mode (critical for SST simulation)
    int flags = fcntl(dev->server_fd, F_GETFL, 0);
    fcntl(dev->server_fd, F_SETFL, flags | O_NONBLOCK);

    // 5. Bind to socket path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, dev->socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(dev->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        output_.fatal(CALL_INFO, -1, "Failed to bind socket %s: %s\n",
                      dev->socket_path.c_str(), strerror(errno));
    }

    // 6. Listen for incoming connection from QEMU
    if (listen(dev->server_fd, 1) < 0) {
        output_.fatal(CALL_INFO, -1, "Failed to listen on socket %s: %s\n",
                      dev->socket_path.c_str(), strerror(errno));
    }

    dev->client_fd = -1;
    dev->socket_ready = false;

    output_.verbose(CALL_INFO, 1, 0, "Device %s socket listening at %s\n",
                    dev->name.c_str(), dev->socket_path.c_str());
}
```

### Non-Blocking Accept Pattern

**File**: qemu-binary/QEMUBinaryComponent.cc:658-687

```cpp
void QEMUBinaryComponent::acceptDeviceConnection(DeviceInfo* dev) {
    if (dev->socket_ready) return;  // Already connected

    // Non-blocking accept: returns immediately if no connection pending
    int client_fd = accept(dev->server_fd, NULL, NULL);

    if (client_fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No connection pending - this is normal, simulation continues
            return;
        }
        // Other errors are logged but non-fatal
        output_.output("Warning: accept() failed for %s: %s\n",
                       dev->name.c_str(), strerror(errno));
        return;
    }

    // Connection established! Set client socket to non-blocking
    int flags = fcntl(client_fd, F_GETFL, 0);
    fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

    dev->client_fd = client_fd;
    dev->socket_ready = true;

    output_.verbose(CALL_INFO, 1, 0, "Device %s connected\n", dev->name.c_str());
}
```

**Why Non-Blocking I/O is Critical**:
- SST simulation must continue even if QEMU hasn't connected yet
- Devices can connect at different times asynchronously
- Avoids deadlock if QEMU crashes before connecting
- Allows graceful degradation and reconnection

### Clock Tick Polling Pattern

**File**: qemu-binary/QEMUBinaryComponent.cc:198-219

```cpp
bool QEMUBinaryComponent::clockTick(SST::Cycle_t cycle) {
    if (use_multi_device_) {
        pollDeviceSockets();  // N-device mode
    } else {
        // Legacy single-device mode (backward compatible)
        if (socket_ready_ && client_fd_ >= 0) {
            handleMMIORequest();
        }
    }

    return false;  // Never unregister clock
}

void QEMUBinaryComponent::pollDeviceSockets() {
    for (size_t i = 0; i < devices_.size(); i++) {
        DeviceInfo& dev = devices_[i];

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

**Performance Characteristics**:
- Time complexity: O(N) per clock tick where N = number of devices
- Space complexity: O(N) for device array
- Optimization opportunity: Use `select()` or `epoll()` for large N (>16)

### QEMU Launch with Environment Variables

**File**: qemu-binary/QEMUBinaryComponent.cc:344-373

Instead of command-line `-device` arguments (which don't work for SysBusDevice), the component sets environment variables that QEMU's virt machine reads:

```cpp
void QEMUBinaryComponent::launchQEMU() {
    // ... (create and fork child process)

    if (qemu_pid_ == 0) {
        // Child process: configure environment and exec QEMU

        if (use_multi_device_) {
            // Configure N devices via environment variables
            char num_buf[16];
            snprintf(num_buf, sizeof(num_buf), "%zu", devices_.size());
            setenv("SST_NUM_DEVICES", num_buf, 1);

            for (size_t i = 0; i < devices_.size(); i++) {
                char env_socket[32], env_base[32], addr_buf[32];

                // SST_DEVICE0_SOCKET=/tmp/qemu-sst-device0.sock
                snprintf(env_socket, sizeof(env_socket), "SST_DEVICE%zu_SOCKET", i);
                setenv(env_socket, devices_[i].socket_path.c_str(), 1);

                // SST_DEVICE0_BASE=10200000 (hex without 0x prefix)
                snprintf(env_base, sizeof(env_base), "SST_DEVICE%zu_BASE", i);
                snprintf(addr_buf, sizeof(addr_buf), "%lx", devices_[i].base_addr);
                setenv(env_base, addr_buf, 1);
            }
        }

        // Execute QEMU (environment variables inherited)
        std::vector<const char*> args;
        args.push_back(qemu_path_.c_str());
        args.push_back("-M");
        args.push_back("virt");
        args.push_back("-nographic");
        args.push_back("-kernel");
        args.push_back(binary_path_.c_str());
        args.push_back(nullptr);

        execv(qemu_path_.c_str(), (char* const*)args.data());

        // If exec fails
        fprintf(stderr, "Failed to launch QEMU: %s\n", strerror(errno));
        exit(1);
    }

    // Parent process continues...
}
```

**QEMU virt.c reads these environment variables** (hw/riscv/virt.c:948-989):
```c
const char *num_devices_str = getenv("SST_NUM_DEVICES");
int num_sst_devices = num_devices_str ? atoi(num_devices_str) : 1;

for (int dev_idx = 0; dev_idx < num_sst_devices; dev_idx++) {
    char env_socket[64], env_base[64];
    snprintf(env_socket, sizeof(env_socket), "SST_DEVICE%d_SOCKET", dev_idx);
    snprintf(env_base, sizeof(env_base), "SST_DEVICE%d_BASE", dev_idx);

    const char *socket_path = getenv(env_socket);
    const char *base_str = getenv(env_base);

    // Create and map device at specified address
    DeviceState *sst_dev = qdev_new("sst-device");
    qdev_prop_set_string(sst_dev, "socket", socket_path);
    qdev_prop_set_uint64(sst_dev, "base_address", base_addr);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(sst_dev), &error_fatal);
    sysbus_mmio_map(SYS_BUS_DEVICE(sst_dev), 0, base_addr);
}
```

---

## SST Python Configuration Customization

### Configuration File Architecture

**Critical Understanding**: SST Python scripts are NOT just configuration files - they are **full Python programs** that execute to build the simulation graph.

```python
import sst

# This is Python code executing at runtime!
# You have full Python language features:
# - Loops, conditionals, functions
# - External libraries (json, yaml, numpy, etc.)
# - File I/O, command-line arguments
# - Dynamic component creation
```

### Pattern 1: Loop-Based Device Creation

**Example**: Creating 8 identical devices with parameterized addresses

```python
import sst

NUM_DEVICES = 8
BASE_ADDR = 0x10200000
ADDR_STRIDE = 0x100000  # 1MB address space per device

# Create QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
params = {
    "clock": "1GHz",
    "binary_path": "/path/to/program.elf",
    "qemu_path": "/home/user/local/qemu-sst/bin/qemu-system-riscv32",
    "num_devices": NUM_DEVICES
}

# Dynamically generate device parameters
for i in range(NUM_DEVICES):
    addr = BASE_ADDR + (i * ADDR_STRIDE)
    params[f"device{i}_base"] = f"0x{addr:x}"
    params[f"device{i}_size"] = 4096
    params[f"device{i}_name"] = f"accel_{i}"

qemu.addParams(params)

# Create device components
devices = []
for i in range(NUM_DEVICES):
    dev = sst.Component(f"accel_{i}", "acalsim.ComputeDevice")
    dev.addParams({
        "clock": "1GHz",
        "latency": 10 + (i * 5),  # Variable latency per device
        "compute_units": 16
    })
    devices.append(dev)

    # Create link
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )

sst.setProgramOption("stop-at", "10ms")
```

### Pattern 2: Heterogeneous Device Mix

**Example**: Different device types with varying characteristics

```python
import sst

# Device type database
DEVICE_CONFIGS = [
    {
        "type": "acalsim.EchoDevice",
        "name": "echo",
        "base": 0x10200000,
        "params": {"latency": 10, "buffer_size": 256}
    },
    {
        "type": "acalsim.ComputeDevice",
        "name": "compute",
        "base": 0x10300000,
        "params": {"latency": 100, "compute_units": 32}
    },
    {
        "type": "acalsim.MemoryDevice",
        "name": "memory",
        "base": 0x10400000,
        "params": {"latency": 50, "size_kb": 1024}
    },
    {
        "type": "acalsim.NPUDevice",
        "name": "npu",
        "base": 0x10500000,
        "params": {"latency": 200, "mac_units": 256}
    }
]

# Create QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
params = {
    "clock": "2GHz",
    "binary_path": "/path/to/heterogeneous_test.elf",
    "qemu_path": "/home/user/local/qemu-sst/bin/qemu-system-riscv32",
    "num_devices": len(DEVICE_CONFIGS)
}

# Configure devices from database
for i, config in enumerate(DEVICE_CONFIGS):
    # QEMU device parameters
    params[f"device{i}_base"] = f"0x{config['base']:x}"
    params[f"device{i}_size"] = 4096
    params[f"device{i}_name"] = config["name"]

    # Create SST device component
    dev = sst.Component(config["name"], config["type"])
    dev_params = {"clock": "2GHz"}
    dev_params.update(config["params"])
    dev.addParams(dev_params)

    # Connect to QEMU
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )

qemu.addParams(params)
sst.setProgramOption("stop-at", "100ms")
```

### Pattern 3: Configuration from External File

**Example**: Read device configuration from JSON file

**devices.json**:
```json
{
  "devices": [
    {"type": "acalsim.GPU", "name": "gpu0", "base": "0x20000000", "cores": 128},
    {"type": "acalsim.NPU", "name": "npu0", "base": "0x30000000", "macs": 512},
    {"type": "acalsim.DSP", "name": "dsp0", "base": "0x40000000", "simd": 16}
  ]
}
```

**config.py**:
```python
import sst
import json
import os

# Read configuration from JSON
config_file = os.getenv("DEVICE_CONFIG", "devices.json")
with open(config_file, 'r') as f:
    config = json.load(f)

devices_list = config["devices"]

# Create QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
params = {
    "clock": "1GHz",
    "binary_path": os.getenv("BINARY_PATH", "test.elf"),
    "qemu_path": os.getenv("QEMU_PATH", "qemu-system-riscv32"),
    "num_devices": len(devices_list)
}

# Build from JSON configuration
for i, dev_cfg in enumerate(devices_list):
    # QEMU parameters
    params[f"device{i}_base"] = dev_cfg["base"]
    params[f"device{i}_size"] = 4096
    params[f"device{i}_name"] = dev_cfg["name"]

    # Create component
    dev = sst.Component(dev_cfg["name"], dev_cfg["type"])
    dev.addParams({
        "clock": "1GHz",
        **{k: v for k, v in dev_cfg.items() if k not in ["type", "name", "base"]}
    })

    # Link
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )

qemu.addParams(params)
```

**Usage**:
```bash
DEVICE_CONFIG=gpus.json BINARY_PATH=gpu_test.elf sst config.py
```

### Pattern 4: Multi-Server Topology

**Example**: Deploy QEMU on rank 0, devices distributed across ranks

```python
import sst

rank = sst.getMPIRank()
num_ranks = sst.getNumRanks()

if rank == 0:
    # Rank 0: QEMU + local devices
    qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
    qemu.addParams({
        "clock": "1GHz",
        "binary_path": "distributed_test.elf",
        "qemu_path": "qemu-system-riscv32",
        "num_devices": 4
    })

    # Local devices (rank 0) - low latency
    for i in range(2):
        dev = sst.Component(f"local_dev_{i}", "acalsim.FastDevice")
        dev.addParams({"clock": "2GHz", "latency": 10})

        link = sst.Link(f"link_{i}")
        link.connect(
            (qemu, f"device_port_{i}", "1ns"),     # Local: 1ns latency
            (dev, "cpu_port", "1ns")
        )

    # Remote devices (rank 1) - higher latency due to network
    for i in range(2, 4):
        dev = sst.Component(f"remote_dev_{i}", "acalsim.SlowDevice")
        dev.addParams({"clock": "1GHz", "latency": 100})

        link = sst.Link(f"link_{i}")
        link.connect(
            (qemu, f"device_port_{i}", "100ns"),   # Network: 100ns latency
            (dev, "cpu_port", "100ns")
        )

elif rank == 1:
    # Rank 1: Only remote devices (created by rank 0 link)
    for i in range(2, 4):
        dev = sst.Component(f"remote_dev_{i}", "acalsim.SlowDevice")
        dev.addParams({"clock": "1GHz", "latency": 100})

        link = sst.Link(f"link_{i}")
        link.connect(
            (dev, "cpu_port", "100ns")
        )

sst.setProgramOption("stop-at", "1s")
```

**Launch**:
```bash
# Create hostfile
cat > hostfile <<EOF
192.168.100.178 slots=1
192.168.100.69 slots=1
EOF

# Run distributed simulation
mpirun -np 2 --hostfile hostfile sst distributed_config.py
```

### Pattern 5: Advanced Memory Hierarchy Integration

**Example**: Connect QEMU to cache hierarchy instead of direct device access

```python
import sst

# QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "2GHz",
    "binary_path": "cache_test.elf",
    "num_devices": 2
})

# L1 Cache
l1_cache = sst.Component("l1_cache", "memHierarchy.Cache")
l1_cache.addParams({
    "cache_frequency": "2GHz",
    "cache_size": "32KiB",
    "associativity": 4,
    "access_latency_cycles": 1,
    "cache_line_size": 64,
    "replacement_policy": "lru"
})

# L2 Cache
l2_cache = sst.Component("l2_cache", "memHierarchy.Cache")
l2_cache.addParams({
    "cache_frequency": "2GHz",
    "cache_size": "256KiB",
    "associativity": 8,
    "access_latency_cycles": 10,
    "cache_line_size": 64,
    "replacement_policy": "lru",
    "mshr_num_entries": 8
})

# Memory Controller
memory = sst.Component("memory", "memHierarchy.MemController")
memory.addParams({
    "clock": "1GHz",
    "backing": "none",
    "backend.mem_size": "512MiB",
    "backend.access_time": "100ns"
})

# Connect cache hierarchy
link_l1_l2 = sst.Link("link_l1_l2")
link_l1_l2.connect(
    (l1_cache, "low_network_0", "1ns"),
    (l2_cache, "high_network_0", "1ns")
)

link_l2_mem = sst.Link("link_l2_mem")
link_l2_mem.connect(
    (l2_cache, "low_network_0", "10ns"),
    (memory, "direct_link", "10ns")
)

# Connect QEMU to L1 instead of direct device access
# (Requires QEMUBinaryComponent to have cache_link port)
link_qemu_l1 = sst.Link("link_qemu_l1")
link_qemu_l1.connect(
    (qemu, "cache_link", "1ns"),
    (l1_cache, "high_network_0", "1ns")
)

# Devices still connected normally
for i in range(2):
    dev = sst.Component(f"device_{i}", "acalsim.GenericDevice")
    dev.addParams({"clock": "1GHz"})

    link = sst.Link(f"link_dev_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )
```

---

## Device Development Guide

### Creating a Custom SST Device Component

**Step 1: Header File** (`MyDevice.hh`)

```cpp
#ifndef _MY_DEVICE_HH
#define _MY_DEVICE_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/event.h>
#include <queue>

namespace AcalSim {

class MyDevice : public SST::Component {
public:
    MyDevice(SST::ComponentId_t id, SST::Params& params);
    ~MyDevice();

    void setup() override;
    void finish() override;

    bool clockTick(SST::Cycle_t cycle);
    void handleCPURequest(SST::Event* event);

    // SST ELI macros
    SST_ELI_REGISTER_COMPONENT(
        MyDevice,
        "acalsim",
        "MyDevice",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Custom accelerator device",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"latency", "Operation latency in cycles", "10"},
        {"buffer_size", "Internal buffer size", "256"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port to QEMU CPU", {}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"read_requests", "Number of read requests", "requests", 1},
        {"write_requests", "Number of write requests", "requests", 1},
        {"total_latency", "Total processing latency", "cycles", 1}
    )

private:
    struct PendingRequest {
        DeviceEvent::Type type;
        uint64_t address;
        uint64_t data;
    };

    uint32_t latency_;
    uint32_t buffer_size_;

    SST::Link* cpu_link_;
    std::queue<PendingRequest> pending_requests_;
    uint64_t cycles_until_ready_;

    // Statistics
    SST::Statistic<uint64_t>* stat_reads_;
    SST::Statistic<uint64_t>* stat_writes_;
    SST::Statistic<uint64_t>* stat_latency_;

    SST::Output output_;
    SST::TimeConverter* clock_tc_;
};

} // namespace AcalSim

#endif
```

**Step 2: Implementation** (`MyDevice.cc`)

```cpp
#include "MyDevice.hh"
#include "DeviceEvent.hh"

using namespace AcalSim;

MyDevice::MyDevice(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
{
    output_.init("MyDevice[@p:@l]: ", 1, 0, SST::Output::STDOUT);

    latency_ = params.find<uint32_t>("latency", 10);
    buffer_size_ = params.find<uint32_t>("buffer_size", 256);

    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    clock_tc_ = registerClock(
        clock_freq,
        new SST::Clock::Handler<MyDevice>(this, &MyDevice::clockTick)
    );

    cpu_link_ = configureLink(
        "cpu_port",
        new SST::Event::Handler<MyDevice>(this, &MyDevice::handleCPURequest)
    );

    if (!cpu_link_) {
        output_.fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
    }

    stat_reads_ = registerStatistic<uint64_t>("read_requests");
    stat_writes_ = registerStatistic<uint64_t>("write_requests");
    stat_latency_ = registerStatistic<uint64_t>("total_latency");

    cycles_until_ready_ = 0;

    output_.verbose(CALL_INFO, 1, 0, "Initialized with latency=%u\n", latency_);
}

MyDevice::~MyDevice() {}

void MyDevice::setup() {
    output_.verbose(CALL_INFO, 1, 0, "Setup complete\n");
}

void MyDevice::finish() {
    output_.verbose(CALL_INFO, 1, 0, "Finishing\n");
}

bool MyDevice::clockTick(SST::Cycle_t cycle) {
    if (cycles_until_ready_ > 0) {
        cycles_until_ready_--;
        return false;
    }

    if (!pending_requests_.empty()) {
        PendingRequest req = pending_requests_.front();
        pending_requests_.pop();

        cycles_until_ready_ = latency_;

        // Send response
        DeviceEvent* resp = new DeviceEvent(req.type, req.address, req.data);
        cpu_link_->send(resp);

        stat_latency_->addData(latency_);
    }

    return false;
}

void MyDevice::handleCPURequest(SST::Event* event) {
    DeviceEvent* dev_event = dynamic_cast<DeviceEvent*>(event);

    if (!dev_event) {
        output_.fatal(CALL_INFO, -1, "Invalid event type\n");
    }

    if (dev_event->getType() == DeviceEvent::READ) {
        stat_reads_->addData(1);
    } else {
        stat_writes_->addData(1);
    }

    PendingRequest req;
    req.type = dev_event->getType();
    req.address = dev_event->getAddress();
    req.data = dev_event->getData();

    pending_requests_.push(req);

    delete event;
}
```

**Step 3: Makefile**

```makefile
CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS)
CXXFLAGS += -std=c++14 -g -O2
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

SOURCES = MyDevice.cc DeviceEvent.cc
HEADERS = MyDevice.hh DeviceEvent.hh
TARGET = libmydevice.so

all: $(TARGET)

$(TARGET): $(SOURCES:.cc=.o)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: $(TARGET)
	install -D $(TARGET) $(shell sst-config --prefix)/lib/sstcore/$(TARGET)

clean:
	rm -f *.o $(TARGET)

.PHONY: all install clean
```

---

## QEMU Integration Details

### sst-device.c Implementation

**File**: qemu-sst-device/sst-device.c

Key functions:

```c
// Device state
typedef struct {
    SysBusDevice parent_obj;
    MemoryRegion mmio;
    int socket_fd;
    char *socket_path;
    uint64_t base_address;  // Global base address
} SSTDeviceState;

// MMIO read operation
static uint64_t sst_device_read(void *opaque, hwaddr offset, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    MMIORequest req;
    req.type = 0;  // READ
    req.addr = s->base_address + offset;  // Send GLOBAL address
    req.data = 0;

    send(s->socket_fd, &req, sizeof(req), 0);

    MMIOResponse resp;
    recv(s->socket_fd, &resp, sizeof(resp), MSG_WAITALL);

    return resp.success ? resp.data : 0;
}

// MMIO write operation
static void sst_device_write(void *opaque, hwaddr offset,
                             uint64_t value, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    MMIORequest req;
    req.type = 1;  // WRITE
    req.addr = s->base_address + offset;  // Send GLOBAL address
    req.data = value;

    send(s->socket_fd, &req, sizeof(req), 0);

    MMIOResponse resp;
    recv(s->socket_fd, &resp, sizeof(resp), MSG_WAITALL);
}

// Property definitions
static Property sst_device_properties[] = {
    DEFINE_PROP_STRING("socket", SSTDeviceState, socket_path),
    DEFINE_PROP_UINT64("base_address", SSTDeviceState, base_address, 0),
    DEFINE_PROP_END_OF_LIST(),
};
```

**Why Global Addresses**:
- Each sst-device knows its global base address (set by virt machine)
- Sends `base_address + offset` to SST
- SST routes by global address to correct device
- Enables true address-based routing

### virt.c Integration (N-Device Support)

**File**: hw/riscv/virt.c (lines 948-989)

```c
/* SST integration device(s) - support N devices via environment variables */
const char *num_devices_str = getenv("SST_NUM_DEVICES");
int num_sst_devices = num_devices_str ? atoi(num_devices_str) : 1;

if (num_sst_devices < 1) num_sst_devices = 1;
if (num_sst_devices > 16) num_sst_devices = 16;

for (int dev_idx = 0; dev_idx < num_sst_devices; dev_idx++) {
    char env_socket[64], env_base[64];
    snprintf(env_socket, sizeof(env_socket), "SST_DEVICE%d_SOCKET", dev_idx);
    snprintf(env_base, sizeof(env_base), "SST_DEVICE%d_BASE", dev_idx);

    const char *socket_path = getenv(env_socket);
    const char *base_str = getenv(env_base);

    // Defaults
    char default_socket[128];
    if (!socket_path) {
        snprintf(default_socket, sizeof(default_socket),
                 "/tmp/qemu-sst-device%d.sock", dev_idx);
        socket_path = default_socket;
    }

    uint64_t base_addr = base_str ? strtoul(base_str, NULL, 0) :
                         (0x10200000 + dev_idx * 0x100000);

    // Create device
    DeviceState *sst_dev = qdev_new("sst-device");
    qdev_prop_set_string(sst_dev, "socket", socket_path);
    qdev_prop_set_uint64(sst_dev, "base_address", base_addr);

    // Realize and map
    sysbus_realize_and_unref(SYS_BUS_DEVICE(sst_dev), &error_fatal);
    sysbus_mmio_map(SYS_BUS_DEVICE(sst_dev), 0, base_addr);

    printf("SST Device %d: socket=%s, base=0x%lx\n",
           dev_idx, socket_path, base_addr);
}
```

---

## Debugging and Profiling

### Debugging SST Components

**Enable Verbose Output**:
```python
qemu.addParams({
    "verbose": 3,  # 0=off, 1=info, 2=debug, 3=trace
})
```

**GDB Debugging**:
```bash
gdb --args sst test_config.py

(gdb) break QEMUBinaryComponent::handleMMIORequest
(gdb) run
(gdb) print dev->name
(gdb) print req.addr
```

**Attach to QEMU**:
```bash
ps aux | grep qemu-system-riscv32
gdb -p <PID>
(gdb) break sst_device_read
```

### Socket Debugging

**Monitor Socket Traffic**:
```bash
strace -e trace=socket,connect,accept,send,recv sst config.py 2>&1 | grep qemu-sst

ls -la /tmp/qemu-sst-device*.sock

# Test socket manually
echo -ne '\x00\x00\x00\x10\x20\x00\x00\x00\x00\x00\x00\x00' | nc -U /tmp/qemu-sst-device0.sock | hexdump -C
```

### Performance Profiling

**SST Statistics**:
```python
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV")
sst.setStatisticOutputOptions({"filepath": "stats.csv"})

qemu.enableAllStatistics()
device.enableAllStatistics()
```

**Perf Profiling**:
```bash
perf record -g sst test_config.py
perf report

perf stat -e cache-references,cache-misses sst config.py
```

---

## Performance Optimization

### Socket Performance (Large N)

**Current**: O(N) polling per clock tick

**Optimization**: Use `select()` for N > 16

```cpp
void QEMUBinaryComponent::pollDeviceSockets() {
    if (devices_.size() > 16) {
        fd_set readfds;
        FD_ZERO(&readfds);
        int max_fd = 0;

        for (auto& dev : devices_) {
            if (dev.socket_ready && dev.client_fd >= 0) {
                FD_SET(dev.client_fd, &readfds);
                max_fd = std::max(max_fd, dev.client_fd);
            }
        }

        struct timeval timeout = {0, 0};
        int ready = select(max_fd + 1, &readfds, NULL, NULL, &timeout);

        if (ready > 0) {
            for (auto& dev : devices_) {
                if (FD_ISSET(dev.client_fd, &readfds)) {
                    handleMMIORequest(&dev);
                }
            }
        }
    } else {
        // Original O(N) loop
        for (auto& dev : devices_) {
            if (dev.socket_ready) handleMMIORequest(&dev);
        }
    }
}
```

### Hash Map Routing (O(1) Lookup)

```cpp
// In header
std::unordered_map<uint64_t, DeviceInfo*> addr_to_device_;

// In constructor
for (auto& dev : devices_) {
    addr_to_device_[dev.base_addr] = &dev;
}

// O(1) lookup
DeviceInfo* QEMUBinaryComponent::findDeviceByAddress(uint64_t addr) {
    uint64_t base = (addr / 0x100000) * 0x100000;  // Round to 1MB
    auto it = addr_to_device_.find(base);
    return (it != addr_to_device_.end()) ? it->second : nullptr;
}
```

---

## Advanced Topics

### Checkpoint/Restart Support

```cpp
void QEMUBinaryComponent::serialize_order(
    SST::Core::Serialization::serializer& ser) override
{
    Component::serialize_order(ser);
    ser& devices_;
    ser& qemu_pid_;
    ser& use_multi_device_;

    for (auto& dev : devices_) {
        ser& dev.num_requests;
        ser& dev.socket_ready;
    }
}
```

### Multi-Threaded QEMU

```cpp
// Add mutex to DeviceInfo
std::mutex socket_mutex;

void QEMUBinaryComponent::handleMMIORequest(DeviceInfo* dev) {
    std::lock_guard<std::mutex> lock(dev->socket_mutex);
    // ... socket I/O
}
```

---

## Conclusion

This developer guide provides the foundation for understanding, extending, and optimizing the QEMU-AcalSim-SST baremetal simulation infrastructure.

**Key Takeaways**:
1. Three-layer architecture enables flexible system simulation
2. N-device support scales linearly with proper optimization
3. SST Python configs are full programs, not static files
4. Per-device sockets provide independent communication
5. Environment variables enable clean QEMU integration

**Next Steps**:
- Review concrete examples in `DEMO_EXAMPLE.md`
- Study existing device implementations
- Experiment with custom SST configurations
- Profile and optimize for your specific use case

**Resources**:
- SST Documentation: https://sst-simulator.org/
- QEMU Documentation: https://www.qemu.org/docs/
- RISC-V Specifications: https://riscv.org/specifications/

---

**Document Version**: 2.0 (Updated for N-Device Support)
**Last Updated**: 2025-11-10
**Phase**: 2D - N-Device Integration Complete
**Status**: Production Ready
