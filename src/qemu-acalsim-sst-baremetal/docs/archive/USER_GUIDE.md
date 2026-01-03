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

# QEMU-ACALSim-SST User Guide

This guide helps you create custom firmware tests and device models using the QEMU-ACALSim-SST framework.

## Table of Contents

1. [Creating Custom Firmware Tests](#creating-custom-firmware-tests)
2. [Creating Custom SST Device Models](#creating-custom-sst-device-models)
3. [Building and Running Tests](#building-and-running-tests)
4. [Debugging Your Code](#debugging-your-code)

---

## Creating Custom Firmware Tests

### Prerequisites

- RISC-V toolchain installed (`riscv64-unknown-elf-gcc`)
- Basic C and RISC-V assembly knowledge
- Understanding of memory-mapped I/O

### Step 1: Write Your Test Program

Create a new C file in `riscv-programs/`:

```c
/*
 * my_custom_test.c - Custom test for my device
 */

#include <stdint.h>

// Your device MMIO base address
#define MY_DEVICE_BASE  0x10300000

// Register definitions
#define MY_REG_CONTROL  (*(volatile uint32_t *)(MY_DEVICE_BASE + 0x00))
#define MY_REG_STATUS   (*(volatile uint32_t *)(MY_DEVICE_BASE + 0x04))
#define MY_REG_DATA     (*(volatile uint32_t *)(MY_DEVICE_BASE + 0x08))

// UART for debug output
#define UART_BASE 0x10000000
#define UART_TX   (*(volatile uint8_t *)UART_BASE)

void uart_putc(char c) {
    UART_TX = c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

// Trap handler required by crt0.S
void trap_handler_c(uint32_t mcause, uint32_t mepc, uint32_t mtval) {
    uart_puts("\n[TRAP] Exception!\n");
    while (1) asm volatile("wfi");
}

// Main function called by crt0.S
int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    uart_puts("My Custom Test\n");

    // Initialize device
    MY_REG_CONTROL = 0x1;  // Start

    // Wait for completion
    while (MY_REG_STATUS & 0x1);  // BUSY bit

    // Read result
    uint32_t result = MY_REG_DATA;

    uart_puts(result == 0xDEADBEEF ? "[PASS]\n" : "[FAIL]\n");

    return 0;
}
```

### Step 2: Add Build Rules

Edit `riscv-programs/Makefile`:

```makefile
# Add your test to the all target
all: ... my_custom_test.elf my_custom_test.bin my_custom_test.dump

# Add build rules
my_custom_test.elf: crt0.o my_custom_test.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) crt0.o my_custom_test.o -o $@

my_custom_test.bin: my_custom_test.elf
	$(OBJCOPY) -O binary $< $@

my_custom_test.dump: my_custom_test.elf
	$(OBJDUMP) -d $< > $@

my_custom_test.o: my_custom_test.c
	$(CC) $(CFLAGS) -c $< -o $@

# Add test target
test-my-custom: my_custom_test.elf
	@echo "Running my custom test..."
	qemu-system-riscv32 -M virt -bios none -nographic -kernel my_custom_test.elf

.PHONY: test-my-custom
```

### Step 3: Build and Test

```bash
cd riscv-programs

# Build your test
make my_custom_test.elf

# Test standalone (without SST device - will trap at MMIO access)
make test-my-custom

# Or test with SST simulation (requires SST setup)
# Update qemu_binary_test.py to use your test binary
```

### Advanced: Adding Assembly Functions

If you need custom assembly routines:

```assembly
# my_asm_functions.S

.section .text

.global my_optimized_copy
.type my_optimized_copy, @function
my_optimized_copy:
    # void my_optimized_copy(uint32_t *dest, const uint32_t *src, int count)
    # a0 = dest, a1 = src, a2 = count

    beqz a2, done
loop:
    lw t0, 0(a1)
    sw t0, 0(a0)
    addi a0, a0, 4
    addi a1, a1, 4
    addi a2, a2, -1
    bnez a2, loop
done:
    ret
```

Call from C:

```c
extern void my_optimized_copy(uint32_t *dest, const uint32_t *src, int count);

// In main()
uint32_t src[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
uint32_t dest[10];
my_optimized_copy(dest, src, 10);
```

Update Makefile:

```makefile
my_custom_test.elf: crt0.o my_custom_test.o my_asm_functions.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) crt0.o my_custom_test.o my_asm_functions.o -o $@

my_asm_functions.o: my_asm_functions.S
	$(CC) $(CFLAGS) -c $< -o $@
```

---

## Creating Custom SST Device Models

### Step 1: Create Device Component Header

Create `my-device/MyDeviceComponent.hh`:

```cpp
#ifndef MY_DEVICE_COMPONENT_HH
#define MY_DEVICE_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/event.h>

namespace SST {
namespace MyDevice {

class MyDeviceComponent : public SST::Component {
public:
    // Constructor
    MyDeviceComponent(SST::ComponentId_t id, SST::Params& params);

    // Destructor
    ~MyDeviceComponent();

    // Setup and finish
    void setup() override;
    void finish() override;

    // Clock handler
    bool clockTick(SST::Cycle_t cycle);

    // Event handlers
    void handleCPURequest(SST::Event *event);

    // SST Registration
    SST_ELI_REGISTER_COMPONENT(
        MyDeviceComponent,
        "mydevice",
        "MyDevice",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Custom device for demonstration",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"base_addr", "Base address of device", "0x10300000"},
        {"verbose", "Verbosity level (0-3)", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port to CPU/QEMU", {"MemoryTransactionEvent"}},
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"total_reads", "Total read operations", "count", 1},
        {"total_writes", "Total write operations", "count", 1}
    )

private:
    // Configuration
    uint64_t baseAddr;
    int verbose;

    // Links
    SST::Link *cpuLink;

    // Statistics
    Statistic<uint64_t>* statReads;
    Statistic<uint64_t>* statWrites;

    // Device state
    uint32_t controlReg;
    uint32_t statusReg;
    uint32_t dataReg;

    // Helper methods
    void handleRead(uint64_t addr, SST::Event *event);
    void handleWrite(uint64_t addr, uint32_t data, SST::Event *event);
};

} // namespace MyDevice
} // namespace SST

#endif
```

### Step 2: Implement Device Component

Create `my-device/MyDeviceComponent.cc`:

```cpp
#include "MyDeviceComponent.hh"
#include <sst/core/event.h>

using namespace SST;
using namespace SST::MyDevice;

MyDeviceComponent::MyDeviceComponent(ComponentId_t id, Params& params) :
    Component(id)
{
    // Get parameters
    std::string clock = params.find<std::string>("clock", "1GHz");
    baseAddr = params.find<uint64_t>("base_addr", 0x10300000);
    verbose = params.find<int>("verbose", 0);

    // Configure link to CPU
    cpuLink = configureLink("cpu_port",
        new Event::Handler<MyDeviceComponent>(this, &MyDeviceComponent::handleCPURequest));

    if (!cpuLink) {
        getSimulationOutput().fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
    }

    // Register clock
    registerClock(clock,
        new Clock::Handler<MyDeviceComponent>(this, &MyDeviceComponent::clockTick));

    // Initialize statistics
    statReads = registerStatistic<uint64_t>("total_reads");
    statWrites = registerStatistic<uint64_t>("total_writes");

    // Initialize device state
    controlReg = 0;
    statusReg = 0;
    dataReg = 0;

    if (verbose > 0) {
        getSimulationOutput().output("MyDevice: Initialized at 0x%lx\n", baseAddr);
    }
}

MyDeviceComponent::~MyDeviceComponent() {
}

void MyDeviceComponent::setup() {
    if (verbose > 0) {
        getSimulationOutput().output("MyDevice: Setup complete\n");
    }
}

void MyDeviceComponent::finish() {
    if (verbose > 0) {
        getSimulationOutput().output("MyDevice: Simulation finished\n");
        getSimulationOutput().output("  Total reads: %lu\n", statReads->getCollectionCount());
        getSimulationOutput().output("  Total writes: %lu\n", statWrites->getCollectionCount());
    }
}

bool MyDeviceComponent::clockTick(Cycle_t cycle) {
    // Perform periodic device operations

    // Check if operation is complete
    if (controlReg & 0x1) {  // START bit set
        // Simulate processing
        dataReg = 0xDEADBEEF;  // Echo pattern
        statusReg &= ~0x1;     // Clear BUSY
        controlReg &= ~0x1;    // Clear START

        if (verbose > 1) {
            getSimulationOutput().output("MyDevice[%lu]: Operation complete\n", cycle);
        }
    }

    return false;  // Keep component active
}

void MyDeviceComponent::handleCPURequest(Event *event) {
    // Handle memory transaction from CPU/QEMU
    // You'll need to define MemoryTransactionEvent based on your protocol

    // Example:
    // MemoryTransactionEvent *req = dynamic_cast<MemoryTransactionEvent*>(event);
    // if (req->isRead()) {
    //     handleRead(req->getAddr(), event);
    // } else {
    //     handleWrite(req->getAddr(), req->getData(), event);
    // }

    delete event;
}

void MyDeviceComponent::handleRead(uint64_t addr, Event *event) {
    uint64_t offset = addr - baseAddr;
    uint32_t data = 0;

    switch (offset) {
        case 0x00:  // Control register
            data = controlReg;
            break;
        case 0x04:  // Status register
            data = statusReg;
            break;
        case 0x08:  // Data register
            data = dataReg;
            break;
        default:
            if (verbose > 0) {
                getSimulationOutput().output("MyDevice: Read from unknown offset 0x%lx\n", offset);
            }
            break;
    }

    statReads->addData(1);

    if (verbose > 1) {
        getSimulationOutput().output("MyDevice: Read 0x%x from offset 0x%lx\n", data, offset);
    }

    // Send response back to CPU
    // (Implementation depends on your event protocol)
}

void MyDeviceComponent::handleWrite(uint64_t addr, uint32_t data, Event *event) {
    uint64_t offset = addr - baseAddr;

    switch (offset) {
        case 0x00:  // Control register
            controlReg = data;
            if (data & 0x1) {  // START bit
                statusReg |= 0x1;  // Set BUSY
            }
            break;
        case 0x08:  // Data register
            dataReg = data;
            break;
        default:
            if (verbose > 0) {
                getSimulationOutput().output("MyDevice: Write to unknown offset 0x%lx\n", offset);
            }
            break;
    }

    statWrites->addData(1);

    if (verbose > 1) {
        getSimulationOutput().output("MyDevice: Write 0x%x to offset 0x%lx\n", data, offset);
    }
}
```

### Step 3: Create Makefile

Create `my-device/Makefile`:

```makefile
CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS) -std=c++17 -Wall -Wextra
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

SST_INSTALL_DIR = $(shell sst-config --prefix)
SST_ELEMENT_DIR = $(SST_INSTALL_DIR)/lib/sstcore

COMPONENT_NAME = libmydevice.so
SOURCES = MyDeviceComponent.cc
OBJECTS = $(SOURCES:.cc=.o)
HEADERS = MyDeviceComponent.hh

all: $(COMPONENT_NAME)

$(COMPONENT_NAME): $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

install: $(COMPONENT_NAME)
	@mkdir -p $(SST_ELEMENT_DIR)
	@cp $(COMPONENT_NAME) $(SST_ELEMENT_DIR)/
	@echo "Installed to $(SST_ELEMENT_DIR)"

clean:
	rm -f $(OBJECTS) $(COMPONENT_NAME)

.PHONY: all install clean
```

### Step 4: Create SST Configuration

Create `my_test.py`:

```python
import sst

# QEMU component
qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "binary_path": "/path/to/my_custom_test.elf",
    "qemu_path": "/path/to/custom/qemu-system-riscv32",
    "socket_path": "/tmp/qemu-sst-mmio.sock",
    "device_base": "0x10300000",
})

# Your custom device
mydev = sst.Component("mydev0", "mydevice.MyDevice")
mydev.addParams({
    "clock": "1GHz",
    "base_addr": "0x10300000",
    "verbose": "2",
})

# Connect them
link = sst.Link("qemu_mydev_link")
link.connect(
    (qemu, "device_port", "1ns"),
    (mydev, "cpu_port", "1ns")
)

# Set simulation parameters
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", "1ms")
```

### Step 5: Build and Test

```bash
# Build device
cd my-device
make
make install

# Verify installation
sst-info mydevice

# Run simulation
cd ..
sst my_test.py
```

---

## Building and Running Tests

### Standard Build Process

```bash
# 1. Build firmware
cd riscv-programs
make clean
make my_custom_test.elf

# 2. Build SST component
cd ../my-device
make clean
make
make install

# 3. Run simulation
cd ..
sst my_test.py
```

### Incremental Builds

```bash
# Rebuild only firmware
cd riscv-programs
make my_custom_test.elf

# Rebuild only device
cd ../my-device
make && make install
```

---

## Debugging Your Code

### Firmware Debugging

**Method 1: UART Debug Output**

```c
// Add debug prints
void debug_print_hex(uint32_t val) {
    char buf[9];
    for (int i = 7; i >= 0; i--) {
        int nibble = (val >> (i * 4)) & 0xF;
        buf[7-i] = nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
    }
    buf[8] = '\0';
    uart_puts(buf);
}

// Usage
debug_print_hex(MY_REG_STATUS);
uart_puts("\n");
```

**Method 2: GDB Debugging**

```bash
# Terminal 1: Start QEMU in debug mode
qemu-system-riscv32 -M virt -bios none -nographic \
    -kernel my_custom_test.elf -s -S

# Terminal 2: Connect GDB
riscv64-unknown-elf-gdb my_custom_test.elf
(gdb) target remote localhost:1234
(gdb) break main
(gdb) continue
```

### SST Device Debugging

**Method 1: Verbose Output**

Set `verbose` parameter to higher values:

```python
mydev.addParams({
    "verbose": "3",  # Maximum verbosity
})
```

**Method 2: Custom Debug Output**

```cpp
if (verbose > 2) {
    getSimulationOutput().output("DEBUG: controlReg=0x%x, statusReg=0x%x\n",
                                 controlReg, statusReg);
}
```

**Method 3: Statistics**

```cpp
// Define custom statistics
Statistic<uint64_t>* statDebugEvents;

// In constructor
statDebugEvents = registerStatistic<uint64_t>("debug_events");

// Increment when needed
statDebugEvents->addData(1);
```

### Common Issues

**Issue: MMIO Access Causes Trap**

- Check device base address matches in firmware and SST config
- Verify custom QEMU has device at correct address
- Ensure SST component is properly connected

**Issue: Device Not Responding**

- Check clock handler is running
- Verify event handling logic
- Enable verbose output to see transactions

**Issue: Build Failures**

- Verify SST environment variables
- Check RISC-V toolchain installation
- Ensure all dependencies are installed

---

## Example Templates

Complete example templates are available in:
- `riscv-programs/mmio_test_main.c` - Firmware example
- `riscv-programs/asm_link_test.c` - Assembly linkage example
- `acalsim-device/ACALSimDeviceComponent.cc` - SST device example

---

## Next Steps

1. Review the [Developer Guide](DEVELOPER_GUIDE.md) for architecture details
2. Study existing examples in `riscv-programs/` and `acalsim-device/`
3. Experiment with modifying existing tests
4. Create your own custom device model

---

**Last Updated**: 2025-11-10
**Version**: 1.0
