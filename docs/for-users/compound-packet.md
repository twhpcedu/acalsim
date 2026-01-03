# How to Use Compound Packets to Model Different Bandwidth Requirements

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

---

- Author: ACALSim Team
- Date: 2025/01/03

([Back To Documentation Portal](/docs/README.md))

## Introduction

In ACALSim, the `SimPort` (`MasterPort`/`SlavePort`) infrastructure has an inherent throughput limitation: **one packet per cycle per port**. This design simplifies the simulation model but can make it challenging to model hardware components with different bandwidth requirements.

For example:
- A memory controller might need to transfer 512 bytes per cycle
- A cache line is typically 64 bytes
- The port can only send 1 packet per cycle

**The Problem:**
```
If 1 packet = 64 bytes and 1 packet/cycle limit:
    Maximum bandwidth = 64 bytes/cycle

But we need to model 512 bytes/cycle!
```

**The Solution: CompoundPacket**

The `CompoundPacket` class allows you to pack multiple logical packets into a single compound packet. This compound packet is sent through the port infrastructure as a single unit (counting as 1 packet), but the receiver can unpack it to retrieve all the individual packets.

```
Pack 8 packets (8 × 64 = 512 bytes) into 1 CompoundPacket
Send 1 CompoundPacket/cycle through SimPort
Effective bandwidth = 512 bytes/cycle
```

## How It Works

### The SimPort Limitation

The `MasterPort` has a single-entry buffer, limiting throughput:

```
┌─────────────┐      Single-Entry       ┌─────────────┐
│  MasterPort │ ──── Buffer ─────────► │  SlavePort  │
│             │   (1 packet/cycle)      │   (Queue)   │
└─────────────┘                         └─────────────┘
```

### CompoundPacket Solution

By wrapping multiple packets into a `CompoundPacket`, you bypass this limitation:

```
┌───────────────────┐                    ┌───────────────────┐
│   CompoundPacket  │                    │   CompoundPacket  │
│  ┌─────┐ ┌─────┐  │      1 packet      │  ┌─────┐ ┌─────┐  │
│  │ Pkt1│ │ Pkt2│  │ ────────────────►  │  │ Pkt1│ │ Pkt2│  │
│  └─────┘ └─────┘  │     per cycle      │  └─────┘ └─────┘  │
│  ┌─────┐ ┌─────┐  │                    │  ┌─────┐ ┌─────┐  │
│  │ Pkt3│ │ Pkt4│  │                    │  │ Pkt3│ │ Pkt4│  │
│  └─────┘ └─────┘  │                    │  └─────┘ └─────┘  │
└───────────────────┘                    └───────────────────┘
     Sender                                  Receiver
  (4 × 64 = 256 B)                      (unpack 4 packets)
```

## API Reference

### CompoundPacket Class

```cpp
#include "packet/CompoundPacket.hh"

namespace acalsim {

template <typename PacketType>
class CompoundPacket : public SimPacket {
public:
    // Construction
    explicit CompoundPacket(uint32_t sourceId = 0, PTYPE pktType = PTYPE::DATA);

    // Add packets
    void addPacket(PacketType* packet);

    // Query
    size_t size() const;                              // Number of contained packets
    bool empty() const;                               // Is empty?
    const std::vector<PacketType*>& getPackets() const;  // Read-only access
    std::vector<PacketType*>& getPackets();              // Mutable access

    // Extract (transfers ownership)
    std::vector<PacketType*> extractPackets();

    // Metadata
    uint32_t getSourceId() const;
    std::string getName() const override;
    bool isCompoundPacket() const;
};

}  // namespace acalsim
```

## Usage Guide

### Step 1: Define Your Packet Type

First, define your data packet class:

```cpp
class MyDataPacket : public acalsim::SimPacket {
public:
    static constexpr size_t PACKET_SIZE_BYTES = 64;  // Each packet = 64 bytes

    explicit MyDataPacket(uint32_t seqNum)
        : acalsim::SimPacket(PTYPE::DATA), seqNum_(seqNum) {}

    void visit(acalsim::Tick, acalsim::SimModule&) override {}
    void visit(acalsim::Tick, acalsim::SimBase&) override {}

    uint32_t getSeqNum() const { return seqNum_; }

private:
    uint32_t seqNum_;
};
```

### Step 2: Create and Send CompoundPackets (Sender)

```cpp
void Producer::sendData() {
    // Define desired bandwidth: 4 packets per compound = 256 bytes/cycle
    const size_t PACKETS_PER_COMPOUND = 4;

    // Create a compound packet
    auto* compound = new acalsim::CompoundPacket<MyDataPacket>(producerId_);

    // Pack multiple packets
    for (size_t i = 0; i < PACKETS_PER_COMPOUND; ++i) {
        compound->addPacket(new MyDataPacket(nextSeqNum_++));
    }

    // Send through port (1 cycle for all 4 packets = 256 bytes)
    if (masterPort_->isPushReady()) {
        masterPort_->push(compound);
        // Effective bandwidth: 4 × 64 = 256 bytes/cycle
    }
}
```

### Step 3: Receive and Unpack CompoundPackets (Receiver)

```cpp
void Consumer::receiveData() {
    if (!slavePort_->isPopValid()) return;

    acalsim::SimPacket* pkt = slavePort_->pop();

    // Try to cast to CompoundPacket
    auto* compound = dynamic_cast<acalsim::CompoundPacket<MyDataPacket>*>(pkt);

    if (compound != nullptr) {
        // Unpack individual packets
        auto packets = compound->extractPackets();
        for (MyDataPacket* dataPkt : packets) {
            processPacket(dataPkt);
            delete dataPkt;  // Clean up individual packet
        }
        delete compound;  // Clean up container
    } else {
        // Handle single packet (backward compatibility)
        auto* dataPkt = dynamic_cast<MyDataPacket*>(pkt);
        if (dataPkt) {
            processPacket(dataPkt);
            delete dataPkt;
        }
    }
}
```

## Bandwidth Modeling Examples

### Example 1: Memory Controller (512 bytes/cycle)

```cpp
// Memory read response: 512 bytes = 8 cache lines
const size_t CACHE_LINE_SIZE = 64;
const size_t BANDWIDTH_BYTES = 512;
const size_t PACKETS_PER_COMPOUND = BANDWIDTH_BYTES / CACHE_LINE_SIZE;  // 8

auto* compound = new CompoundPacket<CacheLinePacket>(controllerId);
for (size_t i = 0; i < PACKETS_PER_COMPOUND; ++i) {
    compound->addPacket(new CacheLinePacket(address + i * CACHE_LINE_SIZE));
}
memPort_->push(compound);  // 512 bytes in 1 cycle
```

### Example 2: PCIe Link (Variable Bandwidth)

```cpp
// PCIe Gen4 x16: ~32 GB/s = ~32 bytes/ns
// At 1 GHz clock: 32 bytes/cycle
const size_t PCIE_BANDWIDTH = 32;
const size_t TLP_SIZE = 8;  // Transaction Layer Packet
const size_t TLPS_PER_COMPOUND = PCIE_BANDWIDTH / TLP_SIZE;  // 4

auto* compound = new CompoundPacket<PCIeTLP>(linkId);
for (size_t i = 0; i < TLPS_PER_COMPOUND; ++i) {
    compound->addPacket(new PCIeTLP(payload[i]));
}
pciePort_->push(compound);
```

### Example 3: GPU Memory (High Bandwidth)

```cpp
// HBM2e: ~460 GB/s per stack
// At 2 GHz: ~230 bytes/cycle per stack
const size_t HBM_BANDWIDTH = 256;  // Round to 256 bytes/cycle
const size_t BURST_SIZE = 32;
const size_t BURSTS_PER_COMPOUND = HBM_BANDWIDTH / BURST_SIZE;  // 8

auto* compound = new CompoundPacket<HBMBurst>(stackId);
for (size_t i = 0; i < BURSTS_PER_COMPOUND; ++i) {
    compound->addPacket(new HBMBurst(channel, bank, row, col));
}
hbmPort_->push(compound);
```

## Complete Example

A complete working example is provided in `src/testCompoundPacket/`. Run it with:

```bash
# Build
cd build && make testCompoundPacket

# Run with default settings (4 packets/compound = 256 bytes/cycle)
./testCompoundPacket

# Model high bandwidth (8 packets/compound = 512 bytes/cycle)
./testCompoundPacket --packets-per-cycle 8

# Model low bandwidth (1 packet/compound = 64 bytes/cycle)
./testCompoundPacket --packets-per-cycle 1

# Large transfer
./testCompoundPacket --total-packets 1000 --packets-per-cycle 16
```

**Expected Output:**
```
========================================
CompoundPacket Bandwidth Modeling Example
========================================
Configuration:
  Total packets:      100
  Packets per cycle:  4
  Bytes per packet:   64
  Effective bandwidth: 256 bytes/cycle
========================================
[Producer] Sent CompoundPacket with 4 packets (seq 0-3) at tick 1
[Consumer] Received CompoundPacket from source 0 with 4 packets at tick 2
[Consumer]   - Unpacked DataPacket[seq=0, producer=0]
[Consumer]   - Unpacked DataPacket[seq=1, producer=0]
[Consumer]   - Unpacked DataPacket[seq=2, producer=0]
[Consumer]   - Unpacked DataPacket[seq=3, producer=0]
...
[Consumer] All 100 packets received!
[Consumer] Average packets per CompoundPacket: 4 (effective bandwidth: 256 bytes/cycle)
```

## Best Practices

### 1. Choose Appropriate Packet Granularity

Define your base packet to represent the smallest meaningful unit of data:
- Cache line (64 bytes) for memory systems
- Flit (16-32 bytes) for NoC
- Transaction (variable) for bus protocols

### 2. Calculate Packets Per Compound Based on Target Bandwidth

```cpp
size_t packetsPerCompound = targetBandwidthBytes / basePacketSize;
```

### 3. Handle Backpressure

Store pending compound packets for retry:

```cpp
void Producer::masterPortRetry(MasterPort* port) {
    if (pendingCompound_ != nullptr) {
        if (port->push(pendingCompound_)) {
            pendingCompound_ = nullptr;
        }
    }
}
```

### 4. Clean Up Properly

- The receiver takes ownership of individual packets via `extractPackets()`
- Delete individual packets after processing
- Delete the compound container after extraction

```cpp
auto packets = compound->extractPackets();
for (auto* pkt : packets) {
    process(pkt);
    delete pkt;
}
delete compound;
```

### 5. Maintain Backward Compatibility

Check for both compound and single packets:

```cpp
auto* compound = dynamic_cast<CompoundPacket<MyPacket>*>(pkt);
if (compound) {
    // Handle compound
} else {
    auto* single = dynamic_cast<MyPacket*>(pkt);
    if (single) {
        // Handle single packet
    }
}
```

## Summary

| Scenario | Packets/Compound | Bytes/Cycle | Use Case |
|----------|------------------|-------------|----------|
| Low bandwidth | 1 | 64 | Simple interconnect |
| Medium bandwidth | 4 | 256 | L2 cache interface |
| High bandwidth | 8 | 512 | Memory controller |
| Very high bandwidth | 16+ | 1024+ | HBM, GPU memory |

The `CompoundPacket` methodology provides a flexible way to model various bandwidth requirements while working within ACALSim's single-packet-per-cycle port constraint. By packing multiple logical packets into compound packets, you can accurately simulate high-bandwidth hardware components.

## Related Documentation

- [SimPort User Guide](simport.md) - Understanding MasterPort/SlavePort
- [Example: testCompoundPacket](../../src/testCompoundPacket/main.cc) - Complete working example
