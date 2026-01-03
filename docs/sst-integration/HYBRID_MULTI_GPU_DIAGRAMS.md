# Hybrid ACALSim-SST Multi-GPU Architecture Diagrams

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

This document explains the diagrams for the **Hybrid ACALSim-SST Multi-Device Architecture** for LLM inference on multi-GPU systems.

## Overview

The hybrid architecture combines:
- **ACALSim**: Cycle-accurate simulation of individual A100 GPUs
- **SST**: Event-driven orchestration of multi-GPU coordination
- **Dual-Port TCP**: Separate job control and NVLink data traffic
- **Flexible Topology**: Configurable NVLink interconnect via SST Links

---

## Diagram Files

### 1. Architecture Diagram (Main)
**File**: `hybrid-multi-gpu-architecture.puml`

**Shows**:
- SST Orchestration Layer with HostSchedulerComponent
- 4 GPUDeviceComponent instances (GPU0-GPU3)
- Full-mesh NVLink SST::Links between GPUs
- Dual TCP ports per GPU (Job Port + NVLink Port)
- 4 independent ACALSim A100 processes
- Complete component relationships

**Generate PNG**:
```bash
plantuml -tpng hybrid-multi-gpu-architecture.puml
```

**Key Features Illustrated**:
- ✅ HostSchedulerComponent distributes inference requests
- ✅ GPUDeviceComponent wraps each A100 with dual TCP connections
- ✅ NVLink SST::Links form full-mesh topology (6 links for 4 GPUs)
- ✅ Dual-port architecture prevents blocking
- ✅ Process isolation between ACALSim instances

---

### 2. Sequence Diagram (Communication Flow)
**File**: `hybrid-multi-gpu-sequence.puml`

**Shows**:
- Complete inference request flow from submission to completion
- Job control messages via Job Ports (low-volume, latency-sensitive)
- NVLink data transfer via NVLink Ports (high-volume, bandwidth-intensive)
- Inter-GPU communication through SST::Links
- Non-blocking architecture: Job completion independent of NVLink traffic

**Generate PNG**:
```bash
plantuml -tpng hybrid-multi-gpu-sequence.puml
```

**Communication Phases**:
1. **Job Submission Phase**: HostScheduler assigns jobs to GPUs via Job Ports
2. **Inter-GPU Communication Phase**: Tensor sharding and KV cache migration via NVLink Ports
3. **Job Completion Phase**: Completion notifications via Job Ports (not blocked by NVLink)

**Example Flow**:
```
External Workload
  └─> HostSchedulerComponent
      ├─> GPU0 (Job Port :9100) - Job submission
      └─> GPU1 (Job Port :9101) - Job submission
          ↓
GPU0 (NVLink Port :9200) - Send tensor shard
  └─> SST NVLink
      └─> GPU1 (NVLink Port :9201) - Receive tensor
          ↓
GPU0 (Job Port :9100) - Job complete (NOT blocked)
GPU1 (Job Port :9101) - Job complete (NOT blocked)
```

---

### 3. Topology Diagram (NVLink Configurations)
**File**: `hybrid-multi-gpu-topology.puml`

**Shows**:
- **Full-Mesh Topology**: NVSwitch fabric (6 bidirectional links)
- **Ring Topology**: PCIe-only systems (4 bidirectional links)
- **2x2 Mesh Topology**: Physical layout-aware (4 bidirectional links)
- SST Python configuration examples for each topology

**Generate PNG**:
```bash
plantuml -tpng hybrid-multi-gpu-topology.puml
```

**Topology Comparison**:

| Topology | Links | Bandwidth | Latency | Use Case |
|----------|-------|-----------|---------|----------|
| **Full-Mesh** | 6 | Maximum | Lowest | NVSwitch systems, best performance |
| **Ring** | 4 | Lower | Multi-hop | PCIe-only, simple coordination |
| **2x2 Mesh** | 4 | Balanced | Medium | Pipeline parallelism, physical layout |

**Configuration Flexibility**:
- Topologies configured in SST Python script
- No ACALSim code changes required
- Per-link latency and bandwidth settings
- Easy experimentation with different topologies

---

## Generate All Diagrams

To generate all three diagrams at once:

```bash
cd docs/sst-integration
plantuml -tpng hybrid-multi-gpu-*.puml
```

This creates:
- `hybrid-multi-gpu-architecture.png`
- `hybrid-multi-gpu-sequence.png`
- `hybrid-multi-gpu-topology.png`

---

## Architecture Details

### Design Rationale

**Problem**: ACALSim uses shared-memory programming model
- Each simulator process models exactly one GPU device
- Efficient intra-device communication
- Cannot directly share memory between multiple GPU instances

**Solution**: Hybrid architecture
- Each GPU runs as independent ACALSim process
- SST components manage inter-device NVLink routing
- SST handles global job scheduling

### Dual-Port TCP Architecture

Each A100 simulator exposes two TCP ports:

#### Job Port (e.g., 9100, 9101, 9102, 9103)
- **Traffic**: Low-volume, latency-sensitive
- **Purpose**: Job control
- **Operations**:
  - Job submission
  - Completion notifications
  - Synchronization signals
- **Characteristics**: Fast response, never blocked

#### NVLink Port (e.g., 9200, 9201, 9202, 9203)
- **Traffic**: High-volume, bandwidth-intensive
- **Purpose**: Memory transfer
- **Operations**:
  - Tensor sharding
  - KV cache migration
  - GPU-to-GPU data packets
- **Characteristics**: High throughput, does NOT block job control

### SST Integration Components

#### 1. GPUDeviceComponent
Wraps each ACALSim A100 process:
- **HPCSimWrapper**: Manages Job Port TCP connection
- **NVLinkWrapper**: Manages NVLink Port TCP connection
- **Routing**: Forwards packets to/from SST::Links

#### 2. HostSchedulerComponent
External workload interface:
- Distributes inference requests across devices
- Load balancing policies
- Tensor parallelism coordination
- Global job scheduling

#### 3. SST::Link for NVLink
Models inter-GPU NVLink topology:
- Configurable latency and bandwidth
- Routes packets between GPUDeviceComponent instances
- Supports various topologies (full-mesh, ring, mesh)

### Architecture Benefits

#### ✓ Preserved Fidelity
- Each GPU maintains cycle-accurate simulation
- Native 2-phase execution preserved
- Backpressure modeling intact
- No loss of simulation accuracy

#### ✓ Process Isolation
- Independent A100 processes
- Parallel simulation
- Fault isolation
- No cross-process memory corruption

#### ✓ Flexible Topology
- SST Link configuration in Python
- No ACALSim code modifications
- Easy topology experimentation
- Per-link parameter tuning

#### ✓ Scalability
- Add GPUs by spawning additional A100 processes
- Configure corresponding SST components
- Linear scaling with GPU count
- No architectural changes needed

#### ✓ Non-Blocking
- Dual-port prevents NVLink from blocking job control
- Responsive scheduling under heavy communication
- Job completion independent of data transfer
- Better overall system responsiveness

---

## Usage Example

### 4-GPU Full-Mesh Configuration

**SST Python Configuration** (`multi_gpu_llm.py`):

```python
import sst

# Create HostSchedulerComponent
scheduler = sst.Component("scheduler", "acalsim.HostScheduler")
scheduler.addParams({
    "clock": "1GHz",
    "load_balancing": "round_robin",
    "tensor_parallelism": "enabled"
})

# Create 4 GPUDeviceComponents
gpus = []
for i in range(4):
    gpu = sst.Component(f"gpu{i}", "acalsim.GPUDevice")
    gpu.addParams({
        "clock": "1.4GHz",
        "gpu_id": i,
        "job_port": 9100 + i,
        "nvlink_port": 9200 + i,
        "acalsim_host": "localhost"
    })
    gpus.append(gpu)
    
    # Connect scheduler to GPU
    link = sst.Link(f"scheduler_gpu{i}")
    link.connect(
        (scheduler, f"gpu_port_{i}", "1ns"),
        (gpu, "scheduler_port", "1ns")
    )

# Create Full-Mesh NVLink topology
link_id = 0
for i in range(4):
    for j in range(i+1, 4):
        nvlink = sst.Link(f"nvlink_{i}_{j}")
        nvlink.connect(
            (gpus[i], "nvlink_port", "10ns"),  # 10ns latency
            (gpus[j], "nvlink_port", "10ns")
        )
        link_id += 1

# Configure statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV", {
    "filepath": "multi_gpu_stats.csv"
})
```

**Launch ACALSim A100 Processes**:

```bash
# Terminal 1: GPU0
./acalsim_a100 --gpu-id 0 --job-port 9100 --nvlink-port 9200

# Terminal 2: GPU1
./acalsim_a100 --gpu-id 1 --job-port 9101 --nvlink-port 9201

# Terminal 3: GPU2
./acalsim_a100 --gpu-id 2 --job-port 9102 --nvlink-port 9202

# Terminal 4: GPU3
./acalsim_a100 --gpu-id 3 --job-port 9103 --nvlink-port 9203

# Terminal 5: SST Orchestration
sst multi_gpu_llm.py
```

---

## Performance Characteristics

### Job Control (Job Port)
- **Latency**: ~100μs per message
- **Throughput**: ~10K messages/sec per GPU
- **Traffic**: <1 MB/sec typical

### NVLink Data (NVLink Port)
- **Latency**: ~10μs + SST routing overhead
- **Throughput**: Limited by TCP, ~1-5 GB/sec per link
- **Traffic**: 100s of MB/sec to GB/sec during tensor transfers

### Scaling
- **4 GPUs**: 6 NVLink connections (full-mesh)
- **8 GPUs**: 28 NVLink connections (full-mesh)
- **Memory**: ~500MB per ACALSim A100 process
- **CPU**: ~1 core per ACALSim process + SST orchestration

---

## Comparison with Alternatives

### vs. Single-Process Multi-GPU ACALSim
❌ Would require major ACALSim refactoring
❌ Shared-memory model doesn't support multiple devices
❌ Less fault isolation

✅ Hybrid preserves ACALSim design
✅ Process isolation
✅ No ACALSim changes needed

### vs. Pure SST Multi-GPU
❌ Would lose ACALSim's cycle-accurate fidelity
❌ Would need to reimplement GPU models in SST
❌ No access to ACALSim's advanced features

✅ Hybrid preserves both strengths
✅ Cycle-accurate intra-device
✅ Event-driven inter-device

### vs. Monolithic Simulator
❌ No parallelism between GPUs
❌ Difficult to scale
❌ Single point of failure

✅ Hybrid enables parallel GPU simulation
✅ Scales linearly
✅ Fault-isolated processes

---

## Future Extensions

### Planned Features

1. **Dynamic Load Balancing**
   - Runtime job migration
   - Adaptive scheduling based on GPU utilization

2. **Heterogeneous GPU Support**
   - Mix of A100, H100, different memory sizes
   - Per-GPU capability modeling

3. **Advanced NVLink Modeling**
   - Congestion and backpressure
   - Per-link bandwidth saturation
   - Quality-of-Service (QoS)

4. **Multi-Node Scaling**
   - Distributed SST across nodes
   - InfiniBand/RoCE modeling
   - NCCL collective simulation

5. **Power Modeling**
   - Per-GPU power consumption
   - Thermal modeling
   - Dynamic frequency scaling

---

## Troubleshooting

### Issue: TCP Connection Refused

**Symptom**: SST cannot connect to ACALSim processes

**Solution**:
1. Ensure ACALSim processes started before SST
2. Check port conflicts: `netstat -an | grep 9100`
3. Verify firewall settings
4. Check ACALSim logs for binding errors

### Issue: NVLink Traffic Blocking Job Control

**Symptom**: Slow job completions during heavy data transfer

**Diagnosis**: Likely single-port architecture being used

**Solution**: Ensure dual-port configuration:
```python
gpu.addParams({
    "job_port": 9100 + i,       # Separate port
    "nvlink_port": 9200 + i     # Separate port
})
```

### Issue: Poor Scaling Beyond 4 GPUs

**Symptom**: Performance doesn't improve with more GPUs

**Diagnosis**: Full-mesh links (N*(N-1)/2) may overwhelm SST

**Solution**: Use hierarchical topology (e.g., 2-level mesh) or switch to ring

---

## References

### Documentation
- [Main Architecture Diagram](architecture-diagram.md)
- [Integration Guide](integration-guide.md)
- [SST Component Documentation](https://sst-simulator.org/)

### Source Code
- GPUDeviceComponent: `include/sst/GPUDeviceComponent.hh`
- HostSchedulerComponent: `include/sst/HostSchedulerComponent.hh`
- A100 Simulator: `src/a100-simulator/`

### Related Papers
- ACALSim: High-Performance Event-Driven Simulation Framework
- SST: Structural Simulation Toolkit for Large-Scale Systems

---

## Quick Reference

**Generate all diagrams**:
```bash
plantuml -tpng hybrid-multi-gpu-*.puml
```

**4-GPU Full-Mesh**:
- 4 ACALSim processes
- 8 TCP ports total (4 job + 4 NVLink)
- 6 NVLink SST::Links
- 1 HostSchedulerComponent

**Dual-Port Benefits**:
- Job control never blocked
- High NVLink bandwidth
- Independent completion
- Better responsiveness

---

**Copyright 2023-2026 Playlab/ACAL**  
Licensed under the Apache License, Version 2.0

