# Deployment Guide

**Single-Server vs Multi-Server SST Simulation**

Copyright 2023-2025 Playlab/ACAL

---

## Table of Contents

1. [Deployment Architectures](#deployment-architectures)
2. [Single-Server Deployment](#single-server-deployment)
3. [Multi-Server Deployment](#multi-server-deployment)
4. [Performance Considerations](#performance-considerations)
5. [Network Configuration](#network-configuration)

---

## Deployment Architectures

### Overview

ACALSim-Linux can be deployed in two modes:

1. **Single-Server**: QEMU and SST run on same machine
2. **Multi-Server**: SST components distributed across multiple machines

```
Single-Server                    Multi-Server
─────────────                    ────────────

┌──────────────┐                ┌──────────────┐
│   Server 1   │                │   Server 1   │
│              │                │              │
│   ┌─QEMU─┐   │                │   ┌─QEMU─┐   │
│   │Linux │   │                │   │Linux │   │
│   └──────┘   │                │   └──────┘   │
│      ↕       │                │      ↕       │
│  ┌────────┐  │                │  ┌────────┐  │
│  │SST Core│  │                │  │SST Core│  │  ← Rank 0 (coordinator)
│  │        │  │                │  │ Rank 0 │  │
│  │All     │  │                │  └────────┘  │
│  │Comps   │  │                │      ↕       │
│  └────────┘  │                │   Network    │
└──────────────┘                └──────────────┘
                                       ↕
                                ┌──────────────┐
                                │   Server 2   │
                                │              │
                                │  ┌────────┐  │
                                │  │SST Core│  │  ← Rank 1 (worker)
                                │  │ Rank 1 │  │
                                │  │        │  │
                                │  │Comp A  │  │
                                │  └────────┘  │
                                └──────────────┘
                                       ↕
                                ┌──────────────┐
                                │   Server 3   │
                                │              │
                                │  ┌────────┐  │
                                │  │SST Core│  │  ← Rank 2 (worker)
                                │  │ Rank 2 │  │
                                │  │        │  │
                                │  │Comp B-C│  │
                                │  └────────┘  │
                                └──────────────┘
```

---

## Single-Server Deployment

### Requirements

**Hardware**:
- CPU: 4+ cores recommended (2 for QEMU, 2+ for SST)
- RAM: 8GB+ (2GB for QEMU Linux, 4GB+ for SST models)
- Disk: 20GB+ for build artifacts

**Software**:
- Docker (acalsim-workspace container) OR
- Native Linux with all dependencies installed
- SST-Core 14.1.0+
- QEMU 8.0+
- Linux kernel 6.1+

### Setup

All components run in Docker container:

```bash
# Start container
docker exec -it acalsim-workspace bash

# Terminal 1: Start SST
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/sst-config
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
sst linux_basic.py

# Terminal 2: Start QEMU (after SST shows "Waiting for QEMU...")
cd /home/user
export QEMU=/home/user/qemu-build/qemu/build/qemu-system-riscv64
export KERNEL=/home/user/linux/arch/riscv/boot/Image
export INITRD=/home/user/initramfs.cpio.gz
export SOCKET=/tmp/qemu-sst-linux.sock

$QEMU \
    -machine virt \
    -cpu rv64 \
    -m 2G \
    -smp 4 \
    -nographic \
    -kernel $KERNEL \
    -initrd $INITRD \
    -append "console=ttyS0 earlycon=sbi" \
    -device virtio-sst-device,socket=$SOCKET,device-id=0
```

### Communication Path

```
QEMU Process              SST Process
────────────              ───────────
 (PID 1234)                (PID 5678)
     │                          │
     └──→ Unix Socket ──────────┘
          /tmp/qemu-sst-linux.sock

IPC Type: Unix Domain Socket (AF_UNIX, SOCK_STREAM)
Latency: ~1-10 microseconds
Bandwidth: ~1 GB/s
```

### Advantages

- **Simple setup**: One machine, one Docker container
- **Low latency**: Unix sockets, no network overhead
- **Easy debugging**: All logs in one place
- **Development**: Ideal for development/testing

### Limitations

- **CPU bound**: Single machine's CPU limit
- **Memory bound**: Single machine's RAM limit
- **No scaling**: Can't add more machines for performance

### Use Cases

- Development and testing
- Small simulations (<1M cycles)
- Simple hardware models (few components)
- Demos and tutorials

---

## Multi-Server Deployment

### Requirements

**Hardware** (per server):
- CPU: 8+ cores recommended
- RAM: 16GB+ recommended
- Network: 10Gbps+ Ethernet or InfiniBand
- Disk: 50GB+ for each server

**Software** (all servers):
- SST-Core built with MPI support
- OpenMPI 4.0+ or MPICH 3.3+
- Shared filesystem (NFS) OR synchronized builds
- Network time synchronization (NTP)

**Network**:
- Low latency (<100 μs)
- High bandwidth (>1 Gbps)
- All servers in same subnet

### SST Distributed Simulation

SST uses **MPI (Message Passing Interface)** for distributed simulation:

```
┌────────────────────────────────────────────────────────┐
│  SST Partitioning: Components distributed across ranks │
└────────────────────────────────────────────────────────┘

Rank 0 (Server 1)        Rank 1 (Server 2)        Rank 2 (Server 3)
─────────────────        ─────────────────        ─────────────────
VirtIO Device    ←Link→  AI Accelerator 1  ←Link→  Memory Controller
                         AI Accelerator 2  ←Link→  Interconnect

Links = SST inter-component communication
   Local links: shared memory (fast)
   Remote links: MPI messages (slower)
```

### Setup

**Step 1**: Build SST with MPI support (on all servers):

```bash
# Install MPI
sudo apt-get install libopenmpi-dev openmpi-bin

# Build SST-Core with MPI
cd /home/user/projects/acalsim/sst-core/build
../sst-core/configure --prefix=$PWD/../sst-core-install \
                      --with-mpi=/usr/lib/x86_64-linux-gnu/openmpi
make -j$(nproc) install
```

**Step 2**: Create distributed SST configuration:

```python
# linux_distributed.py
import sst

# Partition components across ranks
partition = sst.SSTPartitioner()

# Rank 0: VirtIO device (connects to QEMU)
if sst.getRank() == 0:
    virtio_dev = sst.Component("virtio_dev", "acalsim.VirtIODevice")
    virtio_dev.addParams({
        "socket_path": "/tmp/qemu-sst-linux.sock",
        "device_id": 0,
        "verbose": "1",
        "clock": "1GHz"
    })
    partition.addComponent(virtio_dev, 0)  # Assign to rank 0

# Rank 1: AI Accelerator 1
if sst.getRank() == 1:
    accel1 = sst.Component("accel1", "acalsim.HSACompute")
    accel1.addParams({
        "device_id": 1,
        "cores": "64",
        "clock": "2GHz"
    })
    partition.addComponent(accel1, 1)  # Assign to rank 1

# Rank 2: AI Accelerator 2
if sst.getRank() == 2:
    accel2 = sst.Component("accel2", "acalsim.HSACompute")
    accel2.addParams({
        "device_id": 2,
        "cores": "64",
        "clock": "2GHz"
    })
    partition.addComponent(accel2, 2)  # Assign to rank 2

# Create inter-rank links
if sst.getRank() == 0:
    link01 = sst.Link("link_0_1")
    link01.connect(
        (virtio_dev, "accel_link", "1ns"),
        (accel1, "host_link", "1ns")  # This is on rank 1!
    )
    # SST automatically handles MPI communication
```

**Step 3**: Create hostfile:

```bash
# hosts.txt
server1.example.com slots=4
server2.example.com slots=8
server3.example.com slots=8
```

**Step 4**: Launch distributed simulation:

```bash
# On Server 1:
mpirun -np 3 \
       --hostfile hosts.txt \
       --map-by node \
       sst linux_distributed.py

# -np 3: 3 MPI ranks (one per server)
# --hostfile: which servers to use
# --map-by node: one rank per server
```

**Step 5**: Launch QEMU (on Server 1 only):

```bash
# QEMU only runs on Server 1 (where VirtIO device is)
$QEMU -machine virt -cpu rv64 -m 2G -smp 4 -nographic \
      -kernel $KERNEL -initrd $INITRD \
      -append "console=ttyS0" \
      -device virtio-sst-device,socket=/tmp/qemu-sst-linux.sock
```

### Communication Architecture

```
Server 1                Server 2                Server 3
────────                ────────                ────────
QEMU
  ↕ Unix Socket
VirtIO Device
  ↕ MPI (10G Ethernet)
              ↔────────→ AI Accel 1 ↔──────→  AI Accel 2
                         (Rank 1)              (Rank 2)
```

### Synchronization

SST uses **Parallel Discrete Event Simulation (PDES)** with conservative synchronization:

```
Time Advance:
────────────

Rank 0                Rank 1                Rank 2
──────                ──────                ──────
Cycle 0 ──────────→  Cycle 0 ──────────→  Cycle 0
  │ Process events     │ Process events     │ Process events
  │                    │                    │
Cycle 1000            Cycle 1000           Cycle 1000
  │                    │                    │
  │← Barrier: All ranks reach cycle 1000 before any advance to 1001
  │                    │                    │
Cycle 1001 ────────→  Cycle 1001 ───────→  Cycle 1001
```

**Lookahead**: SST requires minimum lookahead (e.g., link latency = 1ns) to prevent deadlock.

### Advantages

- **Scalability**: Linear scaling with servers (for compute-intensive models)
- **Large simulations**: Billion-cycle simulations feasible
- **Memory**: Distributed memory pool
- **Specialization**: Different servers for different hardware models

### Limitations

- **Complexity**: More complex setup and debugging
- **Network latency**: Inter-rank communication adds overhead
- **Synchronization**: Conservative synchronization limits speedup
- **Cost**: Requires multiple servers

### Use Cases

- Large-scale simulations (>1B cycles)
- Complex multi-accelerator systems
- Production workloads
- Performance studies

---

## Performance Considerations

### Single-Server Performance

**Bottlenecks**:
1. Unix socket bandwidth (~1 GB/s)
2. CPU cores (QEMU uses 1-4, SST uses remaining)
3. Memory bandwidth (shared between QEMU and SST)

**Optimization**:
```bash
# Pin QEMU to specific cores
taskset -c 0-3 $QEMU ...

# Pin SST to different cores
taskset -c 4-7 sst linux_basic.py
```

### Multi-Server Performance

**Speedup Formula** (Amdahl's Law):
```
Speedup = 1 / (S + P/N)

S = Sequential fraction (QEMU + VirtIO device)
P = Parallel fraction (SST components)
N = Number of servers

Example:
- S = 10% (QEMU overhead)
- P = 90% (parallelizable SST work)
- N = 4 servers

Speedup = 1 / (0.1 + 0.9/4) = 1 / 0.325 = 3.08x
```

**Network Impact**:
- 1 Gbps: ~10-20% overhead for fine-grained communication
- 10 Gbps: ~2-5% overhead
- InfiniBand: <1% overhead

**Partitioning Strategy**:
```python
# BAD: Fine-grained partitioning (high communication)
Component A (Rank 0) ←→ Component B (Rank 1)  # Every cycle!

# GOOD: Coarse-grained partitioning (low communication)
Subsystem A (Rank 0) ←→ Subsystem B (Rank 1)  # Every 1000 cycles
```

---

## Network Configuration

### SSH Setup

For MPI to work, passwordless SSH must be configured:

```bash
# On Server 1:
ssh-keygen -t rsa  # Generate key
ssh-copy-id user@server2  # Copy to server 2
ssh-copy-id user@server3  # Copy to server 3

# Test:
ssh server2 hostname  # Should work without password
ssh server3 hostname
```

### Shared Filesystem (Optional but Recommended)

**Option 1: NFS**

```bash
# On Server 1 (NFS server):
sudo apt-get install nfs-kernel-server
echo "/home/user/projects *(rw,sync,no_subtree_check)" | \
    sudo tee -a /etc/exports
sudo exportfs -ra

# On Server 2 & 3 (NFS clients):
sudo apt-get install nfs-common
sudo mount server1:/home/user/projects /home/user/projects
```

**Option 2: Git + Rsync**

```bash
# Build on each server independently
# Synchronize SST libraries:
rsync -avz libacalsim.so server2:/path/to/lib/
rsync -avz libacalsim.so server3:/path/to/lib/
```

### Firewall Configuration

```bash
# Allow MPI communication (ports vary by MPI implementation)
# OpenMPI: typically uses dynamic ports in range 1024-65535

# On all servers:
sudo ufw allow from 192.168.1.0/24  # Allow subnet
# Or specific ports:
sudo ufw allow 10000:20000/tcp  # MPI port range
```

---

## Choosing the Right Deployment

| Criterion              | Single-Server       | Multi-Server        |
|------------------------|---------------------|---------------------|
| Simulation size        | < 1M cycles         | > 10M cycles        |
| Components             | < 10 components     | 10+ components      |
| Development phase      | ✅ Ideal            | ❌ Overkill         |
| Production             | ⚠️ Limited          | ✅ Recommended      |
| Debugging              | ✅ Easy             | ⚠️ Harder           |
| Cost                   | ✅ Low (1 machine)  | ❌ High (N machines)|
| Network requirement    | ✅ None             | ⚠️ 10Gbps+          |
| Setup complexity       | ✅ Simple           | ⚠️ Complex          |

**Recommendation**:
- **Start with single-server** for development and validation
- **Scale to multi-server** only when necessary (performance, memory, or scalability requirements)

---

## Next Steps

- **Single-server**: Follow GETTING_STARTED.md
- **Multi-server**: Consult SST documentation for advanced MPI configuration
- **Benchmarking**: See examples/benchmarks/ for performance testing

**Questions?** See ARCHITECTURE.md or open a GitHub issue.
