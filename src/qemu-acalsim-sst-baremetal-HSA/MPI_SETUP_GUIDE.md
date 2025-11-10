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

# MPI Setup Guide for Distributed Simulation

Comprehensive guide for setting up and running ACALSim distributed simulations across multiple servers using MPI.

## Table of Contents

- [Overview](#overview)
- [How MPI Distributed Execution Works](#how-mpi-distributed-execution-works)
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
- [Running Distributed Simulations](#running-distributed-simulations)
- [Verification and Testing](#verification-and-testing)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Performance Considerations](#performance-considerations)
- [Advanced Topics](#advanced-topics)

---

## Overview

ACALSim supports **distributed simulation** where QEMU and SST components can run across multiple physical servers using SST's MPI (Message Passing Interface) support. This enables large-scale simulations that exceed the resources of a single machine.

### Key Concepts

- **Head Node**: The server where you run the `mpirun` command (typically the first server)
- **Compute Nodes**: Additional servers that participate in the simulation
- **MPI Ranks**: Individual processes in the distributed simulation (rank 0, 1, 2, ...)
- **SST Links**: Communication channels between components that automatically work across MPI ranks

---

## How MPI Distributed Execution Works

### Critical Understanding

**You run the `mpirun` command on ONE server only** (the head node). MPI automatically:
1. Launches processes on remote servers via SSH
2. Distributes the SST configuration to all ranks
3. Handles inter-process communication transparently

### Example: 2-Server Execution

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR ACTION:                                               │
│  Run this command ON SERVER1 ONLY:                         │
│  $ mpirun -np 2 -host server1:1,server2:1 sst config.py   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌──────────────────┐                    ┌──────────────────┐
│  Server1         │                    │  Server2         │
│  Rank 0          │ ◄──MPI Links────► │  Rank 1          │
│  - QEMUBinary    │                    │  - Device1       │
│  - QEMU Process  │                    │                  │
└──────────────────┘                    └──────────────────┘
   MPI spawns this                       MPI auto-launches
   process locally                       via SSH to server2
```

### What MPI Does Automatically

- **SSH Connection**: Uses SSH to launch processes on remote servers
- **File Distribution**: Shares the Python configuration file across ranks
- **Communication Setup**: Establishes MPI communication channels
- **Synchronization**: Coordinates simulation time across all ranks
- **Event Passing**: Serializes and transmits SST events between ranks

---

## Prerequisites

### Software Requirements

All servers must have:
- **Same SST-Core version** (built with MPI support: `--with-mpi`)
- **Same MPI implementation** (OpenMPI or MPICH - all servers must match)
- **Same Python version**
- **Same ACALSim installation** (or at least same paths)

### Network Requirements

- All servers must be able to reach each other via network
- Firewalls must allow:
  - SSH (port 22)
  - MPI communication ports (typically ephemeral high ports)

### User Account Requirements

- Same username on all servers (or at least consistent UID/GID)
- Passwordless SSH access from head node to all compute nodes

---

## Step-by-Step Setup

### 1. Install MPI on All Servers

**Ubuntu/Debian:**
```bash
# On ALL servers:
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev

# Verify installation
mpirun --version
# Should show: mpirun (Open MPI) 4.x.x or similar
```

**CentOS/RHEL:**
```bash
# On ALL servers:
sudo yum install -y openmpi openmpi-devel
module load mpi/openmpi-x86_64

# Add to ~/.bashrc for persistence:
echo 'module load mpi/openmpi-x86_64' >> ~/.bashrc
```

### 2. Verify Network Connectivity

```bash
# On head node (server1):
ping server2
ping server3
# All should respond

# Test name resolution
nslookup server2
# Should resolve to IP address

# If using IP addresses instead of hostnames:
ping 192.168.1.11
ping 192.168.1.12
```

### 3. Setup Passwordless SSH Authentication

This is **critical** - MPI uses SSH to launch processes on remote servers.

**On the head node (server1):**
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
# Press Enter to accept defaults

# Copy public key to all compute nodes
ssh-copy-id user@server2
ssh-copy-id user@server3

# Test passwordless SSH
ssh server2 "hostname"
# Should print "server2" WITHOUT asking for password

ssh server3 "hostname"
# Should print "server3" WITHOUT asking for password
```

**If `ssh-copy-id` is not available:**
```bash
# Manual method:
cat ~/.ssh/id_rsa.pub | ssh user@server2 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
cat ~/.ssh/id_rsa.pub | ssh user@server3 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

### 4. Verify SSH Host Keys

First-time SSH connections require accepting host keys:
```bash
# On head node, connect to each server once:
ssh server2  # Type 'yes' when prompted
exit

ssh server3  # Type 'yes' when prompted
exit

# Now test automated SSH:
ssh server2 "echo 'SSH works!'"
# Should print "SSH works!" without any prompts
```

### 5. Install SST-Core with MPI Support

**On ALL servers** (must be identical installation):

```bash
# Download SST-Core (use same version on all servers!)
cd ~/projects/acalsim/sst-core
git clone https://github.com/sstsimulator/sst-core.git
cd sst-core

# Configure with MPI support
./autogen.sh
./configure --prefix=/home/user/projects/acalsim/sst-core/sst-core-install \
            --with-mpi=/usr/lib/x86_64-linux-gnu/openmpi

# Build and install
make -j$(nproc)
make install

# Set environment (add to ~/.bashrc on ALL servers)
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Verify MPI support
sst-config --MPI
# Should print: yes
```

### 6. Sync Project Files

Choose one of the following methods:

**Method A: Shared Filesystem (Recommended)**
```bash
# Setup NFS or similar shared filesystem
# Mount on all servers at same path:
# /home/user/projects/acalsim → shared across all nodes

# Verify on each server:
ssh server2 "ls /home/user/projects/acalsim"
ssh server3 "ls /home/user/projects/acalsim"
# Should show same files
```

**Method B: Manual File Sync**
```bash
# On head node, sync to all compute nodes:
rsync -avz --delete \
  /home/user/projects/acalsim/ \
  server2:/home/user/projects/acalsim/

rsync -avz --delete \
  /home/user/projects/acalsim/ \
  server3:/home/user/projects/acalsim/

# Verify:
ssh server2 "ls /home/user/projects/acalsim/src"
ssh server3 "ls /home/user/projects/acalsim/src"
```

### 7. Build ACALSim Components

**On ALL servers:**
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal

# Build and install SST components
cd qemu-binary && make && make install && cd ..
cd acalsim-device && make && make install && cd ..

# Verify installation on each server
ssh server2 "sst-info qemubinary.QEMUBinary"
ssh server3 "sst-info acalsim.QEMUDevice"
# Should show component info
```

---

## Running Distributed Simulations

### Basic Command Structure

```bash
mpirun -np <num_ranks> \
       -host <host1>:<slots1>,<host2>:<slots2>,... \
       sst <config.py>
```

**Parameters:**
- `-np`: Total number of MPI ranks (processes)
- `-host`: List of servers and how many ranks per server
- `<config.py>`: SST configuration file

### Example 1: Two Servers, Two Ranks

```bash
# On server1 (head node) - run this command ONLY here:
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal

mpirun -np 2 \
       -host server1:1,server2:1 \
       sst distributed_mmio_test.py
```

**What happens:**
- Rank 0 runs on server1 (QEMUBinary + QEMU process)
- Rank 1 runs on server2 (MMIO Device)
- MPI handles communication between them

### Example 2: Three Servers, Four Ranks

```bash
# On server1:
mpirun -np 4 \
       -host server1:1,server2:2,server3:1 \
       sst distributed_multi_device_test.py
```

**Distribution:**
- server1: 1 rank (Rank 0 - QEMU)
- server2: 2 ranks (Rank 1, 2 - Device 1, Device 2)
- server3: 1 rank (Rank 3 - Device 3)

### Example 3: Using Hostfile

Create `hosts.txt`:
```
server1 slots=1 max_slots=4
server2 slots=2 max_slots=4
server3 slots=1 max_slots=4
```

Run with hostfile:
```bash
mpirun -np 4 \
       -hostfile hosts.txt \
       -x SST_CORE_HOME \
       -x PATH \
       -x LD_LIBRARY_PATH \
       sst distributed_multi_device_test.py
```

**Note:** `-x` exports environment variables to remote processes

### Example 4: Using IP Addresses

```bash
# When DNS resolution doesn't work:
mpirun -np 2 \
       -host 192.168.1.10:1,192.168.1.11:1 \
       sst distributed_mmio_test.py
```

---

## Verification and Testing

### Step 1: Test Basic MPI

```bash
# Test MPI across servers (from head node):
mpirun -np 2 -host server1:1,server2:1 hostname

# Expected output:
# server1
# server2
```

### Step 2: Test MPI with Environment

```bash
# Verify environment variables propagate:
mpirun -np 2 \
       -host server1:1,server2:1 \
       -x SST_CORE_HOME \
       bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK: SST_CORE_HOME=$SST_CORE_HOME"'

# Expected output:
# Rank 0: SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
# Rank 1: SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
```

### Step 3: Test SST with MPI

```bash
# Create test configuration (test_ranks.py):
cat > test_ranks.py <<'EOF'
import sst
my_rank, num_ranks = sst.getMPIRankCount()
print(f"Hello from rank {my_rank} of {num_ranks}")
EOF

# Run test:
mpirun -np 2 -host server1:1,server2:1 sst test_ranks.py

# Expected output:
# Hello from rank 0 of 2
# Hello from rank 1 of 2
```

### Step 4: Run Actual Distributed Simulation

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal

# Run distributed MMIO test:
mpirun -np 2 \
       -host server1:1,server2:1 \
       -x SST_CORE_HOME \
       -x PATH \
       -x LD_LIBRARY_PATH \
       sst distributed_mmio_test.py
```

---

## Common Issues and Solutions

### Issue 1: "Host key verification failed"

**Symptom:**
```
Host key verification failed.
ORTE was unable to reliably start one or more daemons.
```

**Solution:**
```bash
# Accept host keys for all servers:
ssh server2  # Type 'yes'
ssh server3  # Type 'yes'
exit

# Or add to SSH config (~/.ssh/config):
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
```

### Issue 2: "Permission denied (publickey)"

**Symptom:**
```
Permission denied (publickey,password).
```

**Solution:**
```bash
# Re-setup SSH keys:
ssh-copy-id user@server2
ssh-copy-id user@server3

# Test:
ssh server2 "hostname"  # Should work without password

# Check permissions:
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
chmod 644 ~/.ssh/authorized_keys  # On remote servers
```

### Issue 3: "sst: command not found"

**Symptom:**
```
bash: sst: command not found
```

**Solution:**
```bash
# Export environment variables explicitly:
mpirun -np 2 \
       -host server1:1,server2:1 \
       -x SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install \
       -x PATH=$SST_CORE_HOME/bin:$PATH \
       -x LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH \
       sst config.py

# Or add to ~/.bashrc on ALL servers:
echo 'export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install' >> ~/.bashrc
echo 'export PATH=$SST_CORE_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue 4: Different MPI Versions

**Symptom:**
```
MPI version mismatch detected
```

**Solution:**
```bash
# Check MPI version on all servers:
ssh server1 "mpirun --version"
ssh server2 "mpirun --version"
ssh server3 "mpirun --version"

# All must match! If not, reinstall MPI:
# - Use same package manager
# - Use same version
# - Rebuild SST-Core with new MPI
```

### Issue 5: Firewall Blocking MPI Communication

**Symptom:**
```
Connection timed out during MPI initialization
```

**Solution:**
```bash
# Allow MPI ports through firewall (on all servers):
sudo ufw allow from 192.168.1.0/24
# Or disable firewall temporarily for testing:
sudo ufw disable

# For OpenMPI, specify network interface:
mpirun --mca btl_tcp_if_include eth0 \
       -np 2 -host server1:1,server2:1 \
       sst config.py
```

### Issue 6: File Not Found

**Symptom:**
```
Error: Cannot find config.py
```

**Solution:**
```bash
# Ensure files are synced on all servers:
# Method 1: Use absolute path
mpirun -np 2 -host server1:1,server2:1 \
  sst /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/config.py

# Method 2: Use same working directory on all servers
mpirun -np 2 -host server1:1,server2:1 \
  --wdir /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal \
  sst config.py

# Method 3: Sync files to all servers
rsync -avz project/ server2:~/project/
```

### Issue 7: QEMU Process Not Found

**Symptom:**
```
Error: QEMU binary not found at /path/to/qemu
```

**Solution:**
```bash
# QEMU must be on the server running rank 0 (QEMUBinary)
# If using distributed deployment, ensure QEMU is on head node:
ssh server1 "which qemu-system-riscv32"
# Should show path

# Update config to use correct path:
qemu.addParams({"qemu_binary": "/usr/local/bin/qemu-system-riscv32"})
```

---

## Performance Considerations

### Network Latency

MPI communication adds latency compared to local execution:

```python
# Adjust SST link latencies for distributed simulation:
# Local simulation:
link.connect((comp1, "port", "1ns"), (comp2, "port", "1ns"))

# Distributed simulation (cross-server):
link.connect((comp1, "port", "100ns"), (comp2, "port", "100ns"))
```

### Synchronization Overhead

More ranks = more synchronization:
- **Event serialization**: Cross-rank events must be serialized
- **Synchronization barriers**: SST synchronizes time across all ranks
- **Recommendation**: Keep tightly-coupled components on same rank

### Load Balancing

Distribute computational load evenly:

```python
# Good: Balance compute-heavy components across ranks
# Rank 0: QEMUBinary (light)
# Rank 1: ComputeDevice1 (heavy)
# Rank 2: ComputeDevice2 (heavy)

# Bad: All heavy components on one rank
# Rank 0: QEMUBinary + ComputeDevice1 + ComputeDevice2 (overloaded!)
# Rank 1: EchoDevice (idle)
```

### Communication Patterns

Minimize cross-rank communication:

```python
# Good: Group communicating components on same rank
if my_rank == 1:
    device1 = sst.Component("dev1", "acalsim.Device")
    device2 = sst.Component("dev2", "acalsim.Device")
    # peer_link connects devices on same rank - efficient!

# Suboptimal: Frequent communication across ranks
# device1 on rank1, device2 on rank2 → high MPI overhead
```

---

## Advanced Topics

### Resource Managers (SLURM, PBS)

For production HPC clusters:

**SLURM:**
```bash
#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=acalsim_distributed

# SLURM automatically sets up MPI environment
srun sst distributed_config.py
```

**PBS/Torque:**
```bash
#!/bin/bash
#PBS -l nodes=3:ppn=1
#PBS -l walltime=01:00:00
#PBS -N acalsim_distributed

cd $PBS_O_WORKDIR
mpirun sst distributed_config.py
```

### Monitoring Distributed Execution

**Watch processes on each server:**
```bash
# Terminal 1 (server1):
watch -n 1 "ps aux | grep sst"

# Terminal 2 (server2):
ssh server2 "watch -n 1 'ps aux | grep sst'"

# Monitor network traffic:
iftop -i eth0  # On each server
```

### MPI Debugging

```bash
# Enable MPI debug output:
mpirun --debug-devel -np 2 -host server1:1,server2:1 sst config.py

# Attach debugger to specific rank:
mpirun -np 2 \
  xterm -e gdb -ex "run" --args sst config.py

# Verbose MPI communication:
mpirun --mca btl_base_verbose 30 \
  -np 2 -host server1:1,server2:1 sst config.py
```

### Performance Profiling

```bash
# Time execution:
time mpirun -np 2 -host server1:1,server2:1 sst config.py

# Profile with SST stats:
# In config.py:
sst.setStatisticOutput("sst.statOutputCSV")
sst.setStatisticOutputOptions({"filepath": "stats.csv"})
```

---

## Quick Reference

### Essential Commands

```bash
# Test MPI connectivity:
mpirun -np 2 -host server1:1,server2:1 hostname

# Test SSH:
ssh server2 "hostname"

# Run distributed simulation:
mpirun -np 2 \
  -host server1:1,server2:1 \
  -x SST_CORE_HOME \
  sst config.py

# With hostfile:
mpirun -np 4 -hostfile hosts.txt sst config.py

# Debug mode:
mpirun --debug-devel -np 2 -host server1:1,server2:1 sst config.py
```

### Checklist Before Running

- [ ] Passwordless SSH working to all compute nodes
- [ ] Same SST-Core version on all servers
- [ ] Same MPI version on all servers
- [ ] Same ACALSim installation on all servers
- [ ] Environment variables set on all servers
- [ ] Network connectivity verified (ping, SSH)
- [ ] Firewall allows MPI communication
- [ ] Project files synced to all servers
- [ ] SST components built and installed on all servers

---

## Related Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - General getting started guide
- [README_MMIO_DEVICE.md](acalsim-device/README_MMIO_DEVICE.md) - Distributed simulation with MMIO devices
- [README.md](README.md) - Main project documentation
- [SST-Core MPI Documentation](http://sst-simulator.org/)

---

## Support

If you encounter issues not covered in this guide:

1. Check firewall and network connectivity
2. Verify all prerequisites are met
3. Test with simple MPI programs first
4. Check SST-Core documentation for MPI-specific issues
5. Report issues at https://github.com/anthropics/acalsim/issues

---

**Last Updated:** 2025-11-10
