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

# Running QEMU-ACALSim in Docker Container

This guide explains how to build and run the QEMU-ACALSim distributed simulation in the `acalsim-workspace` Docker container.

## Quick Start (Docker Container)

### Method 1: Automated Build Script (Recommended)

The `build.sh` script automatically detects the Docker environment and sets up SST:

```bash
# Inside Docker container
cd ~/projects/acalsim/src/qemu-sst

# One command to build, install, and run
./build.sh

# Or step by step
./build.sh prereq    # Check prerequisites
./build.sh build     # Build components
./build.sh install   # Install to SST
./build.sh run       # Run simulation
```

### Method 2: Manual Environment Setup

If you prefer manual control:

```bash
# Inside Docker container
cd ~/projects/acalsim/src/qemu-sst

# Source the environment setup script
source ./setup-env.sh

# Now use make or build.sh
make build
make install
make run
```

### Method 3: Using Make with Environment Variables

```bash
# Inside Docker container
cd ~/projects/acalsim/src/qemu-sst

# Set environment variables
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Now use make
make verify-sst
make build
make install
make run
```

## SST Installation in Docker Container

### Prerequisites: Install SST-Core First

Before building the QEMU-ACALSim project, you need to install SST-Core. Follow these steps from the main project's SST-INTEGRATION.md:

```bash
# Inside Docker container
cd /home/user/projects/acalsim

# Install dependencies (if not already installed)
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev \
                    libtool libtool-bin autoconf python3 python3-dev \
                    automake build-essential git

# Create sst-core directory and clone
mkdir -p sst-core && cd sst-core
git clone https://github.com/sstsimulator/sst-core.git sst-core-src

# Build and install SST
cd sst-core-src && ./autogen.sh && cd ..
mkdir -p build && cd build
../sst-core-src/configure --prefix=$PWD/../sst-core-install
make -j$(nproc)
make install

# Set environment variables
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
export PATH=$SST_CORE_HOME/bin:$PATH

# Verify installation
sst --version
```

After installation, SST will be located at:
- **Installation directory**: `/home/user/projects/acalsim/sst-core/sst-core-install`
- **Binaries**: `/home/user/projects/acalsim/sst-core/sst-core-install/bin`
- **Libraries**: `/home/user/projects/acalsim/sst-core/sst-core-install/lib/sstcore`

## Verifying SST Setup

After sourcing the environment or running build.sh:

```bash
# Check SST is accessible
which sst
# Should show: /home/user/projects/acalsim/sst-core/sst-core-install/bin/sst

# Check SST version
sst --version

# Check sst-config
which sst-config

# List SST info
sst-info
```

## Building Components

### Build Both Components

```bash
cd ~/projects/acalsim/src/qemu-sst

# Method 1: Using build script (auto-detects Docker)
./build.sh build

# Method 2: Using make (after environment setup)
make build
```

### Build Individual Components

```bash
# Build QEMU component
cd qemu-component
make clean
make
make install

# Build ACALSim device component
cd ../acalsim-device
make clean
make
make install
```

## Verifying Installation

After installation, verify components are registered with SST:

```bash
# List all SST elements
sst-info

# Check QEMU component
sst-info qemu

# Check ACALSim device component
sst-info acalsim
```

Expected output:
```
qemu.RISCV (1.0.0)
  QEMU RISC-V emulator wrapper for distributed simulation

acalsim.QEMUDevice (1.0.0)
  ACALSim memory-mapped device for QEMU integration
```

## Running the Simulation

### Using build.sh

```bash
cd ~/projects/acalsim/src/qemu-sst
./build.sh run
```

### Using make

```bash
cd ~/projects/acalsim/src/qemu-sst
make run
```

### Manual execution

```bash
cd ~/projects/acalsim/src/qemu-sst/config
mpirun -n 2 sst echo_device.py
```

## Expected Output

Successful simulation output:

```
SST Configuration: Rank 0 of 2
Rank 0: Creating QEMU component

======================================================================
QEMU-ACALSim Distributed Simulation Configuration
======================================================================
Clock Frequency:      1GHz
Device Base Address:  0x10000000
...

QEMU[0:0]: === Starting Test Iteration 1 ===
QEMU[0:0]: Writing pattern 0xDEADBEEF to DATA_IN
ACALSimDevice[1:0]: Received STORE request
...
QEMU[0:0]: ✓ Test iteration 1 PASSED

...

QEMU[0:0]: *** TEST PASSED ***
```

## Troubleshooting

### Problem: "SST not found in PATH"

**Solution**: The build.sh script should auto-detect this, but if it doesn't:

```bash
# Manually set environment
source ./setup-env.sh

# Or export variables
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
```

### Problem: "Component not found" after installation

**Solution**: Check installation directory:

```bash
# List installed elements
ls -la $SST_CORE_HOME/lib/sst-elements-library/

# Should see:
# libqemu.so
# libacalsim.so
```

If not present, manually install:

```bash
cd qemu-component && make install
cd ../acalsim-device && make install
```

### Problem: MPI errors

**Solution**: Ensure MPI is installed in container:

```bash
# Check MPI
which mpirun
mpirun --version

# If missing, install
sudo apt-get update
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
```

### Problem: "Cannot find component qemu.RISCV"

**Solution**: Verify component name in SST:

```bash
# List all components
sst-info | grep -A5 "qemu"
sst-info | grep -A5 "acalsim"

# Check element library path
echo $SST_CORE_HOME/lib/sst-elements-library
ls -la $SST_CORE_HOME/lib/sst-elements-library/libqemu.so
```

## Permanent Environment Setup (Optional)

To avoid sourcing setup-env.sh every time:

### Option 1: Add to .bashrc

```bash
# Add these lines to ~/.bashrc
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

### Option 2: Create alias

```bash
# Add to ~/.bashrc
alias setup-sst='source /home/user/projects/acalsim/src/qemu-sst/setup-env.sh'

# Usage
setup-sst
```

## Development Workflow in Docker

Typical development cycle:

```bash
# 1. Enter Docker container (if not already inside)
docker exec -it acalsim-workspace bash

# 2. Navigate to project
cd ~/projects/acalsim/src/qemu-sst

# 3. Setup environment (if not permanent)
source ./setup-env.sh

# 4. Edit source files
vim qemu-component/QEMUComponent.cc

# 5. Rebuild and test
cd qemu-component
make clean && make && make install

# 6. Run simulation
cd ../config
mpirun -n 2 sst echo_device.py

# 7. Or use build script for full rebuild
cd ..
./build.sh install
./build.sh run
```

## Running from Host Machine

To run from host (outside container):

```bash
# From host machine
docker exec acalsim-workspace bash -c "\
  cd /home/user/projects/acalsim/src/qemu-sst && \
  source ./setup-env.sh && \
  ./build.sh all"
```

## See Also

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - General quick start guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
