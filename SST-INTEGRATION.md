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

🚀 Quick Start Guide

  Step 0: install open sst-config
  # install open MPI in ubuntu
  sudo apt update;sudo apt install openmpi-bin openmpi-common libopenmpi-dev

  # install sst-config
  DEBIAN_FRONTEND=noninteractive sudo apt install openmpi-bin openmpi-common libtool libtool-bin autoconf python3 python3-dev automake build-essential git
  mkdir sst-core && cd sst-core
  git clone https://github.com/sstsimulator/sst-core.git sst-core-src
  (cd sst-core-src && ./autogen.sh)
  mkdir build && cd build
  ../sst-core-src/configure --prefix=$PWD/../sst-core-install
  make install
  export LD_LIBRARY_PATH=/home/user/projects/acalsim/build/sst-core/sst-core-install/lib/sstcore:$LD_LIBRARY_PATH
  export PATH=/home/user/projects/acalsim/sst-core/build/../sst-core-install/bin:$PATH  

  # verify the installation
  sst --version

  Step 1: Build ACALSim (if not already built)

  # Navigate to ACALSim root
  cd /Users/weifen/work/acal/acalsim-workspace/projects/acalsim

  # Build ACALSim library
  make

  # Verify the library was created
  ls lib/libacalsim.*
  # Should show: lib/libacalsim.a or lib/libacalsim.so

  Step 2: Build and Install SST Integration

  # Navigate to SST integration directory
  cd src/sst-riscv

  # Clean any previous builds
  make clean

  # Build the SST element library (includes RISC-V)
  make

  # Install to SST
  make install

  Expected output:
  Compiling: ACALSimComponent.cc
  Compiling: RISCVSoCComponent.cc
  Compiling: RISCVSoCStandalone.cc
  Compiling: ../../src/riscv/libs/BaseMemory.cc
  Compiling: ../../src/riscv/libs/CPU.cc
  ...
  Linking SST element library: libacalsim_sst.so
  Build complete: libacalsim_sst.so
  Installing ACALSim SST element...
  Installation complete!

  Step 3: Verify Installation

  # Check if SST can see the ACALSim element
  sst-info acalsim

  Expected output:
  Registered Components:
    acalsim.ACALSimComponent
    acalsim.RISCVSoCStandalone
    acalsim.SimpleProcessor

  Step 4: Run RISC-V Examples

  Option 1: Single RISC-V Core

  # Run single RISC-V processor with branch test program
  sst examples/riscv_single_core.py

  Option 2: Dual RISC-V Cores

  # Run dual RISC-V processors with different programs
  sst examples/riscv_dual_core.py

  Option 3: Run with Different Assembly Program

  Edit examples/riscv_single_core.py and change the ASM_FILE variable:

  # Available programs:
  ASM_FILE = "../../src/riscv/asm/branch_simple.txt"      # Branch tests
  # ASM_FILE = "../../src/riscv/asm/load_store_simple.txt"  # Memory tests
  # ASM_FILE = "../../src/riscv/asm/full_test.txt"          # Full ISA test
  # ASM_FILE = "../../src/riscv/asm/test.txt"               # Basic arithmetic

  Then run:
  sst examples/riscv_single_core.py

  📋 Expected Output

  When you run sst examples/riscv_single_core.py, you should see:

  Creating RISC-V RV32I Single-Core System...
  ====================================================================
  RISC-V RV32I Single-Core Configuration
  ====================================================================
  Clock frequency:     1GHz
  Max cycles:          100000
  Memory size:         65536 bytes (64KB)
  Text segment:        0x00000000
  Data segment:        0x00002000
  Assembly program:    ../../src/riscv/asm/branch_simple.txt
  Verbosity level:     2
  ====================================================================

  Running RISC-V simulation...
  The simulator will execute instructions until:
    1. HCF (Halt and Catch Fire) instruction is encountered
    2. Maximum cycles (100000) is reached
    3. No more events to process

  RISCVSoCStandalone[@p:@l]: Initializing Standalone RISC-V SoC
  RISCVSoCStandalone[@p:@l]: Assembly file: ../../src/riscv/asm/branch_simple.txt
  RISCVSoCStandalone[@p:@l]: Clock: 1GHz
  RISCVSoCStandalone[@p:@l]: Creating SOCTop instance
  RISCVSoCStandalone[@p:@l]: Initializing RISC-V simulator
  RISCVSoCStandalone[@p:@l]: Standalone RISC-V SoC initialized
  ...
  RISCVSoCStandalone[@p:@l]: Finish phase
  RISCVSoCStandalone[@p:@l]: Finalizing RISC-V simulator

  === RISC-V Simulation Complete ===
  Total cycles: 156
  ==================================

  🔧 Troubleshooting

  Problem 1: sst-config: command not found

  Solution: Install SST-Core first

  # Check if SST is installed
  which sst-config

  # If not found, install SST-Core
  # See: https://github.com/sstsimulator/sst-core

  Problem 2: Assembly file not found

  Solution: Check the path to assembly files

  # From src/sst-riscv directory, verify assembly files exist
  ls ../../src/riscv/asm/

  # Should show:
  # branch_simple.txt
  # full_test.txt
  # load_store_simple.txt
  # test.txt

  If the path is wrong, use absolute path in the Python file:
  ASM_FILE = "/Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/riscv/asm/branch_simple.txt"

  Problem 3: libacalsim.so: cannot open shared object file

  Solution: Rebuild ACALSim

  cd /Users/weifen/work/acal/acalsim-workspace/projects/acalsim
  make clean
  make
  cd src/sst-riscv
  make clean
  make install

  Problem 4: Element 'acalsim' not found

  Solution: Reinstall the SST element

  cd src/sst-riscv
  make clean
  make install
  sst-info acalsim  # Verify

  📝 Customizing Your Simulation

  Change Clock Frequency

  Edit examples/riscv_single_core.py:
  CLOCK_FREQ = "2GHz"  # Change from 1GHz to 2GHz

  Change Memory Size

  MEMORY_SIZE = 131072  # Change from 64KB to 128KB

  Change Maximum Cycles

  MAX_CYCLES = 200000  # Run longer

  Increase Verbosity

  VERBOSE = 5  # Maximum detail (0-5)

  🎯 Next Steps

  1. Try all assembly programs:
  # Edit riscv_single_core.py and try each program
  sst examples/riscv_single_core.py
  2. Run dual core:
  sst examples/riscv_dual_core.py
  3. Write your own assembly program:
    - Create a new .txt file in src/riscv/asm/
    - Follow the format in existing files
    - Update ASM_FILE in the Python config
    - Run it!
  4. Check the comprehensive documentation:
  cat examples/RISCV_README.md


