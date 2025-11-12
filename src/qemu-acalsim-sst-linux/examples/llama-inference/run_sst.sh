#!/bin/bash
"""
Launch SST Simulation for LLAMA 2 Inference

This script sets up the SST environment and launches the simulation
with the LLAMA accelerator configuration.

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "LLAMA 2 Inference - SST Simulation Launcher"
echo "============================================================"
echo ""

# Check if running inside Docker container
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Running inside Docker container"
    IN_DOCKER=true
else
    echo -e "${YELLOW}⚠${NC}  Not running in Docker container"
    echo "This script is designed to run inside the acalsim-workspace container"
    echo ""
    echo "To run inside Docker:"
    echo "  docker exec -it acalsim-workspace bash"
    echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
    echo "  ./run_sst.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    IN_DOCKER=false
fi

# Set up SST environment
if [ "$IN_DOCKER" = true ]; then
    # Docker paths
    export SST_CORE_HOME=${SST_CORE_HOME:-/home/user/projects/acalsim/sst-core/sst-core-install}
    SST_CONFIG_FILE="sst_config_llama.py"
else
    # Local paths (adjust as needed)
    export SST_CORE_HOME=${SST_CORE_HOME:-$HOME/projects/acalsim/sst-core/sst-core-install}
    SST_CONFIG_FILE="sst_config_llama.py"
fi

export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

echo "Environment:"
echo "  SST_CORE_HOME: $SST_CORE_HOME"
echo "  PATH: $PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check if sst command is available
if ! command -v sst &> /dev/null; then
    echo -e "${RED}✗${NC} SST command not found!"
    echo ""
    echo "Please ensure SST-Core is built and installed at:"
    echo "  $SST_CORE_HOME"
    echo ""
    echo "Build instructions:"
    echo "  cd /home/user/projects/acalsim/sst-core"
    echo "  ./build.sh"
    exit 1
fi

echo -e "${GREEN}✓${NC} SST command found: $(which sst)"
echo ""

# Check if configuration file exists
if [ ! -f "$SST_CONFIG_FILE" ]; then
    echo -e "${RED}✗${NC} SST configuration file not found: $SST_CONFIG_FILE"
    echo ""
    echo "Expected location:"
    echo "  $(pwd)/$SST_CONFIG_FILE"
    exit 1
fi

echo -e "${GREEN}✓${NC} SST configuration: $SST_CONFIG_FILE"
echo ""

# Check if SST components are built
echo "Checking SST components..."
if [ -d "$SST_CORE_HOME/lib/sstcore" ]; then
    ACALSIM_LIB=$(find "$SST_CORE_HOME/lib/sstcore" -name "libacalsim.so" 2>/dev/null | head -1)
    if [ -n "$ACALSIM_LIB" ]; then
        echo -e "${GREEN}✓${NC} ACALSim components found: $ACALSIM_LIB"
    else
        echo -e "${YELLOW}⚠${NC}  ACALSim components not found in $SST_CORE_HOME/lib/sstcore"
        echo ""
        echo "Build components with:"
        echo "  cd /home/user/projects/acalsim/src/sst-integration"
        echo "  make clean && make && make install"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo ""
echo "============================================================"
echo "Starting SST Simulation"
echo "============================================================"
echo ""
echo "Configuration: $SST_CONFIG_FILE"
echo "Socket: /tmp/qemu-sst-llama.sock"
echo ""
echo "Next steps:"
echo "  1. Wait for 'Waiting for QEMU to connect...'"
echo "  2. In another terminal, run: ./run_qemu.sh"
echo ""
echo "Press Ctrl+C to stop the simulation"
echo ""
echo "============================================================"
echo ""

# Run SST simulation
sst --verbose $SST_CONFIG_FILE

# Cleanup (if we get here, simulation finished)
echo ""
echo "============================================================"
echo "SST simulation completed"
echo "============================================================"
