#!/bin/bash

# Copyright 2023-2026 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# QEMU-ACALSim Build and Test Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup SST environment if not already configured (disable exit-on-error for this check)
set +e
command -v sst &>/dev/null
SST_IN_PATH=$?
set -e

if [ $SST_IN_PATH -ne 0 ]; then
	# Try to find SST in known locations
	SST_LOCATIONS=(
		"/home/user/projects/acalsim/sst-core/sst-core-install"
		"/home/user/projects/acalsim/build/sst-core/sst-core-install"
	)

	for location in "${SST_LOCATIONS[@]}"; do
		if [ -d "$location" ] && [ -f "$location/bin/sst" ]; then
			echo "Setting up SST environment from $location..."
			export SST_CORE_HOME=$location
			export PATH=$SST_CORE_HOME/bin:$PATH
			export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
			break
		fi
	done
fi

set -e # Exit on error for rest of script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
	echo ""
	echo -e "${BLUE}====================================================================${NC}"
	echo -e "${BLUE}$1${NC}"
	echo -e "${BLUE}====================================================================${NC}"
	echo ""
}

print_success() {
	echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
	echo -e "${RED}✗ $1${NC}"
}

print_info() {
	echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
	print_header "Checking Prerequisites"

	# Check for SST
	if ! command -v sst &>/dev/null; then
		print_error "SST not found in PATH"
		echo ""
		echo "SST-Core must be installed before building QEMU-ACALSim components."
		echo ""
		echo "Installation instructions:"
		echo "  1. See: /home/user/projects/acalsim/SST-INTEGRATION.md (Step 0)"
		echo "  2. Or: src/qemu-sst/DOCKER.md"
		echo ""
		echo "Quick install (in Docker container):"
		echo "  cd /home/user/projects/acalsim"
		echo "  mkdir -p sst-core && cd sst-core"
		echo "  git clone https://github.com/sstsimulator/sst-core.git sst-core-src"
		echo "  cd sst-core-src && ./autogen.sh && cd .."
		echo "  mkdir build && cd build"
		echo "  ../sst-core-src/configure --prefix=\$PWD/../sst-core-install"
		echo "  make -j\$(nproc) && make install"
		echo ""
		echo "After installation, run this script again."
		echo ""
		exit 1
	fi
	print_success "SST found: $(sst --version 2>&1 | head -1)"

	# Check for sst-config
	if ! command -v sst-config &>/dev/null; then
		print_error "sst-config not found in PATH"
		exit 1
	fi
	print_success "sst-config found"

	# Check for MPI
	if ! command -v mpirun &>/dev/null; then
		print_error "MPI not found (mpirun not in PATH)"
		print_info "Install OpenMPI or MPICH"
		exit 1
	fi
	print_success "MPI found: $(mpirun --version 2>&1 | head -1)"

	# Check for C++ compiler
	CXX_FULL=$(sst-config --CXX)
	# Extract just the compiler name (first word)
	CXX=$(echo $CXX_FULL | awk '{print $1}')
	if ! command -v "$CXX" &>/dev/null; then
		print_error "C++ compiler not found: $CXX"
		exit 1
	fi
	print_success "C++ compiler found: $CXX"
}

# Build components
build_components() {
	print_header "Building Components"

	# Build QEMU component
	echo "Building QEMU component..."
	cd qemu-component
	make clean
	make
	cd ..
	print_success "QEMU component built"

	# Build ACALSim device component
	echo "Building ACALSim device component..."
	cd acalsim-device
	make clean
	make
	cd ..
	print_success "ACALSim device component built"
}

# Install components
install_components() {
	print_header "Installing Components"

	# Install QEMU component
	echo "Installing QEMU component..."
	cd qemu-component
	make install
	cd ..

	# Install ACALSim device component
	echo "Installing ACALSim device component..."
	cd acalsim-device
	make install
	cd ..

	print_success "All components installed to SST"
}

# Verify installation
verify_installation() {
	print_header "Verifying Installation"

	# Check QEMU component
	if sst-info qemu >/dev/null 2>&1; then
		print_success "QEMU component registered with SST"
	else
		print_error "QEMU component not found in SST"
		return 1
	fi

	# Check ACALSim device component
	if sst-info acalsim >/dev/null 2>&1; then
		print_success "ACALSim device component registered with SST"
	else
		print_error "ACALSim device component not found in SST"
		return 1
	fi
}

# Run simulation
run_simulation() {
	print_header "Running Simulation"

	cd config
	print_info "Starting distributed simulation with 2 MPI ranks..."
	echo ""

	if mpirun -n 2 sst echo_device.py; then
		echo ""
		print_success "Simulation completed successfully"
	else
		echo ""
		print_error "Simulation failed"
		cd ..
		return 1
	fi
	cd ..
}

# Main script
main() {
	# Get script directory
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	cd "$SCRIPT_DIR"

	print_header "QEMU-ACALSim Build and Test Script"

	# Parse command line arguments
	case "${1:-all}" in
	prereq)
		check_prerequisites
		;;
	build)
		check_prerequisites
		build_components
		;;
	install)
		check_prerequisites
		build_components
		install_components
		verify_installation
		;;
	verify)
		verify_installation
		;;
	run)
		run_simulation
		;;
	all)
		check_prerequisites
		build_components
		install_components
		verify_installation
		run_simulation
		;;
	clean)
		print_header "Cleaning Build Artifacts"
		cd qemu-component && make clean && cd ..
		cd acalsim-device && make clean && cd ..
		print_success "Clean complete"
		;;
	help | *)
		echo "Usage: $0 [command]"
		echo ""
		echo "Commands:"
		echo "  all       - Run all steps (default)"
		echo "  prereq    - Check prerequisites only"
		echo "  build     - Build components only"
		echo "  install   - Build and install components"
		echo "  verify    - Verify installation"
		echo "  run       - Run simulation"
		echo "  clean     - Clean build artifacts"
		echo "  help      - Show this help"
		echo ""
		echo "Examples:"
		echo "  $0              # Run all steps"
		echo "  $0 build        # Build only"
		echo "  $0 install      # Build and install"
		echo "  $0 run          # Run simulation"
		;;
	esac

	if [ $? -eq 0 ]; then
		echo ""
		print_header "Success!"
		echo "Next steps:"
		echo "  - Review output above for any warnings"
		echo "  - See README.md for detailed documentation"
		echo "  - See QUICKSTART.md for usage guide"
		echo ""
	else
		echo ""
		print_header "Build/Test Failed"
		echo "Check error messages above for details"
		echo ""
		exit 1
	fi
}

# Run main function
main "$@"
