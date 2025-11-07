# Copyright 2023-2025 Playlab/ACAL
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

# ========== ACALSim SST Integration Makefile ==========

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -Wall -Wextra -g -O2

# SST configuration
SST_CONFIG = sst-config
SST_CXXFLAGS = $(shell $(SST_CONFIG) --ELEMENT_CXXFLAGS)
SST_LDFLAGS = $(shell $(SST_CONFIG) --ELEMENT_LDFLAGS)

# ACALSim paths
ACALSIM_ROOT = ../..
ACALSIM_INCLUDE = $(ACALSIM_ROOT)/include
ACALSIM_LIB = $(ACALSIM_ROOT)/lib

# RISC-V paths
RISCV_ROOT = $(ACALSIM_ROOT)/src/riscv
RISCV_INCLUDE = $(RISCV_ROOT)/include
RISCV_LIBS = $(RISCV_ROOT)/libs

# Include paths
INCLUDES = -I$(ACALSIM_INCLUDE) \
           -I$(ACALSIM_INCLUDE)/external \
           -I$(ACALSIM_INCLUDE)/sim \
           -I$(ACALSIM_INCLUDE)/event \
           -I$(ACALSIM_INCLUDE)/packet \
           -I$(ACALSIM_INCLUDE)/port \
           -I$(ACALSIM_INCLUDE)/channel \
           -I$(RISCV_INCLUDE) \
           -I$(RISCV_INCLUDE)/event \
           -I. \
           $(SST_CXXFLAGS)

# Library paths and dependencies
LIBS = -L$(ACALSIM_LIB) -lacalsim $(SST_LDFLAGS)

# Target library name
LIBRARY = libacalsim_sst.so

# Source files
SOURCES = ACALSimComponent.cc \
          RISCVSoCComponent.cc \
          RISCVSoCStandalone.cc

# RISC-V library source files
RISCV_SOURCES = $(RISCV_LIBS)/BaseMemory.cc \
                $(RISCV_LIBS)/CPU.cc \
                $(RISCV_LIBS)/DataMemory.cc \
                $(RISCV_LIBS)/Emulator.cc \
                $(RISCV_LIBS)/EXEStage.cc \
                $(RISCV_LIBS)/IFStage.cc \
                $(RISCV_LIBS)/InstPacket.cc \
                $(RISCV_LIBS)/MemPacket.cc \
                $(RISCV_LIBS)/SOC.cc \
                $(RISCV_LIBS)/TopPipeRegisterManager.cc

# Object files
OBJECTS = $(SOURCES:.cc=.o)
RISCV_OBJECTS = $(RISCV_SOURCES:.cc=.o)

# SST element registration
ELEMENT_NAME = acalsim
ELEMENT_LIB = $(ELEMENT_NAME)

# ========== Build Targets ==========

.PHONY: all clean install test examples help

all: $(LIBRARY)

# Build the SST element library
$(LIBRARY): $(OBJECTS) $(RISCV_OBJECTS)
	@echo "Linking SST element library: $@"
	$(CXX) -shared -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

# Compile source files
%.o: %.cc
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Install to SST
install: $(LIBRARY)
	@echo "Installing ACALSim SST element..."
	@SST_LIB_PATH=$$($(SST_CONFIG) --prefix)/lib/sst-elements-library; \
	if [ ! -d "$$SST_LIB_PATH" ]; then \
		echo "Creating SST library directory: $$SST_LIB_PATH"; \
		mkdir -p "$$SST_LIB_PATH"; \
	fi; \
	echo "Copying $(LIBRARY) to $$SST_LIB_PATH/"; \
	cp $(LIBRARY) "$$SST_LIB_PATH/"; \
	echo "Installation complete!"
	@echo ""
	@echo "To verify installation, run: sst-info $(ELEMENT_NAME)"

# Test the installation
test: install
	@echo "Testing SST element registration..."
	sst-info $(ELEMENT_NAME)

# Build examples
examples:
	@echo "Example Python configurations are in: examples/"
	@echo "To run an example:"
	@echo "  sst examples/simple_system.py"
	@echo "  sst examples/multicore_system.py"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(RISCV_OBJECTS) $(LIBRARY)
	rm -f *.o *.so
	@echo "Clean complete."

# Help target
help:
	@echo "ACALSim SST Integration Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the SST element library (default)"
	@echo "  install   - Install the library to SST"
	@echo "  test      - Test the installation with sst-info"
	@echo "  examples  - Show how to run example configurations"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - SST-Core installed and sst-config in PATH"
	@echo "  - ACALSim built (lib/libacalsim.a or lib/libacalsim.so)"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build the library"
	@echo "  make install      # Build and install to SST"
	@echo "  make test         # Verify installation"
	@echo "  sst examples/simple_system.py  # Run example"

# Build examples
examples-riscv:
	@echo "RISC-V Example Python configurations:"
	@echo "  sst examples/riscv_single_core.py"
	@echo "  sst examples/riscv_dual_core.py"

# ========== Dependencies ==========

ACALSimComponent.o: ACALSimComponent.cc ACALSimComponent.hh
RISCVSoCComponent.o: RISCVSoCComponent.cc RISCVSoCComponent.hh
RISCVSoCStandalone.o: RISCVSoCStandalone.cc RISCVSoCStandalone.hh
