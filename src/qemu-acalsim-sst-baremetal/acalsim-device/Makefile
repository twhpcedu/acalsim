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

# ACALSim Device SST Component Makefile

CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS) -std=c++17 -Wall -Wextra
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

# SST installation directories
SST_INSTALL_DIR = $(shell sst-config --prefix)
SST_ELEMENT_DIR = $(SST_INSTALL_DIR)/lib/sstcore

# Shared framework directories
FRAMEWORK_ROOT = ../../..
HSA_SRC_DIR = $(FRAMEWORK_ROOT)/libs/HSA
HSA_INCLUDE_DIR = $(FRAMEWORK_ROOT)/include

# Include paths
INCLUDES = -I$(HSA_INCLUDE_DIR)

# Component name and files
COMPONENT_NAME = libacalsim.so

# Local device sources
LOCAL_SOURCES = ACALSimDeviceComponent.cc ACALSimComputeDeviceComponent.cc ACALSimMMIODevice.cc
LOCAL_HEADERS = ACALSimDeviceComponent.hh ACALSimComputeDeviceComponent.hh ACALSimMMIODevice.hh

# Shared HSA framework sources
HSA_SOURCES = $(HSA_SRC_DIR)/HSAHostComponent.cc $(HSA_SRC_DIR)/HSAComputeComponent.cc

# All sources and objects
SOURCES = $(LOCAL_SOURCES) $(HSA_SOURCES)
OBJECTS = $(LOCAL_SOURCES:.cc=.o) HSAHostComponent.o HSAComputeComponent.o
HEADERS = $(LOCAL_HEADERS)

# Default target
all: $(COMPONENT_NAME)

# Build shared library
$(COMPONENT_NAME): $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

# Compile local source files
%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Compile HSA framework sources
HSAHostComponent.o: $(HSA_SRC_DIR)/HSAHostComponent.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fPIC -c $< -o $@

HSAComputeComponent.o: $(HSA_SRC_DIR)/HSAComputeComponent.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Install to SST element library directory
install: $(COMPONENT_NAME)
	@echo "Installing $(COMPONENT_NAME) to $(SST_ELEMENT_DIR)"
	@mkdir -p $(SST_ELEMENT_DIR)
	@cp $(COMPONENT_NAME) $(SST_ELEMENT_DIR)/
	@echo "Installation complete"
	@echo "You can now use these components in SST:"
	@echo "  - sst.Component(name, \"acalsim.QEMUDevice\")      # Echo device"
	@echo "  - sst.Component(name, \"acalsim.ComputeDevice\")  # Compute device"
	@echo "  - sst.Component(name, \"acalsim.MMIODevice\")     # MMIO device with interrupts"
	@echo "  - sst.Component(name, \"acalsim.HSAHost\")        # HSA host agent"
	@echo "  - sst.Component(name, \"acalsim.HSACompute\")     # HSA compute agent"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(COMPONENT_NAME)

# Uninstall from SST
uninstall:
	rm -f $(SST_ELEMENT_DIR)/$(COMPONENT_NAME)

# Help target
help:
	@echo "ACALSim Device SST Component Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the component (default)"
	@echo "  install    - Install to SST element library"
	@echo "  clean      - Remove build artifacts"
	@echo "  uninstall  - Remove from SST element library"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make           # Build component"
	@echo "  make install   # Install to SST"
	@echo "  make clean     # Clean build files"

.PHONY: all install clean uninstall help
