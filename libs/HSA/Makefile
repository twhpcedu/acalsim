# HSA Shared Library Makefile
# Copyright 2023-2025 Playlab/ACAL

# SST Configuration
SST_CONFIG = sst-config
CXX = $(shell $(SST_CONFIG) --CXX)
CXXFLAGS = $(shell $(SST_CONFIG) --ELEMENT_CXXFLAGS) -std=c++17
LDFLAGS = $(shell $(SST_CONFIG) --ELEMENT_LDFLAGS)

# Directories
INCLUDE_DIR = ../../include
HSA_INCLUDE = $(INCLUDE_DIR)/HSA
SRC_DIR = .
BUILD_DIR = ../../build/HSA
INSTALL_DIR = $(shell $(SST_CONFIG) --prefix)/lib/sstcore

# Source files
SOURCES = HSAHostComponent.cc HSAComputeComponent.cc
OBJECTS = $(SOURCES:%.cc=$(BUILD_DIR)/%.o)

# Library output
LIBRARY = libacalsim_hsa.so

# Include paths
INCLUDES = -I$(INCLUDE_DIR) -I$(HSA_INCLUDE)

# Build rules
all: $(BUILD_DIR) $(LIBRARY)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cc
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIBRARY): $(OBJECTS)
	@echo "Linking $(LIBRARY)..."
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -fPIC -o $@ $(OBJECTS)
	@echo "Build complete: $(LIBRARY)"

install: $(LIBRARY)
	@echo "Installing $(LIBRARY) to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	@cp $(LIBRARY) $(INSTALL_DIR)/
	@echo "Installation complete"

clean:
	@echo "Cleaning HSA library..."
	@rm -rf $(BUILD_DIR) $(LIBRARY)
	@echo "Clean complete"

.PHONY: all install clean
