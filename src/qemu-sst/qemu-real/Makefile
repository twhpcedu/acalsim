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

# QEMU Real Component Makefile

CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS) -std=c++17 -Wall -Wextra
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

SST_INSTALL_DIR = $(shell sst-config --prefix)
SST_ELEMENT_DIR = $(SST_INSTALL_DIR)/lib/sstcore

COMPONENT_NAME = libqemureal.so
SOURCES = QEMURealComponent.cc
OBJECTS = $(SOURCES:.cc=.o)
HEADERS = QEMURealComponent.hh

all: $(COMPONENT_NAME)

$(COMPONENT_NAME): $(OBJECTS)
	@echo "Linking $(COMPONENT_NAME)..."
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^
	@echo "Build complete: $(COMPONENT_NAME)"

%.o: %.cc $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

install: $(COMPONENT_NAME)
	@echo "Installing to $(SST_ELEMENT_DIR)..."
	@mkdir -p $(SST_ELEMENT_DIR)
	@cp $(COMPONENT_NAME) $(SST_ELEMENT_DIR)/
	@echo "Installation complete"

clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(COMPONENT_NAME)

.PHONY: all install clean
