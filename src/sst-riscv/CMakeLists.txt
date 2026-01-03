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

cmake_minimum_required(VERSION 3.12)

project(ACALSim_SST_Integration VERSION 1.0.0 LANGUAGES CXX)

# Make SST integration optional
option(ENABLE_SST "Enable SST integration" OFF)

if(NOT ENABLE_SST)
    message(STATUS "SST integration disabled (set -DENABLE_SST=ON to enable)")
    return()
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ========== Find SST-Core ==========

# Find sst-config executable
find_program(SST_CONFIG sst-config)
if(NOT SST_CONFIG)
    message(
        WARNING
        "sst-config not found. SST integration will be disabled. Install SST-Core to enable."
    )
    return()
endif()

message(STATUS "Found sst-config: ${SST_CONFIG}")

# Get SST compiler flags
execute_process(
    COMMAND ${SST_CONFIG} --ELEMENT_CXXFLAGS
    OUTPUT_VARIABLE SST_CXXFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get SST linker flags
execute_process(
    COMMAND ${SST_CONFIG} --ELEMENT_LDFLAGS
    OUTPUT_VARIABLE SST_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get SST include directory
execute_process(
    COMMAND ${SST_CONFIG} --includedir
    OUTPUT_VARIABLE SST_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get SST library directory
execute_process(
    COMMAND ${SST_CONFIG} --libdir
    OUTPUT_VARIABLE SST_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "SST Include Dir: ${SST_INCLUDE_DIR}")
message(STATUS "SST Library Dir: ${SST_LIB_DIR}")

# ========== Find ACALSim ==========

# Assume ACALSim is built and installed
set(ACALSIM_ROOT ${PROJECT_SOURCE_DIR}/../..)
set(ACALSIM_INCLUDE_DIR ${ACALSIM_ROOT}/include)
set(ACALSIM_LIB_DIR ${ACALSIM_ROOT}/lib)

if(NOT EXISTS ${ACALSIM_INCLUDE_DIR})
    message(
        FATAL_ERROR
        "ACALSim headers not found at ${ACALSIM_INCLUDE_DIR}. Build ACALSim first."
    )
endif()

message(STATUS "ACALSim Include Dir: ${ACALSIM_INCLUDE_DIR}")
message(STATUS "ACALSim Library Dir: ${ACALSIM_LIB_DIR}")

# ========== Build SST Element ==========

# Source files
set(SST_ELEMENT_SOURCES
    src/SSTACALSimComponent.cc
    # Add more source files as needed
)

# Create shared library for SST element
add_library(acalsim MODULE ${SST_ELEMENT_SOURCES})

# Set library properties
set_target_properties(
    acalsim
    PROPERTIES
        PREFIX "lib"
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Include directories
target_include_directories(
    acalsim
    PRIVATE
        ${SST_INCLUDE_DIR}
        ${ACALSIM_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler flags
target_compile_options(acalsim PRIVATE ${SST_CXXFLAGS_RAW} -fPIC)

# Link libraries
target_link_libraries(
    acalsim
    PRIVATE ${ACALSIM_LIB_DIR}/libacalsim.a
    # Add other required libraries
)

# Linker flags
target_link_options(acalsim PRIVATE ${SST_LDFLAGS_RAW})

# ========== Installation ==========

# Get SST element directory
execute_process(
    COMMAND ${SST_CONFIG} --prefix
    OUTPUT_VARIABLE SST_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(SST_ELEMENT_INSTALL_DIR ${SST_PREFIX}/lib/sstcore-13.1.0)

message(STATUS "SST Element Install Dir: ${SST_ELEMENT_INSTALL_DIR}")

# Install the shared library
install(TARGETS acalsim LIBRARY DESTINATION ${SST_ELEMENT_INSTALL_DIR})

message(STATUS "========================================")
message(STATUS "SST Integration Configuration Complete")
message(STATUS "========================================")
message(STATUS "To install the SST element, run:")
message(STATUS "  make install")
message(STATUS "========================================")
