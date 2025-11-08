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

cmake_minimum_required(VERSION 3.12)
project(ACALSim_SST_Integration VERSION 1.0.0 LANGUAGES CXX)

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
find_program(SST_CONFIG sst-config REQUIRED)
if(NOT SST_CONFIG)
    message(
        FATAL_ERROR
        "sst-config not found. Please install SST-Core and ensure it's in PATH."
    )
endif()

message(STATUS "Found sst-config: ${SST_CONFIG}")

# Get SST compiler flags
execute_process(
    COMMAND ${SST_CONFIG} --ELEMENT_CXXFLAGS
    OUTPUT_VARIABLE SST_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get SST linker flags
execute_process(
    COMMAND ${SST_CONFIG} --ELEMENT_LDFLAGS
    OUTPUT_VARIABLE SST_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get SST prefix
execute_process(
    COMMAND ${SST_CONFIG} --prefix
    OUTPUT_VARIABLE SST_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "SST prefix: ${SST_PREFIX}")
message(STATUS "SST CXXFLAGS: ${SST_CXXFLAGS}")
message(STATUS "SST LDFLAGS: ${SST_LDFLAGS}")

# ========== Find ACALSim ==========

# ACALSim paths (adjust if needed)
set(ACALSIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(ACALSIM_INCLUDE "${ACALSIM_ROOT}/include")
set(ACALSIM_LIB "${ACALSIM_ROOT}/lib")

# Check if ACALSim include directory exists
if(NOT EXISTS "${ACALSIM_INCLUDE}")
    message(
        FATAL_ERROR
        "ACALSim include directory not found: ${ACALSIM_INCLUDE}"
    )
endif()

# Find ACALSim library
find_library(
    ACALSIM_LIBRARY
    NAMES acalsim libacalsim
    PATHS ${ACALSIM_LIB}
    NO_DEFAULT_PATH
)

if(NOT ACALSIM_LIBRARY)
    message(WARNING "ACALSim library not found. Please build ACALSim first.")
    message(WARNING "Searched in: ${ACALSIM_LIB}")
    set(ACALSIM_LIBRARY "")
else()
    message(STATUS "Found ACALSim library: ${ACALSIM_LIBRARY}")
endif()

# ========== SST Element Library ==========

# Source files
set(SOURCES ACALSimComponent.cc)

# Create shared library
add_library(acalsim_sst SHARED ${SOURCES})

# Include directories
target_include_directories(
    acalsim_sst
    PRIVATE
        ${ACALSIM_INCLUDE}
        ${ACALSIM_INCLUDE}/external
        ${ACALSIM_INCLUDE}/sim
        ${ACALSIM_INCLUDE}/event
        ${ACALSIM_INCLUDE}/packet
        ${ACALSIM_INCLUDE}/port
        ${ACALSIM_INCLUDE}/channel
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags
target_compile_options(
    acalsim_sst
    PRIVATE -Wall -Wextra -O2 -g ${SST_CXXFLAGS}
)

# Linker flags
target_link_libraries(acalsim_sst PRIVATE ${ACALSIM_LIBRARY} ${SST_LDFLAGS})

# Set library properties
set_target_properties(
    acalsim_sst
    PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "acalsim_sst"
        POSITION_INDEPENDENT_CODE ON
)

# ========== Installation ==========

# Install to SST element library directory
set(SST_ELEMENT_LIB_DIR "${SST_PREFIX}/lib/sst-elements-library")

install(
    TARGETS acalsim_sst
    LIBRARY
    DESTINATION ${SST_ELEMENT_LIB_DIR}
    RUNTIME
    DESTINATION ${SST_ELEMENT_LIB_DIR}
)

message(STATUS "Install destination: ${SST_ELEMENT_LIB_DIR}")

# ========== Testing ==========

# Custom target to test installation
add_custom_target(
    test_sst
    COMMAND ${SST_CONFIG} --version
    COMMAND sst-info acalsim
    COMMENT "Testing SST element installation"
    VERBATIM
)

# ========== Examples ==========

# Custom target to list examples
add_custom_target(
    examples
    COMMAND ${CMAKE_COMMAND} -E echo "Example configurations:"
    COMMAND
        ${CMAKE_COMMAND} -E echo
        "  sst ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_system.py"
    COMMAND
        ${CMAKE_COMMAND} -E echo
        "  sst ${CMAKE_CURRENT_SOURCE_DIR}/examples/multicore_system.py"
    COMMENT "Listing example configurations"
    VERBATIM
)

# ========== Uninstall ==========

# Add uninstall target
add_custom_target(
    uninstall
    COMMAND ${CMAKE_COMMAND} -E remove ${SST_ELEMENT_LIB_DIR}/libacalsim_sst.so
    COMMENT "Uninstalling ACALSim SST element"
    VERBATIM
)

# ========== Summary ==========

message(STATUS "")
message(STATUS "========================================")
message(STATUS "ACALSim SST Integration Configuration")
message(STATUS "========================================")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "SST prefix:       ${SST_PREFIX}")
message(STATUS "ACALSim include:  ${ACALSIM_INCLUDE}")
message(STATUS "ACALSim library:  ${ACALSIM_LIBRARY}")
message(STATUS "Install dir:      ${SST_ELEMENT_LIB_DIR}")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "  make install")
message(STATUS "  make test_sst")
message(STATUS "========================================")
message(STATUS "")
