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

# Minimum CMake version required
cmake_minimum_required(VERSION 3.15)

# Get the directory name for the executable
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME g${DIR_NAME})

# Collect all source files
file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# Create the test executable
add_executable(${EXE_NAME} ${TEST_SOURCES})

# Include directories
target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(
    ${EXE_NAME}
    PRIVATE ACALSim::acalsim
    PRIVATE GTest::gtest
    PRIVATE GTest::gtest_main
)

# Enable testing
enable_testing()

# Add the test
add_test(
    NAME ${EXE_NAME}
    COMMAND ${EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Set properties for the test
set_tests_properties(
    ${EXE_NAME}
    PROPERTIES ENVIRONMENT "PATH=${CMAKE_BINARY_DIR};$ENV{PATH}"
)
