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

# // clang-format off

# ##########################################################################
# Configuration & Setup
# #########################################################################
include(${PROJECT_SOURCE_DIR}/cmake/ACALSimUtility.cmake)

# ##########################################################################
# User Configure
# ##########################################################################
set(TEST_LIB_NAME gtest_DAG)

set(TEST_PREFIX_LIB_NAME ${PROJECT_NAME}_${TEST_LIB_NAME})
# ##########################################################################
# # Build Source Files
# ##########################################################################
set(LIBS_SRCS DAGSchedulerTest.cc)

# ##########################################################################
# Library and Executable Setup
# ##########################################################################
add_library(${TEST_PREFIX_LIB_NAME} ${LIBS_SRCS})

set_target_properties(${LIBS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${TEST_PREFIX_LIB_NAME} PROPERTY CXX_STANDARD 20)
target_link_libraries(
    ${TEST_PREFIX_LIB_NAME}
    PUBLIC Threads::Threads
    PUBLIC ACALSim::acalsim
    PUBLIC GTest::gtest
    PUBLIC CLI11::CLI11
    PUBLIC GTest::gmock
    PUBLIC GTest::gmock_main
)
target_include_directories(
    ${TEST_PREFIX_LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable testing framework
enable_testing()

# Define executable name based on directory name
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(EXE_NAME g${DIR_NAME})

# Add main executable and link with GoogleTest
add_executable(${EXE_NAME} main.cc)
target_link_libraries(${EXE_NAME} ${TEST_PREFIX_LIB_NAME})

# Integrate GoogleTest discovery for test auto-registration
include(GoogleTest)
gtest_discover_tests(${EXE_NAME} DISCOVERY_MODE "PRE_TEST")
