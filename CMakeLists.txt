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

cmake_minimum_required(VERSION 3.21)

include(cmake/ACALSimUtility.cmake)
include(cmake/GetLibtorch.cmake)

# CMake variables
get_filename_component(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)

# Project data
project(ACALSim VERSION 0.1.0 LANGUAGES CXX)

# Options
option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)
option(
    ACALSIM_VERBOSE
    "Print debugging messages of ACALSim internal operations."
    OFF
)
option(
    ACALSIM_STATISTICS
    "Print statistics of ACALSim internal operations."
    OFF
)
option(MT_DEBUG "Show debugging message for the multi-threaded support" OFF)
option(NO_LOGS "Disable all messages generated during simulation." OFF)

# Directories
set(INC_DIR include)
set(LIB_DIR libs)
set(SC_DIR systemc)
set(THIRD_PARTY_DIR third-party)

if(PROJECT_IS_TOP_LEVEL)
    set(TEST_DIR gtest)
    set(SRC_DIR src)
endif()

# Compile flags
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20 CACHE INTERNAL "This is a read-only variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Debug
        CACHE STRING
        "Choose the type of build, options are: empty, Debug, Release, RelWithDebInfo, MinSizeRel."
        FORCE
    )
endif()

string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)

# Compiler dependent flags
if(PROJECT_IS_TOP_LEVEL)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-ffile-prefix-map=${ROOT_DIR}=.)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang/AppleClang also support -ffile-prefix-map
        add_compile_options(-ffile-prefix-map=${ROOT_DIR}=.)
    else()
        message(
            WARNING
            "File prefix mapping not configured for ${CMAKE_CXX_COMPILER_ID}"
        )
    endif()
endif(PROJECT_IS_TOP_LEVEL)

# Link pthread
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
if(NOT Threads_FOUND)
    find_package(Threads REQUIRED)
endif(NOT Threads_FOUND)

# Link libtorch
get_libtorch()

# CMake options
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Third-party options
set(JSON_BuildTests OFF CACHE INTERNAL "")

# Print configuration status
option_summary()

# Subdirectories
add_subdirectory(${LIB_DIR})
add_subdirectory(${THIRD_PARTY_DIR})
add_subdirectory(${SC_DIR})

if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(${SRC_DIR})
    add_subdirectory(${TEST_DIR})
endif()
