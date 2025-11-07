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

# ##########################################################################
# Configure
# ##########################################################################
include(${PROJECT_SOURCE_DIR}/cmake/ACALSimUtility.cmake)

realpath(APP_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
# ##########################################################################
# User Configure
# ##########################################################################
# Setup the libaray name
## You can modify the `template` below to any library name you prefer.
set(APP_LIB_NAME testPETile)

set(APP_PREFIX_LIB_NAME ${PROJECT_NAME}_${APP_LIB_NAME})

# ##########################################################################
# # Build Source Files
# ##########################################################################
set(LIBS_SRCS
    PETile.cc
    CPUReqEvent.cc
    CPUTraffic.cc
    AXIBus.cc
    MemReq.cc
    SRAM.cc
)
# ##########################################################################
# # Build rules
# ##########################################################################
add_library(${APP_PREFIX_LIB_NAME} ${LIBS_SRCS})

set_target_properties(${LIBS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${LIBS_NAME} PROPERTY CXX_STANDARD 20)

target_link_libraries(
    ${APP_PREFIX_LIB_NAME}
    PUBLIC Threads::Threads
    PUBLIC ACALSim::acalsim
)

add_target_compile_definitions(${APP_PREFIX_LIB_NAME}
    ACALSIM_VERBOSE
    MT_DEBUG
    NO_LOGS
    ACALSIM_STATISTICS
)

target_include_directories(${APP_PREFIX_LIB_NAME} PUBLIC ${APP_INC_DIR})

add_library(${PROJECT_NAME}::${APP_LIB_NAME} ALIAS ${APP_PREFIX_LIB_NAME})
