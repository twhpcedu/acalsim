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

# ##########################################################################
# Configure
# ##########################################################################
include(${PROJECT_SOURCE_DIR}/cmake/ACALSimUtility.cmake)

realpath(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ##########################################################################
# Build Configure
# ##########################################################################
set(LIBS_NAME acalsim_systemc)

# ##########################################################################
# # Build Source Files
# ##########################################################################
set(LIBS_SRCS
    packet/SCSimPacket.cc
    sim/SCInterface.cc
    sim/SCSimBase.cc
    sim/SCSimTop.cc
    sim/SCThreadManager.cc
)

# ##########################################################################
# # Build rules
# ##########################################################################
add_library(${LIBS_NAME} ${LIBS_SRCS})

set_target_properties(${LIBS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${LIBS_NAME} PROPERTY CXX_STANDARD 20)

target_link_libraries(
    ${LIBS_NAME}
    PUBLIC ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS}
    PUBLIC ACALSim::acalsim
    PUBLIC SystemC::systemc
)
target_include_directories(${LIBS_NAME} PUBLIC ${INCLUDE_DIR})

add_target_compile_definitions(${LIBS_NAME}
    ACALSIM_VERBOSE
    MT_DEBUG
    NO_LOGS
    ACALSIM_STATISTICS
)

add_library(${PROJECT_NAME}::systemc ALIAS ${LIBS_NAME})
