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
# User Configure
# ##########################################################################
# Setup the libaray name
## You can modify the `template` below to any library name you prefer.
set(APP_LIB testBlackBear)

set(LIBS_SRCS
    memsim/MemSim.cc
    cachesim/CacheSim.cc
    nocsim/NocSim.cc
    pesim/PESim.cc
    mcpusim/MCPUSim.cc
    system/DataMovementManager.cc
)
# ##########################################################################
# # Build rules
# ##########################################################################
function(configure_target TARGET_NAME)
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
    target_link_libraries(
        ${TARGET_NAME}
        PUBLIC Threads::Threads
        PUBLIC ACALSim::acalsim
    )
    target_include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIR})
endfunction(configure_target)

# Shared Libs for gdb
if(BUILD_SHARED_LIBS)
    add_library(${APP_LIB}_lib SHARED ${LIBS_SRCS})
else()
    add_library(${APP_LIB}_lib STATIC ${LIBS_SRCS})
endif()

configure_target(${APP_LIB}_lib)
