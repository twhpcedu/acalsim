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

realpath(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ##########################################################################
# Build Configure
# ##########################################################################
set(LIBS_NAME acalsim)

# ##########################################################################
# # Build Source Files
# ##########################################################################
set(CHANNEL_LIB_SRCS channel/ChannelPortManager.cc channel/ChannelPort.cc)

set(COMMON_LIB_SRCS common/Arbiter.cc common/BitVector.cc)

set(CONFIG_LIB_SRCS
    config/ACALSimConfig.cc
    config/CLIManager.cc
    config/SimConfig.cc
    config/SimConfigManager.cc
)

set(CONTAINER_LIB_SRCS
    container/RecycleContainer/RecycleContainer.cc
    container/SimTraceContainer.cc
)

set(EVENT_LIB_SRCS event/CallbackEvent.cc event/SimEvent.cc)

set(EXTERNAL_GEM5_LIB_SRCS
    external/gem5/Event.cc
    external/gem5/EventManager.cc
    external/gem5/EventQueue.cc
    external/gem5/base/cprintf.cc
)

set(HW_LIB_SRCS hw/CrossBar.cc)

set(MODULE_LIB_SRCS module/Pytorch.cc)

set(PACKET_LIB_SRCS
    packet/DataPacket.cc
    packet/EventPacket.cc
    packet/SimPacket.cc
)

set(PORT_LIB_SRCS port/MasterPort.cc port/SimPortManager.cc port/SlavePort.cc)

set(SIM_SRCS
    sim/SimBase.cc
    sim/SimTop.cc
    sim/SimModule.cc
    sim/DeviceManager.cc
    sim/SimAddressMap.cc
    sim/STSim.cc
    sim/ThreadManager.cc
    sim/TaskManager.cc
    sim/PipeRegisterManager.cc
    sim/ThreadManagerV1/TaskV1.cc
)

set(WORKLOADS_LIB_SRCS
    workloads/tensor/SimTensor.cc
    workloads/tensor/SimTensorManager.cc
    workloads/graph/DAG.cc
)
set(HW_LIB_SRCS hw/CrossBar.cc)

set(UTILS_LIB_SRCS utils/Logging.cc utils/Arguments.cc)

set(LIBS_SRCS
    ${CHANNEL_LIB_SRCS}
    ${COMMON_LIB_SRCS}
    ${CONFIG_LIB_SRCS}
    ${CONTAINER_LIB_SRCS}
    ${EVENT_LIB_SRCS}
    ${EXTERNAL_GEM5_LIB_SRCS}
    ${MODULE_LIB_SRCS}
    ${PACKET_LIB_SRCS}
    ${PORT_LIB_SRCS}
    ${SIM_SRCS}
    ${WORKLOADS_LIB_SRCS}
    ${UTILS_LIB_SRCS}
    ${HW_LIB_SRCS}
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
    PUBLIC Threads::Threads
    PUBLIC msd_channel
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC CLI11::CLI11
)
target_include_directories(${LIBS_NAME} PUBLIC ${INCLUDE_DIR})

add_target_compile_definitions(${LIBS_NAME}
    ACALSIM_VERBOSE
    MT_DEBUG
    NO_LOGS
    ACALSIM_STATISTICS
)

add_library(${PROJECT_NAME}::${LIBS_NAME} ALIAS ${LIBS_NAME})
