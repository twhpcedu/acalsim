/*
 * Copyright 2023-2026 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Main - Key elements of the ACALSim
#include "sim/PipeRegisterManager.hh"
#include "sim/STSim.hh"
#include "sim/SimBase.hh"
#include "sim/SimModule.hh"
#include "sim/SimTop.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV1/TaskManagerV1.hh"
#include "sim/ThreadManagerV1/ThreadManagerV1.hh"
#include "sim/ThreadManagerV2/TaskManagerV2.hh"
#include "sim/ThreadManagerV2/ThreadManagerV2.hh"
#include "sim/ThreadManagerV3/TaskManagerV3.hh"
#include "sim/ThreadManagerV3/ThreadManagerV3.hh"

// Channel - Software communication for conducting simulation
#include "channel/ChannelPort.hh"
#include "channel/ChannelPortManager.hh"
#include "channel/SimChannel.hh"

// Common - General building blocks
#include "common/Arbiter.hh"
#include "common/BitVector.hh"
#include "common/FifoQueue.hh"
#include "common/LinkManager.hh"
#include "common/UnorderedRequestQueue.hh"

// Config - Hardware parameter management
#include "config/ACALSimConfig.hh"
#include "config/CLIManager.hh"
#include "config/SimConfig.hh"
#include "config/SimConfigManager.hh"

// Container
#include "container/JsonContainer.hh"
#include "container/RecycleContainer/LinkedList.hh"
#include "container/RecycleContainer/LinkedListArray.hh"
#include "container/RecycleContainer/ObjectPool.hh"
#include "container/RecycleContainer/RecyclableObject.hh"
#include "container/RecycleContainer/RecycleContainer.hh"
#include "container/SharedContainer.hh"
#include "container/SimTraceContainer.hh"

// Event
#include "event/CallbackEvent.hh"
#include "event/LambdaEvent.hh"
#include "event/SimEvent.hh"

// Hardware
#include "hw/CrossBar.hh"
#include "hw/SimPipeRegister.hh"

// Modules - Particular building blocks
#include "module/Pytorch.hh"

// Packet
#include "packet/CompoundPacket.hh"
#include "packet/DataPacket.hh"
#include "packet/EventPacket.hh"
#include "packet/SimPacket.hh"

// Port - Hardware interface modeling
#include "port/MasterPort.hh"
#include "port/SimPort.hh"
#include "port/SimPortManager.hh"
#include "port/SlavePort.hh"

// hw - HardwOBare modeling components
#include "hw/SimPipeRegister.hh"

// Workloads
#include "workloads/tensor/SimTensor.hh"
#include "workloads/tensor/SimTensorManager.hh"

// Utils
#include "utils/Arguments.hh"
#include "utils/HashableType.hh"
#include "utils/Logging.hh"
#include "utils/TypeDef.hh"

// Profiler
#include "profiling/Statistics.hh"
