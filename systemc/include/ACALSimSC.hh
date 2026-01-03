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

// SystemC - Utilities for integrating SystemC simulators

#pragma once

#include "ACALSim.hh"

// Main - Key elements of the ACALSim
#include "sim/SCInterface.hh"
#include "sim/SCSimBase.hh"
#include "sim/SCSimTop.hh"
#include "sim/SCThreadManager.hh"

// Packet
#include "packet/SCSimPacket.hh"

// Utils
#include "utils/sc_rv.hh"
