/*
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __ACALSIM_MINIMAL_HH__
#define __ACALSIM_MINIMAL_HH__

/*
 * Minimal ACALSim includes for SST integration
 * This avoids pulling in PyTorch and other heavy dependencies
 */

#include <cstdint>
#include <memory>
#include <string>
#include <map>

// Type definitions
using Tick = uint64_t;

// Forward declarations to avoid pulling in full headers
class SimBase;
class SimTop;
class SimConfig;
class SimPacket;

#endif // __ACALSIM_MINIMAL_HH__
