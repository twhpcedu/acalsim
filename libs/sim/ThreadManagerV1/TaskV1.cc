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

#include "sim/ThreadManagerV1/TaskV1.hh"

#include "sim/SimBase.hh"
#include "sim/SimTop.hh"

namespace acalsim {

void TaskFunctor::operator()() { this->simbase->stepWrapperBase(); }

Tick TaskFunctor::getSimNextTick() const { return this->simbase->getSimNextTick(); }

Tick TaskFunctor::getGlobalTick() const { return top->getGlobalTick(); }

Tick TaskFunctor::isReadyToTerminate() const { return top->isReadyToTerminate(); }

int TaskFunctor::getSimBaseId() const { return this->simbase->getID(); }

}  // namespace acalsim
