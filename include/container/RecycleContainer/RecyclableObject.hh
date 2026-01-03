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

#include <cstdlib>

#include "container/RecycleContainer/LinkedList.hh"
#include "utils/HashableType.hh"

namespace acalsim {

class RecyclableObject : public LinkedList::Node, virtual public HashableType {
	friend class RecycleContainer;

public:
	RecyclableObject() { ; }
	virtual ~RecyclableObject() { ; }

protected:
	virtual void preRecycle() { ; }

#ifndef NDEBUG
private:
	bool is_recycled_ = false;
#endif  // NDEBUG
};

}  // namespace acalsim
