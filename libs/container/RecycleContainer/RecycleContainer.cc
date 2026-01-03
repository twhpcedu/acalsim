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

#include "container/RecycleContainer/RecycleContainer.hh"

#ifdef ACALSIM_STATISTICS
#include "utils/Logging.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

RecycleContainer::RecycleContainer(size_t _defaultInitSegments, size_t _defaultSegmentLength)
    : defaultInitSegments(_defaultInitSegments), defaultSegmentLength(_defaultSegmentLength) {}

RecycleContainer::~RecycleContainer() {
#ifdef ACALSIM_STATISTICS
	double prealloc_cost    = 0;
	double gpool_mutex_cost = 0;
	double balancing_cost   = 0;

	for (const auto& [type, pool] : this->objectPools) {
		prealloc_cost += pool->getPreallocCost();
		gpool_mutex_cost += pool->getGlobalPoolMutexCost();
		balancing_cost += pool->getBalancingPoolCost();
	}

	LABELED_STATISTICS("RecycleContainer") << "Pre-allocation Cost: " << prealloc_cost << " us.";
	LABELED_STATISTICS("RecycleContainer") << "Mutex Cost for Global Pools: " << gpool_mutex_cost << " us.";
	LABELED_STATISTICS("RecycleContainer") << "Rebalancing Cost: " << balancing_cost << " us.";
	LABELED_STATISTICS("RecycleContainer") << "Total Overhead: " << prealloc_cost + balancing_cost << " us.";
#endif  // ACALSIM_STATISTICS

	*this->hasDestructed = true;
	for (auto& [type, pool] : this->objectPools) { delete pool; }
}

void RecycleContainer::Deleter::operator()(RecyclableObject* _obj) const noexcept {
	try {
		if (!(*this->hasCntrDestructed))
			this->cntr->recycle(_obj);
		else
			delete _obj;
	} catch (...) { delete _obj; }
}

}  // namespace acalsim
