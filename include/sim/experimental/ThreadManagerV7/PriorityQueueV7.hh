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

#include <functional>
#include <map>
#include <vector>

namespace acalsim {

template <typename TPriority, typename TElem>
class PriorityQueueV7 {
public:
	PriorityQueueV7();
	~PriorityQueueV7();

	void insert(const TElem& _elem, const TPriority& _priority);

	TElem&    getTopElem() const;
	TElem     popTopElem();
	void      getTopElements(std::function<void(const TElem&)> _func);
	void      getTopElements(std::function<void(std::vector<TElem>*)> _func);
	TPriority getTopPriority() const;

	bool empty() const;
	void remove(const TElem& _elem);

protected:
	std::vector<TElem>* getNewElemVec();

private:
	std::map<TPriority, std::vector<TElem>*> priorityMap;
	std::vector<std::vector<TElem>*>         elemVecReclcyeBin;
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV7/PriorityQueueV7.inl"
