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

#include "workloads/graph/DAG.hh"

#include "container/RecycleContainer/RecycleContainer.hh"
#include "sim/SimTop.hh"

namespace acalsim {
namespace graph {

template <typename NodeType, typename EdgeType>
void DAG<NodeType, EdgeType>::setInputs(std::vector<EdgeType*> _inputs) {
	this->inputs.clear();                  // Clear any existing elements in 'inputs'
	this->inputs.reserve(_inputs.size());  // Pre-allocate memory for efficiency
	for (const auto& tensorPtr : _inputs) {
		if (tensorPtr) {
			// Assuming SimTensor has a copy constructor or a way to copy its data
			this->inputs.push_back(top->getRecycleContainer()->acquire<EdgeType>(*tensorPtr));  // Deep copy
		} else {
			this->inputs.push_back(nullptr);  // Handle null pointers if necessary
		}
	}
}

template <typename NodeType, typename EdgeType>
void DAG<NodeType, EdgeType>::setOutputs(std::vector<EdgeType*> _outputs) {
	this->outputs.clear();                   // Clear any existing elements in 'outputs'
	this->outputs.reserve(_outputs.size());  // Pre-allocate memory for efficiency
	for (const auto& tensorPtr : _outputs) {
		if (tensorPtr) {
			// Assuming SimTensor has a copy constructor or a way to copy its data
			this->outputs.push_back(top->getRecycleContainer()->acquire<EdgeType>(*tensorPtr));  // Deep copy
		} else {
			this->outputs.push_back(nullptr);  // Handle null pointers if necessary
		}
	}
}

}  // namespace graph
}  // namespace acalsim
