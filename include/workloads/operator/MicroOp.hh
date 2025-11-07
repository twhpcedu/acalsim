/*
 * Copyright 2023-2025 Playlab/ACAL
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

#include "workloads/graph/DAG.hh"
#include "workloads/tensor/SimTensor.hh"

namespace acalsim {

// MicroOp class - derives from Node template
class MicroOp : public virtual graph::Node<MicroOp, SimTensor> {
public:
	// Constructor with optional name
	explicit MicroOp(const std::string& _name = "") : graph::Node<MicroOp, SimTensor>(_name) {}

	// Virtual destructor for inheritance
	virtual ~MicroOp() = default;

	// Rename operation
	void renew(const std::string& _name) { this->acalsim::graph::Node<MicroOp, SimTensor>::renew(_name); }

	// For compatibility with the original MicroOp interface
	const std::string& getName() const { return name; }
	graph::NodeStatus  getStatus() const { return status; }
};

}  // namespace acalsim
