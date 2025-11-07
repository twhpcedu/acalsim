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

#include <string>
#include <unordered_map>

#include "container/RecycleContainer/RecycleContainer.hh"
#include "utils/Logging.hh"
#include "workloads/tensor/SimTensor.hh"

namespace acalsim {

class SimTensorManager : virtual public HashableType {
public:
	/**
	 * @brief Construct a new SimTensorManager object.
	 *
	 */
	SimTensorManager(const std::string& name) {
		tensorPool = new RecycleContainer();
		// force objectPool creation
		tensorPool->recycle(tensorPool->acquire<SimTensor>());
	}

	/**
	 * @brief allocate a tensor from the SimTensorManager.
	 *
	 * @param _name : tensor name
	 */
	SimTensor* allocateTensor(const std::string& _name) {
		SimTensor* pTensor        = this->getTensorPool()->acquire<SimTensor>();
		tensors[pTensor->getID()] = pTensor;
		return pTensor;
	}

	/**
	 * @brief recycle a tensor to the SimTensorManager.
	 *
	 * @param pTensor : a smart pointer to the tensor object
	 */
	void freeTensor(SimTensor* pTensor) {
		auto iter = tensors.find(pTensor->getID());
		CLASS_ASSERT_MSG(iter != this->tensors.end(),
		                 "The tensor \'" + std::to_string(pTensor->getID()) + "\' does not exist.");
		tensors.erase(iter);
		tensorPool->recycle(pTensor);
	}

	/**
	 * @brief Get the SimTensor object by its id.
	 * @param id The id of the SimTensor object.
	 * @return Pointer to the SimConfig object.
	 */
	SimTensor* getTensor(uint64_t id) const {
		auto iter = this->tensors.find(id);
		CLASS_ASSERT_MSG(iter != this->tensors.end(), "The tensor \'" + std::to_string(id) + "\' does not exist.");
		return iter->second;
	}

private:
	/// @brief A map of tensor id to SimTensor objects.
	std::unordered_map<uint64_t, SimTensor*> tensors;

	// A recycle container for SimTensor objects
	RecycleContainer* tensorPool = nullptr;
	RecycleContainer* getTensorPool() { return this->tensorPool; }
};

}  // end of namespace acalsim
