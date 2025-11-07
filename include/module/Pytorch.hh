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

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sim/SimModule.hh"
#include "sim/SimTop.hh"
#include "utils/Logging.hh"

namespace acalsim {

class Pytorch : public SimModule {
protected:
	torch::jit::Module model;

public:
	Pytorch(std::string _name) : SimModule(_name) {}
	~Pytorch() {}

	void accept(Tick when, SimPacket& pkt) override { ERROR << "No packet is supported in the Pytorch module yet!"; }
	void loadModel(std::string modelFileName);
	std::vector<std::string>                getLayerNames();
	std::unordered_map<std::string, size_t> getParamSizeByLayers();
	size_t                                  getTotalSize(const std::unordered_map<std::string, size_t>& p);
	std::unordered_map<std::string, size_t> getActivationSize(const at::Tensor& input);
	std::vector<std::vector<int64_t>>       getDefaultInputDimensions(const torch::nn::Module& module);
	std::vector<int64_t>                    getConv2dInputDimensions(const torch::nn::Conv2d& conv);
	std::vector<int64_t>                    getLinearInputDimensions(const torch::nn::Linear& linear);
	std::vector<std::vector<int64_t>>       getInputDimensions(const torch::nn::Module& module);
	std::vector<std::vector<int64_t>>       getModelInputDimensions();
	torch::jit::Module getModuleFromNode(const torch::jit::Node* node, const torch::jit::Module& parentModule);
	void               dumpNodeDependencyTopological(const torch::jit::Graph& graph);
	void               printModelInfo();
};

}  // namespace acalsim
