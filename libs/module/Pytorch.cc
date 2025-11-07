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

#include "module/Pytorch.hh"

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/Logging.hh"

namespace acalsim {

void Pytorch::loadModel(std::string modelFileName) {
	try {
		// Load the model
		model = torch::jit::load(modelFileName);

		std::cout << "Model loaded and traced successfully." << std::endl;
	} catch (const c10::Error& e) {
		std::cerr << "Error loading the model: " << e.what() << std::endl;
		// Handle the error appropriately
	}

	CLASS_INFO << "load the torchscript model, " + modelFileName + ", successfully \n";
}

std::vector<std::string> Pytorch::getLayerNames() {
	std::vector<std::string> layerNames;
	for (auto submodule : model.named_children()) layerNames.push_back(submodule.name);
	return layerNames;
}

std::ostream& operator<<(std::ostream& os, std::unordered_map<std::string, size_t> const& m) {
	os << "{ \n";
	for (const auto& p : m) { os << "\t(" << p.first << " : " << p.second << ")," << std::endl; }
	os << "}";

	return os;
}

std::unordered_map<std::string, size_t> Pytorch::getParamSizeByLayers() {
	std::unordered_map<std::string, size_t> parameters;

	for (const auto& param : model.named_parameters()) {
		auto&  tensor       = param.value;
		size_t element_size = tensor.element_size();  // get element size in bytes
		size_t num_elements = tensor.numel();         // get number of elements
		parameters.insert(std::make_pair(param.name, element_size * num_elements));
	}
	return parameters;
}

size_t Pytorch::getTotalSize(const std::unordered_map<std::string, size_t>& p) {
	size_t total = 0;
	for (const auto& param : p) { total += param.second; }
	return total;
}

torch::jit::Module Pytorch::getModuleFromNode(const torch::jit::Node* node, const torch::jit::Module& parentModule) {
	if (node->kind() == c10::prim::GetAttr) {
		// Get the name of the attribute
		const auto& attrName = node->s(c10::attr::name);

		// Split the attribute name into parts (in case of nested modules)
		std::vector<std::string> attrParts;
		std::istringstream       ss(attrName);
		std::string              part;
		while (std::getline(ss, part, '.')) { attrParts.push_back(part); }

		// Navigate through the module hierarchy
		torch::jit::Module currentModule = parentModule;
		for (const auto& part : attrParts) { currentModule = currentModule.attr(part).toModule(); }

		return currentModule;
	}

	// If it's not a GetAttr node, return an empty module
	return torch::jit::Module();
}

std::unordered_map<std::string, size_t> Pytorch::getActivationSize(const at::Tensor& input) {
	std::unordered_map<std::string, size_t>                          activations;
	std::unordered_map<const torch::jit::Value*, torch::jit::IValue> value_map;

	// Get the graph of the model
	auto graph = model.get_method("forward").graph();

	// Initialize the value map with the input
	value_map[graph->inputs()[0]] = torch::jit::IValue(input);

	// Topological sort
	std::vector<torch::jit::Node*>         topo_order;
	std::unordered_set<torch::jit::Node*>  visited;
	std::function<void(torch::jit::Node*)> dfs = [&](torch::jit::Node* node) {
		if (visited.count(node)) return;
		visited.insert(node);
		for (auto input : node->inputs()) {
			if (input->node() != graph->param_node()) { dfs(input->node()); }
		}
		topo_order.push_back(node);
	};
	for (auto node : graph->nodes()) { dfs(node); }

	// Process nodes in topological order
	for (auto node : topo_order) {
		CLASS_INFO << "Processing node: " << node->kind().toQualString();
		if (node->hasAttribute(torch::jit::attr::name)) {
			CLASS_INFO << " (name: " << node->s(torch::jit::attr::name) << ")";
		}

		try {
			if (node->kind() == torch::jit::prim::Constant) {
				// Handle constant nodes
				for (size_t i = 0; i < node->outputs().size(); ++i) {
					auto output = node->output(i);
					auto ivalue = torch::jit::toIValue(output);
					if (ivalue.has_value()) { value_map[output] = ivalue.value(); }
				}
			} else if (node->kind() == torch::jit::prim::GetAttr) {
				// Handle GetAttr nodes
				const auto& name   = node->s(torch::jit::attr::name);
				auto        parent = value_map[node->inputs()[0]];
				if (parent.isObject()) {
					auto module = parent.toObject();
					auto attr   = module->getAttr(name);
					for (auto output : node->outputs()) { value_map[output] = attr; }

					// If the attribute is a tensor, add it to activations
					if (attr.isTensor()) {
						at::Tensor  tensor         = attr.toTensor();
						size_t      activationSize = tensor.numel() * tensor.element_size();
						std::string key            = "prim::GetAttr." + name;
						activations[key]           = activationSize;
						CLASS_INFO << "Added activation for " << key << ": " << activationSize;
					}
				} else {
					std::cerr << "Parent is not an object for GetAttr: " << name << std::endl;
				}
			} else if (node->kind() == torch::jit::prim::CallMethod) {
				const auto& method_name = node->s(torch::jit::attr::name);

				CLASS_INFO << "Processing CallMethod node: " << method_name;

				// Get inputs
				for (auto input : node->inputs()) { CLASS_INFO << "\tInput: " << input->type()->str(); }

				// Get outputs
				for (auto output : node->outputs()) { CLASS_INFO << "\tOutput: " << output->type()->str(); }

				auto outputs = node->outputs();

				// Handle the outputs
				for (size_t i = 0; i < outputs.size(); ++i) {
					auto output = outputs[i];
					auto type   = output->type();

					if (type->kind() == c10::TypeKind::TensorType) {
						auto tensorType = type->cast<c10::TensorType>();
						if (tensorType->sizes().isComplete()) {
							size_t numel = 1;
							for (size_t i = 0; i < tensorType->sizes().size(); ++i) {
								numel *= tensorType->sizes()[i].value();
							}
							size_t      activationSize = numel * 4;  // Assuming float32
							std::string key =
							    std::string(node->kind().toQualString()) + "." + method_name + "." + std::to_string(i);
							activations[key] = activationSize;
							CLASS_INFO << "Added activation for " << key << ": " << activationSize;
						}
					} else if (type->kind() == c10::TypeKind::ListType &&
					           type->cast<c10::ListType>()->getElementType()->kind() == c10::TypeKind::TensorType) {
						CLASS_INFO << "TensorList output detected for " << method_name << "." << i;
						// Note: We can't accurately determine the size of a TensorList at compile time
					} else {
						CLASS_INFO << "Unhandled output type for " << method_name << "." << i << ": " << type->str();
					}
				}

				// Store the output for subsequent nodes
				for (size_t i = 0; i < outputs.size(); ++i) { value_map[outputs[i]] = outputs[i]; }
			}
		} catch (const std::exception& e) {
			const auto& method_name = node->s(torch::jit::attr::name);
			CLASS_ERROR << "Error processing CallMethod node: " << method_name << " - " << e.what() << std::endl;
		}
	}

	return activations;
}

std::vector<std::vector<int64_t>> Pytorch::getDefaultInputDimensions(const torch::nn::Module& module) {
	std::vector<std::vector<int64_t>> inputDimensions;

	for (const auto& pair : module.named_parameters()) {
		const auto& param = pair.value();
		if (param.dim() > 1) {  // Assuming the first dimension is batch size
			inputDimensions.push_back(param.sizes().vec());
		}
	}

	for (const auto& pair : module.named_buffers()) {
		const auto& buffer = pair.value();
		if (buffer.dim() > 1) { inputDimensions.push_back(buffer.sizes().vec()); }
	}

	return inputDimensions;
}

std::vector<int64_t> Pytorch::getConv2dInputDimensions(const torch::nn::Conv2d& conv) {
	auto weight = conv->weight;
	return {weight.size(1), 0, 0};  // Channels, Height, Width (Height and Width are unknown)
}

std::vector<int64_t> Pytorch::getLinearInputDimensions(const torch::nn::Linear& linear) {
	return {linear->weight.size(1)};
}

std::vector<std::vector<int64_t>> Pytorch::getInputDimensions(const torch::nn::Module& module) {
	if (auto conv = dynamic_cast<const torch::nn::Conv2d*>(&module)) {
		return {getConv2dInputDimensions(*conv)};
	} else if (auto linear = dynamic_cast<const torch::nn::Linear*>(&module)) {
		return {getLinearInputDimensions(*linear)};
	}
	// Add more module types if  needed

	// Fallback to the general approach
	return {getDefaultInputDimensions(module)};
}

std::vector<std::vector<int64_t>> Pytorch::getModelInputDimensions() {
	try {
		// Ensure the model is a jit::Module
		auto jit_module = torch::jit::Module(model);

		// Get the method schema for the forward function
		auto                              method = jit_module.get_method("forward");
		auto                              schema = method.function().getSchema();
		std::vector<std::vector<int64_t>> input_dimensions;

		// Iterate through the arguments of the forward method
		for (const auto& arg : schema.arguments()) {
			if (arg.type()->kind() == c10::TypeKind::TensorType) {
				auto tensor_type = arg.type()->cast<c10::TensorType>();
				if (tensor_type && tensor_type->sizes().isComplete()) {
					auto sizes = tensor_type->sizes().sizes();
					if (sizes.has_value()) {
						std::vector<int64_t> dims;
						for (const auto& size : sizes.value()) {
							if (size.has_value()) {
								dims.push_back(size.value());
							} else {
								dims.push_back(-1);  // for dynamic dimensions
							}
						}
						input_dimensions.push_back(dims);
					}
				}
			}
		}

		return input_dimensions;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return {};
	}
}

void Pytorch::dumpNodeDependencyTopological(const torch::jit::Graph& graph) {
	// Create a directed graph representation
	std::unordered_map<const torch::jit::Value*, std::vector<const torch::jit::Node*>> node_deps;
	for (const auto& node : graph.nodes()) {
		for (const auto& input : node->inputs()) { node_deps[input].push_back(node); }
	}

	// Perform topological sort
	std::queue<const torch::jit::Value*>              q;
	std::unordered_map<const torch::jit::Value*, int> in_degree;

	// Initialize in_degree for all values
	for (const auto& node : graph.nodes()) {
		for (const auto& output : node->outputs()) { in_degree[output] = 0; }
	}

	// Calculate in_degree
	for (const auto& node : graph.nodes()) {
		for (const auto& input : node->inputs()) { in_degree[input]++; }
	}

	// Find starting nodes (those with in_degree 0)
	for (const auto& pair : in_degree) {
		if (pair.second == 0) { q.push(pair.first); }
	}

	std::vector<const torch::jit::Value*> topological_order;
	while (!q.empty()) {
		const torch::jit::Value* value = q.front();
		q.pop();
		topological_order.push_back(value);

		auto it = node_deps.find(value);
		if (it != node_deps.end()) {
			for (const auto& node : it->second) {
				for (const auto& output : node->outputs()) {
					in_degree[output]--;
					if (in_degree[output] == 0) { q.push(output); }
				}
			}
		}
	}

	// Print node dependencies in topological order
	for (const auto& value : topological_order) {
		CLASS_INFO << "Value: " << value->debugName();
		auto it = node_deps.find(value);
		if (it != node_deps.end()) {
			for (const auto& node : it->second) { CLASS_INFO << "  Dependency: " << node->kind().toQualString(); }
		}
	}
}

#include <torch/version.h>
void Pytorch::printModelInfo() {
	CLASS_INFO << "PyTorch version: " << TORCH_VERSION;
	CLASS_INFO << "layers : " << getLayerNames();

	/* [TODO] to be fixed
	 *  the return inputDims is empty for ResNet18 trace.
	 */
	/*
	// need to get the input tensor dimensions to generate random inputs
	auto inputDims = getModelInputDimensions();
	if (!inputDims.empty()) {
	    CLASS_INFO << "Input dimensions: " << inputDims;
	} else {
	    CLASS_ERROR << "Could not determine input dimensions for this model.";
	}

	// The getModelInputDimensions() function returns a vector of vectors to account
	// for potentially multiple inputs, but torch::randn() needs a single vector
	// for creating a single tensor.
	// Assume we're dealing with the first (and possibly only) input tensor
	const auto& firstInputDims = inputDims[0];

	// Create a vector of int64_t for the dimensions
	std::vector<int64_t> dims;
	dims.push_back(1);  // Add batch dimension
	dims.insert(dims.end(), firstInputDims.begin(), firstInputDims.end());
	    */

	// Now create the random input tensor
	torch::Tensor input = torch::randn({(1, 3, 224, 224)});

	const std::unordered_map<std::string, size_t>& parameters = getParamSizeByLayers();
	CLASS_INFO << "# of parameter : " << parameters.size();
	CLASS_INFO << "Total parameter size : " << getTotalSize(parameters);
	CLASS_INFO << "parameters : \n" << parameters;

	/* [TODO] to be fixed. the getActivationSize still has bugs
	 */
	const std::unordered_map<std::string, size_t>& activations = getActivationSize(input);
	CLASS_INFO << "# of activations : " << activations.size();
	CLASS_INFO << "Total activation size : " << getTotalSize(activations);
	CLASS_INFO << "activations : \n" << activations;

	// print layer dependency in the topological order
	/* [TODO] to be fixed. Only one layer is printed out for ResNet18 trace.
	 *
	 */
	try {
		// Get the forward method
		auto method = model.get_method("forward");

		// Get the graph from the method
		auto graph = method.graph();

		// Dump the node dependency
		dumpNodeDependencyTopological(*graph);
	} catch (const std::exception& e) { std::cerr << "Error dumping model topology: " << e.what() << std::endl; }
}

}  // end of namespace acalsim
