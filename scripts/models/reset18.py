#!/usr/bin/python3
"""ResNet-18 Model to TorchScript Converter.

This script converts a pretrained ResNet-18 PyTorch model into a serialized
TorchScript model file for deployment and inference. The script uses torch.jit.trace
to capture the model's forward pass execution and serialize it to a portable format.

ResNet-18 Model Details:
    - Architecture: ResNet-18 (18-layer Residual Network)
    - Input Shape: (batch_size, 3, 224, 224) - RGB images
    - Output Shape: (batch_size, 1000) - ImageNet class predictions
    - Parameters: ~11.7 million trainable parameters
    - Pretrained Weights: ImageNet (if available via torchvision)

TorchScript Export:
    The script uses tracing mode (torch.jit.trace) which records operations
    executed during a forward pass with example inputs. This is suitable for
    models without data-dependent control flow.

Generated Files:
    - traced_resnet_model.pt: Serialized TorchScript model file
        - Can be loaded in C++ using libtorch
        - Can be loaded in Python using torch.jit.load()
        - Portable across different Python environments
        - Optimized for inference (no Python overhead)

Usage:
    Basic execution:
        $ python3 scripts/models/reset18.py

    This will generate 'traced_resnet_model.pt' in the current working directory.

    To load the generated model:
        >>> import torch
        >>> model = torch.jit.load('traced_resnet_model.pt')
        >>> model.eval()
        >>> input_tensor = torch.rand(1, 3, 224, 224)
        >>> output = model(input_tensor)
        >>> print(output.shape)  # torch.Size([1, 1000])

Example Output:
    When executed successfully, the script produces:
        traced_resnet_model.pt (size: ~44.7 MB)

Dependencies:
    - torch: PyTorch deep learning framework
    - torchvision: PyTorch vision models and utilities

Notes:
    - The model uses random initialization if pretrained weights aren't loaded
    - For pretrained weights, modify line to: torchvision.models.resnet18(pretrained=True)
    - The traced model maintains the same numerical behavior as the original
    - TorchScript models are optimized for production deployment
"""

# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision

# ============================================================================
# Model Instantiation
# ============================================================================

# Create an instance of the ResNet-18 model with random initialization.
# Note: For pretrained weights trained on ImageNet, use:
#   model = torchvision.models.resnet18(pretrained=True)
# or with newer API:
#   model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model = torchvision.models.resnet18()

# ============================================================================
# Example Input Tensor Creation
# ============================================================================

# Create a random example input tensor for tracing the model's forward pass.
# Shape: (batch_size=1, channels=3, height=224, width=224)
# - batch_size: Number of images to process simultaneously (1 for single image)
# - channels: RGB color channels (3 for standard color images)
# - height, width: Standard ImageNet input dimensions (224x224 pixels)
#
# This example input is used by torch.jit.trace to record all operations
# performed during the forward pass, creating an optimized execution graph.
example = torch.rand(1, 3, 224, 224)

# ============================================================================
# TorchScript Tracing
# ============================================================================

# Trace the model with the example input to create a TorchScript module.
# Tracing process:
#   1. Executes model.forward(example) and records all operations
#   2. Builds a static computation graph (IR - Intermediate Representation)
#   3. Optimizes the graph for inference (operator fusion, constant folding)
#   4. Returns a torch.jit.ScriptModule that can be serialized
#
# Important: Tracing assumes the control flow is the same for all inputs.
# If your model has data-dependent control flow (if/else, loops), consider
# using torch.jit.script instead of torch.jit.trace.
traced_script_module = torch.jit.trace(model, example)

# ============================================================================
# Model Serialization
# ============================================================================

# Save the traced TorchScript module to disk as a portable .pt file.
# The serialized file contains:
#   - Model architecture and parameters
#   - Optimized computation graph
#   - All necessary metadata for inference
#
# Output file: traced_resnet_model.pt (~44.7 MB)
# This file can be loaded and executed in:
#   - Python environments (torch.jit.load)
#   - C++ applications (torch::jit::load)
#   - Mobile devices (PyTorch Mobile)
#   - Production servers (without Python dependencies)
traced_script_module.save("traced_resnet_model.pt")
