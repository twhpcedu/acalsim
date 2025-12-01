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
"""ACALSim Backend for PyTorch torch.compile().

This package provides a custom TorchDynamo backend that compiles PyTorch models
to run on the ACALSim accelerator simulator via QEMU-SST integration.

Usage:
    import torch
    from acalsim_backend import acalsim_backend

    model = MyModel()
    compiled_model = torch.compile(model, backend=acalsim_backend)
    output = compiled_model(input_tensor)

The backend:
1. Captures the FX graph from TorchDynamo
2. Lowers it to ACALSim IR (intermediate representation)
3. Generates RISC-V bare-metal code for the accelerator
4. Optionally runs the simulation via QEMU-SST
"""

from .backend import acalsim_backend, ACALSimBackend
from .compiler import ACALSimCompiler
from .ir import ACALSimIR, ACALSimOp

__version__ = "0.1.0"
__all__ = [
    "acalsim_backend",
    "ACALSimBackend",
    "ACALSimCompiler",
    "ACALSimIR",
    "ACALSimOp",
]
