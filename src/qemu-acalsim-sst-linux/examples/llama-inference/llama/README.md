<!--
Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLaMA Inference Guide

Complete guide to running LLaMA 2 inference with ACALSIM SST accelerator integration.

## Overview

The LLaMA inference example demonstrates:
- **LLaMA 2 7B** model inference
- **SST (Structural Simulation Toolkit)** integration
- **QEMU RISC-V** Linux environment
- **PyTorch** on RISC-V architecture
- **Hardware accelerator simulation**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ QEMU RISC-V Linux (Debian)                                   │
│   ├── PyTorch 2.4.0                                         │
│   ├── LLaMA 2 7B Model                                      │
│   └── llama_inference.py                                    │
│          ↓ (via Unix socket)                                │
│ SST Simulator (Host)                                        │
│   ├── ACALSim Components                                    │
│   ├── Hardware Accelerator Model                            │
│   └── sst_config_llama.py                                   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Running QEMU Debian with PyTorch

You should have already:
- ✅ QEMU Debian running with PyTorch installed
- ✅ SSH access configured (port 2222)
- ✅ Shared folder mounted at `/mnt/shared`

If not, start with:
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh
```

### 2. SST-Core Installed

Check if SST is installed:
```bash
docker exec acalsim-workspace bash -c "which sst"
```

If not found, build SST-Core:
```bash
docker exec acalsim-workspace bash -c "
cd /home/user/projects/acalsim/sst-core
./build.sh
"
```

### 3. ACALSim SST Components

Build the ACALSIM SST components:
```bash
docker exec acalsim-workspace bash -c "
cd /home/user/projects/acalsim/src/sst-integration
make clean && make && make install
"
```

## Quick Start (Without Actual Model)

### Step 1: Start SST Simulator

In one terminal:
```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama
./run_sst.sh
```

This will:
- Configure SST environment
- Create Unix socket at `/tmp/qemu-sst-llama.sock`
- Wait for QEMU to connect

### Step 2: Start QEMU (In Another Terminal)

```bash
# QEMU should already be running with run_qemu_debian_dqib.sh
# If not, start it:
./run_qemu_debian_dqib.sh
```

### Step 3: Run Inference Demo (In QEMU)

SSH into QEMU:
```bash
ssh -p 2222 debian@localhost
# Password: debian
```

Inside QEMU:
```bash
cd /mnt/shared  # This is your shared folder

# Create a simple test
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print("LLaMA inference would run here")
print("Model loading skipped (demo mode)")
EOF
```

## Full Setup (With Actual LLaMA Model)

### Step 1: Download LLaMA 2 Model

You need the LLaMA 2 7B model weights. Options:

**Option A: Hugging Face (Requires approval)**
```bash
# On host Mac or in QEMU
pip3 install transformers

# Download model (requires HF account and Meta approval)
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
save_path = "/mnt/shared/llama-2-7b"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Model saved to {save_path}")
EOF
```

**Option B: Use Smaller Model for Testing**
```bash
# Download TinyLlama (faster for testing)
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
save_path = "/mnt/shared/tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Model saved to {save_path}")
EOF
```

### Step 2: Install Dependencies in QEMU

```bash
ssh -p 2222 debian@localhost

# Install transformers and other deps
pip3 install --break-system-packages transformers accelerate

# Verify installation
python3 -c "import transformers; print(transformers.__version__)"
```

### Step 3: Copy LLaMA Script to QEMU

```bash
# In QEMU
cd /mnt/shared
mkdir -p llama-demo

# Copy the inference script
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama/llama_inference.py llama-demo/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama/llama_sst_backend.py llama-demo/
```

### Step 4: Modify Script for Your Model

Edit `llama_inference.py` to point to your model:

```python
# Change MODEL_PATH to your model location
MODEL_PATH = "/mnt/shared/llama-2-7b"  # or /mnt/shared/tinyllama
```

### Step 5: Run Complete Inference

**Terminal 1: Start SST**
```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama
./run_sst.sh
```

**Terminal 2: Run Inference in QEMU**
```bash
ssh -p 2222 debian@localhost
cd /mnt/shared/llama-demo

# Run inference
python3 llama_inference.py
```

## Configuration

### SST Configuration

Edit `llama/sst_config_llama.py` to configure:
- Memory hierarchy
- Accelerator parameters
- Network topology
- Clock frequencies

### Inference Parameters

In `llama_inference.py`:
```python
max_new_tokens = 100      # Tokens to generate
temperature = 0.8         # Sampling temperature
top_p = 0.95             # Nucleus sampling
top_k = 50               # Top-k sampling
```

## Prompts

Test prompts are in `llama/test_prompts.txt`:

```
What is the meaning of life?
Explain quantum computing in simple terms.
Write a haiku about computers.
```

Run with prompts:
```bash
python3 llama_inference.py < test_prompts.txt
```

## Troubleshooting

### Issue 1: PyTorch Not Found

```bash
# In QEMU, verify PyTorch
python3 -c "import torch; print(torch.__version__)"

# If not found, PyTorch may not be installed correctly
# See the main README for PyTorch installation
```

### Issue 2: SST Socket Error

```
OSError: [Errno 111] Connection refused
```

**Solution**: Ensure SST is running first before starting QEMU inference.

### Issue 3: Model Not Found

```
FileNotFoundError: Model not found at /mnt/models/llama-2-7b
```

**Solution**:
1. Check shared folder is mounted: `ls /mnt/shared`
2. Verify model path in script matches actual location
3. Download model if not present

### Issue 4: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution** (for CPU-only RISC-V):
```python
# In llama_inference.py, use smaller model or reduce batch size
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Use FP16
    low_cpu_mem_usage=True,
    device_map="auto"
)
```

### Issue 5: Slow Inference

LLaMA inference on QEMU RISC-V CPU is slow (expected). For faster testing:
- Use TinyLlama instead of LLaMA 2 7B
- Reduce `max_new_tokens` to 10-20
- Use smaller prompts

## Performance Expectations

### QEMU RISC-V CPU-only:
- **LLaMA 2 7B**: ~1-5 tokens/second
- **TinyLlama 1.1B**: ~10-20 tokens/second
- **First token latency**: 10-30 seconds

### With SST Accelerator Simulation:
- Simulated performance (not real-time)
- Detailed hardware statistics
- Memory access patterns
- Accelerator utilization

## Example Output

```
============================================================
LLAMA 2 Inference with SST Accelerator
============================================================

Loading model from /mnt/shared/tinyllama...
Model loaded successfully
Model parameters: 1,100,048,384
Model dtype: torch.float16

Prompt: What is the meaning of life?
Generating 50 tokens...

Output: The meaning of life is a question that has puzzled
philosophers and thinkers for centuries. While there is no
single answer that applies to everyone, many believe that
finding purpose and happiness is the key to a fulfilling life.

Generation time: 45.2 seconds
Tokens generated: 50
Tokens per second: 1.11

SST Accelerator Statistics:
  Matrix operations: 15,234
  Memory accesses: 1,256,789
  Cache hit rate: 87.3%
  Average latency: 125.4 cycles

============================================================
```

## Directory Structure

```
llama/
├── llama_inference.py      # Main inference script
├── llama_sst_backend.py    # SST accelerator backend
├── sst_config_llama.py     # SST configuration
├── run_sst.sh              # Launch SST simulator
├── run.sh                  # Combined launcher
├── test_prompts.txt        # Sample prompts
└── Makefile                # Build automation
```

## Advanced Usage

### Custom Accelerator Configuration

Edit `sst_config_llama.py`:

```python
# Configure memory hierarchy
l1_cache_size = "64KB"
l2_cache_size = "1MB"
memory_size = "8GB"

# Configure accelerator
num_processing_elements = 64
systolic_array_size = 16x16
```

### Batch Inference

```python
prompts = [
    "What is AI?",
    "Explain deep learning.",
    "What is PyTorch?"
]

for prompt in prompts:
    generate_text(prompt, tokenizer, model, backend)
```

### Export Statistics

```python
# In llama_sst_backend.py
stats = backend.get_stats()
with open('inference_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

## Next Steps

1. ✅ **Test basic inference** with TinyLlama
2. ✅ **Profile performance** with different models
3. ✅ **Tune accelerator config** for optimization
4. ✅ **Scale to larger models** (LLaMA 2 13B, 70B)
5. ✅ **Integrate with applications** (chatbots, etc.)

## References

- **LLaMA 2 Paper**: https://arxiv.org/abs/2307.09288
- **PyTorch**: https://pytorch.org
- **Transformers**: https://huggingface.co/docs/transformers
- **SST-Core**: http://sst-simulator.org

---

**Last Updated**: 2025-11-23
**Status**: Experimental (QEMU RISC-V + PyTorch)
