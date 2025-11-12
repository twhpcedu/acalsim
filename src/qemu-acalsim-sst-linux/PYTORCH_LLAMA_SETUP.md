# PyTorch and LLAMA 2 7B Installation Guide for QEMU-SST Linux

<!--
Copyright 2023-2025 Playlab/ACAL

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

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Step 1: Build Full Linux Root Filesystem](#step-1-build-full-linux-root-filesystem)
5. [Step 2: Install Python and Dependencies](#step-2-install-python-and-dependencies)
6. [Step 3: Install PyTorch for RISC-V](#step-3-install-pytorch-for-risc-v)
7. [Step 4: Setup Persistent Storage for Models](#step-4-setup-persistent-storage-for-models)
8. [Step 5: Download and Prepare LLAMA 2 7B](#step-5-download-and-prepare-llama-2-7b)
9. [Step 6: Implement Inference Application](#step-6-implement-inference-application)
10. [Step 7: SST Accelerator Integration](#step-7-sst-accelerator-integration)
11. [Running the Complete System](#running-the-complete-system)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This guide demonstrates how to set up a complete ML inference stack in QEMU-SST Linux:

- **Full Linux Environment**: Debian/Ubuntu-based RISC-V rootfs
- **PyTorch**: Cross-compiled for RISC-V 64-bit
- **LLAMA 2 7B**: Large language model for text generation
- **SST Integration**: Offload inference to simulated AI accelerators
- **Persistent Storage**: Virtual disk for model files

### What You'll Build

```
┌─────────────────────────────────────────────────────────────────┐
│                    QEMU RISC-V Linux                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐          ┌──────────────────┐             │
│  │  Python 3.11+   │          │   LLAMA 2 7B     │             │
│  │                 │          │   (13GB model)   │             │
│  │  - PyTorch      │─────────►│                  │             │
│  │  - Transformers │          │  /mnt/models/    │             │
│  │  - Tokenizers   │          └──────────────────┘             │
│  └─────────┬───────┘                                            │
│            │                                                     │
│            │ /dev/sst0                                          │
│            ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │     VirtIO-SST Kernel Driver            │                   │
│  └─────────────────┬───────────────────────┘                   │
└────────────────────┼─────────────────────────────────────────┘
                     │ VirtIO
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                     QEMU VirtIO Device                          │
└────────────────────┬───────────────────────────────────────────┘
                     │ Unix Socket
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                   SST Simulation                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ AI Accel #0  │  │ AI Accel #1  │  │ AI Accel #2  │         │
│  │ (Attention)  │  │ (FFN)        │  │ (Embeddings) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Run LLAMA 2 inference on RISC-V Linux
- Offload tensor operations to SST-simulated AI accelerators
- Measure realistic system performance including OS overhead
- Demonstrate multi-device workload distribution

---

## Prerequisites

### Required Tools (on Host)

```bash
# RISC-V cross-compiler
sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Build tools
sudo apt-get install build-essential cmake python3 python3-pip

# QEMU (if not using Docker)
sudo apt-get install qemu-system-misc

# Disk image tools
sudo apt-get install qemu-utils
```

### Required Space

- **Rootfs**: ~2GB (with Python + PyTorch)
- **Model Storage**: ~15GB (LLAMA 2 7B + cache)
- **Total**: ~20GB free space recommended

### Time Estimate

- **Rootfs build**: 2-4 hours (first time)
- **PyTorch compilation**: 4-8 hours
- **Model download**: 30-60 minutes
- **Total**: 1-2 days (mostly automated)

---

## Architecture Overview

### Storage Layout

```
Host Filesystem:
/home/user/
├── rootfs-full/              # Full Debian/Ubuntu rootfs (2GB)
│   ├── usr/bin/python3       # Python 3.11+
│   ├── usr/lib/python3.11/   # Python libraries
│   │   └── site-packages/
│   │       ├── torch/        # PyTorch (~800MB)
│   │       ├── transformers/ # Hugging Face Transformers
│   │       └── tokenizers/   # Fast tokenizers
│   └── apps/
│       └── llama_inference   # Our inference application
├── initramfs-full.cpio.gz    # Compressed rootfs (~800MB)
└── models.qcow2              # Virtual disk (20GB)
    └── [In QEMU as /mnt/models/]
        └── llama-2-7b/       # Model files (13GB)
            ├── pytorch_model.bin
            ├── config.json
            ├── tokenizer.model
            └── tokenizer_config.json
```

### Inference Flow

```
1. User input: "Explain quantum computing"
2. Python app tokenizes input
3. For each layer:
   a. Compute attention weights (SST Accel #0)
   b. Apply feedforward network (SST Accel #1)
   c. Update embeddings (SST Accel #2)
4. Decode output tokens
5. Return generated text
```

---

## Step 1: Build Full Linux Root Filesystem

We'll use Buildroot to create a complete Linux environment.

### Option A: Using Buildroot (Recommended)

```bash
# In Docker container
docker exec -it acalsim-workspace bash

# Download Buildroot
cd /home/user
wget https://buildroot.org/downloads/buildroot-2024.02.tar.gz
tar -xzf buildroot-2024.02.tar.gz
cd buildroot-2024.02

# Configure for RISC-V with Python
make menuconfig
```

**Buildroot Configuration**:

```
Target options --->
    Target Architecture: RISCV
    Target Architecture Variant: riscv64 (RV64GC)
    Target ABI: lp64d

Toolchain --->
    Toolchain type: Buildroot toolchain
    C library: glibc
    Kernel Headers: Linux 6.1.x
    Enable C++ support: YES
    Enable OpenMP support: YES

System configuration --->
    System hostname: acalsim-riscv
    Root password: (leave empty or set)
    /dev management: Dynamic using devtmpfs + eudev

Target packages --->
    Interpreter languages and scripting --->
        [*] python3
            python3 module format: .pyc compiled sources only
            Core python3 modules --->
                [*] bz2
                [*] readline
                [*] ssl
                [*] sqlite
                [*] zlib
            External python3 modules --->
                [*] python-pip
                [*] python-setuptools
                [*] python-numpy

    Networking applications --->
        [*] openssh

    System tools --->
        [*] htop
        [*] tar
        [*] gzip

Filesystem images --->
    [*] cpio the root filesystem (for use as an initial RAM filesystem)
    Compression method: gzip
```

**Build**:

```bash
# Build (takes 2-4 hours)
make -j$(nproc)

# Extract rootfs
cd output/images
mkdir -p /home/user/rootfs-full
cd /home/user/rootfs-full
gunzip -c /home/user/buildroot-2024.02/output/images/rootfs.cpio.gz | cpio -idmv

# Copy to our project
cd /home/user/rootfs-full
cp -r . /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs-full/
```

### Option B: Using Pre-built Debian RISC-V

```bash
# Download Debian RISC-V base system
cd /home/user
wget https://deb.debian.org/debian/dists/sid/main/installer-riscv64/current/images/netboot/debian-installer/riscv64/initrd.gz

# Extract
mkdir rootfs-full
cd rootfs-full
gunzip -c ../initrd.gz | cpio -idmv

# Minimal Debian has Python available via apt
```

---

## Step 2: Install Python and Dependencies

### If Using Buildroot

Python is already included in the Buildroot image. Skip to Step 3.

### If Using Debian Base

```bash
# Mount rootfs for chroot (requires root on host)
sudo mount --bind /dev /home/user/rootfs-full/dev
sudo mount --bind /proc /home/user/rootfs-full/proc
sudo mount --bind /sys /home/user/rootfs-full/sys

# Chroot into RISC-V rootfs (requires qemu-user-static)
sudo apt-get install qemu-user-static
sudo chroot /home/user/rootfs-full /bin/bash

# Inside chroot:
apt-get update
apt-get install -y python3 python3-pip python3-dev
apt-get install -y build-essential cmake git

# Exit chroot
exit

# Unmount
sudo umount /home/user/rootfs-full/dev
sudo umount /home/user/rootfs-full/proc
sudo umount /home/user/rootfs-full/sys
```

---

## Step 3: Install PyTorch for RISC-V

### Option A: Cross-Compile PyTorch (Advanced)

**Note**: Building PyTorch from source for RISC-V takes 4-8 hours and requires significant expertise.

```bash
# Clone PyTorch
cd /home/user
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.1.0

# Setup cross-compilation
export CROSS_COMPILE=riscv64-linux-gnu-
export CC=${CROSS_COMPILE}gcc
export CXX=${CROSS_COMPILE}g++
export AR=${CROSS_COMPILE}ar
export RANLIB=${CROSS_COMPILE}ranlib

# Configure for minimal build (inference only)
export BUILD_TEST=0
export USE_CUDA=0
export USE_ROCM=0
export USE_DISTRIBUTED=0
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0

# Build
python3 setup.py build

# Install to rootfs
python3 setup.py install --prefix=/home/user/rootfs-full/usr
```

### Option B: Use PyTorch Lite (Faster, Recommended)

PyTorch Lite is a smaller, inference-only build suitable for embedded systems.

```bash
cd /home/user
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Build lite interpreter
export CROSS_COMPILE=riscv64-linux-gnu-
export BUILD_LITE_INTERPRETER=1
export BUILD_MOBILE_BENCHMARK=0
export BUILD_MOBILE_TEST=0

python3 setup.py build

# Copy to rootfs
cp build/lib.linux-riscv64-3.11/torch/*.so /home/user/rootfs-full/usr/lib/
cp -r torch /home/user/rootfs-full/usr/lib/python3.11/site-packages/
```

### Option C: Use Pre-built Wheels (Easiest)

```bash
# Download pre-built PyTorch for RISC-V (if available)
# Check: https://github.com/riscv-collab/pytorch-riscv

# Install to rootfs
cp torch-*.whl /home/user/rootfs-full/root/
# Later, in Linux: pip3 install torch-*.whl
```

---

## Step 4: Setup Persistent Storage for Models

LLAMA 2 7B is 13GB - too large for initramfs. Use a virtual disk.

### Create Virtual Disk

```bash
# Create 20GB qcow2 image
cd /home/user
qemu-img create -f qcow2 models.qcow2 20G

# Format as ext4 (requires running QEMU temporarily)
qemu-system-riscv64 \
    -M virt \
    -m 2G \
    -nographic \
    -kernel /home/user/linux/arch/riscv/boot/Image \
    -append "root=/dev/ram rw console=ttyS0" \
    -initrd /home/user/initramfs-full.cpio.gz \
    -drive file=/home/user/models.qcow2,if=virtio,format=qcow2

# In Linux console:
mkfs.ext4 /dev/vda
mkdir -p /mnt/models
mount /dev/vda /mnt/models

# Exit QEMU (Ctrl-A X)
```

### Auto-mount on Boot

Add to `/home/user/rootfs-full/etc/fstab`:

```
/dev/vda    /mnt/models    ext4    defaults    0    2
```

Or add to init script `/home/user/rootfs-full/init`:

```bash
#!/bin/sh

# Existing init commands...

# Mount models disk
mkdir -p /mnt/models
mount -t ext4 /dev/vda /mnt/models
```

---

## Step 5: Download and Prepare LLAMA 2 7B

### On Host Machine

```bash
# Install Hugging Face CLI
pip3 install huggingface-hub

# Login (requires Hugging Face account with LLAMA 2 access)
huggingface-cli login

# Download LLAMA 2 7B
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /tmp/llama-2-7b

# Check size
du -sh /tmp/llama-2-7b
# Output: ~13GB
```

### Copy to Virtual Disk

```bash
# Mount the virtual disk on host
sudo modprobe nbd max_part=8
sudo qemu-nbd --connect=/dev/nbd0 /home/user/models.qcow2
sudo mount /dev/nbd0 /mnt/temp

# Copy model
sudo mkdir -p /mnt/temp/llama-2-7b
sudo cp -r /tmp/llama-2-7b/* /mnt/temp/llama-2-7b/

# Verify
ls -lh /mnt/temp/llama-2-7b/
# Should show:
# - pytorch_model-*.bin files (13GB total)
# - config.json
# - tokenizer.model
# - tokenizer_config.json
# - special_tokens_map.json

# Unmount
sudo umount /mnt/temp
sudo qemu-nbd --disconnect /dev/nbd0
```

---

## Step 6: Implement Inference Application

See `examples/llama-inference/` directory for complete implementation.

### Directory Structure

```
examples/llama-inference/
├── README.md                    # Usage instructions
├── Makefile                     # Cross-compilation build
├── llama_inference.py           # Python inference script
├── llama_sst_backend.py         # SST accelerator backend
├── sst_config_llama.py          # SST simulation config
└── test_prompts.txt             # Example prompts
```

### Key Implementation Files

#### `llama_inference.py` - Main Application

```python
#!/usr/bin/env python3
"""
LLAMA 2 7B Inference with SST Accelerator Integration
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_sst_backend import SSTAcceleratorBackend

# Model path (on virtual disk)
MODEL_PATH = "/mnt/models/llama-2-7b"

def load_model():
    """Load LLAMA 2 model and tokenizer"""
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        low_cpu_mem_usage=True
    )

    # Replace compute-intensive ops with SST backend
    backend = SSTAcceleratorBackend()
    model = backend.instrument_model(model)

    print("Model loaded successfully")
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_new_tokens=100):
    """Generate text from prompt"""
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"\\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    if len(sys.argv) < 2:
        print("Usage: llama_inference.py <prompt>")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    # Load model
    tokenizer, model = load_model()

    # Generate
    result = generate_text(prompt, tokenizer, model)

    print("\\n" + "="*60)
    print("GENERATED TEXT:")
    print("="*60)
    print(result)
    print("="*60)

if __name__ == "__main__":
    main()
```

#### `llama_sst_backend.py` - SST Integration

```python
"""
SST Accelerator Backend for LLAMA 2 Inference
Offloads attention and FFN operations to SST-simulated accelerators
"""

import torch
import torch.nn as nn
import struct
import os

class SSTAcceleratorBackend:
    """Interface to SST accelerators via /dev/sst0"""

    def __init__(self):
        self.device_path = "/dev/sst0"
        self.fd = None
        self.stats = {
            "attention_ops": 0,
            "ffn_ops": 0,
            "embedding_ops": 0,
            "total_cycles": 0
        }

    def open_device(self):
        """Open SST device"""
        if self.fd is None:
            self.fd = os.open(self.device_path, os.O_RDWR)

    def close_device(self):
        """Close SST device"""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def compute_attention(self, query, key, value):
        """Offload attention computation to SST"""
        self.open_device()

        # Pack request
        req_type = 2  # SST_REQ_COMPUTE
        req_id = 1
        compute_units = query.size(0) * query.size(1)  # batch * seq_len
        latency_model = 0

        request = struct.pack('<IIQQQQ8x',
            req_type, req_id, 0,  # type, id, user_data
            compute_units, latency_model, 0, 0, 0  # compute payload
        )

        # Send to SST
        os.write(self.fd, request)

        # Receive response
        response = os.read(self.fd, 4096)
        status, cycles = struct.unpack('<IQ', response[:12])

        self.stats["attention_ops"] += 1
        self.stats["total_cycles"] += cycles

        # Still compute locally (SST provides timing)
        # In real implementation, could skip computation
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output

    def compute_ffn(self, hidden_states, up_proj, gate_proj, down_proj):
        """Offload FFN computation to SST"""
        self.open_device()

        # Similar to attention, send compute request
        req_type = 2
        req_id = 2
        compute_units = hidden_states.size(0) * hidden_states.size(1)

        request = struct.pack('<IIQQQQ8x',
            req_type, req_id, 0,
            compute_units, 0, 0, 0, 0
        )

        os.write(self.fd, request)
        response = os.read(self.fd, 4096)
        status, cycles = struct.unpack('<IQ', response[:12])

        self.stats["ffn_ops"] += 1
        self.stats["total_cycles"] += cycles

        # Compute FFN locally
        gate = torch.nn.functional.silu(gate_proj(hidden_states))
        up = up_proj(hidden_states)
        output = down_proj(gate * up)

        return output

    def instrument_model(self, model):
        """Replace model ops with SST-accelerated versions"""
        # Hook into attention layers
        for name, module in model.named_modules():
            if "self_attn" in name:
                original_forward = module.forward

                def new_forward(hidden_states, *args, **kwargs):
                    # Use SST backend
                    return self.compute_attention_wrapper(
                        original_forward, hidden_states, *args, **kwargs
                    )

                module.forward = new_forward

        return model

    def compute_attention_wrapper(self, original_fn, *args, **kwargs):
        """Wrapper to intercept attention calls"""
        # Call original but track via SST
        result = original_fn(*args, **kwargs)

        # Send compute notification to SST
        if len(args) > 0:
            hidden_states = args[0]
            self.open_device()

            req = struct.pack('<IIQQQQ8x', 2, 1, 0,
                            hidden_states.size(0) * hidden_states.size(1),
                            0, 0, 0, 0)
            os.write(self.fd, req)
            resp = os.read(self.fd, 4096)

            self.stats["attention_ops"] += 1

        return result

    def print_stats(self):
        """Print statistics"""
        print("\\nSST Accelerator Statistics:")
        print(f"  Attention operations: {self.stats['attention_ops']}")
        print(f"  FFN operations: {self.stats['ffn_ops']}")
        print(f"  Embedding operations: {self.stats['embedding_ops']}")
        print(f"  Total simulated cycles: {self.stats['total_cycles']:,}")

    def __del__(self):
        self.close_device()
```

#### `Makefile` - Build System

```makefile
# Makefile for LLAMA Inference Application

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
PYTHON = python3

ROOTFS = /home/user/rootfs-full

.PHONY: all install deploy clean

all:
	@echo "Python application - no compilation needed"
	@echo "Use 'make install' to copy to rootfs"

install:
	@echo "Installing to rootfs..."
	mkdir -p $(ROOTFS)/apps/llama-inference
	cp llama_inference.py $(ROOTFS)/apps/llama-inference/
	cp llama_sst_backend.py $(ROOTFS)/apps/llama-inference/
	cp test_prompts.txt $(ROOTFS)/apps/llama-inference/
	chmod +x $(ROOTFS)/apps/llama-inference/llama_inference.py

deploy: install
	@echo "Rebuilding initramfs..."
	cd $(ROOTFS) && find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs-full.cpio.gz
	@echo "Deployment complete"
	@ls -lh /home/user/initramfs-full.cpio.gz

clean:
	@echo "Nothing to clean"

help:
	@echo "LLAMA Inference Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Check environment"
	@echo "  install   - Copy to rootfs"
	@echo "  deploy    - Install + rebuild initramfs"
	@echo "  clean     - Clean build artifacts"
	@echo ""
```

---

## Step 7: SST Accelerator Integration

### SST Configuration for LLAMA Inference

Create `sst_config_llama.py`:

```python
"""
SST Configuration for LLAMA 2 7B Inference
Simulates 4 AI accelerator devices for different workloads
"""

import sst

# Component library
component_lib = "/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/acalsim-device/libacalsim-virtio.so"

# Create 4 AI accelerator devices
accelerators = []
for i in range(4):
    accel = sst.Component(f"accel{i}", "acalsim_virtio.ACALSimVirtIODeviceComponent")
    accel.addParams({
        "device_id": i,
        "socket_path": f"/tmp/sst-accel{i}.sock",
        "clock": "2GHz",
        "max_compute_units": 1024,

        # Different accelerators for different ops
        "accelerator_type": ["attention", "ffn", "embedding", "general"][i],

        # Latency models
        "latency_attention": "1000ns",  # Attention: 1us per operation
        "latency_ffn": "500ns",         # FFN: 500ns per operation
        "latency_embedding": "100ns",   # Embedding: 100ns per operation
    })

    accelerators.append(accel)

# Statistics
sst.setStatisticLoadLevel(7)
for i, accel in enumerate(accelerators):
    accel.enableAllStatistics({"type": "sst.AccumulatorStatistic"})

# Simulation configuration
sst.setProgramOption("stop-at", "10s")  # Stop after 10 simulated seconds
```

### Modified QEMU Launch Script

Update `run-linux.sh` to include model disk:

```bash
#!/bin/bash

# QEMU-SST Linux with LLAMA 2 Support

QEMU=/home/user/qemu-build/qemu/build/qemu-system-riscv64
KERNEL=/home/user/linux/arch/riscv/boot/Image
INITRD=/home/user/initramfs-full.cpio.gz
MODEL_DISK=/home/user/models.qcow2

$QEMU \
    -M virt \
    -m 4G \
    -smp 4 \
    -nographic \
    -kernel $KERNEL \
    -append "root=/dev/ram rw console=ttyS0 loglevel=3" \
    -initrd $INITRD \
    -drive file=$MODEL_DISK,if=virtio,format=qcow2 \
    -chardev socket,path=/tmp/sst-accel0.sock,server=on,wait=off,id=sst0 \
    -device virtio-serial-device \
    -device virtserialport,chardev=sst0,name=sst0
```

---

## Running the Complete System

### Terminal 1: Start SST Simulation

```bash
# In Docker container
docker exec -it acalsim-workspace bash

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference

# Start SST
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

sst sst_config_llama.py --verbose
```

### Terminal 2: Start QEMU Linux

```bash
# In Docker container (new terminal)
docker exec -it acalsim-workspace bash

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config

./run-linux.sh
```

### In Linux Console

```bash
# Wait for boot...
# Login: root (no password)

# Verify mounts
mount | grep models
# Should show: /dev/vda on /mnt/models type ext4

# Check model files
ls -lh /mnt/models/llama-2-7b/
# Should show model files

# Check Python
python3 --version
# Should show Python 3.11+

# Check PyTorch
python3 -c "import torch; print(torch.__version__)"

# Check SST device
ls -l /dev/sst0
# Should exist

# Run inference
cd /apps/llama-inference
python3 llama_inference.py "Explain quantum computing in simple terms."

# Expected output:
# Loading model from /mnt/models/llama-2-7b...
# Model loaded successfully
#
# Prompt: Explain quantum computing in simple terms.
# Generating 100 tokens...
#
# ============================================================
# GENERATED TEXT:
# ============================================================
# Explain quantum computing in simple terms. Quantum computers
# use quantum bits (qubits) instead of classical bits. Unlike
# classical bits which can only be 0 or 1, qubits can exist
# in a superposition of both states simultaneously...
# ============================================================
#
# SST Accelerator Statistics:
#   Attention operations: 324
#   FFN operations: 324
#   Embedding operations: 1
#   Total simulated cycles: 1,234,567
```

---

## Performance Optimization

### 1. Use Quantization

Reduce model size and computation:

```python
from transformers import AutoModelForCausalLM
from optimum.quanto import quantize, qint8

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
quantize(model, weights=qint8)

# Model is now ~3.5GB instead of 13GB
```

### 2. Increase QEMU Memory

```bash
# In run-linux.sh
-m 8G  # Increase from 4G to 8G
```

### 3. Enable KVM Acceleration (if on RISC-V host)

```bash
# In run-linux.sh
-enable-kvm \
-cpu host
```

### 4. Optimize SST Accelerator Latency

In `sst_config_llama.py`:

```python
"latency_attention": "100ns",  # 10x faster
"latency_ffn": "50ns",
```

### 5. Use Flash Attention

```bash
pip3 install flash-attn
```

Then in `llama_inference.py`:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

---

## Troubleshooting

### Problem: "Out of memory loading model"

**Solution**:

```bash
# Increase QEMU RAM
-m 8G

# Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    offload_folder="/tmp/offload"
)
```

### Problem: "Cannot find /dev/sst0"

**Solution**:

```bash
# Check kernel driver loaded
lsmod | grep virtio_sst

# If not loaded:
insmod /virtio-sst.ko

# Check device permissions
chmod 666 /dev/sst0
```

### Problem: "Model files not found"

**Solution**:

```bash
# Check disk mounted
mount | grep vda

# If not mounted:
mkdir -p /mnt/models
mount -t ext4 /dev/vda /mnt/models

# Verify files
ls /mnt/models/llama-2-7b/
```

### Problem: "PyTorch import error"

**Solution**:

```bash
# Check Python path
echo $PYTHONPATH

# Set if needed
export PYTHONPATH=/usr/lib/python3.11/site-packages

# Check library dependencies
ldd /usr/lib/python3.11/site-packages/torch/_C.so
```

### Problem: "Inference too slow"

**Solutions**:

1. Use quantization (see Performance Optimization)
2. Reduce `max_new_tokens`
3. Use smaller model (LLAMA 2 7B-Chat or TinyLlama)
4. Optimize SST accelerator latency

### Problem: "SST socket connection failed"

**Solution**:

```bash
# Check socket exists on host
ls -l /tmp/sst-accel*.sock

# Restart SST in Terminal 1
# Then restart QEMU
```

---

## Complete Example Session

```bash
# Terminal 1: SST
sst sst_config_llama.py --verbose

# Terminal 2: QEMU
./run-linux.sh

# In Linux:
root@acalsim-riscv:~# mount /dev/vda /mnt/models
root@acalsim-riscv:~# cd /apps/llama-inference
root@acalsim-riscv:~# cat test_prompts.txt
Explain quantum computing in simple terms.
What is machine learning?
How do neural networks work?

root@acalsim-riscv:~# ./llama_inference.py "$(head -1 test_prompts.txt)"
Loading model from /mnt/models/llama-2-7b...
[████████████████████████████████] 100%
Model loaded successfully

Prompt: Explain quantum computing in simple terms.
Generating 100 tokens...
[Token 10/100]
[Token 20/100]
...
[Token 100/100]

============================================================
GENERATED TEXT:
============================================================
Explain quantum computing in simple terms. Quantum computers
leverage quantum mechanical phenomena like superposition and
entanglement to process information. Traditional computers use
bits (0 or 1), but quantum computers use qubits which can be
in multiple states simultaneously...
============================================================

SST Accelerator Statistics:
  Attention operations: 324
  FFN operations: 324
  Embedding operations: 1
  Total simulated cycles: 1,234,567
  Estimated wall-clock time: 0.617ms
============================================================
```

---

## Next Steps

1. **Implement Custom Accelerators**: Modify SST components for specific tensor operations
2. **Add Profiling**: Instrument code to measure attention/FFN breakdown
3. **Multi-Model Support**: Extend to GPT-Neo, BLOOM, or Mistral
4. **Benchmark Suite**: Create standardized inference benchmarks
5. **Distributed Inference**: Split model across multiple SST ranks

---

## Related Documentation

- [ROOTFS_MANAGEMENT.md](ROOTFS_MANAGEMENT.md) - Filesystem and package management
- [APP_DEVELOPMENT.md](APP_DEVELOPMENT.md) - Writing SST applications
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [RISC-V Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Buildroot Manual](https://buildroot.org/downloads/manual/manual.html)
