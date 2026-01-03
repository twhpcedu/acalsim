# ACALSim SST Integration - Top-Level Architecture

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

## Overview

This document provides a comprehensive top-level architecture diagram for the ACALSim SST integration project, showing all major components, their relationships, and data flows.

## Architecture Layers

The SST integration consists of multiple architectural layers:

### Layer 1: Core ACALSim Framework
The foundational event-driven simulation framework providing the building blocks.

### Layer 2: SST Integration Bridge
Components that bridge ACALSim's event-driven model to SST's discrete-event simulation.

### Layer 3: Application Components
Specific simulator implementations (RISC-V, QEMU, Devices, etc.).

### Layer 4: External Interfaces
Connections to QEMU, PyTorch, Linux kernel drivers, and user applications.

---

## Top-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          SST-Core Framework (External)                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ • Discrete Event Simulation Engine                                                 │ │
│  │ • Python Configuration System (sst.Component, sst.Link, sst.Params)               │ │
│  │ • Component Registry & Element Library Interface (ELI)                            │ │
│  │ • Statistics Collection & Visualization                                           │ │
│  │ • Clock/Time Management                                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ SST Component Interface
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Layer 1: Core ACALSim Framework (libacalsim.so)                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  SimTopBase                                                                        │ │
│  │  • ThreadManager (2-phase execution: Phase1=parallel, Phase2=sync)               │ │
│  │  • EventQueue & Fast-forward mechanism                                           │ │
│  │  • Global tick management                                                        │ │
│  │                                                                                   │ │
│  │  SimBase                                                                          │ │
│  │  • Event-driven simulator base class                                             │ │
│  │  • init() → step() → cleanup() lifecycle                                         │ │
│  │                                                                                   │ │
│  │  Communication Infrastructure                                                     │ │
│  │  • SimPort/SimChannel (port-based communication)                                 │ │
│  │  • SimPacket (data containers)                                                   │ │
│  │  • Channel toggle & dual-queue synchronization                                   │ │
│  │                                                                                   │ │
│  │  Configuration & Utilities                                                        │ │
│  │  • SimConfig (JSON configuration management)                                     │ │
│  │  • RecycleContainer (memory pool management)                                     │ │
│  │  • Profiling & Statistics                                                        │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ Inheritance & Wrapping
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│              Layer 2: SST Integration Bridge (libacalsim_sst.so)                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  ACALSimComponent (Base SST Component Wrapper)                                    │ │
│  │  • Bridges SST::Component ↔ ACALSim SimBase                                      │ │
│  │  • Maps SST clock ticks → ACALSim iterations                                     │ │
│  │  • clockTick() orchestrates Phase1 + Phase2                                      │ │
│  │  • SimPort ↔ SST::Link mapping                                                   │ │
│  │  • SST::Params → SimConfig translation                                           │ │
│  │  • Primary component pattern (registerAsPrimaryComponent)                        │ │
│  │  • 2-phase termination (ExitEvent → cleanup)                                     │ │
│  │                                                                                   │ │
│  │  ACALSimSSTEvent                                                                  │ │
│  │  • SST::Event wrapper for SimPacket                                              │ │
│  │  • Enables ACALSim ↔ SST communication                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ Concrete Implementations
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│              Layer 3: Application Components (SST Elements)                              │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Path A: Standalone RISC-V Simulator                                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  RISCVSoCStandalone (acalsim.RISCVSoCStandalone)                             │ │ │
│  │  │  • Complete RISC-V RV32I simulator as single SST component                  │ │ │
│  │  │  • Wraps SOCTop (SimTopBase-derived)                                        │ │ │
│  │  │                                                                              │ │ │
│  │  │  Internal Components:                                                        │ │ │
│  │  │  ├─ SOC (orchestrator)                                                       │ │ │
│  │  │  ├─ CPU (RV32I ISA implementation)                                           │ │ │
│  │  │  ├─ IFStage (Instruction Fetch)                                              │ │ │
│  │  │  ├─ EXEStage (Execution)                                                     │ │ │
│  │  │  ├─ WBStage (Writeback)                                                      │ │ │
│  │  │  ├─ DataMemory (memory model)                                                │ │ │
│  │  │  └─ PipeRegs (pipeline registers)                                            │ │ │
│  │  │                                                                              │ │ │
│  │  │  Python Config: riscv_single_core.py, riscv_dual_core.py                   │ │ │
│  │  │  Assembly Input: *.txt files (branch_simple.txt, etc.)                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Path B: QEMU Full-System Integration                                             │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  QEMUBinaryComponent (qemubinary.QEMUBinary)                                 │ │ │
│  │  │  • Manages QEMU subprocess (qemu-system-riscv32)                            │ │ │
│  │  │  • Binary MMIO protocol via Unix sockets                                    │ │ │
│  │  │  • N-device routing (multi-device support)                                  │ │ │
│  │  │  • Routes memory transactions to device components                          │ │ │
│  │  │                                                                              │ │ │
│  │  │  Communication:                                                              │ │ │
│  │  │  • Unix Socket: /tmp/qemu-sst-mmio.sock                                     │ │ │
│  │  │  • Protocol: MMIORequest/MMIOResponse (binary, packed structs)             │ │ │
│  │  │  • Multi-socket mode: per-device socket connections                         │ │ │
│  │  │                                                                              │ │ │
│  │  │  SST Links:                                                                  │ │ │
│  │  │  • device_port (legacy single device)                                       │ │ │
│  │  │  • device_port_0, device_port_1, ... (N-device mode)                       │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Path C: Device Components                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  ACALSimDeviceComponent (acalsim.QEMUDevice)                                 │ │ │
│  │  │  • Simple echo device model                                                  │ │ │
│  │  │  • Register map: DATA_IN, DATA_OUT, STATUS, CONTROL                         │ │ │
│  │  │  • Configurable latency                                                      │ │ │
│  │  │                                                                              │ │ │
│  │  │  ACALSimComputeDeviceComponent                                               │ │ │
│  │  │  • Compute accelerator model                                                 │ │ │
│  │  │  • GEMM operations (matrix multiply)                                         │ │ │
│  │  │  • Cycle-accurate timing model                                               │ │ │
│  │  │                                                                              │ │ │
│  │  │  ACALSimVirtIODeviceComponent                                                │ │ │
│  │  │  • VirtIO-SST device model                                                   │ │ │
│  │  │  • Protocol: PING, ECHO, COMPUTE commands                                    │ │ │
│  │  │  • Used by PyTorch device GEMM offloading                                    │ │ │
│  │  │                                                                              │ │ │
│  │  │  HSAComputeComponent / HSAHostComponent                                      │ │ │
│  │  │  • HSA (Heterogeneous System Architecture) support                          │ │ │
│  │  │  • Multi-accelerator coordination                                            │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Path D: Generic Component Template                                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  SimpleProcessor (acalsim.SimpleProcessor)                                   │ │ │
│  │  │  • Example processor for testing                                             │ │ │
│  │  │  • Reference implementation pattern                                          │ │ │
│  │  │                                                                              │ │ │
│  │  │  ACALSimComponentFactory<T>                                                  │ │ │
│  │  │  • Template for custom component creation                                    │ │ │
│  │  │  • T must derive from SimBase                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ External Interfaces
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│              Layer 4: External Interfaces & Applications                                 │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  QEMU Process (qemu-system-riscv32)                                               │ │
│  │  • Full RISC-V system emulation                                                   │ │
│  │  • Custom Linux kernel 6.18.0-rc6                                                 │ │
│  │  • VirtIO-SST device driver (virtio-sst.ko)                                       │ │
│  │  • Debian/Buildroot rootfs                                                        │ │
│  │  • Communicates via Unix sockets to QEMUBinaryComponent                          │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Linux Kernel Drivers                                                             │ │
│  │  • virtio-sst.ko (VirtIO-SST kernel module)                                       │ │
│  │  • Character device: /dev/sst0                                                    │ │
│  │  • ioctl interface for user-space access                                          │ │
│  │  • Protocol parser (PING, ECHO, COMPUTE commands)                                 │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  PyTorch Integration (Device GEMM Offloading)                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Docker Container (PyTorch Training)                                         │ │ │
│  │  │  • PyTorch 2.x with custom operators                                         │ │ │
│  │  │  • device_gemm() operator (autograd function)                                │ │ │
│  │  │  • DeviceLinear() layer (nn.Module)                                          │ │ │
│  │  │  • TCP connection to QEMU device server                                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                          ↓ TCP                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  QEMU Guest (Device Server)                                                  │ │ │
│  │  │  • qemu_device_server_virtio.py                                              │ │ │
│  │  │  • Receives GEMM requests via TCP                                            │ │ │
│  │  │  • Forwards to /dev/sst0 (VirtIO-SST)                                        │ │ │
│  │  │  • Returns cycle-accurate timing                                             │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                          ↓ VirtIO                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  SST Simulator (ACALSimVirtIODeviceComponent)                                │ │ │
│  │  │  • Cycle-accurate GEMM simulation                                            │ │ │
│  │  │  • Returns timing information                                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  User Applications                                                                │ │
│  │  • HSA multi-accelerator demos (hsa_multi_accel_demo.c)                          │ │
│  │  • Llama inference with SST backend (llama_inference.py)                         │ │
│  │  • Custom C/C++ applications using /dev/sst0                                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Data Flow Diagrams

### Data Flow 1: Standalone RISC-V Simulation

```
┌─────────────┐
│ User writes │
│ Python      │
│ config      │
│ (.py)       │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│ SST Python Configuration                            │
│                                                     │
│ comp = sst.Component("riscv",                       │
│                       "acalsim.RISCVSoCStandalone") │
│ comp.addParams({                                    │
│   "asm_file": "branch_simple.txt",                  │
│   "clock": "1GHz",                                  │
│   "memory_size": 65536                              │
│ })                                                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ SST-Core: Launch Simulation                         │
│ • Parse config                                      │
│ • Instantiate RISCVSoCStandalone                    │
│ • Register clock handler                            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ RISCVSoCStandalone::init()                          │
│ • Create SOCTop instance                            │
│ • Load assembly file                                │
│ • Initialize CPU, IFStage, EXEStage, WBStage        │
│ • Register simulators with ThreadManager            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ Simulation Loop (each SST cycle)                    │
│                                                     │
│ RISCVSoCStandalone::clockTick()                     │
│   ├─ SOCTop->startPhase1()                          │
│   │  └─ Parallel: CPU, IFStage, EXEStage, WBStage  │
│   │     execute step()                              │
│   ├─ SOCTop->finishPhase1()                         │
│   ├─ SOCTop->startPhase2()                          │
│   │  └─ Channel toggle, PipeReg updates             │
│   ├─ Check completion: isAllSimulatorDone()         │
│   │  └─ If done: issueExitEvent()                   │
│   ├─ Fast-forward: skip empty ticks                 │
│   └─ Return false (continue) or true (stop)         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ Simulation Complete                                 │
│ • Print statistics                                  │
│ • Cleanup resources                                 │
│ • Exit                                              │
└─────────────────────────────────────────────────────┘
```

### Data Flow 2: QEMU + Device Full-System Simulation

```
┌──────────────┐
│ SST Python   │
│ Config       │
│ linux_*.py   │
└──────┬───────┘
       │
       ↓
┌───────────────────────────────────────────────────────────────┐
│ SST Configuration                                             │
│                                                               │
│ qemu = sst.Component("qemu", "qemubinary.QEMUBinary")         │
│ qemu.addParams({                                              │
│   "binary_path": "/path/to/vmlinux",                          │
│   "socket_path": "/tmp/qemu-sst-mmio.sock",                   │
│   "num_devices": 2                                            │
│ })                                                            │
│                                                               │
│ device0 = sst.Component("dev0", "acalsim.QEMUDevice")         │
│ device1 = sst.Component("dev1", "acalsim.ComputeDevice")      │
│                                                               │
│ link0 = sst.Link("qemu_dev0")                                 │
│ link0.connect((qemu, "device_port_0", "1ns"),                 │
│               (device0, "cpu_port", "1ns"))                   │
│                                                               │
│ link1 = sst.Link("qemu_dev1")                                 │
│ link1.connect((qemu, "device_port_1", "1ns"),                 │
│               (device1, "cpu_port", "1ns"))                   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ QEMUBinaryComponent::setup()                                  │
│ • Launch QEMU subprocess                                      │
│ • Create Unix socket server                                   │
│ • Wait for QEMU to connect                                    │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ QEMU Process Starts                                           │
│ • Boot Linux kernel                                           │
│ • Load VirtIO-SST driver (virtio-sst.ko)                      │
│ • Create /dev/sst0 character device                           │
│ • Mount rootfs                                                │
│ • Run init scripts                                            │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ User Application Execution                                    │
│                                                               │
│ [Inside QEMU Guest]                                           │
│ fd = open("/dev/sst0", O_RDWR)                                │
│ ioctl(fd, SST_CMD_WRITE, &request)                            │
│   ↓                                                           │
│ [Kernel Driver]                                               │
│ virtio_sst_write() → sends to VirtIO queue                    │
│   ↓                                                           │
│ [QEMU Device Model]                                           │
│ virtio_sst_handle_output()                                    │
│   ↓                                                           │
│ [Unix Socket]                                                 │
│ Send MMIORequest to /tmp/qemu-sst-mmio.sock                   │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ QEMUBinaryComponent::handleMMIORequest()                      │
│ • Receive MMIORequest (type, addr, data, size)                │
│ • Determine target device by address                          │
│ • Create MemoryTransactionEvent                               │
│ • Send via SST Link to device                                 │
│ • Store pending request (awaiting response)                   │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ Device Component (e.g., ACALSimComputeDeviceComponent)        │
│ • Receive MemoryTransactionEvent                              │
│ • Process transaction (LOAD or STORE)                         │
│ • Simulate computation (cycle-accurate)                       │
│ • Create MemoryResponseEvent                                  │
│ • Send back via SST Link                                      │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ QEMUBinaryComponent::handleDeviceResponse()                   │
│ • Receive MemoryResponseEvent                                 │
│ • Match to pending request by req_id                          │
│ • Create MMIOResponse                                         │
│ • Send via Unix socket back to QEMU                           │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ QEMU Device Model                                             │
│ • Receive MMIOResponse                                        │
│ • Update VirtIO buffer                                        │
│ • Trigger interrupt to guest                                  │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ Kernel Driver                                                 │
│ • Handle interrupt                                            │
│ • Copy data to user buffer                                    │
│ • Complete ioctl() call                                       │
└───────────────┬───────────────────────────────────────────────┘
                │
                ↓
┌───────────────────────────────────────────────────────────────┐
│ User Application                                              │
│ • Receive response data                                       │
│ • Continue execution                                          │
└───────────────────────────────────────────────────────────────┘
```

### Data Flow 3: PyTorch Device GEMM Offloading

```
┌────────────────────┐
│ Docker Container   │
│ (PyTorch Training) │
└─────────┬──────────┘
          │
          ↓
┌──────────────────────────────────────────────────────┐
│ PyTorch Model Execution                              │
│                                                      │
│ model = MyModel()                                    │
│ layer = DeviceLinear(128, 256)  # Custom layer      │
│ output = layer(input)           # Triggers offload  │
└─────────┬────────────────────────────────────────────┘
          │
          ↓
┌──────────────────────────────────────────────────────┐
│ device_gemm_operator.py                              │
│                                                      │
│ class DeviceGEMM(torch.autograd.Function):           │
│   def forward(ctx, A, B):                            │
│     # Serialize matrices                             │
│     request = pack_gemm_request(A, B)                │
│     # Send via TCP to QEMU                           │
│     response = tcp_client.send(request)              │
│     return torch.from_numpy(response.result)         │
└─────────┬────────────────────────────────────────────┘
          │ TCP (port 5555)
          ↓
┌──────────────────────────────────────────────────────┐
│ QEMU Guest (Device Server)                           │
│ qemu_device_server_virtio.py                         │
│                                                      │
│ while True:                                          │
│   request = tcp_server.recv()                        │
│   # Parse GEMM request                               │
│   cmd = GEMMCommand(op='gemm',                       │
│                     m=32, n=128, k=256,              │
│                     data=matrices)                   │
│   # Forward to VirtIO device                         │
│   fd = open("/dev/sst0", "rw")                       │
│   result = ioctl(fd, SST_CMD_COMPUTE, cmd)           │
│   # Send timing back via TCP                         │
│   tcp_server.send(result)                            │
└─────────┬────────────────────────────────────────────┘
          │ ioctl(/dev/sst0)
          ↓
┌──────────────────────────────────────────────────────┐
│ Kernel Driver (virtio-sst.ko)                        │
│                                                      │
│ static long virtio_sst_ioctl(...)                    │
│ {                                                    │
│   case SST_CMD_COMPUTE:                              │
│     // Write command to VirtIO queue                 │
│     virtqueue_add_outbuf(vq, &cmd, ...)              │
│     virtqueue_kick(vq)                               │
│     // Wait for response                             │
│     wait_for_completion(&comp)                       │
│     return result                                    │
│ }                                                    │
└─────────┬────────────────────────────────────────────┘
          │ VirtIO Queue
          ↓
┌──────────────────────────────────────────────────────┐
│ QEMU VirtIO Device Model                             │
│ (hw/virtio/virtio-sst.c)                             │
│                                                      │
│ static void virtio_sst_handle_output(...)            │
│ {                                                    │
│   // Read from VirtIO queue                          │
│   buf = virtqueue_pop(vq)                            │
│   // Extract COMPUTE command                         │
│   // Send to SST via Unix socket                     │
│   mmio_req = {                                       │
│     type: COMPUTE,                                   │
│     addr: GEMM_DEVICE_BASE,                          │
│     data: serialized_matrices                        │
│   }                                                  │
│   send(socket_fd, &mmio_req, ...)                    │
│ }                                                    │
└─────────┬────────────────────────────────────────────┘
          │ Unix Socket
          ↓
┌──────────────────────────────────────────────────────┐
│ QEMUBinaryComponent                                  │
│ • Route to ACALSimVirtIODeviceComponent              │
│ • Forward COMPUTE command                            │
└─────────┬────────────────────────────────────────────┘
          │ SST Link
          ↓
┌──────────────────────────────────────────────────────┐
│ ACALSimVirtIODeviceComponent                         │
│                                                      │
│ • Parse GEMM parameters (M, N, K)                    │
│ • Simulate matrix multiplication                     │
│ • Compute cycles: M * N * K * CPI                    │
│ • Add memory access latency                          │
│ • Return: {result_matrix, cycles}                    │
└─────────┬────────────────────────────────────────────┘
          │
          ↓ (Response propagates back through stack)
          │
┌──────────────────────────────────────────────────────┐
│ PyTorch receives result + timing                     │
│ • Update gradients if backward pass                  │
│ • Log timing statistics                              │
│ • Continue model training                            │
└──────────────────────────────────────────────────────┘
```

---

## Key Integration Patterns

### Pattern 1: 2-Phase Execution Bridge

ACALSim's 2-phase model is mapped to SST's clock-driven model:

```
SST Clock Tick N:
├─ Phase 1 (Parallel Execution)
│  ├─ ThreadManager::startPhase1()
│  ├─ All simulators execute step() in parallel
│  └─ ThreadManager::finishPhase1()
│
├─ Phase 2 (Synchronization)
│  ├─ ThreadManager::startPhase2()
│  ├─ Channel toggle (dual-queue swap)
│  ├─ PipeReg updates
│  ├─ Fast-forward to next event
│  └─ ThreadManager::finishPhase2()
│
└─ Termination Check
   ├─ If all done: issueExitEvent() + set ready_to_terminate
   └─ Return false (continue) or true (stop)

SST Clock Tick N+1:
└─ If ready_to_terminate: return true (actually stop)
```

### Pattern 2: Primary Component Pattern

To control SST simulation lifetime:

```cpp
Constructor:
  registerAsPrimaryComponent();
  primaryComponentDoNotEndSim();  // "We're not done yet"

During Simulation:
  // Keep running...
  return false;  // Continue clock

When Done Detected:
  primaryComponentOKToEndSim();  // "OK to end now"
  ready_to_terminate_ = true;
  return false;  // One more iteration for cleanup

Next Iteration:
  return true;  // Actually stop clock
```

### Pattern 3: Address-Based Device Routing

For multi-device QEMU integration:

```cpp
struct DeviceInfo {
  uint64_t base_addr;  // e.g., 0x10000000
  uint64_t size;       // e.g., 4096 bytes
  SST::Link* link;     // Link to device component
  string name;         // "accelerator0"
};

DeviceInfo* findDeviceForAddress(uint64_t addr) {
  for (auto& dev : devices_) {
    if (addr >= dev.base_addr && 
        addr < dev.base_addr + dev.size) {
      return &dev;
    }
  }
  return nullptr;  // Unmapped address
}
```

---

## Statistics Collection

### Component-Level Statistics

Each SST component can register statistics:

```cpp
// In component constructor:
registerStatistic<uint64_t>("total_transactions", 
                            "Total transactions processed");
registerStatistic<uint64_t>("total_cycles", 
                            "Total cycles executed");
registerStatistic<double>("avg_latency", 
                          "Average latency per transaction");

// During simulation:
self->total_transactions->addData(1);
self->total_cycles->addData(cycles);
```

### SST Statistics Output

```python
# In Python config:
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV", {
    "filepath": "simulation_stats.csv"
})

comp.enableAllStatistics({
    "type": "sst.AccumulatorStatistic",
    "rate": "1us"
})
```

Output: `simulation_stats.csv` with per-component, per-statistic data.

---

## Configuration Examples

### Example 1: Standalone RISC-V

```python
import sst

comp = sst.Component("riscv0", "acalsim.RISCVSoCStandalone")
comp.addParams({
    "clock": "1GHz",
    "asm_file": "../../src/riscv/asm/branch_simple.txt",
    "memory_size": 65536,
    "text_offset": 0,
    "data_offset": 8192,
    "max_cycles": 100000,
    "verbose": 2
})
```

### Example 2: QEMU + Multi-Device

```python
import sst

# QEMU Component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "binary_path": "/path/to/vmlinux",
    "qemu_path": "qemu-system-riscv32",
    "socket_path": "/tmp/qemu-sst.sock",
    "num_devices": 2,
    "device0_base": "0x10000000",
    "device0_size": "4096",
    "device0_name": "echo_device",
    "device1_base": "0x20000000",
    "device1_size": "4096",
    "device1_name": "compute_device"
})

# Device 0: Echo Device
echo_dev = sst.Component("echo", "acalsim.QEMUDevice")
echo_dev.addParams({
    "clock": "1GHz",
    "base_addr": "0x10000000",
    "size": "4096",
    "echo_latency": "10"
})

# Device 1: Compute Device
compute_dev = sst.Component("compute", "acalsim.ComputeDevice")
compute_dev.addParams({
    "clock": "2GHz",
    "base_addr": "0x20000000",
    "size": "4096"
})

# Links
link0 = sst.Link("qemu_echo")
link0.connect(
    (qemu, "device_port_0", "1ns"),
    (echo_dev, "cpu_port", "1ns")
)

link1 = sst.Link("qemu_compute")
link1.connect(
    (qemu, "device_port_1", "1ns"),
    (compute_dev, "cpu_port", "1ns")
)
```

---

## Build System Overview

### Source Structure

```
src/sst-riscv/              # SST integration directory
├── Makefile                # Build configuration
├── ACALSimComponent.cc/.hh # Base component
├── RISCVSoCStandalone.cc/.hh # RISC-V implementation
├── examples/               # Python configurations
│   ├── riscv_single_core.py
│   ├── riscv_dual_core.py
│   └── simple_system.py
└── build/                  # Build output

include/sst/               # SST component headers
├── ACALSimDeviceComponent.hh
├── ACALSimComputeDeviceComponent.hh
├── ACALSimMMIODevice.hh
└── QEMUBinaryComponent.hh

libs/sst/                  # SST component implementations
├── ACALSimDeviceComponent.cc
├── ACALSimComputeDeviceComponent.cc
├── ACALSimMMIODevice.cc
└── QEMUBinaryComponent.cc
```

### Build Process

```bash
# 1. Build core ACALSim library
cd /path/to/acalsim
make
# Produces: lib/libacalsim.so

# 2. Build SST integration
cd src/sst-riscv
make
# Produces: libacalsim_sst.so

# 3. Install to SST
make install
# Copies to: $SST_PREFIX/lib/sst-elements-library/

# 4. Verify
sst-info acalsim
```

---

## Testing & Validation

### Unit Testing

```bash
# Test component registration
sst-info acalsim
sst-info acalsim.RISCVSoCStandalone
sst-info acalsim.QEMUDevice
```

### Integration Testing

```bash
# Test 1: RISC-V single core
cd src/sst-riscv/examples
sst riscv_single_core.py

# Test 2: RISC-V dual core
sst riscv_dual_core.py

# Test 3: QEMU + Device
cd src/qemu-acalsim-sst-linux/examples/llama-inference
./run_sst.sh  # Terminal 1
./run_qemu_custom_kernel.sh  # Terminal 2
# Inside QEMU:
python3 qemu_device_server_virtio.py  # Terminal 3
python3 test_device_gemm.py  # Terminal 4 (Docker)
```

### Regression Testing

```bash
cd scripts
python3 regression.py --suite sst
```

---

## Performance Characteristics

### RISC-V Standalone

- **Throughput**: ~1M instructions/second (simulated)
- **Overhead**: ~10% SST wrapping overhead
- **Memory**: ~50MB per component

### QEMU + Device Integration

- **MMIO Latency**: ~100μs per transaction (binary protocol)
- **Throughput**: ~10K transactions/second
- **CPU Usage**: <5% (binary protocol vs 50% text protocol)

### PyTorch Device GEMM

- **Offload Latency**: ~1ms per GEMM operation
- **Cycle Accuracy**: ±1% vs hardware measurements
- **Scalability**: Tested up to 512x512x512 matrices

---

## Future Extensions

### Planned Integrations

1. **riscvSimTemplate** - Simplified RISC-V template
2. **testPETile** - Processing element tiles (many-core)
3. **testBlackBear** - Custom accelerator example
4. **testSimPort** - Port communication demonstration

### Potential Enhancements

1. **Distributed SST Support**
   - Implement serialization for ACALSimSSTEvent
   - Add MPI support for parallel simulation

2. **Advanced Memory Models**
   - Integration with Ramulator
   - DRAM timing models
   - Cache hierarchy simulation

3. **Network-on-Chip**
   - Integration with Merlin router
   - Mesh/Torus topologies
   - Congestion modeling

4. **Power Modeling**
   - Energy statistics collection
   - Power-aware scheduling
   - DVFS simulation

---

## References

### Documentation

- [SST Integration Guide](integration-guide.md)
- [PyTorch Device GEMM](pytorch-device-gemm.md)
- [RISC-V Examples](riscv-examples.md)
- [Quick Start Guide](quickstart.md)

### Source Code

- Main SST integration: `src/sst-riscv/`
- QEMU integration: `src/qemu-acalsim-sst-linux/`
- Device components: `include/sst/`, `libs/sst/`

### External Resources

- [SST-Core Documentation](https://sst-simulator.org/sst-docs/)
- [ACALSim Main Documentation](../../README.md)
- [Component Writers Guide](https://sst-simulator.org/sst-docs/docs/guides/component/)

---

**Copyright 2023-2026 Playlab/ACAL**  
Licensed under the Apache License, Version 2.0

