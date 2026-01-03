# ACALSim SST Integration - Simplified Architecture Diagram

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

This document provides a simplified, easy-to-understand architecture diagram using Mermaid. For the comprehensive version with detailed data flows, see [architecture-diagram.md](architecture-diagram.md).

## Simplified Architecture (Mermaid)

```mermaid
graph TB
    subgraph "SST-Core Framework"
        SST[SST Discrete Event Engine]
        SSTPY[Python Configuration System]
        SSTELI[Component Registry ELI]
    end
    
    subgraph "Core ACALSim Framework (libacalsim.so)"
        SimTop[SimTopBase<br/>ThreadManager<br/>EventQueue<br/>Global Tick]
        SimBase[SimBase<br/>init/step/cleanup<br/>lifecycle]
        SimPort[SimPort/SimChannel<br/>Communication]
        SimConfig[SimConfig<br/>JSON Configuration]
    end
    
    subgraph "SST Integration Bridge (libacalsim_sst.so)"
        ACALSimComponent[ACALSimComponent<br/>clockTick orchestration<br/>Phase 1 + Phase 2<br/>Primary Component Pattern]
        SSTEvent[ACALSimSSTEvent<br/>SimPacket wrapper]
    end
    
    subgraph "Application Components"
        RISCV[RISCVSoCStandalone<br/>Complete RV32I CPU<br/>Pipeline stages<br/>✅ WORKING]
        QEMU[QEMUBinaryComponent<br/>QEMU subprocess mgmt<br/>Binary MMIO protocol<br/>✅ PRODUCTION]
        
        subgraph "Device Components"
            EchoDev[ACALSimDeviceComponent<br/>Echo device]
            ComputeDev[ACALSimComputeDeviceComponent<br/>GEMM accelerator]
            VirtioDev[ACALSimVirtIODeviceComponent<br/>PyTorch offloading]
            HSADev[HSAComputeComponent<br/>Multi-accelerator]
        end
    end
    
    subgraph "External Interfaces"
        QEMUProc[QEMU Process<br/>qemu-system-riscv32<br/>Linux + virtio-sst.ko]
        PyTorch[PyTorch Docker<br/>device_gemm operator<br/>TCP → VirtIO → SST]
        UserApps[User Applications<br/>HSA demos<br/>Llama inference<br/>Custom C/C++ apps]
    end
    
    %% Connections
    SSTPY --> SSTELI
    SSTELI --> SST
    
    SST --> ACALSimComponent
    
    ACALSimComponent --> SimTop
    ACALSimComponent --> SimBase
    SimBase --> SimPort
    SimBase --> SimConfig
    
    ACALSimComponent -.extends.-> RISCV
    ACALSimComponent -.extends.-> QEMU
    ACALSimComponent -.extends.-> EchoDev
    ACALSimComponent -.extends.-> ComputeDev
    ACALSimComponent -.extends.-> VirtioDev
    ACALSimComponent -.extends.-> HSADev
    
    QEMU --SST Link--> EchoDev
    QEMU --SST Link--> ComputeDev
    QEMU --SST Link--> VirtioDev
    QEMU --SST Link--> HSADev
    
    QEMUProc --Unix Socket--> QEMU
    PyTorch --TCP--> QEMUProc
    UserApps --> QEMUProc
    
    %% Styling
    classDef sstStyle fill:#ADD8E6,stroke:#333,stroke-width:2px
    classDef acalsimStyle fill:#90EE90,stroke:#333,stroke-width:2px
    classDef bridgeStyle fill:#FFFFE0,stroke:#333,stroke-width:2px
    classDef appStyle fill:#E6E6FA,stroke:#333,stroke-width:2px
    classDef externalStyle fill:#F08080,stroke:#333,stroke-width:2px
    
    class SST,SSTPY,SSTELI sstStyle
    class SimTop,SimBase,SimPort,SimConfig acalsimStyle
    class ACALSimComponent,SSTEvent bridgeStyle
    class RISCV,QEMU,EchoDev,ComputeDev,VirtioDev,HSADev appStyle
    class QEMUProc,PyTorch,UserApps externalStyle
```

## Component Relationships

```mermaid
graph LR
    subgraph "Inheritance Hierarchy"
        SST_Component[SST::Component]
        ACALSim_Component[ACALSimComponent]
        RISCV_Comp[RISCVSoCStandalone]
        QEMU_Comp[QEMUBinaryComponent]
        Device_Comp[Device Components]
        
        SST_Component --> ACALSim_Component
        ACALSim_Component --> RISCV_Comp
        ACALSim_Component --> QEMU_Comp
        ACALSim_Component --> Device_Comp
    end
    
    subgraph "Composition"
        ACALSim_Comp2[ACALSimComponent]
        SimTop2[SimTopBase]
        ThreadMgr[ThreadManager]
        Sims[Simulators]
        
        ACALSim_Comp2 --wraps--> SimTop2
        SimTop2 --manages--> ThreadMgr
        ThreadMgr --executes--> Sims
    end
    
    style SST_Component fill:#ADD8E6
    style ACALSim_Component fill:#FFFFE0
    style RISCV_Comp fill:#E6E6FA
    style QEMU_Comp fill:#E6E6FA
    style Device_Comp fill:#E6E6FA
```

## Data Flow: RISC-V Simulation

```mermaid
sequenceDiagram
    participant User
    participant SST as SST Engine
    participant RISCV as RISCVSoCStandalone
    participant SOC as SOCTop
    participant CPU as CPU/Stages
    
    User->>SST: Run riscv_single_core.py
    SST->>RISCV: Create component
    RISCV->>SOC: Create SOCTop instance
    RISCV->>SOC: init() - load assembly
    SOC->>CPU: Register simulators
    
    loop Every SST Clock Tick
        SST->>RISCV: clockTick(cycle)
        RISCV->>SOC: startPhase1()
        SOC->>CPU: Execute step() in parallel
        CPU-->>SOC: Events generated
        RISCV->>SOC: finishPhase1()
        RISCV->>SOC: startPhase2()
        SOC->>SOC: Channel toggle, PipeReg updates
        RISCV->>SOC: Check if done
        alt All simulators done
            RISCV->>RISCV: issueExitEvent()
            RISCV->>SST: return false (one more cycle)
        else Continue
            RISCV->>SST: return false
        end
    end
    
    RISCV->>SST: return true (terminate)
    SST->>User: Simulation complete
```

## Data Flow: QEMU + Device Integration

```mermaid
sequenceDiagram
    participant App as User App in QEMU
    participant Kernel as Linux Kernel
    participant QEMU_Dev as QEMU VirtIO Device
    participant Socket as Unix Socket
    participant QEMU_Comp as QEMUBinaryComponent
    participant Device as ACALSimDevice
    
    App->>Kernel: ioctl(/dev/sst0, COMPUTE, data)
    Kernel->>QEMU_Dev: VirtIO request
    QEMU_Dev->>Socket: Send MMIORequest (binary)
    Socket->>QEMU_Comp: Receive request
    QEMU_Comp->>QEMU_Comp: Route by address
    QEMU_Comp->>Device: MemoryTransactionEvent via SST Link
    Device->>Device: Process COMPUTE (cycle-accurate)
    Device->>QEMU_Comp: MemoryResponseEvent
    QEMU_Comp->>Socket: Send MMIOResponse
    Socket->>QEMU_Dev: Response received
    QEMU_Dev->>Kernel: VirtIO completion + interrupt
    Kernel->>App: Return result + timing
```

## Data Flow: PyTorch Device GEMM

```mermaid
sequenceDiagram
    participant PyTorch as PyTorch Model
    participant Operator as device_gemm()
    participant TCP as TCP Socket
    participant DevServer as Device Server in QEMU
    participant VirtIO as /dev/sst0
    participant SST as SST VirtIO Device
    
    PyTorch->>Operator: output = DeviceLinear(input)
    Operator->>Operator: Serialize matrices
    Operator->>TCP: Send GEMM request
    TCP->>DevServer: Receive request
    DevServer->>VirtIO: ioctl(COMPUTE, matrices)
    VirtIO->>SST: VirtIO → QEMU → Unix Socket → SST
    SST->>SST: Simulate GEMM (M×N×K cycles)
    SST->>VirtIO: Return result + cycles
    VirtIO->>DevServer: Complete ioctl
    DevServer->>TCP: Send response
    TCP->>Operator: Receive result + timing
    Operator->>PyTorch: Return tensor + log timing
```

## Key Patterns

### Pattern 1: 2-Phase Execution

```mermaid
graph TD
    Start[SST Clock Tick N] --> Phase1Start[startPhase1]
    Phase1Start --> Parallel[Parallel Execution:<br/>All simulators step]
    Parallel --> Phase1End[finishPhase1]
    Phase1End --> Phase2Start[startPhase2]
    Phase2Start --> Sync[Synchronization:<br/>Channel toggle<br/>PipeReg updates<br/>Fast-forward]
    Sync --> Phase2End[finishPhase2]
    Phase2End --> Check{All done?}
    Check -->|Yes| ExitEvent[issueExitEvent<br/>ready_to_terminate = true]
    Check -->|No| Continue[return false]
    ExitEvent --> Continue
    Continue --> NextTick[SST Clock Tick N+1]
    NextTick --> CheckTerminate{ready_to_terminate?}
    CheckTerminate -->|Yes| Stop[return true - STOP]
    CheckTerminate -->|No| Phase1Start
    
    style Phase1Start fill:#90EE90
    style Phase1End fill:#90EE90
    style Phase2Start fill:#ADD8E6
    style Phase2End fill:#ADD8E6
    style ExitEvent fill:#FFD700
    style Stop fill:#FF6B6B
```

### Pattern 2: Primary Component Pattern

```mermaid
stateDiagram-v2
    [*] --> Constructor
    Constructor --> Init: registerAsPrimaryComponent()<br/>primaryComponentDoNotEndSim()
    Init --> Running: Clock started
    Running --> Running: return false (continue)
    Running --> ReadyToEnd: All done detected<br/>primaryComponentOKToEndSim()
    ReadyToEnd --> Cleanup: return false (one more cycle)
    Cleanup --> Terminated: return true (stop clock)
    Terminated --> [*]
    
    note right of Constructor
        Tell SST: "We control termination"
    end note
    
    note right of ReadyToEnd
        Tell SST: "OK to end now"
        But need one more cycle for cleanup
    end note
```

### Pattern 3: Address-Based Device Routing

```mermaid
graph LR
    QEMU[QEMUBinaryComponent] --> Router{Address Router}
    Router -->|0x10000000-0x10000FFF| Dev0[Echo Device]
    Router -->|0x20000000-0x20000FFF| Dev1[Compute Device]
    Router -->|0x30000000-0x30000FFF| Dev2[VirtIO Device]
    Router -->|0x40000000-0x40000FFF| Dev3[HSA Device]
    Router -->|Unknown| Error[Return Error]
    
    style Router fill:#FFFFE0
    style Dev0 fill:#E6E6FA
    style Dev1 fill:#E6E6FA
    style Dev2 fill:#E6E6FA
    style Dev3 fill:#E6E6FA
    style Error fill:#FF6B6B
```

## File Organization

```mermaid
graph TD
    Root[acalsim/] --> SrcSST[src/sst-riscv/]
    Root --> SrcQEMU[src/qemu-acalsim-sst-linux/]
    Root --> IncludeSST[include/sst/]
    Root --> LibsSST[libs/sst/]
    Root --> Docs[docs/sst-integration/]
    
    SrcSST --> RISCV_Files[RISCVSoCStandalone.hh/cc<br/>ACALSimComponent.hh/cc<br/>examples/]
    
    SrcQEMU --> QEMU_Files[acalsim-device/<br/>drivers/<br/>examples/]
    
    IncludeSST --> Headers[ACALSimDeviceComponent.hh<br/>QEMUBinaryComponent.hh<br/>ACALSimComputeDeviceComponent.hh<br/>ACALSimMMIODevice.hh]
    
    LibsSST --> Impls[ACALSimDeviceComponent.cc<br/>QEMUBinaryComponent.cc<br/>etc.]
    
    Docs --> DocFiles[architecture-diagram.md<br/>integration-guide.md<br/>quickstart.md<br/>pytorch-device-gemm.md]
    
    style Root fill:#90EE90
    style SrcSST fill:#E6E6FA
    style SrcQEMU fill:#E6E6FA
    style IncludeSST fill:#ADD8E6
    style LibsSST fill:#ADD8E6
    style Docs fill:#FFFFE0
```

## Build Flow

```mermaid
graph TD
    Start[Developer] --> BuildCore[Build Core ACALSim<br/>make in acalsim/]
    BuildCore --> LibACALSim[lib/libacalsim.so]
    
    LibACALSim --> BuildSST[Build SST Integration<br/>make in src/sst-riscv/]
    BuildSST --> LibSSTElement[libacalsim_sst.so]
    
    LibSSTElement --> Install[make install<br/>Copy to SST elements dir]
    Install --> Verify[sst-info acalsim]
    
    Verify --> Test1[Run Examples<br/>sst riscv_single_core.py]
    Verify --> Test2[Run QEMU Integration<br/>./run_sst.sh + ./run_qemu.sh]
    Verify --> Test3[Run PyTorch Tests<br/>python3 test_device_gemm.py]
    
    Test1 --> Success[✅ Integration Working]
    Test2 --> Success
    Test3 --> Success
    
    style Start fill:#90EE90
    style LibACALSim fill:#ADD8E6
    style LibSSTElement fill:#FFFFE0
    style Success fill:#98FB98
```

## Quick Reference

### Component Status

| Component | Status | Python Name | Description |
|-----------|--------|-------------|-------------|
| RISCVSoCStandalone | ✅ Working | `acalsim.RISCVSoCStandalone` | Complete RV32I CPU |
| QEMUBinaryComponent | ✅ Production | `qemubinary.QEMUBinary` | QEMU integration |
| ACALSimDeviceComponent | ✅ Working | `acalsim.QEMUDevice` | Echo device |
| ACALSimComputeDeviceComponent | ✅ Production | `acalsim.ComputeDevice` | GEMM accelerator |
| ACALSimVirtIODeviceComponent | ✅ Production | `acalsim.VirtIODevice` | PyTorch offloading |
| HSAComputeComponent | ✅ Working | `acalsim.HSACompute` | Multi-accelerator |

### Key Files

| File | Purpose |
|------|---------|
| `src/sst-riscv/RISCVSoCStandalone.cc` | RISC-V SST component implementation |
| `src/sst-riscv/ACALSimComponent.cc` | Base SST component wrapper |
| `include/sst/QEMUBinaryComponent.hh` | QEMU binary component header |
| `include/sst/ACALSimDeviceComponent.hh` | Device component header |
| `docs/sst-integration/integration-guide.md` | Complete integration guide |
| `src/sst-riscv/examples/riscv_single_core.py` | RISC-V example config |
| `src/qemu-acalsim-sst-linux/examples/llama-inference/` | PyTorch integration |

### Performance Metrics

| Metric | Value |
|--------|-------|
| RISC-V Simulation Speed | ~1M instructions/sec |
| QEMU MMIO Latency | ~100μs/transaction |
| QEMU Throughput | ~10K tx/sec |
| PyTorch Offload Latency | ~1ms/GEMM |
| Cycle Accuracy | ±1% vs hardware |

---

For more details, see:
- [Complete Architecture Diagram](architecture-diagram.md)
- [Integration Guide](integration-guide.md)
- [PyTorch Device GEMM](pytorch-device-gemm.md)
- [Quick Start](quickstart.md)

**Copyright 2023-2026 Playlab/ACAL**  
Licensed under the Apache License, Version 2.0

