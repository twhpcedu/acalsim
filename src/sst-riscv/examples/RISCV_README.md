# RISC-V RV32I SST Integration Examples

This directory contains complete examples of running RISC-V RV32I processors in SST using ACALSim.

## Overview

The RISC-V integration demonstrates:
- **Complete RV32I ISA**: All 32 base integer instructions
- **Event-driven timing model**: Accurate cycle-by-cycle simulation
- **Pipeline visualization**: IF (Instruction Fetch), EXE (Execute), WB (Write-Back) stages
- **Hazard detection**: Data and control hazard handling
- **Assembly program execution**: Load and run RISC-V assembly programs
- **Multi-core simulation**: Multiple processors in parallel

## Available Examples

### 1. Single-Core RISC-V (`riscv_single_core.py`)

**Description**: Single RISC-V processor executing an assembly program

**Features**:
- 1 RV32I processor
- Configurable memory size (default: 64KB)
- Pipeline stage visualization
- Register file and memory dumps

**Usage**:
```bash
sst riscv_single_core.py
```

**Configuration**:
```python
CLOCK_FREQ = "1GHz"           # Processor clock
ASM_FILE = "../../src/riscv/asm/branch_simple.txt"  # Program to run
MEMORY_SIZE = 65536           # 64KB memory
MAX_CYCLES = 100000           # Maximum simulation cycles
```

### 2. Dual-Core RISC-V (`riscv_dual_core.py`)

**Description**: Two independent RISC-V processors running different programs

**Features**:
- 2 RV32I processors
- Independent programs per core
- Parallel execution
- Per-core statistics

**Usage**:
```bash
sst riscv_dual_core.py
```

**Configuration**:
```python
NUM_CORES = 2
ASM_FILES = [
    "../../src/riscv/asm/branch_simple.txt",      # Core 0
    "../../src/riscv/asm/load_store_simple.txt"   # Core 1
]
```

## RISC-V RV32I Instruction Set

The simulator implements the complete RV32I base integer instruction set:

### Arithmetic & Logic (R-Type)
- **ADD** rd, rs1, rs2 - Add
- **SUB** rd, rs1, rs2 - Subtract
- **AND** rd, rs1, rs2 - Bitwise AND
- **OR** rd, rs1, rs2 - Bitwise OR
- **XOR** rd, rs1, rs2 - Bitwise XOR
- **SLL** rd, rs1, rs2 - Shift Left Logical
- **SRL** rd, rs1, rs2 - Shift Right Logical
- **SRA** rd, rs1, rs2 - Shift Right Arithmetic
- **SLT** rd, rs1, rs2 - Set Less Than
- **SLTU** rd, rs1, rs2 - Set Less Than Unsigned

### Immediate Operations (I-Type)
- **ADDI** rd, rs1, imm - Add Immediate
- **ANDI** rd, rs1, imm - AND Immediate
- **ORI** rd, rs1, imm - OR Immediate
- **XORI** rd, rs1, imm - XOR Immediate
- **SLLI** rd, rs1, imm - Shift Left Logical Immediate
- **SRLI** rd, rs1, imm - Shift Right Logical Immediate
- **SRAI** rd, rs1, imm - Shift Right Arithmetic Immediate
- **SLTI** rd, rs1, imm - Set Less Than Immediate
- **SLTIU** rd, rs1, imm - Set Less Than Immediate Unsigned

### Memory Access
- **LB** rd, offset(rs1) - Load Byte (sign-extended)
- **LBU** rd, offset(rs1) - Load Byte Unsigned
- **LH** rd, offset(rs1) - Load Half-word (sign-extended)
- **LHU** rd, offset(rs1) - Load Half-word Unsigned
- **LW** rd, offset(rs1) - Load Word
- **SB** rs2, offset(rs1) - Store Byte
- **SH** rs2, offset(rs1) - Store Half-word
- **SW** rs2, offset(rs1) - Store Word

### Branches (B-Type)
- **BEQ** rs1, rs2, label - Branch if Equal
- **BNE** rs1, rs2, label - Branch if Not Equal
- **BLT** rs1, rs2, label - Branch if Less Than
- **BLTU** rs1, rs2, label - Branch if Less Than Unsigned
- **BGE** rs1, rs2, label - Branch if Greater or Equal
- **BGEU** rs1, rs2, label - Branch if Greater or Equal Unsigned

### Jumps
- **JAL** rd, label - Jump and Link
- **JALR** rd, rs1, offset - Jump and Link Register

### Upper Immediate (U-Type)
- **LUI** rd, imm - Load Upper Immediate
- **AUIPC** rd, imm - Add Upper Immediate to PC

### System
- **HCF** - Halt and Catch Fire (terminates simulation)

## Assembly Programs

Pre-built assembly programs are available in `../../src/riscv/asm/`:

| File | Description | Instructions Used |
|------|-------------|-------------------|
| `branch_simple.txt` | Simple branch test | ADDI, BEQ, BNE, HCF |
| `load_store_simple.txt` | Memory access test | ADDI, SW, LW, HCF |
| `full_test.txt` | Comprehensive ISA test | All RV32I instructions |
| `test.txt` | Basic arithmetic | ADD, ADDI, SUB, HCF |

## Writing Custom Assembly Programs

You can create your own RISC-V assembly programs:

```assembly
# Example: Fibonacci sequence calculator
.text
main:
    addi x1, x0, 1       # fib(0) = 1
    addi x2, x0, 1       # fib(1) = 1
    addi x5, x0, 10      # counter = 10

loop:
    add x3, x1, x2       # fib(n) = fib(n-1) + fib(n-2)
    addi x1, x2, 0       # shift: x1 = x2
    addi x2, x3, 0       # shift: x2 = x3
    addi x5, x5, -1      # counter--
    bne x5, x0, loop     # if counter != 0, loop

    hcf                  # halt simulation
```

### Assembly File Format

1. **Labels**: End with `:` (e.g., `main:`, `loop:`)
2. **Instructions**: One per line, format: `OPCODE rd, rs1, rs2/imm`
3. **Comments**: Start with `#`
4. **Directives**: `.text` for code section
5. **Termination**: Must end with `hcf` instruction

## Register File (RISC-V ABI)

The simulator uses the standard RISC-V register file:

| Register | ABI Name | Description | Saver |
|----------|----------|-------------|-------|
| x0 | zero | Hardwired zero | N/A |
| x1 | ra | Return address | Caller |
| x2 | sp | Stack pointer | Callee |
| x3 | gp | Global pointer | - |
| x4 | tp | Thread pointer | - |
| x5-x7 | t0-t2 | Temporaries | Caller |
| x8-x9 | s0-s1 | Saved registers | Callee |
| x10-x11 | a0-a1 | Function args/return values | Caller |
| x12-x17 | a2-a7 | Function arguments | Caller |
| x18-x27 | s2-s11 | Saved registers | Callee |
| x28-x31 | t3-t6 | Temporaries | Caller |

## Memory Layout

Default memory configuration (configurable in Python):

```
0x00000000 ┌─────────────────────────┐
           │  Text Segment (.text)   │  Instruction memory
0x00002000 ├─────────────────────────┤  (TEXT_OFFSET = 0)
           │  Data Segment (.data)   │  Data memory
           │  Heap                   │  (DATA_OFFSET = 8192)
           │  Stack (grows down)     │
0x0000FFFF └─────────────────────────┘  (MEMORY_SIZE = 65536)
```

## Pipeline Architecture

The RISC-V simulator uses a visualized 3-stage pipeline:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ IF Stage │───▶│ EXE Stage│───▶│ WB Stage │
└──────────┘    └──────────┘    └──────────┘
     │               │               │
     ▼               ▼               ▼
  Fetch          Execute         Writeback
  Hazard         Process         Retire
  Detection      Packet          Instruction
```

### Stage Details

1. **IF (Instruction Fetch)**:
   - Fetches instruction from IMEM
   - Detects data/control hazards
   - Stalls pipeline when necessary
   - Forwards packet to EXE stage

2. **EXE (Execute)**:
   - Processes instruction packet
   - Performs ALU operations
   - Handles memory requests
   - Detects control hazards
   - Forwards to WB stage

3. **WB (Write-Back)**:
   - Retires instruction
   - Updates statistics
   - Completes execution

### Hazard Handling

- **Data Hazards (RAW)**: Detected in IF stage, pipeline stalls
- **Control Hazards**: Branch/jump causes pipeline flush
- **Structural Hazards**: Backpressure mechanism with retry

## Configuration Parameters

All parameters are configurable in Python:

```python
# Processor Configuration
CLOCK_FREQ = "1GHz"          # Clock frequency
MAX_CYCLES = 100000          # Max simulation cycles (0 = unlimited)

# Memory Configuration
MEMORY_SIZE = 65536          # Total memory size in bytes
TEXT_OFFSET = 0              # Instruction memory start address
DATA_OFFSET = 8192           # Data memory start address (8KB)

# Program
ASM_FILE = "path/to/file.txt"  # Assembly program to execute

# Output
VERBOSE = 2                  # Verbosity level (0-5)
```

## Statistics Collection

The simulator tracks:
- **Instructions executed**: Total instruction count
- **Cycles**: Total simulation cycles
- **IPC**: Instructions per cycle
- **Branches taken**: Branch instruction count
- **Loads/Stores**: Memory access statistics
- **Pipeline stalls**: Stall cycle count

## Expected Output

When running `sst riscv_single_core.py`:

```
Creating RISC-V RV32I Single-Core System...
====================================================================
RISC-V RV32I Single-Core Configuration
====================================================================
Clock frequency:     1GHz
Max cycles:          100000
Memory size:         65536 bytes (64KB)
Text segment:        0x00000000
Data segment:        0x00002000
Assembly program:    ../../src/riscv/asm/branch_simple.txt
====================================================================

RISCVSoCStandalone[@p:@l]: Initializing Standalone RISC-V SoC
...
=== RISC-V Simulation Complete ===
Total cycles: 156
==================================
```

## Troubleshooting

### Issue: "Assembly file not found"
**Solution**: Use absolute path or path relative to where you run `sst`
```python
ASM_FILE = os.path.abspath("../../src/riscv/asm/branch_simple.txt")
```

### Issue: "Simulation hangs"
**Cause**: Missing `hcf` instruction at end of program
**Solution**: Always end programs with `hcf`

### Issue: Register x0 not zero
**Note**: x0 is hardwired to zero, writes are ignored (correct behavior)

## Advanced Usage

### Connecting to SST Memory Model

To connect RISC-V to external SST memory:

```python
# Create RISC-V CPU
cpu = sst.Component("riscv_cpu", "acalsim.RISCVSoCStandalone")

# Create SST memory
mem = sst.Component("memory", "memHierarchy.MemController")

# Connect via link
link = sst.Link("cpu_mem_link")
link.connect((cpu, "mem_port", "10ns"), (mem, "cpu_port", "10ns"))
```

### Multi-Core with Shared Memory

```python
# Create cores
cores = []
for i in range(4):
    cpu = sst.Component(f"cpu{i}", "acalsim.RISCVSoCStandalone")
    cores.append(cpu)

# Create shared memory
shared_mem = sst.Component("shared_mem", "acalsim.SimpleMemory")

# Connect all cores to shared memory via NoC
# (Implementation depends on NoC model)
```

## Performance Notes

- **Simulation Speed**: ~100K cycles/second (depends on program complexity)
- **Memory Overhead**: ~1MB per core
- **Scalability**: Linear with number of cores (independent execution)

## Further Reading

- [RISC-V ISA Specification](https://riscv.org/specifications/)
- [ACALSim Documentation](../../../docs/README.md)
- [SST Documentation](https://sst-simulator.org/sst-docs/)
- [RISC-V Assembly Programming](https://github.com/riscv/riscv-asm-manual)

## Support

For issues or questions:
- Check existing assembly programs in `../../src/riscv/asm/`
- Review `../../src/riscv/main.cc` for standalone usage
- Open GitHub issue with minimal reproducible example

---

**Ready to simulate!** Start with `sst riscv_single_core.py` and explore from there.
