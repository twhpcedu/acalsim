/*
 * Copyright 2023-2026 Playlab/ACAL
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

/**
 * @file Emulator.cc
 * @brief RISC-V Assembly Parser and ISA Functional Model
 *
 * @details
 * This file implements a complete RISC-V assembly language parser that converts
 * textual assembly code into executable instruction structures for the CPU timing
 * model. This is a **functional model** (not a timing model) that runs during
 * initialization to prepare the instruction and data memory.
 *
 * **Key Responsibilities:**
 * - Parse RISC-V assembly source files (.s files)
 * - Recognize and convert instruction mnemonics to internal opcodes
 * - Handle assembler directives (.text, .data, .word, .byte, .half)
 * - Resolve symbolic labels to absolute memory addresses
 * - Expand pseudo-instructions into base instructions
 * - Initialize instruction memory (IMEM) and data memory (DMEM)
 * - Validate instruction syntax and operand formats
 *
 * **Assembly File Structure:**
 * ```
 * ┌──────────────────────────────────────────────────────┐
 * │  RISC-V Assembly Source File (.s)                   │
 * │                                                       │
 * │  .text                    ← Code section             │
 * │  main:                    ← Label definition         │
 * │      addi x1, x0, 5       ← I-type instruction       │
 * │      addi x2, x0, 10      ← Another instruction      │
 * │  loop:                    ← Branch target label      │
 * │      add  x3, x1, x2      ← R-type instruction       │
 * │      sw   x3, 0(x2)       ← S-type (store)           │
 * │      beq  x1, x2, done    ← B-type (branch)          │
 * │      j    loop            ← Pseudo-instruction       │
 * │  done:                                                │
 * │      hcf                  ← Halt instruction         │
 * │                                                       │
 * │  .data                    ← Data section             │
 * │  value:                   ← Data label               │
 * │      .word 42             ← 32-bit word              │
 * │  array:                                               │
 * │      .byte 1, 2, 3, 4     ← Byte array               │
 * │      .half 1000, 2000     ← Half-word array          │
 * └──────────────────────────────────────────────────────┘
 *
 *              Emulator::parse()
 *                     │
 *                     ▼
 * ┌──────────────────────────────────────────────────────┐
 * │  Parsed Instruction Structures                       │
 * │                                                       │
 * │  IMEM[0]: {op: ADDI, rd: x1, rs1: x0, imm: 5}        │
 * │  IMEM[1]: {op: ADDI, rd: x2, rs1: x0, imm: 10}       │
 * │  IMEM[2]: {op: ADD,  rd: x3, rs1: x1, rs2: x2}       │
 * │  IMEM[3]: {op: SW,   rs2: x3, rs1: x2, imm: 0}       │
 * │  IMEM[4]: {op: BEQ,  rs1: x1, rs2: x2, label: "done"}│
 * │  IMEM[5]: {op: JAL,  rd: x0, label: "loop"}          │
 * │  IMEM[6]: {op: HCF}                                  │
 * │                                                       │
 * │  DMEM[data_offset]: 0x0000002A  (42 in .word)        │
 * │  DMEM[data_offset+4]: 0x01, 0x02, 0x03, 0x04         │
 * │  DMEM[data_offset+8]: 0x03E8, 0x07D0 (1000, 2000)    │
 * └──────────────────────────────────────────────────────┘
 *
 *              Emulator::normalize_labels()
 *                     │
 *                     ▼
 * ┌──────────────────────────────────────────────────────┐
 * │  Labels Resolved to Absolute Addresses               │
 * │                                                       │
 * │  IMEM[4]: {op: BEQ,  rs1: x1, rs2: x2, imm: 0x0018}  │
 * │           (branch to address 24 = "done")            │
 * │  IMEM[5]: {op: JAL,  rd: x0, imm: 0x0008}            │
 * │           (jump to address 8 = "loop")               │
 * └──────────────────────────────────────────────────────┘
 * ```
 *
 * **Parsing Process:**
 *
 * ```
 * ┌─────────────────────────────────────────┐
 * │  1. Read Assembly File Line-by-Line     │
 * │     - Open file stream                  │
 * │     - Convert to lowercase              │
 * │     - Tokenize on whitespace/commas     │
 * └──────────────┬──────────────────────────┘
 *                │
 *                ▼
 * ┌─────────────────────────────────────────┐
 * │  2. Process Each Line                   │
 * │     ────────────────────────────────     │
 * │     Comment (#)?      → Skip line       │
 * │     Directive (.)?    → Handle directive│
 * │     Label (:)?        → Record location │
 * │     Instruction?      → Parse & store   │
 * └──────────────┬──────────────────────────┘
 *                │
 *                ▼
 * ┌─────────────────────────────────────────┐
 * │  3. Handle Assembler Directives         │
 * │     ────────────────────────────────     │
 * │     .text    → Switch to code section   │
 * │     .data    → Switch to data section   │
 * │     .word    → Store 32-bit value       │
 * │     .half    → Store 16-bit value       │
 * │     .byte    → Store 8-bit value        │
 * └──────────────┬──────────────────────────┘
 *                │
 *                ▼
 * ┌─────────────────────────────────────────┐
 * │  4. Parse Instructions                  │
 * │     ────────────────────────────────     │
 * │     • Check if pseudo-instruction       │
 * │     • Expand pseudo → base instructions │
 * │     • Parse opcode mnemonic             │
 * │     • Parse operands (registers/imms)   │
 * │     • Store in IMEM array               │
 * └──────────────┬──────────────────────────┘
 *                │
 *                ▼
 * ┌─────────────────────────────────────────┐
 * │  5. Record Label Locations              │
 * │     ────────────────────────────────     │
 * │     main:   → labels[0] = {name: "main", │
 * │                            loc: 0x0000}  │
 * │     loop:   → labels[1] = {name: "loop", │
 * │                            loc: 0x0008}  │
 * │     done:   → labels[2] = {name: "done", │
 * │                            loc: 0x0018}  │
 * └──────────────┬──────────────────────────┘
 *                │
 *                ▼
 * ┌─────────────────────────────────────────┐
 * │  6. Normalize Labels                    │
 * │     ────────────────────────────────     │
 * │     • Find all label references         │
 * │     • Look up label in labels array     │
 * │     • Replace symbolic → absolute addr  │
 * │     • Validate branch/jump distances    │
 * └─────────────────────────────────────────┘
 * ```
 *
 * **Pseudo-Instruction Expansion:**
 *
 * RISC-V defines pseudo-instructions that expand to base instructions:
 *
 * ```
 * Pseudo          Expands To                      Description
 * ──────────────────────────────────────────────────────────────
 * li rd, imm      lui  rd, imm[31:12]            Load immediate
 *                 addi rd, rd, imm[11:0]
 *
 * la rd, label    lui  rd, label[31:12]          Load address
 *                 addi rd, rd, label[11:0]
 *
 * mv rd, rs       addi rd, rs, 0                 Copy register
 *
 * j label         jal  x0, label                 Unconditional jump
 *
 * ret             jalr x0, ra, 0                 Return from function
 *
 * bnez rs, label  bne  rs, x0, label             Branch if not zero
 *
 * beqz rs, label  beq  rs, x0, label             Branch if zero
 * ```
 *
 * **Example Parsing:**
 *
 * ```
 * Input Assembly:
 *   li x1, 0x12345678   # Load large immediate
 *
 * Pseudo-Instruction Expansion:
 *   lui  x1, 0x12345    # Load upper 20 bits
 *   addi x1, x1, 0x678  # Add lower 12 bits
 *
 * Parsed Instructions:
 *   IMEM[0]: {op: LUI,  rd: x1, imm: 0x12345}
 *   IMEM[1]: {op: ADDI, rd: x1, rs1: x1, imm: 0x678}
 * ```
 *
 * **Register Parsing:**
 *
 * Supports both numeric (x0-x31) and ABI names:
 *
 * ```
 * ABI Name    Register    Usage
 * ───────────────────────────────────────────
 * zero        x0          Constant 0
 * ra          x1          Return address
 * sp          x2          Stack pointer
 * gp          x3          Global pointer
 * tp          x4          Thread pointer
 * t0-t2       x5-x7       Temporaries
 * s0/fp       x8          Saved / Frame pointer
 * s1          x9          Saved register
 * a0-a1       x10-x11     Arguments / Return values
 * a2-a7       x12-x17     Arguments
 * s2-s11      x18-x27     Saved registers
 * t3-t6       x28-x31     Temporaries
 * ```
 *
 * **Immediate Value Parsing:**
 *
 * ```
 * Format              Example         Parsed As
 * ──────────────────────────────────────────────
 * Decimal             42              0x0000002A
 * Hexadecimal         0xFF            0x000000FF
 * Negative decimal    -10             0xFFFFFFF6
 * Negative hex        -0x10           0xFFFFFFF0
 *
 * Sign Extension:
 *   12-bit immediate: -10 → 0xFFF6 → sign_extend → 0xFFFFFFF6
 *   20-bit immediate: 0x80000 → sign_extend if needed
 * ```
 *
 * **Memory Offset Parsing:**
 *
 * Load/Store instructions use format: `offset(base_register)`
 *
 * ```
 * Assembly            Parsed As
 * ──────────────────────────────────────
 * lw x1, 0(x2)        base: x2, offset: 0
 * sw x3, 100(sp)      base: sp(x2), offset: 100
 * lh x4, -4(x5)       base: x5, offset: -4
 * ```
 *
 * **Label Resolution:**
 *
 * Labels are resolved in two passes:
 *
 * **Pass 1 (parse)**: Record label locations
 * ```
 * main:     address 0x0000
 * loop:     address 0x0008
 * done:     address 0x0020
 * ```
 *
 * **Pass 2 (normalize_labels)**: Replace symbolic → absolute
 * ```
 * Before: beq x1, x2, done      (symbolic)
 * After:  beq x1, x2, 0x0020    (absolute address)
 * ```
 *
 * **Error Handling:**
 *
 * The parser detects and reports various syntax errors:
 *
 * ```
 * Error Type                  Example                      Detection
 * ────────────────────────────────────────────────────────────────────
 * Invalid register name       add x1, x33, x2              parse_reg()
 * Malformed immediate         addi x1, x0, 9999999         parse_imm()
 * Unknown opcode              foo x1, x2, x3               parse_instr()
 * Undefined label             beq x1, x2, undefined        label_addr()
 * Branch target too far       beq x1, x2, distant_label    normalize_labels()
 * Wrong operand count         add x1, x2 (missing x3)      syntax check
 * Invalid directive           .foo                         parse_assembler_directive()
 * ```
 *
 * **Configuration Parameters:**
 *
 * ```
 * Parameter           Default     Description
 * ───────────────────────────────────────────────────────────
 * text_offset         0x0000      Start address of .text section
 * data_offset         0x8000      Start address of .data section
 * memory_size         65536       Total DMEM size (bytes)
 * max_label_count     1024        Maximum number of labels
 * max_src_len         65536       Maximum source string length
 * ```
 *
 * **Usage Example:**
 *
 * ```cpp
 * // Create emulator
 * Emulator* emu = new Emulator("RISC-V Emulator");
 *
 * // Allocate memory
 * uint8_t* dmem = new uint8_t[65536];
 * instr* imem = new instr[8192];  // 32KB / 4 bytes per instruction
 *
 * // Parse assembly file
 * emu->parse("program.s", dmem, imem);
 * // → Populates imem with parsed instructions
 * // → Initializes dmem with .data section
 *
 * // Resolve labels
 * emu->normalize_labels(imem);
 * // → Converts symbolic labels to absolute addresses
 *
 * // IMEM and DMEM now ready for CPU execution
 * ```
 *
 * **Key Functions:**
 * - `parse()`: Main entry point, orchestrates file parsing
 * - `parse_instr()`: Parses individual instruction mnemonics
 * - `parse_pseudoinstructions()`: Expands pseudo-instructions
 * - `parse_reg()`: Converts register names to indices
 * - `parse_imm()`: Parses immediate values with sign extension
 * - `parse_mem()`: Handles offset(register) format for loads/stores
 * - `normalize_labels()`: Resolves symbolic labels to addresses
 * - `label_addr()`: Looks up label in symbol table
 * - `signextend()`: Sign-extends immediate values
 *
 * @see CPU for instruction execution
 * @see SOC for emulator integration
 * @see DataStruct.hh for instruction structure definitions
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @copyright Apache License 2.0
 */

#include "Emulator.hh"

#include "SystemConfig.hh"

Emulator::Emulator(std::string _name) : label_count(0), memoff(0) {
	CLASS_INFO << "asm_file_path : " << acalsim::top->getParameter<std::string>("Emulator", "asm_file_path");

	CLASS_INFO << "memory_size : " << acalsim::top->getParameter<int>("Emulator", "memory_size") << " Bytes";

	auto max_label_count = acalsim::top->getParameter<int>("Emulator", "max_label_count");
	auto max_src_len     = acalsim::top->getParameter<int>("Emulator", "max_src_len");
	this->labels         = (label_loc*)malloc(max_label_count * sizeof(label_loc));
	this->src.offset     = 0;
	this->src.src        = (char*)malloc(sizeof(char) * max_src_len);
}

void Emulator::init() {}

void Emulator::append_source(const char* _op, const char* _a1, const char* _a2, const char* _a3, source* _src,
                             instr* _i) {
	char tbuf[128];  // not safe... static size... but should be okay since label length enforced
	if (_op && _a1 && !_a2 && !_a3) {
		sprintf(tbuf, "%s %s", _op, _a1);
	} else if (_op && _a1 && _a2 && !_a3) {
		sprintf(tbuf, "%s %s, %s", _op, _a1, _a2);
	} else if (_op && _a1 && _a2 && _a3) {
		sprintf(tbuf, "%s %s, %s, %s", _op, _a1, _a2, _a3);
	} else {
		return;
	}
	int  slen        = strlen(tbuf);
	auto max_src_len = acalsim::top->getParameter<int>("Emulator", "max_src_len");
	if (slen + _src->offset < max_src_len) {
		strncpy(_src->src + _src->offset, tbuf, strlen(tbuf));

		_i->psrc = _src->src + _src->offset;
		_src->offset += slen + 1;
	}
}

int Emulator::parse_reg(char* _tok, int _line, bool _strict) {
	if (_tok[0] == 'x') {
		int ri = atoi(_tok + 1);
		if (ri < 0 || ri > 32) {
			if (_strict) print_syntax_error(_line, "Malformed register name");
			return -1;
		}
		return ri;
	}
	if (streq(_tok, "zero")) return 0;
	if (streq(_tok, "ra")) return 1;
	if (streq(_tok, "sp")) return 2;
	if (streq(_tok, "gp")) return 3;
	if (streq(_tok, "tp")) return 4;
	if (streq(_tok, "t0")) return 5;
	if (streq(_tok, "t1")) return 6;
	if (streq(_tok, "t2")) return 7;
	if (streq(_tok, "s0")) return 8;
	if (streq(_tok, "s1")) return 9;
	if (streq(_tok, "a0")) return 10;
	if (streq(_tok, "a1")) return 11;
	if (streq(_tok, "a2")) return 12;
	if (streq(_tok, "a3")) return 13;
	if (streq(_tok, "a4")) return 14;
	if (streq(_tok, "a5")) return 15;
	if (streq(_tok, "a6")) return 16;
	if (streq(_tok, "a7")) return 17;
	if (streq(_tok, "s2")) return 18;
	if (streq(_tok, "s3")) return 19;
	if (streq(_tok, "s4")) return 20;
	if (streq(_tok, "s5")) return 21;
	if (streq(_tok, "s6")) return 22;
	if (streq(_tok, "s7")) return 23;
	if (streq(_tok, "s8")) return 24;
	if (streq(_tok, "s9")) return 25;
	if (streq(_tok, "s10")) return 26;
	if (streq(_tok, "s11")) return 27;
	if (streq(_tok, "t3")) return 28;
	if (streq(_tok, "t4")) return 29;
	if (streq(_tok, "t5")) return 30;
	if (streq(_tok, "t6")) return 31;

	if (_strict) print_syntax_error(_line, "Malformed register name");
	return -1;
}

uint32_t Emulator::parse_imm(char* _tok, int _bits, int _line, bool _strict) {
	if (!(_tok[0] >= '0' && _tok[0] <= '9') && _tok[0] != '-' && _strict) {
		print_syntax_error(_line, "Malformed immediate value");
	}
	long int imml = strtol(_tok, NULL, 0);

	if (imml > ((1 << _bits) - 1) || imml < -(1 << (_bits - 1))) {
		printf("Syntax error at token %s\n", _tok);
		exit(1);
	}
	uint64_t uv = *(uint64_t*)&imml;
	uint32_t hv = (uv & UINT32_MAX);

	return hv;
}

void Emulator::parse_mem(char* _tok, int* _reg, uint32_t* _imm, int _bits, int _line) {
	char* imms = strtok(_tok, "(");
	char* regs = strtok(NULL, ")");
	*_imm      = parse_imm(imms, _bits, _line);
	*_reg      = parse_reg(regs, _line);
}

int Emulator::parse_assembler_directive(int _line, char* _ftok, uint8_t* _mem, int _memoff) {
	// printf( "assembler directive %s\n", ftok );
	if (0 == memcmp(_ftok, ".text", strlen(_ftok))) {
		if (strtok(NULL, " \t\r\n")) { print_syntax_error(_line, "Tokens after assembler directive"); }
		// cur_section = SECTION_TEXT;
		auto text_offset = acalsim::top->getParameter<int>("Emulator", "text_offset");
		_memoff          = text_offset;
		// printf( "starting text section\n" );
	} else if (0 == memcmp(_ftok, ".data", strlen(_ftok))) {
		// cur_section = SECTION_TEXT;
		auto data_offset = acalsim::top->getParameter<int>("Emulator", "data_offset");
		_memoff          = data_offset;
		// printf( "starting data section\n" );
	} else if (0 == memcmp(_ftok, ".byte", strlen(_ftok)))
		_memoff = parse_data_element(_line, 1, _mem, _memoff);
	else if (0 == memcmp(_ftok, ".half", strlen(_ftok)))
		_memoff = parse_data_element(_line, 2, _mem, _memoff);
	else if (0 == memcmp(_ftok, ".word", strlen(_ftok)))
		_memoff = parse_data_element(_line, 4, _mem, _memoff);
	else {
		printf("Undefined assembler directive at line %d: %s\n", _line, _ftok);
		exit(3);
	}
	return _memoff;
}

int Emulator::parse_instr(int _line, char* _ftok, instr* _imem, int _memoff, label_loc* _labels, source* _src) {
	auto data_offset = acalsim::top->getParameter<int>("Emulator", "data_offset");
	if (_memoff + 4 > data_offset) {
		printf("Instructions in data segment!\n");
		exit(1);
	}
	char* o1 = strtok(NULL, " \t\r\n,");
	char* o2 = strtok(NULL, " \t\r\n,");
	char* o3 = strtok(NULL, " \t\r\n,");
	char* o4 = strtok(NULL, " \t\r\n,");

	int ioff  = _memoff / 4;
	int pscnt = parse_pseudoinstructions(_line, _ftok, _imem, ioff, _labels, o1, o2, o3, o4, _src);
	if (pscnt > 0) {
		return pscnt;
	} else {
		instr* i      = &_imem[ioff];
		i->str        = _ftok;
		instr_type op = parse_instr(_ftok);
		i->op         = op;
		i->orig_line  = _line;
		append_source(_ftok, o1, o2, o3, _src, i);
		switch (op) {
			case UNIMPL: return 1;

			case JAL:
				if (o2) {  // two operands, reg, label
					if (!o1 || !o2 || o3 || o4) print_syntax_error(_line, "Invalid format");
					i->a1.type = OPTYPE_REG;
					i->a1.reg  = parse_reg(o1, _line);
					i->a2.type = OPTYPE_LABEL;
					strncpy(i->a2.label, o2, MAX_LABEL_LEN);
				} else {  // one operand, label
					if (!o1 || o2 || o3 || o4) print_syntax_error(_line, "Invalid format");

					i->a1.type = OPTYPE_REG;
					i->a1.reg  = 1;
					i->a2.type = OPTYPE_LABEL;
					strncpy(i->a2.label, o1, MAX_LABEL_LEN);
				}
				return 1;
			case JALR:
				if (!o1 || !o2 || o3 || o4) print_syntax_error(_line, "Invalid format");
				i->a1.reg = parse_reg(o1, _line);
				parse_mem(o2, &i->a2.reg, &i->a3.imm, 12, _line);
				return 1;
			case ADD:
			case SUB:
			case SLT:
			case SLTU:
			case AND:
			case OR:
			case XOR:
			case SLL:
			case SRL:
			case SRA:
				if (!o1 || !o2 || !o3 || o4) print_syntax_error(_line, "Invalid format");
				i->a1.reg = parse_reg(o1, _line);
				i->a2.reg = parse_reg(o2, _line);
				i->a3.reg = parse_reg(o3, _line);
				return 1;
			case LB:
			case LBU:
			case LH:
			case LHU:
			case LW:
			case SB:
			case SH:
			case SW:
				if (!o1 || !o2 || o3 || o4) print_syntax_error(_line, "Invalid format");
				i->a1.reg = parse_reg(o1, _line);
				parse_mem(o2, &i->a2.reg, &i->a3.imm, 12, _line);
				return 1;
			case ADDI:
			case SLTI:
			case SLTIU:
			case ANDI:
			case ORI:
			case XORI:
			case SLLI:
			case SRLI:
			case SRAI:
				if (!o1 || !o2 || !o3 || o4) print_syntax_error(_line, "Invalid format");

				i->a1.reg = parse_reg(o1, _line);
				i->a2.reg = parse_reg(o2, _line);
				i->a3.imm = signextend(parse_imm(o3, 12, _line), 12);
				return 1;
			case BEQ:
			case BGE:
			case BGEU:
			case BLT:
			case BLTU:
			case BNE:
				if (!o1 || !o2 || !o3 || o4) print_syntax_error(_line, "Invalid format");
				i->a1.reg  = parse_reg(o1, _line);
				i->a2.reg  = parse_reg(o2, _line);
				i->a3.type = OPTYPE_LABEL;
				strncpy(i->a3.label, o3, MAX_LABEL_LEN);
				return 1;
			case LUI:
			case AUIPC:  // how to deal with LSB correctly? FIXME
				if (!o1 || !o2 || o3 || o4) print_syntax_error(_line, "Invalid format");
				i->a1.reg = parse_reg(o1, _line);
				i->a2.imm = (parse_imm(o2, 20, _line));
				return 1;
			case HCF: return 1;
		}
	}
	return 1;
}

instr_type Emulator::parse_instr(char* _tok) {
	if (streq(_tok, "add")) return ADD;
	if (streq(_tok, "sub")) return SUB;
	if (streq(_tok, "slt")) return SLT;
	if (streq(_tok, "sltu")) return SLTU;
	if (streq(_tok, "and")) return AND;
	if (streq(_tok, "or")) return OR;
	if (streq(_tok, "xor")) return XOR;
	if (streq(_tok, "sll")) return SLL;
	if (streq(_tok, "srl")) return SRL;
	if (streq(_tok, "sra")) return SRA;

	// 1r, imm -> 1r
	if (streq(_tok, "addi")) return ADDI;
	if (streq(_tok, "slti")) return SLTI;
	if (streq(_tok, "sltiu")) return SLTIU;
	if (streq(_tok, "andi")) return ANDI;
	if (streq(_tok, "ori")) return ORI;
	if (streq(_tok, "xori")) return XORI;
	if (streq(_tok, "slli")) return SLLI;
	if (streq(_tok, "srli")) return SRLI;
	if (streq(_tok, "srai")) return SRAI;

	// load/store
	if (streq(_tok, "lb")) return LB;
	if (streq(_tok, "lbu")) return LBU;
	if (streq(_tok, "lh")) return LH;
	if (streq(_tok, "lhu")) return LHU;
	if (streq(_tok, "lw")) return LW;
	if (streq(_tok, "sb")) return SB;
	if (streq(_tok, "sh")) return SH;
	if (streq(_tok, "sw")) return SW;

	// branch
	if (streq(_tok, "beq")) return BEQ;
	if (streq(_tok, "bge")) return BGE;
	if (streq(_tok, "bgeu")) return BGEU;
	if (streq(_tok, "blt")) return BLT;
	if (streq(_tok, "bltu")) return BLTU;
	if (streq(_tok, "bne")) return BNE;

	// jal
	if (streq(_tok, "jal")) return JAL;
	if (streq(_tok, "jalr")) return JALR;

	// lui
	if (streq(_tok, "auipc")) return AUIPC;
	if (streq(_tok, "lui")) return LUI;

	// unimpl
	// if ( streq(tok, "unimpl") ) return UNIMPL;
	if (streq(_tok, "hcf")) return HCF;
	return UNIMPL;
}

int Emulator::parse_pseudoinstructions(int _line, char* _ftok, instr* _imem, int _ioff, label_loc* _labels, char* _o1,
                                       char* _o2, char* _o3, char* _o4, source* _src) {
	if (streq(_ftok, "li")) {
		if (!_o1 || !_o2 || _o3) print_syntax_error(_line, "Invalid format");

		int      reg  = parse_reg(_o1, _line);
		long int imml = strtol(_o2, NULL, 0);

		if (reg < 0 || imml > UINT32_MAX || imml < INT32_MIN) {
			printf("Syntax error at line %d -- %lx, %x\n", _line, imml, INT32_MAX);
			exit(1);
		}
		uint64_t uv = *(uint64_t*)&imml;
		uint32_t hv = (uv & UINT32_MAX);

		char areg[4];
		sprintf(areg, "x%02d", reg);
		char immu[12];
		sprintf(immu, "0x%08x", (hv >> 12));
		char immd[12];
		sprintf(immd, "0x%08x", (hv & ((1 << 12) - 1)));

		instr* i     = &_imem[_ioff];
		i->op        = LUI;
		i->a1.type   = OPTYPE_REG;
		i->a1.reg    = reg;
		i->a2.type   = OPTYPE_IMM;
		i->a2.imm    = hv >> 12;
		i->orig_line = _line;
		append_source("lui", areg, immu, NULL, _src, i);
		instr* i2 = &_imem[_ioff + 1];

		i2->op        = ADDI;
		i2->a1.type   = OPTYPE_REG;
		i2->a1.reg    = reg;
		i2->a2.type   = OPTYPE_REG;
		i2->a2.reg    = reg;
		i2->a3.type   = OPTYPE_IMM;
		i2->a3.imm    = (hv & ((1 << 12) - 1));
		i2->orig_line = _line;
		append_source("addi", areg, areg, immd, _src, i2);
		return 2;
	}
	if (streq(_ftok, "la")) {
		if (!_o1 || !_o2 || _o3) print_syntax_error(_line, "Invalid format");

		int reg = parse_reg(_o1, _line);

		instr* i   = &_imem[_ioff];
		i->op      = LUI;
		i->a1.type = OPTYPE_REG;
		i->a1.reg  = reg;
		i->a2.type = OPTYPE_LABEL;
		strncpy(i->a2.label, _o2, MAX_LABEL_LEN);
		i->orig_line = _line;
		// append_source(ftok, o1, o2, o3, src, i); // done in normalize
		instr* i2   = &_imem[_ioff + 1];
		i2->op      = ADDI;
		i2->a1.type = OPTYPE_REG;
		i2->a1.reg  = reg;
		i2->a2.type = OPTYPE_REG;
		i2->a2.reg  = reg;
		i2->a3.type = OPTYPE_LABEL;
		strncpy(i2->a3.label, _o2, MAX_LABEL_LEN);
		i2->orig_line = _line;
		// append_source(ftok, o1, o2, o3, src, i2); // done in normalize
		return 2;
	}
	if (streq(_ftok, "ret")) {
		if (_o1) print_syntax_error(_line, "Invalid format");

		instr* i     = &_imem[_ioff];
		i->op        = JALR;
		i->a1.type   = OPTYPE_REG;
		i->a1.reg    = 0;
		i->a2.type   = OPTYPE_REG;
		i->a2.reg    = 1;
		i->a3.type   = OPTYPE_IMM;
		i->a3.imm    = 0;
		i->orig_line = _line;
		append_source("jalr", "x0", "x1", "x0", _src, i);
		return 1;
	}
	if (streq(_ftok, "j")) {
		if (!_o1 || _o2) print_syntax_error(_line, "Invalid format");

		instr* i   = &_imem[_ioff];
		i->op      = JAL;
		i->a1.type = OPTYPE_REG;
		i->a1.reg  = 0;
		i->a2.type = OPTYPE_LABEL;
		strncpy(i->a2.label, _o1, MAX_LABEL_LEN);
		i->orig_line = _line;
		append_source("j", "x0", _o1, NULL, _src, i);
		return 1;
	}
	if (streq(_ftok, "mv")) {
		if (!_o1 || !_o2 || _o3) print_syntax_error(_line, "Invalid format");
		instr* i     = &_imem[_ioff];
		i->op        = ADDI;
		i->a1.type   = OPTYPE_REG;
		i->a1.reg    = parse_reg(_o1, _line);
		i->a2.type   = OPTYPE_REG;
		i->a2.reg    = parse_reg(_o2, _line);
		i->a3.type   = OPTYPE_IMM;
		i->a3.imm    = 0;
		i->orig_line = _line;
		append_source("addi", _o1, _o2, NULL, _src, i);
		return 1;
	}
	if (streq(_ftok, "bnez")) {
		if (!_o1 || !_o2 || _o3) print_syntax_error(_line, "Invalid format");
		instr* i   = &_imem[_ioff];
		i->op      = BNE;
		i->a1.type = OPTYPE_REG;
		i->a1.reg  = parse_reg(_o1, _line);
		i->a2.type = OPTYPE_REG;
		i->a2.reg  = 0;
		i->a3.type = OPTYPE_LABEL;
		strncpy(i->a3.label, _o2, MAX_LABEL_LEN);
		i->orig_line = _line;
		append_source("bne", "x0", _o1, _o2, _src, i);
		return 1;
	}
	if (streq(_ftok, "beqz")) {
		if (!_o1 || !_o2 || _o3) print_syntax_error(_line, "Invalid format");
		instr* i   = &_imem[_ioff];
		i->op      = BEQ;
		i->a1.type = OPTYPE_REG;
		i->a1.reg  = parse_reg(_o1, _line);
		i->a2.type = OPTYPE_REG;
		i->a2.reg  = 0;
		i->a3.type = OPTYPE_LABEL;
		strncpy(i->a3.label, _o2, MAX_LABEL_LEN);
		i->orig_line = _line;
		append_source("beq", "x0", _o1, _o2, _src, i);
		return 1;
	}
	return 0;
}

int Emulator::parse_data_element(int _line, int _size, uint8_t* _mem, int _offset) {
	while (char* t = strtok(NULL, " \t\r\n")) {
		errno      = 0;
		int64_t v  = strtol(t, NULL, 0);
		int64_t vs = (v >> (_size * 8));
		if (errno == ERANGE || (vs > 0 && vs != -1)) {
			printf("Value out of bounds at line %d : %s\n", _line, t);
			exit(2);
		}
		// printf ( "parse_data_element %d: %d %ld %d %d\n", line, size, v, errno, sizeof(long int));
		memcpy(&_mem[_offset], &v, _size);
		_offset += _size;
		// strtok(NULL, ",");
	}
	return _offset;
}

void Emulator::print_syntax_error(int _line, const char* _msg) {
	ERROR << "Line " << _line << ": Syntax error! " << _msg;
}

bool Emulator::streq(char* _s, const char* _q) {
	if (strcmp(_s, _q) == 0) return true;

	return false;
}

uint32_t Emulator::signextend(uint32_t _in, int _bits) {
	if (_in & (1 << (_bits - 1))) return ((-1) << _bits) | _in;
	return _in;
}

void Emulator::parse(const std::string& _file_path, uint8_t* _mem, instr* _imem) {
	this->parse(_file_path, _mem, _imem, this->memoff, this->labels, this->label_count, &(this->src));
}

void Emulator::parse(const std::string& _file_path, uint8_t* _mem, instr* _imem, int& _memoff, label_loc* _labels,
                     int& _label_count, source* _src) {
	FILE* fin = fopen(_file_path.c_str(), "r");
	if (!fin) { ERROR << _file_path << ": No such file"; }
	int line = 0;

	CLASS_INFO << "Parsing input file";

	// sectionType cur_section = SECTION_NONE;
	char rbuf[1024];
	while (!feof(fin)) {
		if (!fgets(rbuf, 1024, fin)) break;
		for (char* p = rbuf; *p; ++p) *p = tolower(*p);
		line++;

		char* _ftok = strtok(rbuf, " \t\r\n");
		if (!_ftok) continue;

		if (_ftok[0] == '#') continue;
		if (_ftok[0] == '.') {
			_memoff = parse_assembler_directive(line, _ftok, _mem, _memoff);
		} else if (_ftok[strlen(_ftok) - 1] == ':') {
			_ftok[strlen(_ftok) - 1] = 0;
			if (strlen(_ftok) >= MAX_LABEL_LEN) {
				printf("Exceeded maximum length of label: %s\n", _ftok);
				exit(3);
			}
			auto max_label_count = acalsim::top->getParameter<int>("Emulator", "max_label_count");
			if (_label_count >= max_label_count) {
				printf("Exceeded maximum number of supported labels");
				exit(3);
			}
			strncpy(_labels[_label_count].label, _ftok, MAX_LABEL_LEN);
			_labels[_label_count].loc = _memoff;
			_label_count++;
			// printf( "Parsing label %s at mem %x\n", ftok, memoff );

			char* ntok = strtok(NULL, " \t\r\n");
			// there is more code after label
			if (ntok) {
				if (ntok[0] == '.') {
					_memoff = parse_assembler_directive(line, ntok, _mem, _memoff);
				} else {
					int count = parse_instr(line, ntok, _imem, _memoff, _labels, _src);
					for (int i = 0; i < count; i++) *(uint32_t*)&_mem[_memoff + (i * 4)] = 0xcccccccc;
					_memoff += count * 4;
				}
			}
		} else {
			int count = parse_instr(line, _ftok, _imem, _memoff, _labels, _src);
			for (int i = 0; i < count; i++) *(uint32_t*)&_mem[_memoff + (i * 4)] = 0xcccccccc;
			_memoff += count * 4;
		}
	}
}

void Emulator::normalize_labels(instr* _imem) {
	this->normalize_labels(_imem, this->labels, this->label_count, &(this->src));
}

void Emulator::normalize_labels(instr* _imem, label_loc* _labels, int _label_count, source* _src) {
	auto data_offset = acalsim::top->getParameter<int>("Emulator", "data_offset");
	for (int i = 0; i < data_offset / 4; i++) {
		instr* ii = &_imem[i];
		if (ii->op == UNIMPL) continue;

		if (ii->a1.type == OPTYPE_LABEL) {
			ii->a1.type = OPTYPE_IMM;
			ii->a1.imm  = label_addr(ii->a1.label, _labels, _label_count, ii->orig_line);
		}
		if (ii->a2.type == OPTYPE_LABEL) {
			ii->a2.type = OPTYPE_IMM;
			ii->a2.imm  = label_addr(ii->a2.label, _labels, _label_count, ii->orig_line);
			switch (ii->op) {
				case LUI: {
					ii->a2.imm = (ii->a2.imm >> 12);
					char areg[4];
					sprintf(areg, "x%02d", ii->a1.reg);
					char immu[12];
					sprintf(immu, "0x%08x", ii->a2.imm);
					// printf( "LUI %d 0x%x %s\n", ii->a1.reg, ii->a2.imm, immu );
					append_source("lui", areg, immu, NULL, _src, ii);
					break;
				}
				case JAL:
					int pc     = (i * 4);
					int target = ii->a3.imm;
					int diff   = pc - target;
					if (diff < 0) diff = -diff;

					if (diff >= (1 << 21)) {
						printf("JAL instruction target out of bounds\n");
						exit(3);
					}
					break;
			}
		}
		if (ii->a3.type == OPTYPE_LABEL) {
			ii->a3.type = OPTYPE_IMM;
			ii->a3.imm  = label_addr(ii->a3.label, _labels, _label_count, ii->orig_line);
			switch (ii->op) {
				case ADDI: {
					ii->a3.imm = ii->a3.imm & ((1 << 12) - 1);
					char a1reg[4];
					sprintf(a1reg, "x%02d", ii->a1.reg);
					char a2reg[4];
					sprintf(a2reg, "x%02d", ii->a2.reg);
					char immd[12];
					sprintf(immd, "0x%08x", ii->a3.imm);
					// printf( "ADDI %d %d 0x%x %s\n", ii->a1.reg, ii->a2.reg, ii->a3.imm, immd );
					append_source("addi", a1reg, a2reg, immd, _src, ii);
					break;
				}
				case BEQ:
				case BGE:
				case BGEU:
				case BLT:
				case BLTU:
				case BNE: {
					int pc     = (i * 4);
					int target = ii->a3.imm;
					int diff   = pc - target;
					if (diff < 0) diff = -diff;

					if (diff >= (1 << 13)) {
						printf("Branch instruction target out of bounds\n");
						exit(3);
					}
					break;
				}
			}
		}
	}
}

uint32_t Emulator::label_addr(char* _label, label_loc* _labels, int _label_count, int _orig_line) {
	for (int i = 0; i < _label_count; i++) {
		if (streq(_labels[i].label, _label)) return _labels[i].loc;
	}
	print_syntax_error(_orig_line, "Undefined label");
	return -1;
}
