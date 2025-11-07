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

/**
 * @file Emulator.cc
 * @brief RISC-V Assembly Parser and ISA Functional Model
 *
 * @details
 * This file implements the Emulator class, which provides the **ISA functional model**
 * for the RISC-V simulator template. The Emulator is responsible for parsing RISC-V
 * assembly code and converting it into an internal instruction representation that the
 * CPU timing model can execute.
 *
 * **Role in Simulator:**
 * The Emulator is a **compile-time component** that runs once during initialization:
 * - Parses assembly source file (.s or .asm)
 * - Converts mnemonics to internal instruction structures
 * - Initializes instruction and data memory
 * - Resolves labels to addresses
 * - Does NOT participate in runtime execution (that's the CPU's job)
 *
 * **Assembly File Format:**
 * Supports standard RISC-V assembly with GCC-style directives:
 * @code{.asm}
 * .text                    # Code section
 * .global _start
 * _start:
 *     li a0, 10            # Pseudo-instruction: load immediate
 *     li a1, 20
 *     add a2, a0, a1       # Real instruction
 *     jal ra, print        # Jump and link
 *     hcf                  # Halt simulation
 *
 * .data                    # Data section
 *     msg: .word 0x12345678
 *     arr: .byte 1, 2, 3, 4
 * @endcode
 *
 * **Parsing Architecture:**
 * @code
 *                       ┌─────────────────────────────┐
 *                       │    Emulator::parse()         │
 *                       └──────────┬──────────────────┘
 *                                  │
 *                   ┌──────────────┴──────────────┐
 *                   │                             │
 *                   ▼                             ▼
 *        ┌──────────────────┐         ┌──────────────────┐
 *        │  .text section   │         │  .data section   │
 *        │   (Instructions) │         │  (Variables)     │
 *        └────────┬─────────┘         └────────┬─────────┘
 *                 │                            │
 *                 ├─> parse_instr()            ├─> parse_data_element()
 *                 │   ├─> parse_pseudoinstr() │   ├─> .byte
 *                 │   └─> parse_reg()         │   ├─> .half
 *                 │                            │   └─> .word
 *                 ▼                            ▼
 *         ┌──────────────┐            ┌──────────────┐
 *         │ Instruction  │            │  DataMemory  │
 *         │   Memory     │            │  (uint8_t*)  │
 *         │  (instr[])   │            └──────────────┘
 *         └──────┬───────┘
 *                │
 *                ├─> normalize_labels()
 *                │   └─> Resolve label references to addresses
 *                │
 *                ▼
 *         ┌──────────────┐
 *         │   Ready for  │
 *         │  Execution   │
 *         └──────────────┘
 * @endcode
 *
 * **Instruction Representation:**
 * Assembly mnemonics are converted to internal `instr` structures:
 * @code{.cpp}
 * struct instr {
 *     instr_type op;        // Opcode (ADD, ADDI, LW, etc.)
 *     operand a1;           // Destination/source register
 *     operand a2;           // Source register or address
 *     operand a3;           // Source register or immediate
 *     char* psrc;           // Original source text for debugging
 *     int orig_line;        // Line number in assembly file
 * };
 * @endcode
 *
 * **Pseudo-Instruction Expansion:**
 * The Emulator expands RISC-V pseudo-instructions into real instructions:
 * | Pseudo         | Expansion                                     |
 * |----------------|-----------------------------------------------|
 * | li rd, imm     | lui rd, imm[31:12] + addi rd, rd, imm[11:0]  |
 * | la rd, label   | lui rd, label[31:12] + addi rd, rd, label[11:0] |
 * | mv rd, rs      | addi rd, rs, 0                               |
 * | j label        | jal x0, label                                |
 * | ret            | jalr x0, ra, 0                               |
 * | beqz rs, label | beq rs, x0, label                            |
 * | bnez rs, label | bne rs, x0, label                            |
 *
 * **Label Resolution:**
 * Two-pass process:
 * 1. **First pass (parse):** Store label names and addresses
 * 2. **Second pass (normalize_labels):** Replace label operands with addresses
 *
 * Example:
 * @code{.asm}
 * loop:           # Label stored at address 0x100
 *     addi a0, a0, 1
 *     bne a0, a1, loop   # "loop" -> 0x100 during normalization
 * @endcode
 *
 * **Memory Layout:**
 * The Emulator initializes two memory regions:
 * @code
 * ┌──────────────────┐ 0x00000000 (text_offset)
 * │  Instruction     │ Stored in CPU::imem
 * │  Memory          │ Array of instr structures
 * │  (.text section) │
 * ├──────────────────┤ 0x00010000 (data_offset)
 * │  Data Memory     │ Stored in DataMemory
 * │  (.data section) │ Raw uint8_t array
 * │                  │
 * └──────────────────┘ memory_size
 * @endcode
 *
 * **Error Handling:**
 * The Emulator performs extensive syntax checking:
 * - Invalid register names (must be x0-x31 or ABI names)
 * - Out-of-range immediates (checked against bit width)
 * - Undefined labels
 * - Malformed instructions
 * - Branch/jump targets out of range
 *
 * **Configuration:**
 * Parsing behavior controlled by configs.json:
 * @code{.json}
 * {
 *   "Emulator": {
 *     "asm_file_path": "program.s",
 *     "memory_size": 1048576,      // 1MB total
 *     "text_offset": 0,             // Instructions start at 0
 *     "data_offset": 65536,         // Data starts at 64KB
 *     "max_label_count": 256,       // Max labels
 *     "max_src_len": 65536          // Max source text length
 *   }
 * }
 * @endcode
 *
 * **Key Functions:**
 * - parse(): Main entry point for assembly parsing
 * - normalize_labels(): Resolves label references to addresses
 * - parse_instr(): Parses individual instruction mnemonics
 * - parse_pseudoinstructions(): Expands pseudo-instructions
 * - parse_reg(): Converts register names to numbers
 * - parse_imm(): Parses immediate values (decimal, hex, binary)
 * - parse_data_element(): Handles .byte, .half, .word directives
 *
 * **Differences from Full Simulator:**
 * Both template and full simulators use the same Emulator:
 * - Same assembly syntax support
 * - Same pseudo-instruction expansion
 * - Same label resolution
 * - Same memory initialization
 *
 * The difference is in what happens AFTER parsing:
 * - Template: CPU executes directly (single-cycle)
 * - Full: Instructions traverse multi-stage pipeline
 *
 * @see CPU Timing model that executes parsed instructions
 * @see SOC System integration that calls parse()
 * @see DataStruct.hh Instruction and operand data structures
 * @see main.cc Overall simulator flow
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 * @version 1.0
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
		instr*     i  = &_imem[ioff];
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
