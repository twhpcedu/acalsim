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

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#define MAX_LABEL_LEN 32

typedef enum {
	UNIMPL = 0,
	ADD,
	ADDI,
	AND,
	ANDI,
	AUIPC,
	BEQ,
	BGE,
	BGEU,
	BLT,
	BLTU,
	BNE,
	JAL,
	JALR,
	LB,
	LBU,
	LH,
	LHU,
	LUI,
	LW,
	OR,
	ORI,
	SB,
	SH,
	SLL,
	SLLI,
	SLT,
	SLTI,
	SLTIU,
	SLTU,
	SRA,
	SRAI,
	SRL,
	SRLI,
	SUB,
	SW,
	XOR,
	XORI,
	HCF
} instr_type;

typedef struct {
	char* src;
	int   offset;
} source;

typedef enum {
	OPTYPE_NONE,  // more like "don't care"
	OPTYPE_REG,
	OPTYPE_IMM,
	OPTYPE_LABEL,
} operand_type;

struct operand {
	operand_type type = OPTYPE_NONE;
	char         label[MAX_LABEL_LEN];
	int          reg;
	uint32_t     imm;
};

struct instr {
	std::string str;
	instr_type  op;
	operand     a1;
	operand     a2;
	operand     a3;
	char*       psrc       = NULL;
	int         orig_line  = -1;
	bool        breakpoint = false;
};

struct label_loc {
	char label[MAX_LABEL_LEN];
	int  loc = -1;
};
