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

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>

#include "ACALSim.hh"
#include "DataMemory.hh"
#include "DataStruct.hh"

class Emulator : virtual public acalsim::HashableType {
public:
	Emulator(std::string _name = "Emulator");
	virtual ~Emulator() {}

	void init();

	// Lab7 Emulator Function Definition
	uint32_t label_addr(char* _label, label_loc* _labels, int _label_count, int _orig_line);
	void     append_source(const char* _op, const char* _a1, const char* _a2, const char* _a3, source* _src, instr* _i);
	int      parse_reg(char* _tok, int _line, bool _strict = true);
	uint32_t parse_imm(char* _tok, int _bits, int _line, bool _strict = true);
	void     parse_mem(char* _tok, int* _reg, uint32_t* _imm, int _bits, int _line);
	int      parse_assembler_directive(int _line, char* _ftok, uint8_t* _mem, int _memoff);
	int      parse_instr(int _line, char* _ftok, instr* _imem, int _memoff, label_loc* _labels, source* _src);
	instr_type parse_instr(char* _tok);
	int        parse_pseudoinstructions(int _line, char* _ftok, instr* _imem, int _ioff, label_loc* _labels, char* _o1,
	                                    char* _o2, char* _o3, char* _o4, source* _src);
	int        parse_data_element(int _line, int _size, uint8_t* _mem, int _offset);

	void     print_syntax_error(int _line, const char* _msg);
	bool     streq(char* _s, const char* _q);
	uint32_t signextend(uint32_t _in, int _bits);
	void     parse(const std::string& _file_path, uint8_t* _mem, instr* _imem);
	void     parse(const std::string& _file_path, uint8_t* _mem, instr* _imem, int& _memoff, label_loc* _labels,
	               int& _label_count, source* _src);
	void     normalize_labels(instr* _imem);
	void     normalize_labels(instr* _imem, label_loc* _labels, int _label_count, source* _src);

private:
	label_loc* labels;
	int        label_count;
	int        memoff;
	source     src;
};
