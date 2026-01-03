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

#pragma once

#include <string>

#include "ACALSim.hh"
#include "DataMemory.hh"
#include "DataStruct.hh"
#include "Emulator.hh"
#include "InstPacket.hh"
#include "MemPacket.hh"

class SOC;

/**
 * @class CPU A CPU model integrated with CPU ISA Emulator
 * @brief Implements a basic CPU with instruction execution, memory operations, and register file
 */
class CPU : public acalsim::SimModule {
public:
	/**
	 * @brief Constructor for the CPU class
	 * @param _name Name identifier for the CPU instance
	 * @param _emulator Pointer to the ISA emulator
	 */
	CPU(std::string _name, SOC* _soc);

	/**
	 * @brief Destructor that frees instruction memory
	 */
	virtual ~CPU() { delete[] this->imem; }

	/**
	 * @brief Execute one instruction
	 */
	void execOneInstr();

	/**
	 * @brief Execute an instruction
	 * @param _i The instruction to execute
	 */
	void processInstr(const instr& _i, InstPacket* instPacket);

	/**
	 * @brief Commits an instruction after execution
	 * @param _i The instruction to commit
	 */
	void commitInstr(const instr& _i, InstPacket* instPacket);

	void retrySendInstPacket(MasterPort* mp);
	/**
	 * @brief Performs a memory read operation
	 * @param _i The instruction requesting the read
	 * @param _op Type of instruction
	 * @param _addr Memory address to read from
	 * @param _a1 Operand for storing the read data
	 * @return Whether the memory access is done or not
	 */
	bool memRead(const instr& _i, instr_type _op, uint32_t _addr, operand _a1, InstPacket* instPacket);

	/**
	 * @brief Performs a memory write operation
	 * @param _i The instruction requesting the write
	 * @param _op Type of instruction
	 * @param _addr Memory address to write to
	 * @param _data Data to write
	 * @return Whether the memory access is done or not
	 */
	bool memWrite(const instr& _i, instr_type _op, uint32_t _addr, uint32_t _data, InstPacket* instPacket);

	/**
	 * @brief Returns pointer to instruction memory
	 * @return Pointer to instruction memory array
	 */
	inline instr* getIMemPtr() const { return this->imem; }

	/**
	 * @brief Prints the contents of the register file
	 */
	void printRegfile() const;

protected:
	/**
	 * @brief Fetches an instruction from instruction memory
	 * @param _pc Program counter value indicating instruction address
	 * @return The fetched instruction
	 */
	instr fetchInstr(uint32_t _pc) const;

	/**
	 * @brief Converts instruction type to string representation
	 * @param _op Instruction type to convert
	 * @return String representation of the instruction
	 */
	std::string instrToString(instr_type _op) const;

	/**
	 * @brief Increments the instruction count
	 */
	inline void incrementInstCount() { this->inst_cnt++; }

	/**
	 * @brief Returns the current instruction count
	 * @return Reference to instruction count
	 */
	inline const int& getInstCount() const { return this->inst_cnt; }

private:
	instr*      imem;         ///< Pointer to instruction memory
	Emulator*   isaEmulator;  ///< Pointer to the ISA emulator
	uint32_t    rf[32];       ///< Register file with 32 general-purpose registers
	uint32_t    pc;           ///< Program counter
	int         inst_cnt;     ///< Counter for executed instructions
	InstPacket* pendingInstPacket;
	SOC*        soc;
};
