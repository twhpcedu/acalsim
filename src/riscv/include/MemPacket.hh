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
#include <functional>

#include "ACALSim.hh"
#include "DataStruct.hh"

// Forward declaration for MemReadRespPacket
class MemReadRespPacket;
// Forward declaration for MemWriteRespPacket
class MemWriteRespPacket;

/**
 * @class MemReadReqPacket
 * @brief Memory read request packet class for memory operations
 * @details Handles memory read requests with instruction information and callback handling
 */
class MemReadReqPacket : public acalsim::SimPacket {
public:
	/** @brief Default constructor */
	MemReadReqPacket() {}

	/**
	 * @brief Parameterized constructor for memory read request
	 * @param _callback Function to handle the response packet
	 * @param _i Instruction requesting the memory read
	 * @param _op Type of instruction operation
	 * @param _addr Memory address to read from
	 * @param _a1 Operand for the read operation
	 */
	MemReadReqPacket(std::function<void(MemReadRespPacket*)> _callback, const instr& _i, instr_type _op, uint32_t _addr,
	                 operand _a1)
	    : acalsim::SimPacket(), callback(_callback), i(_i), op(_op), addr(_addr), a1(_a1) {}

	/** @brief Virtual destructor */
	virtual ~MemReadReqPacket() {}

	/**
	 * @brief Renews the packet with new parameters
	 * @param _callback New callback function
	 * @param _i New instruction
	 * @param _op New operation type
	 * @param _addr New memory address
	 * @param _a1 New operand
	 */
	void renew(std::function<void(MemReadRespPacket*)> _callback, const instr& _i, instr_type _op, uint32_t _addr,
	           operand _a1);

	/** @brief Visit function for module interaction */
	void visit(acalsim::Tick _when, acalsim::SimModule& _module) override;
	/** @brief Visit function for simulator interaction */
	void visit(acalsim::Tick _when, acalsim::SimBase& _simulator) override;

	/** @return The instruction associated with this request */
	const instr& getInstr() { return this->i; }
	/** @return The operation type */
	const instr_type& getOP() { return this->op; }
	/** @return The memory address */
	const uint32_t& getAddr() { return this->addr; }
	/** @return The operand */
	const operand getA1() { return this->a1; }
	/** @return The callback function */
	auto getCallback() { return this->callback; }

private:
	instr                                   i;         ///< Associated instruction
	instr_type                              op;        ///< Operation type
	uint32_t                                addr;      ///< Memory address
	operand                                 a1;        ///< Operand
	std::function<void(MemReadRespPacket*)> callback;  ///< Response callback function
};

/**
 * @class MemWriteReqPacket
 * @brief Memory write request packet class
 * @details Handles memory write requests with data and callback handling
 */
class MemWriteReqPacket : public acalsim::SimPacket {
public:
	/** @brief Default constructor */
	MemWriteReqPacket() {}

	/**
	 * @brief Parameterized constructor for memory write request
	 * @param _callback Function to handle the write completion
	 * @param _i Instruction requesting the write
	 * @param _op Type of instruction operation
	 * @param _addr Memory address to write to
	 * @param _data Data to write (default: 0)
	 */
	MemWriteReqPacket(std::function<void(MemWriteRespPacket*)> _callback, const instr& _i, instr_type _op,
	                  uint32_t _addr, uint32_t _data = 0)
	    : acalsim::SimPacket(), callback(_callback), i(_i), op(_op), addr(_addr), data(_data) {}

	/** @brief Virtual destructor */
	virtual ~MemWriteReqPacket() {}

	/**
	 * @brief Renews the packet with new parameters
	 * @param _callback New callback function
	 * @param _i New instruction
	 * @param _op New operation type
	 * @param _addr New memory address
	 * @param _data New data to write
	 */
	void renew(std::function<void(MemWriteRespPacket*)> _callback, const instr& _i, instr_type _op, uint32_t _addr,
	           uint32_t _data = 0);

	/** @brief Visit function for module interaction */
	void visit(acalsim::Tick _when, acalsim::SimModule& _module) override;
	/** @brief Visit function for simulator interaction */
	void visit(acalsim::Tick _when, acalsim::SimBase& _simulator) override;

	/** @return The instruction associated with this request */
	const instr& getInstr() { return this->i; }
	/** @return The operation type */
	const instr_type& getOP() { return this->op; }
	/** @return The memory address */
	const uint32_t& getAddr() { return this->addr; }
	/** @return The data to write */
	const uint32_t& getData() { return this->data; }
	/** @return The callback function */
	auto getCallback() { return this->callback; }

private:
	instr                                    i;         ///< Associated instruction
	instr_type                               op;        ///< Operation type
	uint32_t                                 addr;      ///< Memory address
	uint32_t                                 data;      ///< Data to write
	std::function<void(MemWriteRespPacket*)> callback;  ///< Write completion callback
};

/**
 * @class MemReadRespPacket
 * @brief Memory read response packet class
 * @details Contains the data read from memory and associated instruction information
 */
class MemReadRespPacket : public acalsim::SimPacket {
public:
	/** @brief Default constructor */
	MemReadRespPacket() {}

	/**
	 * @brief Parameterized constructor for memory read response
	 * @param _i Original instruction
	 * @param _op Operation type
	 * @param _data Data read from memory
	 * @param _a1 Associated operand
	 */
	MemReadRespPacket(const instr& _i, instr_type _op, uint32_t _data, operand _a1)
	    : acalsim::SimPacket(), i(_i), op(_op), data(_data), a1(_a1) {}

	/** @brief Virtual destructor */
	virtual ~MemReadRespPacket() {}

	/** @return The original instruction */
	const instr& getInstr() { return this->i; }
	/** @return The operation type */
	const instr_type& getOP() { return this->op; }
	/** @return The data read from memory */
	const uint32_t& getData() { return this->data; }
	/** @return The associated operand */
	const operand& getA1() { return this->a1; }

	/**
	 * @brief Renews the packet with new parameters
	 * @param _i New instruction
	 * @param _op New operation type
	 * @param _data New data
	 * @param _a1 New operand
	 */
	void renew(const instr& _i, instr_type _op, uint32_t _data, operand _a1);

	/** @brief Visit function for module interaction */
	void visit(acalsim::Tick _when, acalsim::SimModule& _module) override;
	/** @brief Visit function for simulator interaction */
	void visit(acalsim::Tick _when, acalsim::SimBase& _simulator) override;

private:
	instr      i;     ///< Associated instruction
	instr_type op;    ///< Operation type
	uint32_t   data;  ///< Data read from memory
	operand    a1;    ///< Associated operand
};

/**
 * @class MemWriteRespPacket
 * @brief Memory write response packet class
 * @details Confirms completion of memory write operations
 */
class MemWriteRespPacket : public acalsim::SimPacket {
public:
	/** @brief Default constructor */
	MemWriteRespPacket() {}

	/**
	 * @brief Parameterized constructor for write response
	 * @param _i Original instruction that requested the write
	 */
	MemWriteRespPacket(const instr& _i) : acalsim::SimPacket(), i(_i) {}

	/** @brief Virtual destructor */
	virtual ~MemWriteRespPacket() {}

	/** @return The original instruction */
	const instr& getInstr() { return this->i; }

	/**
	 * @brief Renews the packet with a new instruction
	 * @param _i New instruction
	 */
	void renew(const instr& _i);

	/** @brief Visit function for module interaction */
	void visit(acalsim::Tick _when, acalsim::SimModule& _module) override;
	/** @brief Visit function for simulator interaction */
	void visit(acalsim::Tick _when, acalsim::SimBase& _simulator) override;

private:
	instr i;  ///< Associated instruction
};
