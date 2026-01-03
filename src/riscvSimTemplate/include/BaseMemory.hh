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

#include <cstddef>
#include <cstdint>

#include "ACALSim.hh"

class BaseMemory {
public:
	/**
	 * @brief Construct a new `BaseMemory` object.
	 *
	 * @param _size The size of this memory.
	 */
	BaseMemory(size_t _size);
	~BaseMemory();

	/**
	 * @brief Get the size of this memory in bytes.
	 *
	 * @return size_t
	 */
	size_t getSize() const;

	/**
	 * @brief Read the data from this memory.
	 *
	 * @param _addr The starting address of the read operation.
	 * @param _size The size of the data in bytes.
	 * @param _deep_copy Create a duplicate of the target data. Default is `false`.
	 * @return void* The pointer that points to the asked data.
	 *
	 * @warning The caller should make sure all the operations against the returned data are within the asked size by
	 * itself.
	 */
	void* readData(uint32_t _addr, size_t _size, bool _deep_copy = false) const;

	/**
	 * @brief Write the given data into memory.
	 *
	 * @param _data The pointer of the data to be written.
	 * @param _addr The starting address of the write operation.
	 * @param _size The size of the `_data` in bytes.
	 *
	 * @warning If the `_size` exceeds the actual size of `_data`, some unknown data would be saved into memory.
	 */
	void writeData(void* _data, uint32_t _addr, size_t _size);

	void* getMemPtr() { return this->mem; }

private:
	void*        mem = nullptr;
	const size_t size;
};
