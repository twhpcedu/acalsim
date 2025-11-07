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

#include "common/BitVector.hh"

#include <string>

#include "utils/Logging.hh"

namespace acalsim {

const uint64_t BitVector::ALL_ONE = UINT64_MAX;

BitVector::BitVector(size_t _size, bool _initial) : size(_size), bitvec((_size + 63) >> 6, (uint64_t)0) {
	if (_initial) {
		for (size_t idx = 0; idx < bitvec.size(); ++idx) {
			size_t offset = _size - (idx << 6);  // the index in the current uint64_t
			if (offset >= 64)
				bitvec[idx] = BitVector::ALL_ONE;
			else
				// Only set used bits to 1
				bitvec[idx] = ((uint64_t)1 << offset) - 1;
		}
	}
}

BitVector::BitVector(const BitVector& _other) : size(_other.size), bitvec(_other.bitvec) {}

void BitVector::setBit(size_t _idx, bool _value) {
	ASSERT_MSG(size > _idx, "The argument _idx=" + std::to_string(_idx) + " is out of range.");

	size_t vec_idx = _idx >> 6;
	size_t int_vec = _idx % 64;

	if (_value)
		bitvec[vec_idx] |= ((uint64_t)1 << int_vec);
	else
		bitvec[vec_idx] &= ~((uint64_t)1 << int_vec);
}

bool BitVector::getBit(size_t _idx) const {
	ASSERT_MSG(size > _idx, "The argument _idx=" + std::to_string(_idx) + " is out of range.");

	size_t vec_idx = _idx >> 6;
	size_t int_vec = _idx % 64;

	return ((bitvec[vec_idx] >> int_vec) & 1) == 1;
}

void BitVector::reset() {
	for (size_t idx = 0; idx < bitvec.size(); ++idx) { bitvec[idx] = 0; }
}

bool BitVector::allEqual(bool _value) const {
	if (_value) {
		for (size_t idx = 0; idx < bitvec.size(); ++idx) {
			size_t offset = size - (idx << 6);  // the index in the current uint64_t
			if (offset >= 64) {
				if (bitvec[idx] != BitVector::ALL_ONE) return false;
			} else {
				// The unused higher-bits should be set to 1 before comparison
				if ((bitvec[idx] | ~(((uint64_t)1 << offset) - 1)) != BitVector::ALL_ONE) return false;
			}
		}
	} else {
		for (size_t idx = 0; idx < bitvec.size(); ++idx) {
			size_t offset = size - (idx << 6);  // the index in the current uint64_t
			if (offset >= 64) {
				if (bitvec[idx] != 0) return false;
			} else {
				// The unused higher-bits should be set to 0 before comparison
				if ((bitvec[idx] & (((uint64_t)1 << offset) - 1)) != 0) return false;
			}
		}
	}

	return true;
}

}  // end of namespace acalsim
