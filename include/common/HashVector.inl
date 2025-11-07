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

#include "common/HashVector.hh"

namespace acalsim {

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
HashVector<Key, T, Hash, KeyEqual, Allocator>::HashVector(const size_t& _bucket_count, const Hash& _hash,
                                                          const KeyEqual& _equal, const Allocator& _alloc)
    : umap_(_bucket_count, _hash, _equal, _alloc) {}

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
bool HashVector<Key, T, Hash, KeyEqual, Allocator>::insert(const std::pair<Key, T>& _value) {
	using UMapIterator   = typename std::unordered_map<Key, VecElemRef, Hash, KeyEqual>::iterator;
	using VectorIterator = typename std::vector<T, Allocator>::iterator;

	// Try to insert the key with an arbitrary value into the unordered map
	std::pair<UMapIterator, bool> umap_result =
	    this->umap_.insert({_value.first, VecElemRef(this->vec_, this->vec_.size())});

	// Handle failed insertion
	if (!umap_result.second) { return false; }

	// Push the value to the vector if the unordered map insetion is successful
	this->vec_.push_back(_value.second);

	return true;
}

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
void HashVector<Key, T, Hash, KeyEqual, Allocator>::reserve(size_t _count) {
	this->umap_.reserve(_count);
	this->vec_.reserve(_count);
}

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
void HashVector<Key, T, Hash, KeyEqual, Allocator>::clear() noexcept {
	this->umap_.clear();
	this->vec_.clear();
}

}  // namespace acalsim
