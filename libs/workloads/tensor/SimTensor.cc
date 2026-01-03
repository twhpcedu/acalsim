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

#include "workloads/tensor/SimTensor.hh"

#include <string>

#include "container/RecycleContainer/RecycleContainer.hh"
#include "sim/SimTop.hh"

namespace acalsim {

std::atomic<uint64_t> SimTensor::uniqueTID = 0;

// Re-initializes the tensor with new properties.
void SimTensor::renew(const std::string _name, const size_t _addr, const DTYPE _dtype,
                      const std::vector<size_t>& _shapes, const TENSORTYPE _type, const std::vector<size_t>& _strides) {
	this->graph::Edge<SimTensor>::renew(_name);
	this->id_      = this->uniqueTID++;
	this->type_    = _type;
	this->dtype_   = _dtype;
	this->addr_    = _addr;
	this->shapes_  = _shapes;
	this->strides_ = _strides.empty() ? initStrides(_shapes) : _strides;
}

// Copies the properties from another tensor to this tensor.
void SimTensor::copyByPointer(const SimTensor* others) {
	this->graph::Edge<SimTensor>::renew(others->name);
	this->id_      = others->id_;
	this->type_    = others->type_;
	this->dtype_   = others->dtype_;
	this->addr_    = others->addr_;
	this->shapes_  = others->shapes_;
	this->strides_ = others->strides_;
}

// Copies the properties from another tensor to this tensor.
void SimTensor::copyByObject(const SimTensor& others) {
	this->graph::Edge<SimTensor>::renew(others.name);
	this->id_      = others.id_;
	this->type_    = others.type_;
	this->dtype_   = others.dtype_;
	this->addr_    = others.addr_;
	this->shapes_  = others.shapes_;
	this->strides_ = others.strides_;
}

size_t SimTensor::getAddr(const std::vector<size_t>& _indices) const {
	LABELED_ASSERT_MSG(_indices.size() == this->shapes_.size(), this->name,
	                   "Dimension mismatch: Expected " + std::to_string(this->shapes_.size()) + " indices, but got " +
	                       std::to_string(_indices.size()) + ".");

	size_t addr = this->addr_;
	for (size_t i = 0; i < _indices.size(); ++i) {
		LABELED_ASSERT_MSG(_indices[i] <= this->shapes_[i], this->name,
		                   "Index out of bounds: Dimension " + std::to_string(i) + " has size " +
		                       std::to_string(this->shapes_[i]) + ", but received index " +
		                       std::to_string(_indices[i]) + ".");

		addr += _indices[i] * this->strides_[i];
	}
	return addr;
}

size_t SimTensor::size() const {
	size_t total_elements = this->shapes_.size() > 0 ? 1 : 0;
	for (auto dim_size : this->shapes_) { total_elements *= dim_size; }
	return total_elements * getTorchTypeByte(this->dtype_);
}

bool SimTensor::contiguous(const std::pair<size_t, size_t>& dims) const {
	size_t dim1 = dims.first;
	size_t dim2 = dims.second;

	if (dim1 == dim2) { return true; }
	if (dim1 > dim2) { std::swap(dim1, dim2); }

	LABELED_ASSERT_MSG(dim1 >= 0 && dim2 < this->shapes_.size(), this->name,
	                   "Invalid dimension range for contiguity check.");

	for (size_t i = dim1 + 1; i <= dim2; ++i) {
		if (this->strides_[i - 1] != this->strides_[i] * this->shapes_[i]) { return false; }
	}
	return true;
}

void SimTensor::flatten(const std::pair<size_t, size_t>& _dims) {
	size_t dim1 = _dims.first;
	size_t dim2 = _dims.second;

	if (dim1 == dim2) { return; }
	if (dim1 > dim2) { std::swap(dim1, dim2); }

	LABELED_ASSERT_MSG(dim1 >= 0 && dim2 < this->shapes_.size(), this->name,
	                   "Dimensions out of bounds for flatten operation.");
	LABELED_ASSERT_MSG(this->contiguous(_dims), this->name, "The dimensions to be flattened are not contiguous.");

	// Initialize new shape and stride vectors
	std::vector<size_t> new_shape;
	std::vector<size_t> new_stride;

	new_shape.insert(new_shape.end(), this->shapes_.begin(), this->shapes_.begin() + dim1);
	new_stride.insert(new_stride.end(), this->strides_.begin(), this->strides_.begin() + dim1);

	// Calculate the flattened dimension size
	int64_t flattened_dim = 1;
	for (size_t i = dim1; i <= dim2; ++i) { flattened_dim *= this->shapes_[i]; }
	new_shape.push_back(flattened_dim);

	// Add all dimensions after dim2
	new_shape.insert(new_shape.end(), this->shapes_.begin() + dim2 + 1, this->shapes_.end());
	new_stride.insert(new_stride.end(), this->strides_.begin() + dim2, this->strides_.end());

	// Update the tensor's shape and strides
	this->shapes_  = new_shape;
	this->strides_ = new_stride;
}

void SimTensor::unflatten(size_t _dim, size_t _num_splits) {
	LABELED_ASSERT_MSG(_dim < this->shapes_.size() && _dim >= 0, this->name,
	                   "Unflatten error: Dimension to unflatten is out of bounds.");

	LABELED_ASSERT_MSG(this->shapes_[_dim] % _num_splits == 0, this->name,
	                   "Unflatten error: Dimension size is not divisible by the number of splits.");

	size_t split_size = this->shapes_[_dim] / _num_splits;

	// Construct the new shape by replacing `dim` with the split dimensions
	this->shapes_[_dim] = _num_splits;
	this->shapes_.insert(this->shapes_.begin() + _dim + 1, split_size);

	// Update the strides after unflattening
	this->strides_.insert(this->strides_.begin() + _dim,
	                      this->strides_[_dim] * split_size);  // The stride of the second dimension (split size)
}

SimTensor SimTensor::slice(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices) {
	// Validate inputs: ensure start_indices and end_indices are the same size as the tensor's number of dimensions.
	if (_start_indices.size() != this->shapes_.size() || _end_indices.size() != this->shapes_.size()) {
		LABELED_ERROR(name) << "Start and end indices must match the number of dimensions of the tensor."
		                    << " start_indices size: " << _start_indices.size()
		                    << " end_indices size: " << _end_indices.size()
		                    << " tensor dimensions: " << this->shapes_.size();
		throw std::invalid_argument("Start and end indices size mismatch with tensor dimensions.");
	}

	// Create a new tensor object for the sliced portion
	SimTensor sliced_tensor = SimTensor(this);  // Start with the current tensor's properties

	// Calculate the new shape and strides for the sliced tensor
	std::vector<size_t> new_shape;
	std::vector<size_t> new_strides = this->strides_;  // Copy the original strides

	// Compute the shape and adjust the strides for each dimension based on slicing
	for (size_t dim = 0; dim < this->shapes_.size(); ++dim) {
		size_t start_idx = _start_indices[dim];
		size_t end_idx   = _end_indices[dim];
		size_t bound     = this->shapes_[dim];  // The maximum valid index for this dimension

		// Check if start_idx is greater than the bound (throw error if true)
		if (start_idx > bound) {
			LABELED_ERROR(this->name) << "Start index is out of bounds. Dimension: " << dim
			                          << ", start index: " << start_idx << ", bound: " << bound
			                          << ", tensor shape: " << this->str();
			throw std::invalid_argument("Start index out of bounds for dimension " + std::to_string(dim));
		}

		// If end_idx is greater than the bound, update it to the bound
		if (end_idx > bound) { end_idx = bound; }

		// Ensure slice size is positive
		if (start_idx >= end_idx) {
			LABELED_ERROR(this->name) << "Invalid slice range. Dimension: " << dim << ", start index: " << start_idx
			                          << ", end index: " << end_idx << ", tensor shape: " << this->str();
			throw std::invalid_argument("Invalid slice range for dimension " + std::to_string(dim));
		}

		// Calculate the size of the slice for this dimension
		size_t slice_size = end_idx - start_idx;
		new_shape.push_back(slice_size);

		// Adjust the stride if necessary
		new_strides[dim] = this->strides_[dim];
	}

	// Set the new shape and strides for the sliced tensor
	sliced_tensor.shapes_  = new_shape;
	sliced_tensor.strides_ = new_strides;
	sliced_tensor.addr_    = this->getAddr(_start_indices);

	return sliced_tensor;
}

SimTensor* SimTensor::slice_ptr(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices,
                                bool _recyclable) {
	SimTensor tsr = this->slice(_start_indices, _end_indices);

	if (_recyclable) {
		if (top == nullptr) { throw std::invalid_argument("SimTop hasn't been initizalized yet"); }
		return top->getRecycleContainer()->acquire<SimTensor>(&SimTensor::copyByObject, tsr);
	} else {
		return new SimTensor(tsr);
	}
}

std::vector<SimTensor> SimTensor::chunk(size_t _n_chunks, size_t _dim) {
	std::vector<SimTensor> chunks;

	size_t dim_size = this->shapes_[_dim];

	if (_n_chunks == 0 || _n_chunks > dim_size) {
		LABELED_ERROR(this->name) << "Invalid number of chunks: " << _n_chunks << ". Must be in range (0, " << dim_size
		                          << "]. Tensor: " << this->str();
		throw std::invalid_argument("Invalid number of chunks.");
	}

	size_t chunk_size = dim_size / _n_chunks;
	size_t remainder  = dim_size % _n_chunks;

	size_t element_size   = this->getTorchTypeByte(this->dtype_);
	auto   chunk_base_idx = std::vector<size_t>(this->shapes_.size(), 0);

	// chunk the tensor along the specified dimension
	for (size_t i = 0; i < _n_chunks; ++i) {
		size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);  // Adjust for remainder

		// Directly modify chunk_shape to avoid copy
		auto chunk_shape  = this->shapes_;
		chunk_shape[_dim] = current_chunk_size;  // Update only the chunked dimension

		// Calculate the chunk tensor address offset
		size_t chunk_offset = this->getAddr(chunk_base_idx);

		// Create the tensor with the appropriate chunk
		chunks.emplace_back(this->name + "." + std::to_string(i), chunk_offset, this->dtype_, chunk_shape, this->type_,
		                    this->strides_);

		// Update the base index
		chunk_base_idx[_dim] += current_chunk_size;
	}

	return chunks;
}

std::vector<SimTensor*> SimTensor::chunk_ptr(size_t _n_chunks, size_t _dim, bool _recyclable) {
	std::vector<SimTensor*> chunk_ptrs;
	chunk_ptrs.reserve(_n_chunks);
	if (_recyclable) {
		if (top == nullptr) { throw std::invalid_argument("SimTop hasn't been initizalized yet"); }
		for (SimTensor& tsr : this->chunk(_n_chunks, _dim)) {
			chunk_ptrs.push_back(top->getRecycleContainer()->acquire<SimTensor>(&SimTensor::copyByObject, tsr));
		}
	} else {
		for (SimTensor& tsr : this->chunk(_n_chunks, _dim)) { chunk_ptrs.push_back(new SimTensor(tsr)); }
	}
	return chunk_ptrs;
}

void SimTensor::updateShape(size_t dim, size_t size, bool updateStride) {
	this->shapes_[dim] = size;
	this->strides_     = updateStride ? initStrides(this->shapes_) : this->strides_;
}

void SimTensor::assignAddr(size_t _addr) {
	this->addr_    = _addr;
	this->strides_ = this->initStrides(this->shapes_);
}

std::string SimTensor::str() const {
	std::stringstream ss;

	ss << "Tensor: " << std::setw(4) << std::setfill(' ') << this->id_;
	ss << " | Dims: [" << (this->shapes_.empty() ? "None" : join(this->shapes_, ", ")) << "]";
	ss << " | Strides: [" << (this->strides_.empty() ? "None" : join(this->strides_, ", ")) << "]";
	ss << " | Addr: 0x" << std::hex << this->addr_;
	ss << " | Name: " << this->name;
	return ss.str();
}

// Helper method to initialize tensor strides based on shape.
std::vector<size_t> SimTensor::initStrides(const std::vector<size_t>& _shape) {
	std::vector<size_t> stride_vec(_shape.size(), 0);

	size_t current_stride = 1;

	// Traverse dimensions in reverse order (starting from the last dimension)
	for (size_t i = _shape.size(); i > 0; --i) {
		stride_vec[i - 1] = current_stride;
		current_stride *= _shape[i - 1];
	}

	return stride_vec;
}

}  // namespace acalsim
