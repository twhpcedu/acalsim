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

#include <string>

#include "workloads/graph/DAG.hh"

namespace acalsim {

/**
 * @class SimTensor
 * @brief Represents a tensor & derives from Edge template.
 */
class SimTensor : public graph::Edge<SimTensor> {
public:
	/**
	 * @enum TENSORTYPE
	 * @brief Tensor types for different uses in computation.
	 *
	 * Defines the intended usage of the tensor within a computational graph, such as:
	 *   - WEIGHT: Represents a weight tensor used in model parameters.
	 *   - INPUT: Represents an input tensor to a computation.
	 *   - OUTPUT: Represents an output tensor from a computation.
	 *   - ACTIVATION: Represents an activation tensor, the result of an activation function.
	 *   - KVCACHE: Represents a Key-Value cache tensor.
	 */
	enum class TENSORTYPE { Unknown = -1, WEIGHT, INPUT, OUTPUT, ACTIVATION, KVCACHE };

	/**
	 * @enum DTYPE
	 * @brief Data types supported by the tensor.
	 *
	 * Specifies the data type of the tensor elements, aligning with common data types
	 * used in machine learning frameworks like PyTorch. Includes various floating-point,
	 * integer, and boolean types.
	 */
	enum class DTYPE {
		Unknown = -1,
		Float32,   // torch.float32 (torch.float)
		Float,     // torch.float (alias for Float32)
		Float16,   // torch.float16 (torch.half)
		Half,      // torch.half (alias for Float16)
		BFloat16,  // torch.bfloat16
		Float64,   // torch.float64 (torch.double)
		Double,    // torch.double (alias for Float64)
		Int8,      // torch.int8
		UInt8,     // torch.uint8
		Int16,     // torch.int16 (torch.short)
		Short,     // torch.short (alias for Int16)
		Int32,     // torch.int32 (torch.int)
		Int,       // torch.int (alias for Int32)
		Int64,     // torch.int64 (torch.long)
		Long,      // torch.long (alias for Int64)
		Bool       // torch.bool
	};

	/**
	 * @brief Default constructor with unique ID generation.
	 *
	 * Initializes a `SimTensor` object with default values. The tensor will have an automatically
	 * generated name based on a unique ID, an unknown data type, a memory address of 0,
	 * no shape (empty vector), and is initialized as a WEIGHT tensor.
	 *
	 * @note The order of `_shapes` and `_strides` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	SimTensor()
	    : graph::Edge<SimTensor>("Tensor-" + std::to_string(uniqueTID)),
	      id_(SimTensor::uniqueTID++),
	      addr_(0),
	      shapes_(std::vector<size_t>()),
	      dtype_(DTYPE::Unknown),
	      strides_(std::vector<size_t>()),
	      type_(TENSORTYPE::WEIGHT) {}

	/**
	 * @brief Parameterized constructor
	 *
	 * This constructor allows setting specific values for the tensor's dtype, address,
	 * shapes, strides, and type. If strides are not provided, they are initialized using
	 * the default stride calculation based on the given shapes.
	 *
	 * @param _dtype The data type of the tensor
	 * @param _addr The address of the tensor
	 * @param _shapes The shape of the tensor as a vector of size_t.
	 * @param _strides The strides of the tensor as a vector of size_t (optional, default is calculated based on
	 * shapes).
	 * @param _type The type of the tensor (optional, default is TENSORTYPE::WEIGHT).
	 *
	 * @note The order of `_shapes` and `_strides` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	SimTensor(DTYPE _dtype, uint64_t _addr, const std::vector<size_t>& _shapes, TENSORTYPE _type = TENSORTYPE::WEIGHT,
	          const std::vector<size_t>& _strides = std::vector<size_t>())
	    : graph::Edge<SimTensor>("Tensor-" + std::to_string(uniqueTID)),
	      id_(SimTensor::uniqueTID++),
	      addr_(_addr),
	      shapes_(_shapes),
	      strides_(_strides.empty() ? initStrides(_shapes) : _strides),
	      type_(_type),
	      dtype_(_dtype) {}

	/**
	 * @brief Named constructor for SimTensor with optional shapes and strides.
	 *
	 * This constructor allows the creation of a SimTensor object with a user-defined
	 * name, dtype, address, shapes, and strides. If shapes or strides are not provided,
	 * they default to empty or calculated values respectively.
	 *
	 * @param _name The name of the tensor.
	 * @param _dtype The data type of the tensor (default is DTYPE::Unknown).
	 * @param _addr The address of the tensor (default is 0).
	 * @param _shapes The shape of the tensor as a vector of size_t.
	 * @param _strides The strides of the tensor as a vector of size_t (optional, defaults to calculated strides).
	 * @param _type The type of the tensor (optional, default is TENSORTYPE::WEIGHT).
	 *
	 * @note The order of `_shapes` and `_strides` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	SimTensor(const std::string _name, const uint64_t _addr = 0, const DTYPE _dtype = DTYPE::Unknown,
	          const std::vector<size_t>& _shapes = std::vector<size_t>(), const TENSORTYPE _type = TENSORTYPE::WEIGHT,
	          const std::vector<size_t>& _strides = std::vector<size_t>())
	    : graph::Edge<SimTensor>(_name),
	      id_(SimTensor::uniqueTID++),
	      addr_(_addr),
	      shapes_(_shapes),
	      strides_(_strides.empty() ? initStrides(_shapes) : _strides),
	      type_(_type),
	      dtype_(_dtype) {}

	virtual ~SimTensor() {}

	/**
	 * @brief Copy Constructor for the Tensor class.
	 *
	 * Copies the properties of another tensor.
	 *
	 * @param _others The tensor to copy.
	 */
	SimTensor(const SimTensor& _others)
	    : graph::Edge<SimTensor>(_others.name),
	      type_(_others.type_),
	      id_(_others.id_),
	      addr_(_others.addr_),
	      shapes_(_others.shapes_),
	      dtype_(_others.dtype_),
	      strides_(_others.strides_) {}

	/**
	 * @brief Copy Constructor for the Tensor class (for pointer).
	 *
	 * Copies the properties of another tensor (pointer to a tensor).
	 *
	 * @param _others Pointer to the tensor to copy.
	 */
	SimTensor(const SimTensor* _others)
	    : graph::Edge<SimTensor>(_others->name),
	      type_(_others->type_),
	      id_(_others->id_),
	      addr_(_others->addr_),
	      shapes_(_others->shapes_),
	      dtype_(_others->dtype_),
	      strides_(_others->strides_) {}

	/**
	 * @brief Overloads the output stream operator to allow printing SimTensor objects directly.
	 *
	 * @param os The output stream.
	 * @param tensor The SimTensor object to print.
	 * @return A reference to the output stream.
	 */
	friend std::ostream& operator<<(std::ostream& os, const SimTensor& tensor) { return os << tensor.str(); }

	/**
	 * @brief Returns a string representation of the SimTensor object.
	 *
	 * @return A string containing the name, ID, type, data type, shape, strides, and address of the tensor.
	 */
	std::string str() const;

	/**
	 * @brief Get the number of bytes required to store a given data type.
	 *
	 * This function determines the size in bytes of various data types used in SimTensor.
	 *
	 * @param dtype The data type for which the byte size is needed.
	 * @return constexpr size_t The size in bytes of the given data type. Returns 0 for unknown types.
	 */
	static constexpr size_t getTorchTypeByte(const DTYPE dtype);

	/**
	 * @brief Re-initializes the tensor with new properties.
	 *
	 * This method resets the tensor with new name, address, shape, data type, and optional strides.
	 *
	 * @param _name The new name of the tensor.
	 * @param _addr The new memory address of the tensor.
	 * @param _shapes The new shape of the tensor.
	 * @param _dtype The new data type of the tensor.
	 * @param _strides The new strides of the tensor, optional. If empty, strides are auto-calculated.
	 * @param _type The new type of the tensor.
	 * @ingroup metadata_operation
	 *
	 * @note The order of `_shapes` and `_strides` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	void renew(const std::string _name, const size_t _addr, const DTYPE _dtype, const std::vector<size_t>& _shapes,
	           const TENSORTYPE           _type    = TENSORTYPE::WEIGHT,
	           const std::vector<size_t>& _strides = std::vector<size_t>());

	/**
	 * @brief Copies the content of another SimTensor object (pointed to by a pointer) to this SimTensor object.
	 *
	 * @param _others A pointer to the SimTensor object to copy.
	 */
	void copyByPointer(const SimTensor* _others);

	/**
	 * @brief Copies the content of another SimTensor object to this SimTensor object.
	 *
	 * @param _others The SimTensor object to copy.
	 */
	void copyByObject(const SimTensor& _others);

	/**
	 * @brief Calculates the total size (number of elements) of the tensor.
	 *
	 * @return The total number of elements in the tensor.
	 */
	size_t size() const;

	/**
	 * @brief Checks if the tensor is contiguous over the specified dimensions.
	 *
	 * A tensor is contiguous if its elements are stored in a contiguous block of memory
	 * along the specified dimensions.
	 *
	 * @param _dims A pair representing the range of dimensions to check for contiguity.
	 *              The first element is the starting dimension, and the second is the ending dimension.
	 * @return True if the tensor is contiguous over the given dimensions, otherwise false.
	 */
	bool contiguous(const std::pair<size_t, size_t>& _dims) const;

	/**
	 * @brief Checks if the entire tensor is contiguous.
	 *
	 * @return True if the tensor is contiguous in all dimensions, otherwise false.
	 */
	bool contiguous() const { return this->contiguous({0, this->getNumDims() - 1}); }

	/**
	 * @brief Assigns a new memory address to the tensor.
	 *
	 * @param _addr The new memory address.
	 */
	void assignAddr(size_t _addr);

	/**
	 * @brief Retrieves the memory address for specific indices in the tensor.
	 *
	 * This function calculates the memory address of a tensor element given its indices,
	 * taking into account the tensor's shape and strides.
	 *
	 * @param _indices The indices for which to retrieve the address.
	 * @return The memory address of the specified indices.
	 *
	 * @note The order of `_indices` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	size_t getAddr(const std::vector<size_t>& _indices) const;

	/**
	 * @brief Retrieves the memory address of the tensor.
	 *
	 * @return The memory address of the tensor.
	 */
	size_t getAddr() const { return this->addr_; }

	/**
	 * @brief Flattens specified dimensions into a single dimension.
	 *
	 * This operation combines a range of dimensions in the tensor's shape into a single dimension,
	 * effectively reducing the number of dimensions in the tensor.
	 *
	 * @param _dims A pair of dimensions to flatten into a single dimension.
	 *              The first element is the starting dimension, and the second is the ending dimension.
	 */
	void flatten(const std::pair<size_t, size_t>& _dims);

	/**
	 * @brief Unflattens a single dimension into multiple dimensions.
	 *
	 * This operation splits a single dimension in the tensor's shape into multiple dimensions,
	 * effectively increasing the number of dimensions in the tensor.
	 *
	 * @param _dim The dimension to unflatten.
	 * @param _num_splits The number of splits to create.
	 */
	void unflatten(size_t _dim, size_t _num_splits);

	/**
	 * @brief Slices the tensor along specified start and end indices.
	 *
	 * This operation extracts a sub-tensor from the current tensor, based on the provided
	 * start and end indices for each dimension.
	 *
	 * @param _start_indices The start indices for slicing.
	 * @param _end_indices The end indices for slicing.
	 * @return The sliced tensor.
	 *
	 * @note The order of `_start_indices` and `_end_indices` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	SimTensor slice(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices);

	/**
	 * @brief Slices the tensor along specified start and end indices and returns a pointer to the new `SimTensor`.
	 *
	 * This operation extracts a sub-tensor from the current tensor, based on the provided
	 * start and end indices for each dimension.
	 *
	 * @param _start_indices The start indices for slicing.
	 * @param _end_indices The end indices for slicing.
	 * @param recyclable A boolean indicating whether the tensor is recyclable.
	 * @return The sliced tensor.
	 *
	 * @note The order of `_start_indices` and `_end_indices` follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	SimTensor* slice_ptr(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices,
	                     bool recyclable = false);

	/**
	 * @brief Divides the tensor into multiple chunks along a specific dimension.
	 *
	 * This operation splits the tensor into a specified number of smaller tensors (chunks)
	 * along a given dimension.
	 *
	 * @param _n_chunks The number of chunks to divide the tensor into.
	 * @param _dim The dimension along which to chunk the tensor.
	 * @return A vector of chunked tensors.
	 */
	std::vector<SimTensor> chunk(size_t _n_chunks, size_t _dim);

	/**
	 * @brief Divides the tensor into multiple chunks along a specific dimension and returns vector of pointers to new
	 * `SimTensor`s.
	 *
	 * This operation splits the tensor into a specified number of smaller tensors (chunks)
	 * along a given dimension.
	 *
	 * @param _n_chunks The number of chunks to divide the tensor into.
	 * @param _dim The dimension along which to chunk the tensor.
	 * @param recyclable A boolean indicating whether the tensor is recyclable.
	 * @return A vector of chunked tensors.
	 */
	std::vector<SimTensor*> chunk_ptr(size_t _n_chunks, size_t _dim, bool recyclable = false);

	/**
	 * @brief Updates the shape of the tensor.
	 *
	 * This function allows modifying the size of a specific dimension in the tensor's shape.
	 * Optionally, it can also update the strides of the tensor to reflect the new shape.
	 *
	 * @param _dim The dimension to update.
	 * @param _size The new size for the specified dimension.
	 * @param _update_stride Whether to update the strides based on the new shape.
	 */
	void updateShape(size_t _dim, size_t _size, bool _update_stride);

	/**
	 * @brief Retrieves the number of dimensions in the tensor.
	 *
	 * @return The number of dimensions.
	 */
	size_t getNumDims() const { return this->shapes_.size(); }

	/**
	 * @brief Retrieves the shape of the tensor.
	 *
	 * @return A vector representing the shape of the tensor.
	 *
	 * @note The order of retrun value follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	const std::vector<size_t>& getShapes() const { return this->shapes_; }

	/**
	 * @brief Retrieves the strides of the tensor.
	 *
	 * @return A vector representing the strides of the tensor.
	 *
	 * @note The order of retrun value follows the convention from outermost dimension
	 * to innermost dimension.
	 */
	const std::vector<size_t>& getStrides() const { return this->strides_; }

	/**
	 * @brief Retrieves the size of a specific dimension.
	 *
	 * @param _idx The dimension index. The index of the outermost dimension is 0.
	 * @return The size of the specified dimension.
	 */
	size_t getShape(int _idx) const { return this->shapes_[_idx]; }

	/**
	 * @brief Retrieves the stride of a specific dimension.
	 *
	 * @param _idx The dimension index. The index of the outermost dimension is 0.
	 * @return The stride of the specified dimension.
	 */
	size_t getStride(int _idx) const { return this->strides_[_idx]; }

	/**
	 * @brief Retrieves the data type of the tensor.
	 *
	 * @return The tensor's data type.
	 */
	DTYPE getDType() const { return this->dtype_; }

	/**
	 * @brief Retrieves the unique ID of the tensor.
	 *
	 * @return The unique ID of the tensor.
	 */
	uint64_t getID() const { return this->id_; }

	/**
	 * @brief Retrieves the type of the tensor.
	 *
	 * @return The tensor's type.
	 */
	TENSORTYPE getType() { return this->type_; }

protected:
	/**
	 * @brief Helper method to initialize tensor strides based on shape.
	 *
	 * This function calculates the strides for each dimension of the tensor based on its shape.
	 * Strides indicate the number of bytes to jump in memory to move to the next element along each dimension.
	 *
	 * @param _shape The shape of the tensor.
	 * @return The initialized strides of the tensor.
	 */
	static std::vector<size_t> initStrides(const std::vector<size_t>& _shape);

protected:
	uint64_t   id_;     ///< A unique tensor ID.
	TENSORTYPE type_;   ///< The type of the tensor (e.g., WEIGHT, INPUT, OUTPUT).
	DTYPE      dtype_;  ///< The data type of the tensor elements.

	// tensor description
	std::vector<size_t> shapes_;   ///< Tensor shape dimensions.
	std::vector<size_t> strides_;  ///< Tensor strides for each dimension.
	size_t              addr_;     ///< Memory address of the tensor.

private:
	static std::atomic<uint64_t> uniqueTID;  ///< Static member to generate unique tensor IDs.
};

/**
 * @brief Determines the size in bytes of various data types used in SimTensor.
 *
 * This function provides a mapping from the `DTYPE` enum to the corresponding size in bytes.
 * It is used to calculate memory requirements for tensors based on their data type.
 *
 * @param dtype The data type for which the byte size is needed.
 * @return constexpr size_t The size in bytes of the given data type. Returns 0 for unknown types.
 */
constexpr size_t SimTensor::getTorchTypeByte(const DTYPE dtype) {
	switch (dtype) {
		case DTYPE::Float32:
		case DTYPE::Float: return 4;
		case DTYPE::Float16:
		case DTYPE::Half:
		case DTYPE::BFloat16: return 2;
		case DTYPE::Float64:
		case DTYPE::Double: return 8;
		case DTYPE::Int8:
		case DTYPE::UInt8:
		case DTYPE::Bool: return 1;
		case DTYPE::Int16:
		case DTYPE::Short: return 2;
		case DTYPE::Int32:
		case DTYPE::Int: return 4;
		case DTYPE::Int64:
		case DTYPE::Long: return 8;
		case DTYPE::Unknown:
		default: return 0;
	}
}

/**
 * @brief Joins elements of a container into a string with a delimiter.
 *
 * This template function takes a container of any type and concatenates its elements
 * into a single string, separated by a specified delimiter.
 *
 * @tparam T The type of elements in the container.
 * @param container The container whose elements are to be joined.
 * @param delimiter The string to insert between each element.
 * @return A string containing the joined elements.
 */
template <typename T>
std::string join(const std::vector<T>& container, const std::string& delimiter) {
	if (container.empty()) return "";

	std::ostringstream oss;
	oss << container[0];

	for (size_t i = 1; i < container.size(); ++i) { oss << delimiter << container[i]; }

	return oss.str();
}

}  // namespace acalsim
