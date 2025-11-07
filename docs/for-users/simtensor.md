<!--
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ACALSim User Guide - SimTensor


- Author: Yen-Po Chen \<yenpo@twhpcedu.org\>
- Date: 2025/03/13

([Back To Documentation Portal](/docs/README.md))

## Introduction
The `SimTensor` class is a core component within ACALSim, designed to represent and manipulate tensors within a simulated environment. It provides a flexible and feature-rich abstraction for modeling tensor operations, memory layout, and data types, enabling detailed analysis of tensor-based workloads. This guide provides an overview of the `SimTensor` class, its key functionalities, and how to effectively utilize it within your simulations.
## QuickStart

To use the `SimTensor` class, you need to include the header file:

- **`#include "workloads/tensor/SimTensor.hh"`**

Here's a basic example of creating and using a `SimTensor`:

```cpp
#include "workloads/tensor/SimTensor.hh"
#include <iostream>
#include <vector>
using namespace acalsim;
int main() {
    // Create a tensor with a shape of (1, 3, 224, 224), Float32 data type, and starting address 0x1000
    std::vector<size_t> shape = {1, 3, 224, 224};
    SimTensor ipt(SimTensor::DTYPE::Float32, 0x1000, shape, SimTensor::TENSORTYPE::INPUT);
    // Print the tensor's information
    std::cout << ipt << std::endl;

    // Get the number of dimensions
    size_t numDims = ipt.getNumDims();
    std::cout << "Number of dimensions: " << numDims << std::endl;
    return 0;
}
```

## Key APIs

This section details the most important APIs of the `SimTensor` class, grouped by functionality.

### Construction and Initialization
- Default constructor:
    - **`SimTensor()`**:
    - Creates an empty tensor with default properties.
- Parameterized constructor:
    - **`SimTensor(DTYPE _dtype, uint64_t _addr, const std::vector<size_t>& _shapes, TENSORTYPE _type = TENSORTYPE::WEIGHT), const std::vector<size_t>& _strides = std::vector<size_t>()`**:
    - Allows specifying the data type, address, shape, strides, and tensor type during creation.
	- The order of `_shapes` and `_strides` follows the convention from outermost dimension to innermost dimension.
- Named constructor:
    - **`SimTensor(const std::string _name, const uint64_t _addr = 0, const DTYPE _dtype = DTYPE::Unknown, const std::vector<size_t>& _shapes  = std::vector<size_t>(), const TENSORTYPE _type = TENSORTYPE::WEIGHT, const std::vector<size_t>& _strides = std::vector<size_t>())`**:
    - Similar to the parameterized constructor, but allows assigning a custom name to the tensor.
- Copy constructor:
    - **`SimTensor(const SimTensor& _others)`**:
    - Creates a new tensor as a copy of an existing one.
- Pointer copy constructor:
    - **`SimTensor(const SimTensor* _others)`**:
    - Creates a new tensor as a copy of an existing one (pointed to by a pointer).
- Re-initializes(RecycleContainer Relative API)
    - **`void renew(const std::string _name, const size_t _addr, const DTYPE _dtype, const std::vector<size_t>& _shapes, const TENSORTYPE _type = TENSORTYPE::WEIGHT, const std::vector<size_t>& _strides = std::vector<size_t>())`**:
    - Re-initializes an existing tensor with new properties.

> **Note**: The order of all the dimensions, shapes, and indices in `SimTensor` follows the convention from outermost dimension to innermost dimension.

### Data Access and Information
- **`DTYPE getDType() const`**:
    - Returns the data type of the tensor.
- **`size_t getNumDims() const`**:
    - Returns the number of dimensions in the tensor.
- **`const std::vector<size_t>& getShapes() const`**:
    - Returns a reference to the vector containing the shape of the tensor.
- **`size_t getShape(int _idx) const`**:
    - Returns the size of the dimension at the specified index.
- **`const std::vector<size_t>& getStrides() const`**:
    - Returns a reference to the vector containing the strides of the tensor.
- **`size_t getStride(int _idx) const`**:
    - Returns the stride of the dimension at the specified index.
- **`size_t getAddr() const`**:
    - Returns the base memory address of the tensor.
- **`size_t getAddr(const std::vector<size_t>& _indices) const`**:
    - Returns the memory address for the element at the specified indices.
- **`size_t size() const`**:
    - Returns the total number of elements in the tensor.
- **`TENSORTYPE getType()`**:
    - Returns the tensor type (e.g., WEIGHT, INPUT).
- **`uint64_t getID() const`**:
    - Returns the unique ID of the tensor.
- **`static constexpr size_t getTorchTypeByte(const DTYPE dtype)`**:
    - Returns the size in bytes of the given data type.
- **`std::string str() const`**:
    - Returns a string representation of the tensor.

### Tensor Manipulation

- **`void assignAddr(size_t _addr)`**:
    - Assigns a new base memory address to the tensor.
- **`void updateShape(size_t _dim, size_t _size, bool _update_stride)`**:
    - Updates the size of a specific dimension in the tensor's shape.  Optionally updates the strides.
- **`void flatten(const std::pair<size_t, size_t>& _dims)`**:
    - Flattens a range of dimensions into a single dimension.
- **`void unflatten(size_t _dim, size_t _num_splits)`**:
    - Unflattens a single dimension into multiple dimensions.
- **`SimTensor slice(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices)`**:
    - Creates a new tensor that is a slice of the original tensor.
- **`SimTensor* slice_ptr(const std::vector<size_t>& _start_indices, const std::vector<size_t>& _end_indices, bool recyclable = false)`**:
    - Creates a new tensor (pointer) that is a slice of the original tensor.
- **`std::vector<SimTensor> chunk(size_t _n_chunks, size_t _dim)`**:
    - Divides the tensor into multiple chunks along a specified dimension.
- **`std::vector<SimTensor*> chunk_ptr(size_t _n_chunks, size_t _dim, bool recyclable = false)`**:
    - Divides the tensor into multiple chunks along a specified dimension (pointer version).
- **`bool contiguous(const std::pair<size_t, size_t>& _dims) const`**:
    - Checks if the tensor is contiguous over the specified dimensions.
- **`bool contiguous() const`**:
    - Checks if the entire tensor is contiguous.
- **`void copyByPointer(const SimTensor* _others)`**:
    - Copies the content of another SimTensor object (pointed to by a pointer) to this SimTensor object.
- **`void copyByObject(const SimTensor& _others)`**:
    - Copies the content of another SimTensor object to this SimTensor object.

---

([Back To Documentation Portal](/docs/README.md))
