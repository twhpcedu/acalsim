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

/**
 * @file TestSimTensor.cc
 * @brief Comprehensive GoogleTest unit tests for SimTensor class functionality
 *
 * @details
 * This test suite provides exhaustive validation of the SimTensor class, which is the
 * fundamental tensor abstraction used throughout the ACAL simulator for representing
 * multi-dimensional data structures in neural network workloads. SimTensor encapsulates
 * tensor metadata including shapes, strides, data types, memory addresses, and provides
 * operations for tensor manipulation without actual data storage.
 *
 * # Purpose
 * The test suite validates:
 * - **Constructor Correctness**: Multiple constructor variants and unique ID generation
 * - **Metadata Management**: Shape, stride, address, type, and name operations
 * - **Memory Layout**: Contiguous memory checking and stride calculations
 * - **Tensor Operations**: Flatten, unflatten, slice, chunk transformations
 * - **Data Type Handling**: Torch-compatible type byte size calculations
 * - **Copy Semantics**: Object and pointer-based copying mechanisms
 * - **Edge Cases**: Boundary conditions, error handling, and exception throwing
 *
 * # Test Coverage
 *
 * ## SimTensorTest Suite
 * Tests fundamental constructor and data type functionality:
 * - Default constructor with unique ID generation
 * - Parameterized constructor with shapes and strides
 * - Named constructor with optional metadata
 * - Copy constructor (both object and pointer forms)
 * - Torch data type byte size calculations for all supported types
 *
 * ## SimTensorMetadataTest Suite
 * Tests metadata manipulation operations:
 * - Renew operation (re-initialization with new parameters)
 * - Copy by object (deep copy semantics)
 * - Copy by pointer (pointer-based copy)
 * - ID regeneration on renewal
 *
 * ## SimTensorBasicTest Suite
 * Tests core tensor operations:
 * - Contiguous memory layout validation
 * - Flatten operations (merging dimensions)
 * - Unflatten operations (splitting dimensions)
 * - Slice operations (creating sub-tensor views)
 * - Stride calculations and updates
 * - Chunk operations (partitioning tensors)
 *
 * # Architecture Overview
 * @code
 * +-----------------------------------------------------------------------+
 * |                    TestSimTensor.cc Test Suite                        |
 * +-----------------------------------------------------------------------+
 * |                                                                       |
 * |  +----------------------------------------------------------------+   |
 * |  |                      SimTensorTest                             |   |
 * |  |  - ConstructorTest: Validates all constructor variants         |   |
 * |  |  - GetTorchTypeByteTest: Data type size calculations           |   |
 * |  +----------------------------------------------------------------+   |
 * |                                  |                                    |
 * |                                  v                                    |
 * |  +----------------------------------------------------------------+   |
 * |  |                  SimTensorMetadataTest                         |   |
 * |  |  - RenewTest: Metadata re-initialization                       |   |
 * |  |  - CopyTest: Object and pointer copying                        |   |
 * |  +----------------------------------------------------------------+   |
 * |                                  |                                    |
 * |                                  v                                    |
 * |  +----------------------------------------------------------------+   |
 * |  |                   SimTensorBasicTest                           |   |
 * |  |  - ContiguousTest: Memory layout validation                    |   |
 * |  |  - FlattenTest: Dimension merging                              |   |
 * |  |  - SliceTest: Sub-tensor view creation                         |   |
 * |  |  - StrideTest: Stride calculation validation                   |   |
 * |  |  - ChunkTest: Tensor partitioning                              |   |
 * |  +----------------------------------------------------------------+   |
 * |                                                                       |
 * +-----------------------------------------------------------------------+
 *                                  |
 *                                  v
 *                    +----------------------------+
 *                    |       SimTensor Class      |
 *                    |  (Subject Under Test)      |
 *                    +----------------------------+
 *                    | - shapes: vector<size_t>   |
 *                    | - strides: vector<size_t>  |
 *                    | - addr: Addr               |
 *                    | - dtype: DTYPE             |
 *                    | - type: TENSORTYPE         |
 *                    | - name: string             |
 *                    | - id: size_t (unique)      |
 *                    +----------------------------+
 *                    | + getShapes()              |
 *                    | + getStrides()             |
 *                    | + flatten()                |
 *                    | + unflatten()              |
 *                    | + slice()                  |
 *                    | + chunk()                  |
 *                    | + contiguous()             |
 *                    | + renew()                  |
 *                    | + copyByObject()           |
 *                    +----------------------------+
 * @endcode
 *
 * # SimTensor Class Overview
 *
 * ## Purpose and Design
 * SimTensor is a metadata-only tensor representation that:
 * - Does NOT store actual tensor data (values)
 * - Only maintains shape, stride, address, and type information
 * - Enables efficient tensor operation simulation without data movement
 * - Supports PyTorch-like tensor operations for workload modeling
 * - Provides unique ID for tracking tensor lifecycle
 *
 * ## Key Concepts
 *
 * ### Shapes
 * The dimensions of the tensor:
 * @code
 * shapes = {10, 20, 30}  // 3D tensor: 10 x 20 x 30
 * Total elements = 10 * 20 * 30 = 6000
 * @endcode
 *
 * ### Strides
 * Number of elements to skip in memory to move to the next element in each dimension:
 * @code
 * For contiguous tensor {10, 20, 30}:
 * strides = {600, 30, 1}
 * - Moving in dim 0: skip 600 elements (20*30)
 * - Moving in dim 1: skip 30 elements
 * - Moving in dim 2: skip 1 element
 * @endcode
 *
 * ### Memory Address
 * Starting address in simulated memory:
 * @code
 * addr = 0x1000  // Base address for this tensor's data
 * @endcode
 *
 * ### Data Type (DTYPE)
 * Specifies element type and size:
 * @code
 * DTYPE::Float32 -> 4 bytes per element
 * DTYPE::Int16   -> 2 bytes per element
 * DTYPE::Float64 -> 8 bytes per element
 * @endcode
 *
 * ### Tensor Type (TENSORTYPE)
 * Categorizes tensor role:
 * @code
 * TENSORTYPE::WEIGHT      // Model weights
 * TENSORTYPE::ACTIVATION  // Activation values
 * TENSORTYPE::GRADIENT    // Gradients
 * TENSORTYPE::Unknown     // Default/uncategorized
 * @endcode
 *
 * # Test Suite Details
 *
 * ## TEST(SimTensorTest, ConstructorTest)
 * Validates all constructor variants and unique ID generation.
 *
 * ### Default Constructor
 * @code{.cpp}
 * SimTensor tsr0, tsr1;
 * EXPECT_NE(tsr0.getID(), tsr1.getID());  // Unique IDs
 * @endcode
 *
 * ### Parameterized Constructor
 * @code{.cpp}
 * SimTensor tsr(SimTensor::DTYPE::Int16, 0x00, {10, 2});
 * EXPECT_EQ(tsr.getShapes(), std::vector<size_t>{10, 2});
 * EXPECT_EQ(tsr.getStrides(), std::vector<size_t>{2, 1});
 * @endcode
 * - Shapes: {10, 2} means 10 rows x 2 columns
 * - Strides: {2, 1} for row-major contiguous layout
 *
 * ### Named Constructor
 * @code{.cpp}
 * SimTensor tsr("test_tensor");
 * EXPECT_EQ(tsr.getName(), "test_tensor");
 * @endcode
 *
 * ### Copy Constructor
 * @code{.cpp}
 * SimTensor tsr0("original", 0x3000, DTYPE::Int32, {3, 3});
 * SimTensor tsr1 = SimTensor(tsr0);
 * EXPECT_EQ(tsr0.getID(), tsr1.getID());  // IDs match in copy
 * @endcode
 *
 * ## TEST(SimTensorTest, GetTorchTypeByteTest)
 * Validates byte size calculations for all PyTorch-compatible data types.
 *
 * ### Supported Data Types
 * @code
 * Float types:
 *   Float32/Float -> 4 bytes
 *   Float16/Half  -> 2 bytes
 *   BFloat16      -> 2 bytes
 *   Float64/Double-> 8 bytes
 *
 * Integer types:
 *   Int8          -> 1 byte
 *   UInt8         -> 1 byte
 *   Int16/Short   -> 2 bytes
 *   Int32/Int     -> 4 bytes
 *   Int64/Long    -> 8 bytes
 *
 * Other:
 *   Bool          -> 1 byte
 *   Unknown       -> 0 bytes
 * @endcode
 *
 * ## TEST(SimTensorMetadataTest, RenewTest)
 * Validates tensor re-initialization with new metadata.
 *
 * ### Renew Operation
 * @code{.cpp}
 * SimTensor tsr;
 * size_t original_id = tsr.getID();
 * tsr.renew("renewed", 0x3000, DTYPE::Int32, {3, 3}, TENSORTYPE::Unknown, {1, 3});
 * EXPECT_NE(tsr.getID(), original_id);  // New ID assigned
 * EXPECT_EQ(tsr.getName(), "renewed");
 * @endcode
 * - Renew generates a new unique ID
 * - All metadata is updated
 * - Useful for tensor reuse in memory pools
 *
 * ## TEST(SimTensorMetadataTest, CopyTest)
 * Validates both object-based and pointer-based copying.
 *
 * ### Copy by Object
 * @code{.cpp}
 * SimTensor tsr0("source", 0x3000, DTYPE::Int32, {3, 3});
 * SimTensor tsr1;
 * tsr1.copyByObject(tsr0);
 * EXPECT_EQ(tsr0.getID(), tsr1.getID());  // IDs match
 * @endcode
 *
 * ### Copy by Pointer
 * @code{.cpp}
 * SimTensor* tsr0_ptr = new SimTensor("source", 0x3000, DTYPE::Int32, {3, 3});
 * SimTensor* tsr1_ptr = new SimTensor();
 * tsr1_ptr->copyByPointer(tsr0_ptr);
 * EXPECT_EQ(tsr0_ptr->getID(), tsr1_ptr->getID());
 * @endcode
 *
 * ## TEST(SimTensorBasicTest, ContiguousTest)
 * Validates contiguous memory layout detection.
 *
 * ### Contiguous Layout
 * A tensor is contiguous if strides follow row-major order:
 * @code{.cpp}
 * SimTensor tsr("test", 0x1000, DTYPE::Float32, {100, 30, 40, 50},
 *               TENSORTYPE::Unknown, {60000, 2000, 50, 1});
 * EXPECT_TRUE(tsr.contiguous());
 * // stride[i] = stride[i+1] * shape[i+1] for all i
 * @endcode
 *
 * ### Non-Contiguous Layout
 * @code{.cpp}
 * SimTensor tsr("test", 0x1000, DTYPE::Float32, {100, 30, 40, 50},
 *               TENSORTYPE::Unknown, {70000, 4000, 100, 2});
 * EXPECT_FALSE(tsr.contiguous());  // Overall not contiguous
 * EXPECT_TRUE(tsr.contiguous({1, 2}));  // Dims 1-2 are contiguous
 * EXPECT_TRUE(tsr.contiguous({2, 3}));  // Dims 2-3 are contiguous
 * @endcode
 *
 * ## TEST(SimTensorBasicTest, FlattenTest)
 * Validates dimension merging and unmerging operations.
 *
 * ### Flatten Operation
 * Merges consecutive dimensions:
 * @code{.cpp}
 * SimTensor tsr("test", 0x1000, DTYPE::Float32, {10, 10, 5, 6, 7});
 * tsr.flatten({0, 1});  // Merge dims 0 and 1
 * EXPECT_EQ(tsr.getShapes(), std::vector<size_t>{100, 5, 6, 7});
 * // {10, 10, 5, 6, 7} -> {100, 5, 6, 7}
 * @endcode
 *
 * ### Unflatten Operation
 * Splits a dimension back:
 * @code{.cpp}
 * tsr.unflatten(0, 10);  // Split dim 0 by factor of 10
 * EXPECT_EQ(tsr.getShapes(), std::vector<size_t>{10, 10, 5, 6, 7});
 * // {100, 5, 6, 7} -> {10, 10, 5, 6, 7}
 * @endcode
 *
 * ## TEST(SimTensorBasicTest, SliceTest)
 * Validates sub-tensor view creation without data copying.
 *
 * ### Slice Operation
 * Creates a view into a subset of the tensor:
 * @code{.cpp}
 * SimTensor tsr("test", 0x1000, DTYPE::Float32, {10, 10, 5, 6, 7});
 * SimTensor slice = tsr.slice(
 *     {0, 0, 0, 0, 1},  // Start indices
 *     {1, 2, 4, 5, 6}   // End indices (exclusive)
 * );
 * // Sliced shapes: {1, 2, 4, 5, 5}
 * // Elements from [0:1, 0:2, 0:4, 0:5, 1:6]
 * @endcode
 *
 * ### Address Offset Calculation
 * @code{.cpp}
 * EXPECT_EQ(slice.getAddr(), tsr.getAddr() + 1);
 * // Offset by 1 element in the last dimension (start index 1)
 * @endcode
 *
 * ### Boundary Checking
 * @code{.cpp}
 * EXPECT_THROW(
 *     tsr.slice({0, 0, 0, 0, 8}, {1, 2, 4, 5, 10}),
 *     std::runtime_error
 * );
 * // Throws because end index 10 exceeds dimension size 7
 * @endcode
 *
 * ## TEST(SimTensorBasicTest, StrideTest)
 * Validates stride calculations through various operations.
 *
 * ### Initial Strides
 * @code{.cpp}
 * SimTensor tsr("test", 0x00, DTYPE::Float32, {4, 5, 6, 7, 8});
 * EXPECT_EQ(tsr.getStrides(), std::vector<size_t>{1680, 336, 56, 8, 1});
 * // stride[i] = product of all dimensions after i
 * // stride[0] = 5*6*7*8 = 1680
 * // stride[1] = 6*7*8 = 336
 * // ...
 * @endcode
 *
 * ### Strides After Flatten
 * @code{.cpp}
 * tsr.flatten({2, 3});  // Merge dims 2 and 3
 * EXPECT_EQ(tsr.getShapes(), std::vector<size_t>{4, 5, 42, 8});
 * EXPECT_EQ(tsr.getStrides(), std::vector<size_t>{1680, 336, 8, 1});
 * @endcode
 *
 * ### Strides After Unflatten
 * @code{.cpp}
 * tsr.unflatten(2, 6);
 * EXPECT_EQ(tsr.getShapes(), std::vector<size_t>{4, 5, 6, 7, 8});
 * EXPECT_EQ(tsr.getStrides(), std::vector<size_t>{1680, 336, 56, 8, 1});
 * @endcode
 *
 * ## TEST(SimTensorBasicTest, ChunkTest)
 * Validates tensor partitioning into smaller chunks.
 *
 * ### Chunk Operation
 * Splits a tensor along a dimension:
 * @code{.cpp}
 * SimTensor tsr("test", 0x00, DTYPE::Float32, {5, 5, 5, 5, 5});
 * auto chunks = tsr.chunk(4, 1);  // Split dim 1 into 4 chunks
 * // Original dim 1 has size 5, split into 4 chunks
 * // Chunk sizes: 2, 1, 1, 1 (ceil(5/4)=2 for first chunk, 1 for others)
 * @endcode
 *
 * ### Chunk Shapes
 * @code{.cpp}
 * EXPECT_EQ(chunks[0].getShapes(), std::vector<size_t>{5, 2, 5, 5, 5});
 * EXPECT_EQ(chunks[1].getShapes(), std::vector<size_t>{5, 1, 5, 5, 5});
 * EXPECT_EQ(chunks[2].getShapes(), std::vector<size_t>{5, 1, 5, 5, 5});
 * EXPECT_EQ(chunks[3].getShapes(), std::vector<size_t>{5, 1, 5, 5, 5});
 * @endcode
 *
 * ### Chunk Address Offsets
 * @code{.cpp}
 * EXPECT_EQ(chunks[0].getAddr(), tsr.getAddr() + 0 * 125);
 * EXPECT_EQ(chunks[1].getAddr(), tsr.getAddr() + 2 * 125);
 * EXPECT_EQ(chunks[2].getAddr(), tsr.getAddr() + 3 * 125);
 * EXPECT_EQ(chunks[3].getAddr(), tsr.getAddr() + 4 * 125);
 * // Offset by stride[1] * chunk_start
 * @endcode
 *
 * # How to Run
 *
 * ## Building the Tests
 * @code{.sh}
 * cd acalsim-workspace/projects/acalsim
 * mkdir -p build && cd build
 * cmake ..
 * make TestSimTensor
 * @endcode
 *
 * ## Running All Tests
 * @code{.sh}
 * # Run complete test suite
 * ./gtest/UnitTest/TestSimTensor
 *
 * # Run with verbose output
 * ./gtest/UnitTest/TestSimTensor --gtest_verbose
 * @endcode
 *
 * ## Running Specific Tests
 * @code{.sh}
 * # Run only constructor tests
 * ./gtest/UnitTest/TestSimTensor --gtest_filter=SimTensorTest.ConstructorTest
 *
 * # Run only metadata tests
 * ./gtest/UnitTest/TestSimTensor --gtest_filter=SimTensorMetadataTest.*
 *
 * # Run only basic operation tests
 * ./gtest/UnitTest/TestSimTensor --gtest_filter=SimTensorBasicTest.*
 *
 * # Run specific test
 * ./gtest/UnitTest/TestSimTensor --gtest_filter=SimTensorBasicTest.SliceTest
 * @endcode
 *
 * # Expected Outcomes
 *
 * ## Successful Test Run
 * @code
 * [==========] Running 7 tests from 3 test suites.
 * [----------] 2 tests from SimTensorTest
 * [ RUN      ] SimTensorTest.ConstructorTest
 * [       OK ] SimTensorTest.ConstructorTest (0 ms)
 * [ RUN      ] SimTensorTest.GetTorchTypeByteTest
 * [       OK ] SimTensorTest.GetTorchTypeByteTest (0 ms)
 * [----------] 2 tests from SimTensorTest (0 ms total)
 *
 * [----------] 2 tests from SimTensorMetadataTest
 * [ RUN      ] SimTensorMetadataTest.RenewTest
 * [       OK ] SimTensorMetadataTest.RenewTest (0 ms)
 * [ RUN      ] SimTensorMetadataTest.CopyTest
 * [       OK ] SimTensorMetadataTest.CopyTest (0 ms)
 * [----------] 2 tests from SimTensorMetadataTest (0 ms total)
 *
 * [----------] 5 tests from SimTensorBasicTest
 * [ RUN      ] SimTensorBasicTest.ContiguousTest
 * [       OK ] SimTensorBasicTest.ContiguousTest (0 ms)
 * [ RUN      ] SimTensorBasicTest.FlattenTest
 * [       OK ] SimTensorBasicTest.FlattenTest (0 ms)
 * [ RUN      ] SimTensorBasicTest.SliceTest
 * [       OK ] SimTensorBasicTest.SliceTest (0 ms)
 * [ RUN      ] SimTensorBasicTest.StrideTest
 * [       OK ] SimTensorBasicTest.StrideTest (0 ms)
 * [ RUN      ] SimTensorBasicTest.ChunkTest
 * [       OK ] SimTensorBasicTest.ChunkTest (0 ms)
 * [----------] 5 tests from SimTensorBasicTest (0 ms total)
 *
 * [==========] 7 tests from 3 test suites ran. (1 ms total)
 * [  PASSED  ] 9 tests.
 * @endcode
 *
 * # Code Examples
 *
 * ## Creating and Manipulating Tensors
 * @code{.cpp}
 * // Create a 3D tensor
 * SimTensor tensor("activation", 0x1000, SimTensor::DTYPE::Float32,
 *                  {32, 64, 128}, SimTensor::TENSORTYPE::ACTIVATION);
 *
 * // Check if contiguous
 * if (tensor.contiguous()) {
 *     // Efficient memory access possible
 * }
 *
 * // Flatten first two dimensions
 * tensor.flatten({0, 1});  // {32, 64, 128} -> {2048, 128}
 *
 * // Create a slice
 * auto slice = tensor.slice({0, 0}, {1024, 128});
 *
 * // Partition into chunks
 * auto chunks = tensor.chunk(4, 0);  // Split batch dimension
 * @endcode
 *
 * ## Testing Custom Tensor Operations
 * @code{.cpp}
 * TEST(CustomTensorTest, TransposeOperation) {
 *     SimTensor tsr("test", 0x0, DTYPE::Float32, {10, 20});
 *     // Test your custom transpose operation
 *     auto transposed = myTranspose(tsr);
 *     EXPECT_EQ(transposed.getShapes(), std::vector<size_t>{20, 10});
 * }
 * @endcode
 *
 * # Related Components
 *
 * @see SimTensor - The tensor metadata class being tested
 * @see SimTensor.hh - Header file with class definition
 * @see TensorWorkload - Workload classes using SimTensor
 * @see MemoryManager - Memory allocation for tensor data
 *
 * # Performance Considerations
 *
 * - Tests execute in <1ms (metadata-only operations)
 * - No actual data allocation or copying
 * - Efficient for large tensor shape validation
 * - Suitable for integration into CI/CD pipelines
 *
 * # Debugging Tips
 *
 * 1. **Print Tensor State**:
 *    @code{.cpp}
 *    std::cout << tensor << std::endl;  // Uses operator<<
 *    @endcode
 *
 * 2. **Check Stride Calculations**:
 *    @code{.cpp}
 *    auto strides = tensor.getStrides();
 *    for (size_t i = 0; i < strides.size(); i++) {
 *        std::cout << "stride[" << i << "] = " << strides[i] << std::endl;
 *    }
 *    @endcode
 *
 * 3. **Verify Contiguity**:
 *    @code{.cpp}
 *    for (size_t i = 0; i < tensor.getRank() - 1; i++) {
 *        bool dim_contig = tensor.contiguous({i, i+1});
 *        std::cout << "Dims [" << i << "," << i+1 << "] contiguous: "
 *                  << dim_contig << std::endl;
 *    }
 *    @endcode
 *
 * @author ACAL/Playlab Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This test suite does not test actual data manipulation, only metadata operations
 * @warning Some operations may throw exceptions for invalid parameters - tests validate this behavior
 */

#include <gtest/gtest.h>

#include "workloads/tensor/SimTensor.hh"

namespace unit_test {

using namespace acalsim;

// Test the default constructor with unique ID generation
TEST(SimTensorTest, ConstructorTest) {
	{  // Default constructor with unique ID generation
		SimTensor  tsr0, tsr1;
		SimTensor *tsr0_ptr = new SimTensor(), *tsr1_ptr = new SimTensor();

		// unique ID generation
		EXPECT_NE(tsr0.getID(), tsr1.getID()) << "The unique IDs for the two tensors should be different.";
		EXPECT_NE(tsr0_ptr->getID(), tsr1_ptr->getID())
		    << "The unique IDs for the two tensor pointers should be different.";
		delete tsr0_ptr, delete tsr1_ptr;
	}
	{  // Parameterized constructor
		SimTensor  tsr(SimTensor::DTYPE::Int16, 0x00, {10, 2});
		SimTensor* tsr_ptr = new SimTensor(SimTensor::DTYPE::Int16, 0x00, {10, 2});

		// Check shapes and strides
		EXPECT_EQ(tsr.getShapes(), (std::vector<size_t>{10, 2})) << "The shapes of the tensor should be {10, 2}.";
		EXPECT_EQ(tsr_ptr->getShapes(), (std::vector<size_t>{10, 2}))
		    << "The shapes of the tensor pointer should be {10, 2}.";
		EXPECT_EQ(tsr.getStrides(), (std::vector<size_t>{2, 1})) << "The strides of the tensor should be {2, 1}.";
		EXPECT_EQ(tsr_ptr->getStrides(), (std::vector<size_t>{2, 1}))
		    << "The strides of the tensor pointer should be {2, 1}.";
		delete tsr_ptr;
	}
	{  // Named constructor for SimTensor with optional shapes and strides.
		SimTensor  tsr("test_tensor");
		SimTensor* tsr_ptr = new SimTensor("test_tensor");

		// Check the name
		EXPECT_EQ(tsr.getName(), "test_tensor") << "The name of the tensor should be 'test_tensor'.";
		EXPECT_EQ(tsr_ptr->getName(), "test_tensor") << "The name of the tensor pointer should be 'test_tensor'.";
		delete tsr_ptr;
	}
	{  // Copy constructor for SimTensor
		SimTensor tsr0("Object_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3}, SimTensor::TENSORTYPE::WEIGHT,
		               {1, 3});
		SimTensor tsr1 = SimTensor(tsr0);

		EXPECT_EQ(tsr0.getName(), tsr1.getName()) << "The name of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getAddr(), tsr1.getAddr()) << "The address of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getType(), tsr1.getType()) << "The type of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getDType(), tsr1.getDType()) << "The dtype of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getShapes(), tsr1.getShapes()) << "The shapes of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getStrides(), tsr1.getStrides())
		    << "The strides of the copied tensor should match the original.";
		EXPECT_EQ(tsr0.getID(), tsr1.getID()) << "The IDs of the copied tensor and the original should match.";

		SimTensor* tsr0_ptr = new SimTensor("Pointer_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3},
		                                    SimTensor::TENSORTYPE::Unknown, {1, 3});
		SimTensor* tsr1_ptr = new SimTensor(tsr0_ptr);
		EXPECT_EQ(tsr0_ptr->getName(), tsr1_ptr->getName())
		    << "The name of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getAddr(), tsr1_ptr->getAddr())
		    << "The address of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getType(), tsr1_ptr->getType())
		    << "The type of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getDType(), tsr1_ptr->getDType())
		    << "The dtype of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getShapes(), tsr1_ptr->getShapes())
		    << "The shapes of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getStrides(), tsr1_ptr->getStrides())
		    << "The strides of the copied tensor should match the original.";
		EXPECT_EQ(tsr0_ptr->getID(), tsr1_ptr->getID())
		    << "The IDs of the copied tensor and the original should match.";

		delete tsr0_ptr, delete tsr1_ptr;
	}
}

// Test the getTorchTypeByte function
TEST(SimTensorTest, GetTorchTypeByteTest) {
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Float32), 4);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Float), 4);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Float16), 2);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Half), 2);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::BFloat16), 2);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Float64), 8);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Double), 8);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Int8), 1);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::UInt8), 1);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Bool), 1);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Int16), 2);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Short), 2);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Int32), 4);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Int), 4);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Int64), 8);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Long), 8);
	EXPECT_EQ(SimTensor::getTorchTypeByte(SimTensor::DTYPE::Unknown), 0);
}

// Test renew method
TEST(SimTensorMetadataTest, RenewTest) {
	SimTensor tsr;
	size_t    tsr_id = tsr.getID();

	tsr.renew("Object_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3}, SimTensor::TENSORTYPE::Unknown, {1, 3});

	EXPECT_EQ(tsr.getName(), "Object_renewed") << "The name after renew should be 'Object_renewed'.";
	EXPECT_EQ(tsr.getAddr(), 0x3000) << "The address after renew should be 0x3000.";
	EXPECT_EQ(tsr.getShapes(), std::vector<size_t>({3, 3})) << "The shapes after renew should be {3, 3}.";
	EXPECT_EQ(tsr.getDType(), SimTensor::DTYPE::Int32) << "The dtype after renew should be Int32.";
	EXPECT_EQ(tsr.getStrides(), std::vector<size_t>({1, 3})) << "The strides after renew should be {1, 3}.";
	EXPECT_NE(tsr.getID(), tsr_id) << "The ID after renew should be different from the original ID.";

	SimTensor* tsr_ptr    = new SimTensor();
	size_t     tsr_ptr_id = tsr_ptr->getID();
	tsr_ptr->renew("Pointer_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3}, SimTensor::TENSORTYPE::Unknown, {1, 3});

	EXPECT_EQ(tsr_ptr->getName(), "Pointer_renewed") << "The name after renew should be 'Pointer_renewed'.";
	EXPECT_EQ(tsr_ptr->getAddr(), 0x3000) << "The address after renew should be 0x3000.";
	EXPECT_EQ(tsr_ptr->getShapes(), std::vector<size_t>({3, 3})) << "The shapes after renew should be {3, 3}.";
	EXPECT_EQ(tsr_ptr->getDType(), SimTensor::DTYPE::Int32) << "The dtype after renew should be Int32.";
	EXPECT_EQ(tsr_ptr->getStrides(), std::vector<size_t>({1, 3})) << "The strides after renew should be {1, 3}.";
	EXPECT_NE(tsr_ptr->getID(), tsr_ptr_id) << "The ID after renew should be different from the original ID.";
	delete tsr_ptr;
}

// Test copy method
TEST(SimTensorMetadataTest, CopyTest) {
	SimTensor tsr0("Object_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3}, SimTensor::TENSORTYPE::Unknown, {1, 3});

	SimTensor tsr1 = SimTensor();
	tsr1.copyByObject(tsr0);

	EXPECT_EQ(tsr0.getName(), tsr1.getName()) << "The name of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getAddr(), tsr1.getAddr()) << "The address of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getType(), tsr1.getType()) << "The type of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getDType(), tsr1.getDType()) << "The dtype of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getShapes(), tsr1.getShapes()) << "The shapes of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getStrides(), tsr1.getStrides()) << "The strides of the copied tensor should match the original.";
	EXPECT_EQ(tsr0.getID(), tsr1.getID()) << "The IDs of the copied tensor and the original should match.";

	SimTensor* tsr0_ptr = new SimTensor("Pointer_renewed", 0x3000, SimTensor::DTYPE::Int32, {3, 3},
	                                    SimTensor::TENSORTYPE::Unknown, {1, 3});
	SimTensor* tsr1_ptr = new SimTensor();
	tsr1_ptr->copyByPointer(tsr0_ptr);
	EXPECT_EQ(tsr0_ptr->getName(), tsr1_ptr->getName()) << "The name of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getAddr(), tsr1_ptr->getAddr())
	    << "The address of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getType(), tsr1_ptr->getType()) << "The type of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getDType(), tsr1_ptr->getDType())
	    << "The dtype of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getShapes(), tsr1_ptr->getShapes())
	    << "The shapes of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getStrides(), tsr1_ptr->getStrides())
	    << "The strides of the copied tensor should match the original.";
	EXPECT_EQ(tsr0_ptr->getID(), tsr1_ptr->getID()) << "The IDs of the copied tensor and the original should match.";
	delete tsr0_ptr, delete tsr1_ptr;
}

// Test suite for general Tensor operations
TEST(SimTensorBasicTest, ContiguousTest) {
	{
		SimTensor tsr("test_tensor", 0x1000, SimTensor::DTYPE::Float32, {100, 30, 40, 50},
		              SimTensor::TENSORTYPE::Unknown, {60000, 2000, 50, 1});
		EXPECT_TRUE(tsr.contiguous()) << tsr;

		SimTensor* tsr_ptr = new SimTensor("test_tensor", 0x1000, SimTensor::DTYPE::Float32, {100, 30, 40, 50},
		                                   SimTensor::TENSORTYPE::Unknown, {60000, 2000, 50, 1});
		EXPECT_TRUE(tsr_ptr->contiguous()) << *tsr_ptr;

		delete tsr_ptr;
	}
	{
		SimTensor tsr("test_tensor", 0x1000, SimTensor::DTYPE::Float32, {100, 30, 40, 50},
		              SimTensor::TENSORTYPE::Unknown, {70000, 4000, 100, 2});
		EXPECT_FALSE(tsr.contiguous()) << tsr;
		EXPECT_TRUE(tsr.contiguous({1, 2})) << tsr;
		EXPECT_TRUE(tsr.contiguous({2, 3})) << tsr;
		EXPECT_FALSE(tsr.contiguous({0, 1})) << tsr;

		auto tsr_ptr = new SimTensor("test_tensor", 0x1000, SimTensor::DTYPE::Float32, {100, 30, 40, 50},
		                             SimTensor::TENSORTYPE::Unknown, {70000, 4000, 100, 2});
		EXPECT_FALSE(tsr_ptr->contiguous()) << *tsr_ptr;
		EXPECT_TRUE(tsr_ptr->contiguous({1, 2})) << *tsr_ptr;
		EXPECT_TRUE(tsr_ptr->contiguous({2, 3})) << *tsr_ptr;
		EXPECT_FALSE(tsr_ptr->contiguous({0, 1})) << *tsr_ptr;
		delete tsr_ptr;
	}
}

// Test basic flatten and unflatten operations
TEST(SimTensorBasicTest, FlattenTest) {
	SimTensor tsr("flatten_test", 0x1000, SimTensor::DTYPE::Float32, {10, 10, 5, 6, 7});
	tsr.flatten({0, 1});
	EXPECT_EQ(tsr.getShapes(), (std::vector<size_t>{100, 5, 6, 7})) << tsr;

	tsr.unflatten(0, 10);
	EXPECT_EQ(tsr.getShapes(), (std::vector<size_t>{10, 10, 5, 6, 7})) << tsr;

	SimTensor* tsr_ptr = new SimTensor("flatten_test", 0x1000, SimTensor::DTYPE::Float, {10, 10, 5, 6, 7});
	tsr_ptr->flatten({0, 1});
	EXPECT_EQ(tsr_ptr->getShapes(), (std::vector<size_t>{100, 5, 6, 7})) << *tsr_ptr;
	tsr_ptr->unflatten(0, 10);
	EXPECT_EQ(tsr_ptr->getShapes(), (std::vector<size_t>{10, 10, 5, 6, 7})) << *tsr_ptr;
	delete tsr_ptr;
}

// Test basic flatten and unflatten operations
TEST(SimTensorBasicTest, SliceTest) {
	{
		SimTensor tsr("slice_test", 0x1000, SimTensor::DTYPE::Float32, {10, 10, 5, 6, 7});
		SimTensor new_tsr = tsr.slice({0, 0, 0, 0, 1}, {1, 2, 4, 5, 6});
		EXPECT_EQ(new_tsr.getShapes(), std::vector<size_t>({1, 2, 4, 5, 5}));
		EXPECT_EQ(new_tsr.getStrides(), std::vector<size_t>({2100, 210, 42, 7, 1}));
		EXPECT_EQ(new_tsr.getAddr(), tsr.getAddr() + 1);
		EXPECT_EQ(new_tsr.size(), 800);
		EXPECT_THROW(tsr.slice({0, 0, 0, 0, 8}, {1, 2, 4, 5, 10}), std::runtime_error);
	}

	{
		SimTensor* tsr_ptr     = new SimTensor("slice_test", 0x1000, SimTensor::DTYPE::Float32, {10, 10, 5, 6, 7});
		SimTensor* new_tsr_ptr = tsr_ptr->slice_ptr({0, 0, 0, 0, 1}, {1, 2, 4, 5, 6}, false);
		EXPECT_EQ(new_tsr_ptr->getShapes(), std::vector<size_t>({1, 2, 4, 5, 5}));
		EXPECT_EQ(new_tsr_ptr->getStrides(), std::vector<size_t>({2100, 210, 42, 7, 1}));
		EXPECT_EQ(new_tsr_ptr->getAddr(), tsr_ptr->getAddr() + 1);
		EXPECT_EQ(new_tsr_ptr->size(), 800);
		EXPECT_THROW(tsr_ptr->slice_ptr({0, 0, 0, 0, 8}, {1, 2, 4, 5, 10}), std::runtime_error);
		delete tsr_ptr, delete new_tsr_ptr;
	}
}

// Grouped test for stride calculations (initial, flatten, unflatten)
TEST(SimTensorBasicTest, StrideTest) {
	auto tsr     = SimTensor("test_tensor", 0x00, SimTensor::DTYPE::Float32, {4, 5, 6, 7, 8});
	auto tsr_ptr = new SimTensor("test_tensor_ptr", 0x00, SimTensor::DTYPE::Float32, {4, 5, 6, 7, 8});
	// Initial stride check
	EXPECT_EQ(tsr.getStrides(), (std::vector<size_t>{1680, 336, 56, 8, 1})) << tsr;
	EXPECT_EQ(tsr_ptr->getStrides(), (std::vector<size_t>{1680, 336, 56, 8, 1})) << *tsr_ptr;

	// Flatten dimensions 2 and 3 and check stride update
	tsr.flatten({2, 3});
	tsr_ptr->flatten({2, 3});
	EXPECT_EQ(tsr.getShapes(), (std::vector<size_t>{4, 5, 42, 8})) << tsr;
	EXPECT_EQ(tsr_ptr->getShapes(), (std::vector<size_t>{4, 5, 42, 8})) << *tsr_ptr;
	EXPECT_EQ(tsr.getStrides(), (std::vector<size_t>{1680, 336, 8, 1})) << tsr;
	EXPECT_EQ(tsr_ptr->getStrides(), (std::vector<size_t>{1680, 336, 8, 1})) << *tsr_ptr;

	// Unflatten and check stride restoration
	tsr.unflatten(2, 6);       // Unflatten the 42 back into dimensions 6 and 7
	tsr_ptr->unflatten(2, 6);  // Unflatten the 42 back into dimensions 6 and 7
	EXPECT_EQ(tsr.getShapes(), (std::vector<size_t>{4, 5, 6, 7, 8})) << tsr;
	EXPECT_EQ(tsr_ptr->getShapes(), (std::vector<size_t>{4, 5, 6, 7, 8})) << *tsr_ptr;
	EXPECT_EQ(tsr.getStrides(), (std::vector<size_t>{1680, 336, 56, 8, 1})) << tsr;
	EXPECT_EQ(tsr_ptr->getStrides(), (std::vector<size_t>{1680, 336, 56, 8, 1})) << *tsr_ptr;

	delete tsr_ptr;
}

// Test that partitioning does not alter original tensor's strides and properties
TEST(SimTensorBasicTest, ChunkTest) {
	{
		auto tsr = SimTensor("test_tensor", 0x00, SimTensor::DTYPE::Float32, {5, 5, 5, 5, 5});
		// chunk the tensor along dimension 1 (splitting the tensor into 5 parts)
		auto chunks = tsr.chunk(4, 1);

		// Check if the tensor strides are correct (they should remain unchanged)
		EXPECT_EQ(tsr.getStrides(), (std::vector<size_t>{625, 125, 25, 5, 1})) << chunks[0];

		// Check that the partitioned tensors have the correct shapes
		EXPECT_EQ(chunks[0].getShapes(), (std::vector<size_t>{5, 2, 5, 5, 5})) << chunks[0];
		EXPECT_EQ(chunks[1].getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << chunks[1];
		EXPECT_EQ(chunks[2].getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << chunks[2];
		EXPECT_EQ(chunks[3].getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << chunks[3];

		EXPECT_EQ(chunks[0].getAddr(), tsr.getAddr() + 0 * 125) << chunks[0];
		EXPECT_EQ(chunks[1].getAddr(), tsr.getAddr() + 2 * 125) << chunks[1];
		EXPECT_EQ(chunks[2].getAddr(), tsr.getAddr() + 3 * 125) << chunks[2];
		EXPECT_EQ(chunks[3].getAddr(), tsr.getAddr() + 4 * 125) << chunks[3];
	}

	{
		auto tsr_ptr = new SimTensor("test_tensor", 0x00, SimTensor::DTYPE::Float32, {5, 5, 5, 5, 5});
		// chunk the tensor along dimension 1 (splitting the tensor into 5 parts)
		auto chunks = tsr_ptr->chunk_ptr(4, 1, false);

		// Check if the tensor strides are correct (they should remain unchanged)
		EXPECT_EQ(tsr_ptr->getStrides(), (std::vector<size_t>{625, 125, 25, 5, 1})) << *chunks[0];

		// Check that the partitioned tensors have the correct shapes
		EXPECT_EQ(chunks[0]->getShapes(), (std::vector<size_t>{5, 2, 5, 5, 5})) << *chunks[0];
		EXPECT_EQ(chunks[1]->getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << *chunks[1];
		EXPECT_EQ(chunks[2]->getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << *chunks[2];
		EXPECT_EQ(chunks[3]->getShapes(), (std::vector<size_t>{5, 1, 5, 5, 5})) << *chunks[3];

		EXPECT_EQ(chunks[0]->getAddr(), tsr_ptr->getAddr() + 0 * 125) << *chunks[0];
		EXPECT_EQ(chunks[1]->getAddr(), tsr_ptr->getAddr() + 2 * 125) << *chunks[1];
		EXPECT_EQ(chunks[2]->getAddr(), tsr_ptr->getAddr() + 3 * 125) << *chunks[2];
		EXPECT_EQ(chunks[3]->getAddr(), tsr_ptr->getAddr() + 4 * 125) << *chunks[3];
		for (auto chunk : chunks) { delete chunk; }
	}
}

}  // namespace unit_test
