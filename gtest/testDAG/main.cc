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

/**
 * @file main.cc
 * @brief GoogleTest test suite for DAG (Directed Acyclic Graph) scheduler and operations
 *
 * @details
 * This file contains comprehensive unit tests for the DAG scheduler implementation in the
 * ACALSIM project. The test suite validates the correctness of DAG construction, dependency
 * management, topological sorting, and scheduling algorithms across various graph topologies.
 *
 * ## Test Coverage
 *
 * The test suite covers the following key areas:
 *
 * ### 1. Basic DAG Operations (DAGTest)
 * - Node creation and lookup functionality
 * - Tensor (edge) creation and retrieval
 * - Dependency relationship establishment
 * - Ready pool initialization and state management
 * - Graph merging operations
 *
 * ### 2. Scheduler Behavior (DAGSchedulerTest)
 * - Round-robin scheduling across multiple graphs
 * - Complex graph topology handling (diamond patterns, tiled GEMM)
 * - Empty graph edge cases
 * - Concurrent ready node management
 * - Dependency order preservation
 *
 * ### 3. Priority-Based Scheduling (PriorityTestScheduler)
 * - Priority queue management
 * - High-priority graph execution precedence
 * - Fair scheduling with priority weights
 *
 * ## Graph Topologies Tested
 *
 * ### Basic Linear Graph
 * @code
 * op1 -> op2 -> op3
 *  |      |      |
 *  t1    t2,t3  t4
 * @endcode
 * A simple three-node sequential graph where each operation depends on the previous one.
 * Used to test fundamental scheduling and dependency resolution.
 *
 * ### Diamond Graph
 * @code
 *       opA
 *      /   \
 *   tA_out1 tA_out2
 *    /       \
 *   opB      opC
 *    |        |
 * tB_out   tC_out
 *    \       /
 *     \     /
 *      opD
 * @endcode
 * Tests parallel path convergence and ensures both branches complete before sink node.
 * Validates that the scheduler correctly handles multiple dependencies.
 *
 * ### Tiled GEMM Graph
 * A complex 20-node graph representing tiled matrix multiplication with data loading,
 * computation, and storage stages. Tests realistic workload scheduling with multiple
 * interdependent operations.
 *
 * @code
 * Stage 1: Load initial tiles
 *   loadA00, loadB00 -> gemmC00-1
 *
 * Stage 2: Compute and load next tiles
 *   gemmC00-1 -> loadA01, loadB10 -> gemmC00-2
 *
 * Stage 3: Continue tiled computation
 *   Multiple GEMM operations with dependencies on loads
 *
 * Stage 4: Store final results
 *   Final operations -> storeC00, storeC01, storeC10, storeC11
 * @endcode
 *
 * ## Test Architecture
 *
 * The test suite uses GoogleTest fixtures with three main fixture classes:
 *
 * - **DAGTest**: Base fixture providing graph construction helpers
 * - **DAGSchedulerTest**: Tests round-robin scheduling behavior
 * - **PriorityTestScheduler**: Tests priority-based scheduling
 *
 * Test fixtures provide helper methods to create predefined graph topologies:
 * - createBasicGraph(): Three-node sequential graph
 * - createDiamondGraph(): Four-node diamond topology
 * - createTiledGEMMGraph(): 20-node tiled matrix multiplication
 *
 * ## Running the Tests
 *
 * ### Build and Execute
 * @code{.sh}
 * # From the build directory
 * cd build
 * cmake ..
 * make testDAG
 * ./gtest/testDAG/testDAG
 * @endcode
 *
 * ### Run Specific Tests
 * @code{.sh}
 * # Run only DAGTest fixtures
 * ./testDAG --gtest_filter=DAGTest.*
 *
 * # Run only scheduler tests
 * ./testDAG --gtest_filter=DAGSchedulerTest.*
 *
 * # Run a specific test
 * ./testDAG --gtest_filter=DAGTest.NodeCreationAndLookup
 * @endcode
 *
 * ### Verbose Output
 * @code{.sh}
 * # Show detailed test information
 * ./testDAG --gtest_verbose=1
 * @endcode
 *
 * ## Expected Test Outcomes
 *
 * All tests should pass with the following assertions validated:
 *
 * - **Node/Edge Lookup**: Valid nodes return non-null pointers, invalid lookups return nullptr
 * - **Dependency Counts**: Correct parent/child relationships and dependency counts
 * - **Ready Pool**: Only source nodes (zero dependencies) initially ready
 * - **Execution Order**: Topological ordering preserved (parents before children)
 * - **Round-Robin Fairness**: Alternating execution between graphs when possible
 * - **Priority Ordering**: Higher-priority graphs execute first
 * - **Empty Graphs**: Graceful handling with empty execution order
 *
 * ## Key Testing Patterns
 *
 * ### TEST() Macros
 * @code{.cpp}
 * TEST_F(DAGTest, NodeCreationAndLookup) {
 *     graph::DAG<MicroOp, SimTensor> graph = createBasicGraph("Test Graph");
 *     MicroOp* op1 = graph.getNode("op1");
 *     ASSERT_NE(op1, nullptr);  // Fails test if null
 *     EXPECT_EQ(op1->getName(), "op1");  // Continues on failure
 * }
 * @endcode
 *
 * ### EXPECT vs ASSERT
 * - **EXPECT_***: Non-fatal, test continues after failure
 * - **ASSERT_***: Fatal, test stops immediately on failure
 *
 * ### Common Assertions Used
 * - EXPECT_EQ/ASSERT_EQ: Equality comparison
 * - EXPECT_NE/ASSERT_NE: Inequality comparison
 * - EXPECT_TRUE/EXPECT_FALSE: Boolean checks
 * - EXPECT_LT/EXPECT_GT: Ordering comparisons
 *
 * ## Dependencies
 *
 * @see workloads/graph/DAG.hh - Main DAG implementation
 * @see workloads/operator/MicroOp.hh - Operation node representation
 * @see workloads/tensor/SimTensor.hh - Data edge representation
 * @see DAGSchedulerTest.hh - Test fixture definitions and helpers
 *
 * @author ACAL/Playlab
 * @date 2023-2025
 * @version 1.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "DAGSchedulerTest.hh"
#include "workloads/graph/DAG.hh"
#include "workloads/operator/MicroOp.hh"
#include "workloads/tensor/SimTensor.hh"

using namespace acalsim;
using ::testing::Contains;
using ::testing::ElementsAre;

/**
 * @brief Test basic node creation and lookup functionality in DAG
 *
 * @details
 * Validates that nodes can be properly created, registered, and retrieved from the DAG.
 * This test ensures the fundamental node management system works correctly.
 *
 * **Test Steps:**
 * 1. Create a basic three-node graph using the test fixture helper
 * 2. Attempt to retrieve an existing node by name
 * 3. Verify the node pointer is non-null and has correct name
 * 4. Attempt to retrieve a non-existent node
 * 5. Verify that invalid lookup returns nullptr
 *
 * **Assertions:**
 * - ASSERT_NE: Node pointer must not be null (fatal)
 * - EXPECT_EQ: Node name must match expected value
 * - EXPECT_EQ: Non-existent node lookup must return nullptr
 *
 * @see DAGTest::createBasicGraph()
 * @see graph::DAG::getNode()
 */
TEST_F(DAGTest, NodeCreationAndLookup) {
	graph::DAG<MicroOp, SimTensor> graph = createBasicGraph("Test Graph");

	// Test node lookup
	MicroOp* op1 = graph.getNode("op1");
	ASSERT_NE(op1, nullptr);
	EXPECT_EQ(op1->getName(), "op1");

	// Test non-existent node lookup
	MicroOp* nonExistent = graph.getNode("nonExistent");
	EXPECT_EQ(nonExistent, nullptr);
}

/**
 * @brief Test tensor (edge) creation and lookup functionality in DAG
 *
 * @details
 * Validates that tensor edges can be created, registered, and retrieved from the DAG.
 * Tensors represent data dependencies between operations in the computation graph.
 *
 * **Test Steps:**
 * 1. Create a basic graph with predefined tensors
 * 2. Retrieve an existing tensor by name
 * 3. Verify the tensor pointer is valid and has correct name
 * 4. Attempt to retrieve a non-existent tensor
 * 5. Verify that invalid lookup returns nullptr
 *
 * **Assertions:**
 * - ASSERT_NE: Tensor pointer must not be null (fatal)
 * - EXPECT_EQ: Tensor name must match expected value
 * - EXPECT_EQ: Non-existent tensor lookup must return nullptr
 *
 * @see DAGTest::createBasicGraph()
 * @see graph::DAG::getEdge()
 */
TEST_F(DAGTest, TensorCreationAndLookup) {
	graph::DAG<MicroOp, SimTensor> graph = createBasicGraph("Test Graph");

	// Test tensor lookup
	SimTensor* t1 = graph.getEdge("t1");
	ASSERT_NE(t1, nullptr);
	EXPECT_EQ(t1->getName(), "t1");

	// Test non-existent tensor lookup
	SimTensor* nonExistent = graph.getEdge("nonExistent");
	EXPECT_EQ(nonExistent, nullptr);
}

/**
 * @brief Test dependency creation and relationship establishment
 *
 * @details
 * Validates that dependencies between nodes are correctly established and tracked.
 * This test ensures parent-child relationships are properly maintained and dependency
 * counts are accurate for topological sorting.
 *
 * **Test Steps:**
 * 1. Create a basic graph with known dependencies (op1 -> op2 -> op3)
 * 2. Retrieve two connected nodes
 * 3. Verify op1 is in op2's consumers list (parent relationship)
 * 4. Check dependency counts for each node
 * 5. Verify op1 has zero dependencies (source node)
 * 6. Verify op2 has exactly one dependency (depends on op1)
 *
 * **Assertions:**
 * - ASSERT_NE: Node pointers must be valid (fatal)
 * - EXPECT_TRUE: Parent must be in child's consumers list
 * - EXPECT_EQ: Source node must have zero dependency count
 * - EXPECT_EQ: Dependent node must have correct count
 *
 * @see graph::DAG::addDependency()
 * @see MicroOp::consumers
 * @see MicroOp::dependency_count
 */
TEST_F(DAGTest, DependencyCreation) {
	graph::DAG<MicroOp, SimTensor> graph = createBasicGraph("Test Graph");

	MicroOp* op1 = graph.getNode("op1");
	MicroOp* op2 = graph.getNode("op2");

	ASSERT_NE(op1, nullptr);
	ASSERT_NE(op2, nullptr);

	// Check that op1 is a parent of op2
	EXPECT_TRUE(std::find(op1->consumers.begin(), op1->consumers.end(), op2) != op1->consumers.end());

	// Check dependency count
	EXPECT_EQ(op1->dependency_count, 0);  // op1 has no dependencies
	EXPECT_EQ(op2->dependency_count, 1);  // op2 depends on op1
}

/**
 * @brief Test ready pool initialization with source nodes
 *
 * @details
 * Validates that the ready pool is correctly initialized with only source nodes
 * (nodes with zero dependencies). The ready pool is critical for topological
 * scheduling as it contains nodes that can execute immediately.
 *
 * **Test Steps:**
 * 1. Create a basic linear graph (op1 -> op2 -> op3)
 * 2. Initialize the ready pool
 * 3. Verify ready pool contains exactly one node
 * 4. Verify that node is op1 (the only source node)
 *
 * **Expected Behavior:**
 * - Only nodes with dependency_count == 0 should be in ready pool
 * - For basic graph, only op1 qualifies as a source node
 *
 * **Assertions:**
 * - EXPECT_EQ: Ready pool size must be 1
 * - EXPECT_EQ: First ready node must be "op1"
 *
 * @see graph::DAG::initializeReadyPool()
 * @see graph::DAG::readyPool
 */
TEST_F(DAGTest, ReadyPoolInitialization) {
	graph::DAG<MicroOp, SimTensor> graph = createBasicGraph("Test Graph");

	// Only op1 should be in the ready pool initially
	EXPECT_EQ(graph.readyPool.size(), 1);

	if (!graph.readyPool.empty()) { EXPECT_EQ(graph.readyPool[0]->getName(), "op1"); }
}

/**
 * @brief Test round-robin scheduling behavior across multiple graphs
 *
 * @details
 * Validates that the DAG scheduler implements fair round-robin scheduling when
 * managing multiple concurrent graphs. This ensures balanced progress across
 * all graphs rather than completing one graph before starting another.
 *
 * **Test Setup:**
 * - Two identical basic graphs with different prefixes (G1_, G2_)
 * - Each graph has 3 nodes in linear dependency chain
 * - Both graphs initialized with one ready node each
 *
 * **Test Steps:**
 * 1. Create two basic graphs with identifiable prefixes
 * 2. Initialize scheduler with both graphs
 * 3. Run scheduler to completion
 * 4. Verify execution order shows round-robin pattern
 * 5. Check that nodes alternate between graphs when possible
 *
 * **Expected Execution Order:**
 * @code
 * G1_op1, G2_op1,  // First nodes from each graph
 * G1_op2, G2_op2,  // Second nodes alternate
 * G1_op3, G2_op3   // Final nodes alternate
 * @endcode
 *
 * **Assertions:**
 * - ASSERT_FALSE: Execution order must not be empty
 * - EXPECT_EQ: Specific positions verify round-robin pattern
 * - EXPECT_EQ: Total execution count must be 6 (3 ops × 2 graphs)
 *
 * @see DAGSchedulerTest::TestDAGScheduler
 * @see graph::DAGScheduler::run()
 */
TEST_F(DAGSchedulerTest, RoundRobinMultiGraphScheduling) {
	// Create two simple graphs with different prefixes for easy identification
	graph::DAG<MicroOp, SimTensor> graph1 = createBasicGraph("Graph1", "G1_");
	graph::DAG<MicroOp, SimTensor> graph2 = createBasicGraph("Graph2", "G2_");

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {graph1, graph2};

	// Create scheduler with 2 graphs
	TestDAGScheduler scheduler(2);
	scheduler.run(graphs);

	// Check that nodes from both graphs are in the execution order
	ASSERT_FALSE(scheduler.executionOrder.empty());

	// Verify round-robin behavior: nodes should alternate between graphs when possible
	// In this case, only one node from each graph is initially ready
	EXPECT_EQ(scheduler.executionOrder[0], "G1_op1");
	EXPECT_EQ(scheduler.executionOrder[1], "G2_op1");

	// Verify that ready nodes were properly tracked and processed in each graph
	EXPECT_EQ(scheduler.executionOrder[2], "G1_op2");
	EXPECT_EQ(scheduler.executionOrder[3], "G2_op2");
	EXPECT_EQ(scheduler.executionOrder[4], "G1_op3");
	EXPECT_EQ(scheduler.executionOrder[5], "G2_op3");

	// Verify total executed nodes
	EXPECT_EQ(scheduler.executionOrder.size(), 6);  // 3 ops from each graph
}

/**
 * @brief Test scheduling with complex graph topologies (diamond pattern)
 *
 * @details
 * Validates scheduler behavior with more complex graph structures, specifically
 * testing a diamond-shaped dependency graph alongside a basic linear graph.
 * This ensures the scheduler correctly handles multiple dependencies and
 * parallel execution paths.
 *
 * **Test Setup:**
 * - Diamond graph: 4 nodes with parallel paths converging
 *   @code
 *       D_opA
 *       /   \
 *   D_opB   D_opC
 *       \   /
 *       D_opD
 *   @endcode
 * - Basic graph: 3 nodes in linear chain (B_op1 -> B_op2 -> B_op3)
 *
 * **Test Steps:**
 * 1. Create diamond and basic graphs with distinct prefixes
 * 2. Run scheduler with both graphs
 * 3. Verify source nodes execute first (D_opA, B_op1)
 * 4. Verify total execution count is correct (7 nodes)
 * 5. Validate topological ordering: D_opB and D_opC before D_opD
 *
 * **Critical Constraint:**
 * D_opD must execute AFTER both D_opB AND D_opC complete
 * (tests multi-dependency resolution)
 *
 * **Assertions:**
 * - ASSERT_FALSE: Execution order must not be empty
 * - EXPECT_EQ: First two nodes must be source nodes
 * - EXPECT_EQ: Total count must be 7 (4 + 3)
 * - EXPECT_LT: Parent positions must precede child positions
 *
 * @see DAGTest::createDiamondGraph()
 * @see DAGTest::createBasicGraph()
 */
TEST_F(DAGSchedulerTest, ComplexGraphScheduling) {
	// Create a diamond graph and a basic graph
	graph::DAG<MicroOp, SimTensor> diamondGraph = createDiamondGraph("DiamondGraph", "D_");
	graph::DAG<MicroOp, SimTensor> basicGraph   = createBasicGraph("BasicGraph", "B_");

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {diamondGraph, basicGraph};

	// Create scheduler
	TestDAGScheduler scheduler(2);
	scheduler.run(graphs);

	// Check execution order
	ASSERT_FALSE(scheduler.executionOrder.empty());

	// Verify first two nodes (should be source nodes from both graphs)
	EXPECT_EQ(scheduler.executionOrder[0], "D_opA");
	EXPECT_EQ(scheduler.executionOrder[1], "B_op1");

	// Total execution count should be 7 (4 from diamond + 3 from basic)
	EXPECT_EQ(scheduler.executionOrder.size(), 7);

	// Check that diamond middle layer nodes were executed before sink node
	auto posB = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "D_opB");
	auto posC = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "D_opC");
	auto posD = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "D_opD");

	ASSERT_NE(posB, scheduler.executionOrder.end());
	ASSERT_NE(posC, scheduler.executionOrder.end());
	ASSERT_NE(posD, scheduler.executionOrder.end());

	// Verify D comes after both B and C
	EXPECT_LT(posB, posD);
	EXPECT_LT(posC, posD);
}

/**
 * @brief Test tiled GEMM graph scheduling with complex dependencies
 *
 * @details
 * Validates scheduler behavior with a realistic workload: tiled matrix multiplication.
 * This 20-node graph represents a complex computation with load, compute, and store
 * stages, testing the scheduler's ability to handle intricate dependency patterns.
 *
 * **Graph Structure:**
 * Tiled GEMM (General Matrix Multiply) computation graph with:
 * - Load operations: Reading matrix tiles A and B
 * - Compute operations: GEMM operations on tiles
 * - Store operations: Writing result tiles C
 *
 * **Dependency Pattern Example:**
 * @code
 * loadA00, loadB00 -> gemmC00-1 -> loadA01, loadB10 -> gemmC00-2
 *                                                    |
 *                                            ... (more dependencies)
 *                                                    |
 *                                                 storeC11
 * @endcode
 *
 * **Test Steps:**
 * 1. Create tiled GEMM graph with 20 operations
 * 2. Run scheduler to completion
 * 3. Verify all 20 nodes execute
 * 4. Validate critical dependencies:
 *    - GEMM operations execute after their load dependencies
 *    - Store operations execute last
 *    - storeC11 is the final operation
 *
 * **Assertions:**
 * - ASSERT_FALSE: Execution must occur
 * - EXPECT_EQ: Total node count must be 20
 * - EXPECT_LT: Load operations before dependent GEMMs
 * - EXPECT_EQ: storeC11 must be final operation
 *
 * @see DAGTest::createTiledGEMMGraph()
 */
TEST_F(DAGSchedulerTest, TiledGEMMScheduling) {
	graph::DAG<MicroOp, SimTensor> gemmGraph = createTiledGEMMGraph("GEMMGraph", "G_");

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {gemmGraph};

	// Create scheduler
	TestDAGScheduler scheduler(1);
	scheduler.run(graphs);

	// Check that execution happened
	ASSERT_FALSE(scheduler.executionOrder.empty());

	// Total nodes should be 20 (as defined in createTiledGEMMGraph)
	EXPECT_EQ(scheduler.executionOrder.size(), 20);

	// Check a few key dependencies in the execution order
	auto posLoadA00   = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "G_loadA00");
	auto posLoadB00   = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "G_loadB00");
	auto posGemmC00_1 = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "G_gemmC00-1");
	auto posStoreC11  = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "G_storeC11");

	ASSERT_NE(posLoadA00, scheduler.executionOrder.end());
	ASSERT_NE(posLoadB00, scheduler.executionOrder.end());
	ASSERT_NE(posGemmC00_1, scheduler.executionOrder.end());
	ASSERT_NE(posStoreC11, scheduler.executionOrder.end());

	// Verify dependencies: gemm
	// Verify dependencies: gemmC00-1 must come after loadA00 and loadB00
	EXPECT_LT(posLoadA00, posGemmC00_1);
	EXPECT_LT(posLoadB00, posGemmC00_1);

	// storeC11 should be the last operation
	EXPECT_EQ(posStoreC11, scheduler.executionOrder.end() - 1);
}

/**
 * @brief Test scheduler behavior with empty graphs (edge case)
 *
 * @details
 * Validates that the scheduler gracefully handles empty graphs without crashing
 * or producing invalid execution orders. This edge case ensures robustness when
 * no operations are scheduled.
 *
 * **Test Steps:**
 * 1. Create two empty DAG instances (no nodes or edges)
 * 2. Initialize scheduler with empty graphs
 * 3. Run scheduler
 * 4. Verify execution order is empty
 * 5. Ensure no crashes or undefined behavior
 *
 * **Expected Behavior:**
 * - No nodes should be executed
 * - Execution order should be empty vector
 * - Scheduler should return without errors
 *
 * **Assertions:**
 * - EXPECT_TRUE: Execution order must be empty
 *
 * @see graph::DAG constructor
 * @see graph::DAGScheduler::run()
 */
TEST_F(DAGSchedulerTest, EmptyGraphScheduling) {
	// Create empty graphs
	graph::DAG<MicroOp, SimTensor> emptyGraph1("EmptyGraph1");
	graph::DAG<MicroOp, SimTensor> emptyGraph2("EmptyGraph2");

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {emptyGraph1, emptyGraph2};

	// Create scheduler
	TestDAGScheduler scheduler(2);
	scheduler.run(graphs);

	// Execution order should be empty
	EXPECT_TRUE(scheduler.executionOrder.empty());
}

/**
 * @brief Test handling of concurrent ready nodes from the same graph
 *
 * @details
 * Validates scheduler behavior when multiple nodes become ready simultaneously
 * within a single graph. This tests the scheduler's ability to manage parallel
 * execution opportunities and maintain correct dependency ordering.
 *
 * **Graph Structure:**
 * @code
 *   opA (ready)    opB (ready)
 *    |              |
 *   tA             tB
 *    |              |
 *   opC            opD
 * @endcode
 * Two independent paths that can execute concurrently.
 *
 * **Test Steps:**
 * 1. Create custom graph with two source nodes (opA, opB)
 * 2. Each source has one dependent (opC depends on opA, opD on opB)
 * 3. Initialize ready pool (should contain both opA and opB)
 * 4. Run scheduler
 * 5. Verify all 4 nodes execute
 * 6. Validate dependency constraints are maintained
 *
 * **Critical Constraints:**
 * - opA and opB can execute in any order (both initially ready)
 * - opC must execute AFTER opA
 * - opD must execute AFTER opB
 *
 * **Assertions:**
 * - EXPECT_EQ: Total execution count must be 4
 * - EXPECT_NE: opA and opB must be in first two positions
 * - EXPECT_LT: Parent must execute before child in each path
 *
 * @see graph::DAG::initializeReadyPool()
 * @see graph::DAG::addDependency()
 */
TEST_F(DAGSchedulerTest, ConcurrentReadyNodesHandling) {
	// Create a custom graph with multiple initial ready nodes
	graph::DAG<MicroOp, SimTensor> graph("ConcurrentGraph");

	// Create nodes
	MicroOp* opA = new MicroOp("opA");
	MicroOp* opB = new MicroOp("opB");
	MicroOp* opC = new MicroOp("opC");
	MicroOp* opD = new MicroOp("opD");

	// Create tensors
	SimTensor* tA = new SimTensor("tA");
	SimTensor* tB = new SimTensor("tB");
	SimTensor* tC = new SimTensor("tC");
	SimTensor* tD = new SimTensor("tD");

	// Register tensors
	graph.addEdge(tA);
	graph.addEdge(tB);
	graph.addEdge(tC);
	graph.addEdge(tD);

	// Add operations with their tensors - A and B have no dependencies
	graph.addNode(opA, {}, {tA});
	graph.addNode(opB, {}, {tB});
	graph.addNode(opC, {tA}, {tC});
	graph.addNode(opD, {tB}, {tD});

	// Add dependencies - C depends on A, D depends on B
	graph.addDependency(opA, opC);
	graph.addDependency(opB, opD);

	graph.initializeReadyPool();

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {graph};

	// Create scheduler
	TestDAGScheduler scheduler(1);
	scheduler.run(graphs);

	// Should have 4 nodes in execution order
	EXPECT_EQ(scheduler.executionOrder.size(), 4);

	// First two should be A and B (could be in either order)
	auto posA = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.begin() + 2, "opA");
	auto posB = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.begin() + 2, "opB");

	EXPECT_NE(posA, scheduler.executionOrder.begin() + 2);
	EXPECT_NE(posB, scheduler.executionOrder.begin() + 2);

	// C must come after A
	auto posC = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "opC");
	ASSERT_NE(posC, scheduler.executionOrder.end());
	EXPECT_LT(posA, posC);

	// D must come after B
	auto posD = std::find(scheduler.executionOrder.begin(), scheduler.executionOrder.end(), "opD");
	ASSERT_NE(posD, scheduler.executionOrder.end());
	EXPECT_LT(posB, posD);
}

/**
 * @brief Test priority-based scheduling across multiple graphs
 *
 * @details
 * Validates that the priority scheduler correctly prioritizes graph execution
 * based on assigned priorities. Higher-priority graphs should have their nodes
 * executed before lower-priority graphs.
 *
 * **Test Setup:**
 * - High-priority graph (priority=10): 3 operations with prefix "High_"
 * - Low-priority graph (priority=1): 3 operations with prefix "Low_"
 * - Both graphs added to scheduler in reverse priority order (tests sorting)
 *
 * **Test Steps:**
 * 1. Create two identical basic graphs
 * 2. Assign different priorities (10 vs 1)
 * 3. Add to scheduler in low-to-high order
 * 4. Run scheduler
 * 5. Verify high-priority graph executes completely first
 *
 * **Expected Execution Order:**
 * @code
 * High_op1, High_op2, High_op3,  // High-priority graph completes first
 * Low_op1,  Low_op2,  Low_op3    // Then low-priority graph
 * @endcode
 *
 * **Scheduling Policy:**
 * Priority-based scheduling ensures critical workloads execute before
 * less important ones, useful for QoS guarantees in simulation.
 *
 * **Assertions:**
 * - ASSERT_FALSE: Execution order must not be empty
 * - EXPECT_TRUE: First three nodes must have "High_" prefix
 * - EXPECT_TRUE: Last three nodes must have "Low_" prefix
 *
 * @see graph::DAG::setPriority()
 * @see PriorityTestScheduler::TestSchedulerPriority
 */
TEST_F(PriorityTestScheduler, PriorityBasedScheduling) {
	// Create two graphs with different priorities
	graph::DAG<MicroOp, SimTensor> highPriorityGraph = createBasicGraph("HighPriority", "High_");
	graph::DAG<MicroOp, SimTensor> lowPriorityGraph  = createBasicGraph("LowPriority", "Low_");

	// Set priorities (assuming DAG class has priority field)
	highPriorityGraph.setPriority(10);
	lowPriorityGraph.setPriority(1);

	std::vector<graph::DAG<MicroOp, SimTensor>> graphs = {lowPriorityGraph, highPriorityGraph};

	TestSchedulerPriority scheduler(2);
	scheduler.run(graphs);

	// Check that higher priority graph's nodes are executed first
	ASSERT_FALSE(scheduler.executionOrder.empty());

	// Check the first three should be from high priority graph
	EXPECT_TRUE(scheduler.executionOrder[0].find("High_") == 0);
	EXPECT_TRUE(scheduler.executionOrder[1].find("High_") == 0);
	EXPECT_TRUE(scheduler.executionOrder[2].find("High_") == 0);

	// Then the low priority nodes should execute
	EXPECT_TRUE(scheduler.executionOrder[3].find("Low_") == 0);
	EXPECT_TRUE(scheduler.executionOrder[4].find("Low_") == 0);
	EXPECT_TRUE(scheduler.executionOrder[5].find("Low_") == 0);
}

/**
 * @brief Test graph merging functionality
 *
 * @details
 * Validates the ability to merge multiple DAGs into a single unified graph.
 * This functionality is critical for combining subgraphs or creating composite
 * computational workflows.
 *
 * **Test Setup:**
 * - Graph1: 3 nodes (G1_op1 -> G1_op2 -> G1_op3) with 4 tensors
 * - Graph2: 3 nodes (G2_op1 -> G2_op2 -> G2_op3) with 4 tensors
 * - Create empty merged graph and merge both graphs into it
 *
 * **Test Steps:**
 * 1. Create two independent basic graphs with unique prefixes
 * 2. Create new empty graph for merging
 * 3. Merge graph1 into merged graph
 * 4. Merge graph2 into merged graph
 * 5. Verify all nodes from both graphs exist in merged graph
 * 6. Verify all edges from both graphs exist in merged graph
 * 7. Verify dependencies are preserved after merging
 * 8. Verify ready pool contains source nodes from both graphs
 *
 * **Expected Results:**
 * - Merged graph contains all 6 nodes (3 from each source graph)
 * - Merged graph contains all 8 tensors (4 from each source graph)
 * - All parent-child relationships preserved
 * - Ready pool has 2 nodes (G1_op1 and G2_op1)
 *
 * **Assertions:**
 * - EXPECT_NE: All node lookups must succeed (non-null)
 * - EXPECT_NE: All edge lookups must succeed (non-null)
 * - EXPECT_TRUE: Dependencies must be preserved
 * - EXPECT_EQ: Ready pool size must be 2
 *
 * @see graph::DAG::merge()
 * @see graph::DAG::getNode()
 * @see graph::DAG::getEdge()
 */
TEST_F(DAGTest, GraphMerging) {
	graph::DAG<MicroOp, SimTensor> graph1 = createBasicGraph("Graph1", "G1_");
	graph::DAG<MicroOp, SimTensor> graph2 = createBasicGraph("Graph2", "G2_");

	// Create a merged graph
	graph::DAG<MicroOp, SimTensor> mergedGraph("MergedGraph");
	mergedGraph.merge(graph1);
	mergedGraph.merge(graph2);

	// Check that all nodes from both graphs are in the merged graph
	EXPECT_NE(mergedGraph.getNode("G1_op1"), nullptr);
	EXPECT_NE(mergedGraph.getNode("G1_op2"), nullptr);
	EXPECT_NE(mergedGraph.getNode("G1_op3"), nullptr);
	EXPECT_NE(mergedGraph.getNode("G2_op1"), nullptr);
	EXPECT_NE(mergedGraph.getNode("G2_op2"), nullptr);
	EXPECT_NE(mergedGraph.getNode("G2_op3"), nullptr);

	// Check that all edges from both graphs are in the merged graph
	EXPECT_NE(mergedGraph.getEdge("G1_t1"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G1_t2"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G1_t3"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G1_t4"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G2_t1"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G2_t2"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G2_t3"), nullptr);
	EXPECT_NE(mergedGraph.getEdge("G2_t4"), nullptr);

	// Check that dependencies are preserved
	MicroOp* g1Op1 = mergedGraph.getNode("G1_op1");
	MicroOp* g1Op2 = mergedGraph.getNode("G1_op2");
	ASSERT_NE(g1Op1, nullptr);
	ASSERT_NE(g1Op2, nullptr);

	// Op1 should be a parent of Op2
	EXPECT_TRUE(std::find(g1Op1->consumers.begin(), g1Op1->consumers.end(), g1Op2) != g1Op1->consumers.end());

	// Ready pool should include both graphs' source nodes
	EXPECT_EQ(mergedGraph.readyPool.size(), 2);
}

/**
 * @brief Main entry point for GoogleTest test suite
 *
 * @details
 * Initializes the GoogleTest framework and executes all registered tests.
 * This function processes command-line arguments for test filtering and
 * configuration, then runs the complete test suite.
 *
 * **Command-Line Options:**
 * - --gtest_filter=PATTERN: Run only tests matching pattern
 * - --gtest_repeat=COUNT: Repeat tests COUNT times
 * - --gtest_verbose=1: Enable verbose output
 * - --gtest_list_tests: List all available tests
 *
 * **Usage Examples:**
 * @code{.sh}
 * # Run all tests
 * ./testDAG
 *
 * # Run only DAGTest fixture tests
 * ./testDAG --gtest_filter=DAGTest.*
 *
 * # Run specific test
 * ./testDAG --gtest_filter=DAGTest.NodeCreationAndLookup
 *
 * # Repeat tests 5 times
 * ./testDAG --gtest_repeat=5
 * @endcode
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 if all tests pass, non-zero otherwise
 *
 * @see testing::InitGoogleTest()
 * @see RUN_ALL_TESTS()
 */
int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
