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
 * @file DAGSchedulerTest.cc
 * @brief Implementation of DAG test fixture helper functions and graph generators
 *
 * @details
 * This file implements the helper methods for the DAG test fixtures, providing
 * predefined graph topologies used across the test suite. These graph generators
 * create consistent, well-defined DAG structures for testing scheduler behavior,
 * dependency resolution, and topological sorting algorithms.
 *
 * ## Purpose
 *
 * The primary purpose of this file is to provide reusable graph construction
 * utilities that:
 * - Reduce code duplication across test cases
 * - Ensure consistent graph structures for reproducible testing
 * - Provide well-documented reference topologies
 * - Enable easy creation of graphs with identifiable node/edge names
 *
 * ## Graph Generators Implemented
 *
 * ### 1. createBasicGraph()
 * Creates a simple three-node linear dependency chain suitable for testing
 * fundamental scheduling and dependency tracking.
 *
 * ### 2. createDiamondGraph()
 * Generates a diamond-shaped graph with parallel paths converging at a sink,
 * testing multi-dependency resolution.
 *
 * ### 3. createTiledGEMMGraph()
 * Constructs a complex 20-node graph representing tiled matrix multiplication,
 * simulating realistic computational workloads.
 *
 * ## Design Patterns
 *
 * All graph generators follow a consistent pattern:
 * 1. Create node (MicroOp) objects
 * 2. Create edge (SimTensor) objects
 * 3. Register edges with the graph
 * 4. Add nodes with their input/output edges
 * 5. Establish dependencies between nodes
 * 6. Initialize the ready pool
 * 7. Return the constructed graph
 *
 * ## Naming Conventions
 *
 * Graphs support optional prefix parameters for node/edge identification:
 * - Enables multiple instances of the same graph type
 * - Facilitates tracking execution order in multi-graph tests
 * - Prevents name collisions when merging graphs
 *
 * Example: createBasicGraph("Graph1", "G1_") produces nodes:
 * G1_op1, G1_op2, G1_op3
 *
 * ## Memory Management
 *
 * Node and tensor objects are allocated with `new` and ownership is transferred
 * to the DAG. The DAG is responsible for cleanup through its destructor.
 *
 * @see DAGSchedulerTest.hh - Test fixture class definitions
 * @see main.cc - Test implementations using these helpers
 * @see workloads/graph/DAG.hh - DAG implementation
 *
 * @author ACAL/Playlab
 * @date 2023-2025
 * @version 1.0
 */

#include "DAGSchedulerTest.hh"

/**
 * @brief Create a basic three-node linear dependency graph
 *
 * @details
 * Constructs a simple sequential graph with three operations connected in a
 * linear chain. This is the fundamental test graph used for basic scheduler
 * validation and dependency tracking tests.
 *
 * **Graph Structure:**
 * @code
 * op1 (source)
 *  |
 *  t1 (output)
 *  |
 * op2 (intermediate)
 *  |
 * t2, t3 (outputs)
 *  |
 * op3 (sink)
 *  |
 *  t4 (output)
 * @endcode
 *
 * **Node Details:**
 * - op1: Source node, no dependencies, produces tensor t1
 * - op2: Intermediate node, consumes t1, produces t2 and t3
 * - op3: Sink node, consumes t2 and t3, produces t4
 *
 * **Dependency Chain:**
 * op1 -> op2 -> op3
 *
 * **Use Cases:**
 * - Testing basic topological sort
 * - Validating sequential execution order
 * - Testing ready pool initialization (only op1 initially ready)
 * - Verifying dependency count tracking
 *
 * @param graphName Name identifier for the DAG instance
 * @param prefix Optional prefix for all node and tensor names (default: "")
 * @return Fully constructed and initialized DAG with 3 nodes and 4 tensors
 *
 * @note The returned graph has its ready pool initialized with op1
 *
 * @see graph::DAG::addNode()
 * @see graph::DAG::addEdge()
 * @see graph::DAG::addDependency()
 * @see graph::DAG::initializeReadyPool()
 */
graph::DAG<MicroOp, SimTensor> DAGTest::createBasicGraph(const std::string& graphName, const std::string& prefix) {
	graph::DAG<MicroOp, SimTensor> graph(graphName);

	MicroOp* op1 = new MicroOp(prefix + "op1");
	MicroOp* op2 = new MicroOp(prefix + "op2");
	MicroOp* op3 = new MicroOp(prefix + "op3");

	SimTensor* t1 = new SimTensor(prefix + "t1");
	SimTensor* t2 = new SimTensor(prefix + "t2");
	SimTensor* t3 = new SimTensor(prefix + "t3");
	SimTensor* t4 = new SimTensor(prefix + "t4");

	graph.addEdge(t1);
	graph.addEdge(t2);
	graph.addEdge(t3);
	graph.addEdge(t4);

	graph.addNode(op1, {}, {t1});
	graph.addNode(op2, {t1}, {t2, t3});
	graph.addNode(op3, {t2, t3}, {t4});

	graph.addDependency(op1, op2);
	graph.addDependency(op2, op3);

	graph.initializeReadyPool();

	return graph;
}

/**
 * @brief Create a diamond-shaped graph with parallel paths
 *
 * @details
 * Constructs a four-node graph with a diamond topology, featuring two parallel
 * execution paths that converge at a sink node. This topology is critical for
 * testing multi-dependency resolution and parallel path handling.
 *
 * **Graph Structure:**
 * @code
 *         opA (source)
 *        /   \
 *   tA_out1  tA_out2
 *      /       \
 *    opB       opC (parallel middle layer)
 *      |        |
 *   tB_out   tC_out
 *      \       /
 *       \     /
 *        opD (sink - depends on both opB and opC)
 *         |
 *      tD_out
 * @endcode
 *
 * **Node Details:**
 * - opA: Source node, no dependencies, produces two output tensors
 * - opB: Left path node, consumes tB_in, produces tB_out
 * - opC: Right path node, consumes tC_in, produces tC_out
 * - opD: Sink node, consumes both tD_in1 and tD_in2, produces tD_out
 *
 * **Dependency Relationships:**
 * - opA -> opB (A must complete before B)
 * - opA -> opC (A must complete before C)
 * - opB -> opD (B must complete before D)
 * - opC -> opD (C must complete before D)
 *
 * **Critical Test Property:**
 * opD cannot execute until BOTH opB AND opC have completed, testing the
 * scheduler's ability to handle nodes with multiple dependencies.
 *
 * **Use Cases:**
 * - Testing multi-dependency resolution
 * - Validating parallel path execution
 * - Ensuring convergence nodes wait for all parents
 * - Testing scheduler fairness across parallel branches
 *
 * @param graphName Name identifier for the DAG instance
 * @param prefix Optional prefix for all node and tensor names (default: "")
 * @return Fully constructed and initialized DAG with 4 nodes and 9 tensors
 *
 * @note The returned graph has its ready pool initialized with only opA
 *
 * @see graph::DAG::addDependency()
 * @see graph::DAG::initializeReadyPool()
 */
graph::DAG<MicroOp, SimTensor> DAGTest::createDiamondGraph(const std::string& graphName, const std::string& prefix) {
	graph::DAG<MicroOp, SimTensor> graph(graphName);

	// Create nodes for a diamond pattern:
	//      A
	//     / \
        //    B   C
	//     \ /
	//      D
	MicroOp* opA = new MicroOp(prefix + "opA");
	MicroOp* opB = new MicroOp(prefix + "opB");
	MicroOp* opC = new MicroOp(prefix + "opC");
	MicroOp* opD = new MicroOp(prefix + "opD");

	// Create tensors
	SimTensor* tA_out1 = new SimTensor(prefix + "tA_out1");
	SimTensor* tA_out2 = new SimTensor(prefix + "tA_out2");
	SimTensor* tB_in   = new SimTensor(prefix + "tB_in");
	SimTensor* tB_out  = new SimTensor(prefix + "tB_out");
	SimTensor* tC_in   = new SimTensor(prefix + "tC_in");
	SimTensor* tC_out  = new SimTensor(prefix + "tC_out");
	SimTensor* tD_in1  = new SimTensor(prefix + "tD_in1");
	SimTensor* tD_in2  = new SimTensor(prefix + "tD_in2");
	SimTensor* tD_out  = new SimTensor(prefix + "tD_out");

	// Register tensors
	graph.addEdge(tA_out1);
	graph.addEdge(tA_out2);
	graph.addEdge(tB_in);
	graph.addEdge(tB_out);
	graph.addEdge(tC_in);
	graph.addEdge(tC_out);
	graph.addEdge(tD_in1);
	graph.addEdge(tD_in2);
	graph.addEdge(tD_out);

	// Add operations with their tensors
	graph.addNode(opA, {}, {tA_out1, tA_out2});
	graph.addNode(opB, {tB_in}, {tB_out});
	graph.addNode(opC, {tC_in}, {tC_out});
	graph.addNode(opD, {tD_in1, tD_in2}, {tD_out});

	// Add dependencies
	graph.addDependency(opA, opB);  // A -> B
	graph.addDependency(opA, opC);  // A -> C
	graph.addDependency(opB, opD);  // B -> D
	graph.addDependency(opC, opD);  // C -> D

	graph.initializeReadyPool();

	return graph;
}

/**
 * @brief Create a complex tiled GEMM (matrix multiplication) graph
 *
 * @details
 * Constructs a realistic 20-node graph representing tiled matrix multiplication,
 * a common pattern in high-performance computing and deep learning. This graph
 * simulates a blocked matrix multiply operation with separate load, compute,
 * and store phases.
 *
 * **Algorithm Overview:**
 * Tiled GEMM divides large matrices into smaller tiles that fit in cache,
 * computing: C = A × B using blocked operations. For a 2×2 tile configuration:
 * @code
 * C00 = A00×B00 + A01×B10
 * C01 = A00×B01 + A01×B11
 * C10 = A10×B00 + A11×B10
 * C11 = A10×B01 + A11×B11
 * @endcode
 *
 * **Graph Stages:**
 *
 * 1. **Load Stage**: Load matrix tiles from memory
 *    - loadA00, loadA01, loadA10, loadA11 (matrix A tiles)
 *    - loadB00, loadB01, loadB10, loadB11 (matrix B tiles)
 *
 * 2. **Compute Stage**: Perform GEMM operations on tiles
 *    - gemmC00-1, gemmC00-2 (compute C00 in two accumulation steps)
 *    - gemmC01-1, gemmC01-2 (compute C01)
 *    - gemmC10-1, gemmC10-2 (compute C10)
 *    - gemmC11-1, gemmC11-2 (compute C11)
 *
 * 3. **Store Stage**: Write result tiles back to memory
 *    - storeC00, storeC01, storeC10, storeC11
 *
 * **Critical Dependencies:**
 * @code
 * loadA00, loadB00 -> gemmC00-1 (first accumulation needs both inputs)
 * gemmC00-1 -> loadA01, loadB10 (trigger next tile loads)
 * loadA01, loadB10 -> gemmC00-2 (second accumulation)
 * gemmC00-2 -> storeC00 (store when computation complete)
 * @endcode
 *
 * **Complete Operation List (20 nodes):**
 * 1. loadA00    2. loadB00    3. gemmC00-1  4. loadA01    5. loadB10
 * 6. gemmC00-2  7. loadB01    8. gemmC01-1  9. loadB11   10. storeC00
 * 11. gemmC01-2 12. loadA10   13. gemmC10-1 14. loadA11  15. storeC01
 * 16. gemmC10-2 17. gemmC11-1 18. storeC10  19. gemmC11-2 20. storeC11
 *
 * **Dependency Complexity:**
 * The graph contains 25+ dependency relationships, testing the scheduler's
 * ability to handle:
 * - Multi-stage pipelines (load -> compute -> store)
 * - Data dependencies between computation phases
 * - Multiple consumers for single producers
 * - Long dependency chains
 *
 * **Use Cases:**
 * - Testing complex dependency resolution
 * - Validating scheduler with realistic workloads
 * - Stress testing topological sort algorithms
 * - Benchmarking scheduler performance with large graphs
 *
 * @param graphName Name identifier for the DAG instance
 * @param prefix Optional prefix for all node and tensor names (default: "")
 * @return Fully constructed DAG with 20 nodes and 40 tensors (2 per node)
 *
 * @note The returned graph has its ready pool initialized with loadA00 and loadB00
 *
 * @see graph::DAG::addDependency()
 * @see MicroOp - Operation node representation
 * @see SimTensor - Data edge representation
 */
graph::DAG<MicroOp, SimTensor> DAGTest::createTiledGEMMGraph(const std::string& graphName, const std::string& prefix) {
	graph::DAG<MicroOp, SimTensor> graph(graphName);

	// Define operation nodes
	std::vector<MicroOp*>   nodes;
	std::vector<SimTensor*> tensors;

	// Define operation names for each stage of tiled GEMM
	std::vector<std::string> nodeNames = {"loadA00",   "loadB00",   "gemmC00-1", "loadA01",   "loadB10",
	                                      "gemmC00-2", "loadB01",   "gemmC01-1", "loadB11",   "storeC00",
	                                      "gemmC01-2", "loadA10",   "gemmC10-1", "loadA11",   "storeC01",
	                                      "gemmC10-2", "gemmC11-1", "storeC10",  "gemmC11-2", "storeC11"};

	// Create all operation nodes
	for (const auto& name : nodeNames) { nodes.push_back(new MicroOp(prefix + name)); }

	// Create tensors and add operations to graph
	for (size_t i = 0; i < nodes.size(); i++) {
		// Create named input and output tensors
		std::string inName       = prefix + "in_" + nodes[i]->name;
		std::string outName      = prefix + "out_" + nodes[i]->name;
		SimTensor*  inputTensor  = new SimTensor(inName);
		SimTensor*  outputTensor = new SimTensor(outName);

		tensors.push_back(inputTensor);
		tensors.push_back(outputTensor);

		// Register tensors with graph
		graph.addEdge(inputTensor);
		graph.addEdge(outputTensor);

		// Add operation with its tensors
		graph.addNode(nodes[i], {inputTensor}, {outputTensor});
	}

	// Define operation dependencies for tiled GEMM pattern
	std::vector<std::pair<int, int>> dependencies = {
	    {0, 2},   {1, 2},    // loadA00, loadB00 -> gemmC00-1
	    {2, 3},   {2, 4},    // gemmC00-1 -> loadA01, loadB10
	    {3, 5},   {4, 5},    // loadA01, loadB10 -> gemmC00-2
	    {5, 6},              // gemmC00-2 -> loadB01
	    {0, 7},   {6, 7},    // loadA00, loadB01 -> gemmC01-1
	    {7, 8},   {7, 9},    // gemmC01-1 -> loadB11, storeC00
	    {3, 10},  {6, 10},   // loadA01, loadB01 -> gemmC01-2
	    {10, 11},            // gemmC01-2 -> loadA10
	    {4, 12},  {11, 12},  // loadB10, loadA10 -> gemmC10-1
	    {12, 13}, {12, 14},  // gemmC10-1 ->
	    {13, 14},            // gemmC10-1 -> loadA11, storeC01
	    {6, 15},  {13, 15},  // loadB01, loadA11 -> gemmC10-2
	    {4, 16},  {15, 16},  // loadB10, gemmC10-2 -> gemmC11-1
	    {16, 17}, {16, 18},  // gemmC11-1 -> storeC10, gemmC11-2
	    {8, 18},             // loadB11 -> gemmC11-2
	    {9, 17},             // storeC00 -> storeC10
	    {14, 19}, {18, 19}   // storeC01, gemmC11-2 -> storeC11
	};

	// Set up all dependencies
	for (auto& [parent, consumer] : dependencies) { graph.addDependency(nodes[parent], nodes[consumer]); }

	graph.initializeReadyPool();

	return graph;
}
