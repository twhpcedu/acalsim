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
 * @file testPETile.cc
 * @brief GoogleTest unit tests for PETile (Processing Element Tile) component validation
 *
 * @details
 * This test suite provides comprehensive validation of the PETile component, which is a
 * fundamental building block in the ACAL simulator architecture. The PETile represents
 * a processing element tile that integrates CPU traffic management, AXI bus communication,
 * and memory access patterns.
 *
 * # Purpose
 * The primary purpose of this test file is to verify:
 * - Correct operation of the PETile component in the simulator
 * - Proper communication between CPU traffic generator and AXI bus
 * - Memory response packet handling and validation
 * - Bit mask status checking mechanism for test validation
 *
 * # Test Coverage
 * The test suite covers the following areas:
 * - **PETile Basic Functionality**: Validates that the PETile can be instantiated and configured
 * - **CPU Memory Access**: Tests CPU traffic generation and memory request/response handling
 * - **AXI Bus Communication**: Verifies AXI bus protocol compliance and packet routing
 * - **Packet Return Validation**: Ensures all memory responses are correctly received
 * - **Status Bit Checking**: Validates the GTest bit mask mechanism for test pass/fail
 *
 * # Architecture Overview
 * The test architecture consists of:
 * @code
 * +-------------------------------------------------------------------+
 * |                         testPETile.cc                             |
 * |  (GoogleTest Framework + Test Configuration)                      |
 * +-------------------------------------------------------------------+
 *                                  |
 *                                  v
 * +-------------------------------------------------------------------+
 * |                         PETileTop                                 |
 * |  - Simulation top-level coordinator                               |
 * |  - Configuration loading from configs.json                        |
 * |  - Bit mask management for test status                            |
 * +-------------------------------------------------------------------+
 *                                  |
 *                                  v
 * +-------------------------------------------------------------------+
 * |                           PETile                                  |
 * |  - Processing Element Tile implementation                         |
 * |  - CPU traffic generator                                          |
 * |  - AXI bus interface                                              |
 * |  - Memory subsystem connection                                    |
 * +-------------------------------------------------------------------+
 *        |                         |                          |
 *        v                         v                          v
 * +--------------+        +------------------+       +------------------+
 * | CPUTraffic   |        |    AXI Bus       |       |  Memory System   |
 * | - Request    |  <-->  |  - Transactions  | <-->  |  - Responses     |
 * | - Response   |        |  - Routing       |       |  - Data storage  |
 * +--------------+        +------------------+       +------------------+
 * @endcode
 *
 * # Component Interactions
 *
 * ## Test Flow Sequence
 * 1. **Initialization Phase**:
 *    - Instantiate PETileTop with configuration file
 *    - Load simulation parameters from configs.json
 *    - Initialize CPU traffic generator and AXI bus
 *
 * 2. **Execution Phase**:
 *    - CPU generates memory read/write requests
 *    - Requests traverse AXI bus to memory system
 *    - Memory system processes requests and generates responses
 *    - Responses return through AXI bus to CPU
 *
 * 3. **Validation Phase**:
 *    - Each component sets corresponding bit in test mask when complete
 *    - CPUTraffic sets bit[0] when all responses received
 *    - AXIBus sets bit[1] when packet count matches
 *    - Test checks bits[1:0] == 0x3 for pass condition
 *
 * ## Bit Mask Status Encoding
 * The test uses a bit mask mechanism where:
 * - Bit 0: CPUTraffic received all expected memory responses
 * - Bit 1: AXI Bus validated correct packet routing and count
 * - Expected value: 0x3 (both bits set) indicates test pass
 *
 * # How to Run
 *
 * ## Building the Test
 * @code{.sh}
 * # From the project root directory
 * cd acalsim-workspace/projects/acalsim
 * mkdir -p build && cd build
 * cmake ..
 * make testPETile
 * @endcode
 *
 * ## Running the Test
 * @code{.sh}
 * # Run with default configuration
 * ./gtest/testPETile/testPETile
 *
 * # Run with verbose output
 * ./gtest/testPETile/testPETile --gtest_verbose
 *
 * # Run with specific test filter
 * ./gtest/testPETile/testPETile --gtest_filter=PETile.Check
 *
 * # Run with custom simulator arguments
 * ./gtest/testPETile/testPETile --config=custom_configs.json --verbose
 * @endcode
 *
 * ## Configuration File
 * The test requires a configuration file (configs.json) that specifies:
 * - Memory hierarchy parameters
 * - AXI bus configuration (width, latency, etc.)
 * - CPU traffic patterns (read/write ratios, address ranges)
 * - Simulation duration and termination conditions
 *
 * # Expected Outcomes
 *
 * ## Success Criteria
 * A successful test run will:
 * - Complete simulation without crashes or assertions
 * - Show CPUTraffic receiving all expected memory responses
 * - Validate AXI bus packet count matches expected value
 * - Return checkGTestBitMask(0, 0x3) == true
 * - Display "PASSED" status from GoogleTest framework
 *
 * ## Example Success Output
 * @code
 * [==========] Running 1 test from 1 test suite.
 * [----------] Global test environment set-up.
 * [----------] 1 test from PETile
 * [ RUN      ] PETile.Check
 * [INFO] Simulation starting...
 * [INFO] CPUTraffic: Generated 1000 memory requests
 * [INFO] AXIBus: Routed 1000 packets
 * [INFO] CPUTraffic: Received 1000 responses - bit[0] set
 * [INFO] AXIBus: Packet count validated - bit[1] set
 * [       OK ] PETile.Check (1234 ms)
 * [----------] 1 test from PETile (1234 ms total)
 * [==========] 1 test from 1 test suite ran. (1234 ms total)
 * [  PASSED  ] 1 test.
 * @endcode
 *
 * ## Failure Scenarios
 * The test will fail if:
 * - CPU doesn't receive expected number of memory responses (bit[0] not set)
 * - AXI bus packet count mismatch (bit[1] not set)
 * - Simulation hangs or deadlocks
 * - Configuration file not found or invalid
 *
 * ## Example Failure Output
 * @code
 * [ RUN      ] PETile.Check
 * [INFO] Simulation starting...
 * [INFO] CPUTraffic: Generated 1000 memory requests
 * [INFO] AXIBus: Routed 1000 packets
 * [WARN] CPUTraffic: Only received 995 responses - bit[0] NOT set
 * [INFO] AXIBus: Packet count validated - bit[1] set
 * testPETile.cc:36: Failure
 * Value of: top->checkGTestBitMask(0, 0x3)
 *   Actual: false
 * Expected: true
 * CPU doesn't receive the mem response
 * [  FAILED  ] PETile.Check (1234 ms)
 * @endcode
 *
 * # Test Implementation Details
 *
 * The test uses the GoogleTest framework TEST() macro:
 * @code{.cpp}
 * TEST(PETile, Check) {
 *     // Validates bits[1:0] are all set to 1
 *     // Bit 0: CPUTraffic received all responses
 *     // Bit 1: AXI bus validated packet count
 *     EXPECT_EQ(top->checkGTestBitMask(0, 0x3), true)
 *         << "CPU doesn't receive the mem response";
 * }
 * @endcode
 *
 * # Code Examples
 *
 * ## Basic Test Pattern
 * This demonstrates the standard pattern for creating PETile tests:
 * @code{.cpp}
 * #include <gtest/gtest.h>
 * #include "ACALSim.hh"
 * #include "PETile.hh"
 * #include "PETileTop.hh"
 *
 * // Test fixture for more complex scenarios
 * class PETileTest : public ::testing::Test {
 * protected:
 *     void SetUp() override {
 *         top = std::make_shared<PETileTop>("PESTSim", "configs.json");
 *         top->init(0, nullptr);
 *     }
 *
 *     void TearDown() override {
 *         top->finish();
 *     }
 * };
 *
 * // Additional test case example
 * TEST(PETile, MemoryBandwidth) {
 *     // Test memory bandwidth utilization
 *     EXPECT_GT(top->getMemoryBandwidthUtilization(), 0.8);
 * }
 * @endcode
 *
 * ## Custom Configuration Example
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *     // Parse custom config file from command line
 *     std::string config_file = "src/testPETile/configs.json";
 *     for (int i = 1; i < argc; i++) {
 *         if (std::string(argv[i]) == "--config" && i + 1 < argc) {
 *             config_file = argv[i + 1];
 *         }
 *     }
 *
 *     top = std::make_shared<PETileTop>("PESTSim", config_file);
 *     top->init(argc, argv);
 *     top->run();
 *     top->finish();
 *
 *     testing::InitGoogleTest(&argc, argv);
 *     return RUN_ALL_TESTS();
 * }
 * @endcode
 *
 * # Related Components
 *
 * @see PETile - The main processing element tile implementation
 * @see PETileTop - Top-level simulation coordinator for PETile
 * @see CPUTraffic - CPU traffic generator component
 * @see AXIBus - AXI bus communication protocol implementation
 * @see SimTop - Base class for simulation top-level
 * @see SimPacket - Base packet class for simulator communication
 *
 * # Performance Considerations
 *
 * - Test execution time depends on simulation duration in configs.json
 * - Typical test run: 1-5 seconds for basic validation
 * - Extended tests with more traffic can take 10-60 seconds
 * - Memory requirements scale with packet buffer sizes
 *
 * # Debugging Tips
 *
 * 1. **Enable Verbose Logging**: Use --verbose flag to see detailed packet traces
 * 2. **Check Bit Mask Values**: Print individual bits to isolate which component failed
 * 3. **Validate Configuration**: Ensure configs.json has valid memory addresses and sizes
 * 4. **Review Packet Counts**: Compare sent vs. received packet counts in logs
 * 5. **Check Timing**: Verify AXI bus latencies are configured correctly
 *
 * @author ACAL/Playlab Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This test requires a valid configuration file at src/testPETile/configs.json
 * @warning Ensure sufficient memory is available for packet buffering during test execution
 */

/* --------------------------------------------------------------------------------------
 *  An example to demonstrate how to create your own testbench for your application with google test framework.
 *  Step 1. Include <gtest/gtest.hh>
 *  Step 2. Create test fixture which allows you to reuse the same configuration of objects for several different tests.
 * 			Test fixture can help us build up the specific scenarios which we are interested in for testing the objects.
 *  Step 3. Write TEST_F() with assertion for checking whether the behavior of object corresponds to our expectation.
 *
 * --------------------------------------------------------------------------------------*/
#include <gtest/gtest.h>

#include "ACALSim.hh"
using namespace acalsim;

#include "PETile.hh"
#include "PETileTop.hh"
TEST(PETile, Check) {
	// The test is supposed to check bits[1:0] to all ones since CPUTraffic
	// and AXIBus need to check the number of return packets
	// each modules set the corresponding bits when the equirement is met
	EXPECT_EQ(top->checkGTestBitMask(0, 0x3), true) << "CPU doesn't receive the mem response";
}

int main(int argc, char** argv) {
	// instantiate your system
	top = std::make_shared<PETileTop>("PESTSim", "src/testPETile/configs.json");
	top->init(argc, argv);
	top->run();
	top->finish();

	// run google test framework
	testing::InitGoogleTest(&argc, argv);
	bool result = false;
	result      = RUN_ALL_TESTS();

	return result;
}
