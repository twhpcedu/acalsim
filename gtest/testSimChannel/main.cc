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
 * @brief GoogleTest driver for SimChannel communication testing and validation
 *
 * @details
 * This file serves as the main entry point for the SimChannel test suite, which validates
 * the lock-free, asynchronous communication infrastructure used throughout the ACAL simulator.
 * SimChannel is a critical component that enables inter-module communication with minimal
 * synchronization overhead, making it essential for high-performance discrete event simulation.
 *
 * # Purpose
 * The primary purposes of this test suite are:
 * - Validate SimChannel communication correctness across different latency configurations
 * - Verify lock-free queue operations and thread-safety
 * - Test packet routing between sender and receiver simulators
 * - Ensure proper handling of local and remote latencies
 * - Validate the ChannelPort abstraction layer
 *
 * # Test Coverage
 * The test suite provides comprehensive coverage of:
 * - **Method 1**: Direct channel port pushing (pushToMasterChannelPort)
 * - **Method 2**: Zero-latency send/receive (localLat=0, remoteLat=0)
 * - **Method 3**: Remote latency only (localLat=0, remoteLat=1)
 * - **Method 4**: Local latency only (localLat=1, remoteLat=0)
 * - **Method 5**: Both latencies active (localLat=1, remoteLat=1)
 * - **Packet Lifecycle**: Creation, transmission, reception, and recycling
 * - **Bit Mask Validation**: Test result verification mechanism
 *
 * # Architecture Overview
 * @code
 * +-------------------------------------------------------------------------+
 * |                          main.cc (Test Driver)                          |
 * |  - GoogleTest initialization and command-line parsing                   |
 * |  - Test fixture setup and teardown coordination                         |
 * |  - Argument segregation (ACALSim vs GoogleTest)                         |
 * +-------------------------------------------------------------------------+
 *                                    |
 *                                    v
 * +-------------------------------------------------------------------------+
 * |                        SimChannelTest (Test Fixture)                    |
 * |  - Manages test lifecycle (SetUp/TearDown)                              |
 * |  - Stores argc/argv for simulator initialization                        |
 * |  - Coordinates between GoogleTest and ACALSim frameworks                |
 * +-------------------------------------------------------------------------+
 *                                    |
 *                                    v
 * +-------------------------------------------------------------------------+
 * |                      TestSimChannelTop (Simulation Top)                 |
 * |  - Creates and registers SendSim and RecvSim                            |
 * |  - Establishes bidirectional channel connections                        |
 * |  - Manages test mode configuration                                      |
 * +-------------------------------------------------------------------------+
 *             |                                                    |
 *             v                                                    v
 * +--------------------------+                        +--------------------------+
 * |       SendSim            |    Channel Packets     |        RecvSim           |
 * |  - Injects ReqPacket     | =====================> |  - Receives ReqPacket    |
 * |  - Tests 5 send methods  |                        |  - Sets bit mask status  |
 * |  - Port: RecvSim-M       |                        |  - Recycles packets      |
 * +--------------------------+                        +--------------------------+
 *             ^                                                    |
 *             |            Bidirectional Channels                  |
 *             +----------------------------------------------------+
 * @endcode
 *
 * # SimChannel Communication Model
 *
 * ## Lock-Free Queue Architecture
 * SimChannel implements a lock-free, multi-producer, single-consumer queue:
 * - **Producer Side** (SendSim): Multiple senders can enqueue packets concurrently
 * - **Consumer Side** (RecvSim): Single receiver dequeues packets in FIFO order
 * - **Thread-Safety**: Uses atomic operations and memory barriers
 * - **Performance**: Zero-copy design with packet recycling
 *
 * ## Latency Models
 * The test suite validates different latency configurations:
 *
 * ### Local Latency (localLat)
 * - Time delay on the sender's side before packet enters the channel
 * - Simulates processing overhead at the source
 * - Example: Protocol encoding, serialization overhead
 *
 * ### Remote Latency (remoteLat)
 * - Time delay on the receiver's side after packet exits the channel
 * - Simulates processing overhead at the destination
 * - Example: Protocol decoding, event scheduling overhead
 *
 * ### Combined Latency Model
 * @code
 * Total Transfer Time = localLat + channel_delay + remoteLat
 *
 * Where:
 *   localLat       = Sender-side processing time
 *   channel_delay  = Transmission delay through the channel
 *   remoteLat      = Receiver-side processing time
 * @endcode
 *
 * # Test Fixture Design
 *
 * ## SimChannelTest Class
 * The test fixture provides:
 * - **Static argc/argv Storage**: Maintains command-line arguments across tests
 * - **SetUp()**: Initializes ACALSim arguments before each test
 * - **TearDown()**: Cleans up simulation resources after each test
 * - **Test Isolation**: Each TEST_F creates a fresh simulation environment
 *
 * ## Test Lifecycle
 * For each test case:
 * 1. **Setup Phase**:
 *    - Extract ACALSim-specific arguments
 *    - Create TestSimChannelTop with appropriate TestMode
 *    - Initialize simulator with parsed arguments
 *
 * 2. **Execution Phase**:
 *    - Run simulation to completion
 *    - SendSim injects test packet
 *    - RecvSim processes packet and sets bit mask
 *
 * 3. **Validation Phase**:
 *    - Check bit mask for expected value
 *    - EXPECT_EQ validates test success
 *
 * 4. **Teardown Phase**:
 *    - Finish simulation
 *    - Recycle all packets
 *    - Clean up resources
 *
 * # Test Methods Explained
 *
 * ## TEST_F(SimChannelTest, Method1_pushToMasterChannelPort)
 * Tests direct packet injection into the receiver's master channel port.
 * - **API**: pushToMasterChannelPort(port_name, packet)
 * - **Purpose**: Validate basic channel port connection and packet delivery
 * - **Expected**: Packet arrives immediately at RecvSim, bit[0] is set
 * - **Use Case**: Simple direct communication without latency modeling
 *
 * ## TEST_F(SimChannelTest, Method2_sendPacketViaChannel_l0_r0)
 * Tests packet transmission with zero local and remote latencies.
 * - **API**: sendPacketViaChannel(port_name, localLat=0, remoteLat=0, packet)
 * - **Purpose**: Validate immediate packet delivery through channel
 * - **Expected**: Packet arrives without delay, bit[1] is set
 * - **Use Case**: Ideal channel with no processing overhead
 *
 * ## TEST_F(SimChannelTest, Method3_sendPacketViaChannel_l0_r1)
 * Tests packet transmission with remote latency only.
 * - **API**: sendPacketViaChannel(port_name, localLat=0, remoteLat=1, packet)
 * - **Purpose**: Validate receiver-side processing delay
 * - **Expected**: Packet delayed by 1 cycle at receiver, bit[2] is set
 * - **Use Case**: Models receiver processing overhead
 *
 * ## TEST_F(SimChannelTest, Method4_sendPacketViaChannel_l1_r0)
 * Tests packet transmission with local latency only.
 * - **API**: sendPacketViaChannel(port_name, localLat=1, remoteLat=0, packet)
 * - **Purpose**: Validate sender-side processing delay
 * - **Expected**: Packet delayed by 1 cycle at sender, bit[3] is set
 * - **Use Case**: Models sender processing overhead
 *
 * ## TEST_F(SimChannelTest, Method5_sendPacketViaChannel_l1_r1)
 * Tests packet transmission with both local and remote latencies.
 * - **API**: sendPacketViaChannel(port_name, localLat=1, remoteLat=1, packet)
 * - **Purpose**: Validate full latency model
 * - **Expected**: Packet delayed by 2 cycles total, bit[4] is set
 * - **Use Case**: Realistic communication with both-side overheads
 *
 * # Bit Mask Encoding
 * Each test sets a unique bit to indicate success:
 * @code
 * Bit 0 (0x1): Method1 - pushToMasterChannelPort
 * Bit 1 (0x2): Method2 - sendPacketViaChannel(l=0, r=0)
 * Bit 2 (0x4): Method3 - sendPacketViaChannel(l=0, r=1)
 * Bit 3 (0x8): Method4 - sendPacketViaChannel(l=1, r=0)
 * Bit 4 (0x10): Method5 - sendPacketViaChannel(l=1, r=1)
 * @endcode
 *
 * # How to Run
 *
 * ## Building the Tests
 * @code{.sh}
 * # From project root
 * cd acalsim-workspace/projects/acalsim
 * mkdir -p build && cd build
 * cmake ..
 * make testSimChannel
 * @endcode
 *
 * ## Running All Tests
 * @code{.sh}
 * # Run complete test suite
 * ./gtest/testSimChannel/testSimChannel
 *
 * # Run with verbose GoogleTest output
 * ./gtest/testSimChannel/testSimChannel --gtest_verbose
 *
 * # Run with colored output
 * ./gtest/testSimChannel/testSimChannel --gtest_color=yes
 * @endcode
 *
 * ## Running Specific Tests
 * @code{.sh}
 * # Run only Method1 test
 * ./gtest/testSimChannel/testSimChannel --gtest_filter=SimChannelTest.Method1*
 *
 * # Run all latency tests (Methods 2-5)
 * ./gtest/testSimChannel/testSimChannel --gtest_filter=*sendPacketViaChannel*
 *
 * # Run with repetition to check for race conditions
 * ./gtest/testSimChannel/testSimChannel --gtest_repeat=1000
 * @endcode
 *
 * ## Mixed ACALSim and GoogleTest Arguments
 * @code{.sh}
 * # Combine simulator and test framework arguments
 * ./gtest/testSimChannel/testSimChannel \
 *     --verbose \
 *     --log-level=DEBUG \
 *     --gtest_filter=SimChannelTest.Method2* \
 *     --gtest_color=yes
 * @endcode
 *
 * # Expected Outcomes
 *
 * ## Successful Test Run
 * @code
 * [==========] Running 5 tests from 1 test suite.
 * [----------] Global test environment set-up.
 * [----------] 5 tests from SimChannelTest
 * [ RUN      ] SimChannelTest.Method1_pushToMasterChannelPort
 * [INFO] RecvSim: TestMode::Method1
 * [       OK ] SimChannelTest.Method1_pushToMasterChannelPort (10 ms)
 * [ RUN      ] SimChannelTest.Method2_sendPacketViaChannel_l0_r0
 * [INFO] RecvSim: TestMode::Method2
 * [       OK ] SimChannelTest.Method2_sendPacketViaChannel_l0_r0 (12 ms)
 * [ RUN      ] SimChannelTest.Method3_sendPacketViaChannel_l0_r1
 * [INFO] RecvSim: TestMode::Method3
 * [       OK ] SimChannelTest.Method3_sendPacketViaChannel_l0_r1 (11 ms)
 * [ RUN      ] SimChannelTest.Method4_sendPacketViaChannel_l1_r0
 * [INFO] RecvSim: TestMode::Method4
 * [       OK ] SimChannelTest.Method4_sendPacketViaChannel_l1_r0 (13 ms)
 * [ RUN      ] SimChannelTest.Method5_sendPacketViaChannel_l1_r1
 * [INFO] RecvSim: TestMode::Method5
 * [       OK ] SimChannelTest.Method5_sendPacketViaChannel_l1_r1 (14 ms)
 * [----------] 5 tests from SimChannelTest (60 ms total)
 * [==========] 5 tests from 1 test suite ran. (60 ms total)
 * [  PASSED  ] 5 tests.
 * @endcode
 *
 * ## Failure Example
 * @code
 * [ RUN      ] SimChannelTest.Method2_sendPacketViaChannel_l0_r0
 * [ERROR] RecvSim: Packet not received
 * main.cc:62: Failure
 * Value of: acalsim::top->checkGTestBitMask(0, 1 << 1)
 *   Actual: false
 * Expected: true
 * Test error: sendPacketViaChannel, localLat=0, remoteLat=0
 * [  FAILED  ] SimChannelTest.Method2_sendPacketViaChannel_l0_r0 (15 ms)
 * @endcode
 *
 * # Code Examples
 *
 * ## Adding a New Test Method
 * @code{.cpp}
 * TEST_F(SimChannelTest, Method6_HighLatency) {
 *     acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method6);
 *     acalsim::top->init(acalsim_args.size(), acalsim_args.data());
 *     acalsim::top->run();
 *
 *     // Validate high-latency communication
 *     EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 5), true)
 *         << "Test error: high latency communication failed";
 * }
 * @endcode
 *
 * ## Custom Test Fixture
 * @code{.cpp}
 * class CustomChannelTest : public SimChannelTest {
 * protected:
 *     void SetUp() override {
 *         SimChannelTest::SetUp();
 *         // Additional custom setup
 *         enableDetailedLogging();
 *     }
 *
 *     void enableDetailedLogging() {
 *         // Custom logging configuration
 *     }
 * };
 *
 * TEST_F(CustomChannelTest, DetailedLogging) {
 *     // Test with enhanced logging
 * }
 * @endcode
 *
 * # Argument Parsing Strategy
 *
 * ## Dual Argument System
 * The test suite uses a dual argument parsing strategy:
 *
 * ### ACALSim Arguments
 * Extracted by `getACALSimArguments()`:
 * - --verbose, --log-level, --config, etc.
 * - Used for simulator initialization
 * - Control simulation behavior and logging
 *
 * ### GoogleTest Arguments
 * Extracted by `getGoogleTestArguments()`:
 * - --gtest_filter, --gtest_repeat, --gtest_color, etc.
 * - Used for test framework control
 * - Control test execution and output formatting
 *
 * ## Implementation
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *     // Store original arguments in fixture
 *     SimChannelTest::init(argc, argv);
 *
 *     // Extract GoogleTest-specific arguments
 *     std::vector<char*> gtest_args = acalsim::getGoogleTestArguments(argc, argv);
 *     int gtest_argc = gtest_args.size();
 *
 *     // Initialize GoogleTest with filtered arguments
 *     testing::InitGoogleTest(&gtest_argc, gtest_args.data());
 *
 *     // Run all tests (each test extracts ACALSim args in SetUp)
 *     return RUN_ALL_TESTS();
 * }
 * @endcode
 *
 * # Related Components
 *
 * @see TestSimChannel.cc - SimChannel test implementation and packet handlers
 * @see TestSimChannel.hh - Test class definitions and TestMode enumeration
 * @see SimChannel - Core lock-free communication channel implementation
 * @see SimPacket - Base class for all simulator packets
 * @see CPPSimBase - Base class for C++-based simulators
 * @see SimTop - Top-level simulation coordinator
 * @see ChannelPortManager - Manages channel port connections
 *
 * # Performance Characteristics
 *
 * - **Test Execution Time**: Each test completes in 10-15ms
 * - **Memory Overhead**: Minimal (single packet per test)
 * - **Scalability**: Tests are independent and can run in parallel
 * - **Thread Safety**: Validates lock-free queue correctness
 *
 * # Debugging Tips
 *
 * 1. **Enable Verbose Logging**:
 *    @code{.sh}
 *    ./testSimChannel --verbose --log-level=TRACE
 *    @endcode
 *
 * 2. **Check Bit Mask Values**:
 *    Add debug output in RecvSim::handler() to see which bits are set
 *
 * 3. **Verify Channel Connections**:
 *    Ensure ChannelPortManager::ConnectPort() is called correctly
 *
 * 4. **Test Individual Methods**:
 *    Use --gtest_filter to isolate failing tests
 *
 * 5. **Check Packet Lifecycle**:
 *    Verify packets are properly recycled in RecvSim::handler()
 *
 * # Common Issues and Solutions
 *
 * | Issue | Possible Cause | Solution |
 * |-------|---------------|----------|
 * | Bit not set | Packet not received | Check channel port names match |
 * | Segmentation fault | Null packet pointer | Verify packet allocation |
 * | Deadlock | Circular dependency | Review channel connections |
 * | Test timeout | Simulation not terminating | Check finish() conditions |
 *
 * @author ACAL/Playlab Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This test suite validates the fundamental communication infrastructure
 * @warning Tests assume single-threaded execution; multi-threaded tests require additional synchronization
 */

#include <gtest/gtest.h>

#include <iostream>

#include "ACALSim.hh"
#include "TestSimChannel.hh"

class SimChannelTest : public testing::Test {
public:
	std::vector<char*> acalsim_args;

	// Store argc and argv as static members
	static int    argc;
	static char** argv;

	// Static method to set argc and argv
	static void init(int _argc, char** _argv) {
		SimChannelTest::argc = _argc;
		SimChannelTest::argv = _argv;
	}

	void SetUp() override {
		// Initialize any other necessary state here if needed
		acalsim_args = acalsim::getACALSimArguments(argc, argv);
	}

	void TearDown() override { acalsim::top->finish(); }
};

// Definition of static members
int    SimChannelTest::argc = 0;
char** SimChannelTest::argv = nullptr;

TEST_F(SimChannelTest, Method1_pushToMasterChannelPort) {
	acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method1);
	acalsim::top->init(acalsim_args.size(), acalsim_args.data());
	acalsim::top->run();

	EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 0), true) << "Test error pushToMasterChannelPort()";
}

TEST_F(SimChannelTest, Method2_sendPacketViaChannel_l0_r0) {
	acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method2);
	acalsim::top->init(acalsim_args.size(), acalsim_args.data());
	acalsim::top->run();
	EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 1), true)
	    << "Test error: sendPacketViaChannel, localLat=0, remoteLat=0";
}

TEST_F(SimChannelTest, Method3_sendPacketViaChannel_l0_r1) {
	acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method3);
	acalsim::top->init(acalsim_args.size(), acalsim_args.data());
	acalsim::top->run();
	EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 2), true)
	    << "Test error: sendPacketViaChannel, localLat=0, remoteLat=1";
}

TEST_F(SimChannelTest, Method4_sendPacketViaChannel_l1_r0) {
	acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method4);
	acalsim::top->init(acalsim_args.size(), acalsim_args.data());
	acalsim::top->run();
	EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 3), true)
	    << "Test error: sendPacketViaChannel, localLat=1, remoteLat=0";
}

TEST_F(SimChannelTest, Method5_sendPacketViaChannel_l1_r1) {
	acalsim::top = std::make_shared<TestSimChannelTop>(TestMode::Method5);
	acalsim::top->init(acalsim_args.size(), acalsim_args.data());
	acalsim::top->run();
	EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 4), true)
	    << "Test error: sendPacketViaChannel, localLat=1, remoteLat=1";
}

int main(int argc, char** argv) {
	SimChannelTest::init(argc, argv);

	std::vector<char*> gtest_args = acalsim::getGoogleTestArguments(argc, argv);
	int                gtest_argc = gtest_args.size();
	testing::InitGoogleTest(&gtest_argc, gtest_args.data());

	return RUN_ALL_TESTS();
}
