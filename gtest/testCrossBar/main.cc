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
 * @brief Main entry point for CrossBar component GoogleTest suite
 *
 * @details
 * This file implements the GoogleTest-based unit testing framework for validating the
 * CrossBar interconnect component in the ACALSim simulation environment. The CrossBar
 * is a critical on-chip communication component that provides multi-master to multi-slave
 * connectivity with arbitration and backpressure management.
 *
 * ## Purpose
 *
 * The test suite validates:
 * - **Multi-Master Arbitration**: Ensures fair and correct arbitration when multiple
 *   masters simultaneously access the same slave device
 * - **Backpressure Handling**: Verifies proper flow control when slaves cannot accept
 *   new requests or when the crossbar becomes congested
 * - **Request-Response Ordering**: Validates that responses are correctly routed back
 *   to the originating master with matching transaction IDs
 * - **Throughput and Latency**: Measures timing performance under various load conditions
 * - **Channel Integrity**: Ensures no packet loss or corruption during transmission
 *
 * ## Test Architecture
 *
 * The test bench creates a configurable topology:
 *
 * @code
 *     MasterTBSim[0]  MasterTBSim[1]  ...  MasterTBSim[N-1]
 *            |             |                      |
 *            +-------------+----------------------+
 *                          |
 *                    [ CrossBar ]
 *                    (N×M Arbitration)
 *                          |
 *            +-------------+----------------------+
 *            |             |                      |
 *     SlaveTBSim[0]   SlaveTBSim[1]   ...  SlaveTBSim[M-1]
 * @endcode
 *
 * **Request Path**: Master → CrossBar Request Channel → Slave
 * **Response Path**: Slave → CrossBar Response Channel → Master
 *
 * ## Configuration Parameters
 *
 * Tests are parameterized via command-line arguments:
 * - `--n_master <N>`: Number of master devices (default: configurable)
 * - `--n_slave <M>`: Number of slave devices (default: configurable)
 * - `--n_requests <K>`: Number of requests per master (default: configurable)
 *
 * ## Test Scenarios Covered
 *
 * 1. **Single Master to Single Slave**: Basic connectivity validation
 * 2. **Multiple Masters to Single Slave**: Arbitration stress test
 * 3. **Single Master to Multiple Slaves**: Round-robin distribution
 * 4. **Full Mesh Traffic**: All masters accessing all slaves simultaneously
 * 5. **Backpressure Scenarios**: Slaves intentionally stalling to trigger retry logic
 * 6. **Transaction ID Validation**: Ensuring correct request-response pairing
 *
 * ## Running the Tests
 *
 * Execute the test suite using standard GoogleTest commands:
 *
 * @code{.sh}
 * # Run all tests with default configuration
 * ./testCrossBar
 *
 * # Run with custom parameters
 * ./testCrossBar --n_master 4 --n_slave 2 --n_requests 100
 *
 * # Enable verbose logging
 * ./testCrossBar --gtest_output=xml:test_results.xml
 *
 * # Run specific test patterns
 * ./testCrossBar --gtest_filter=CrossBarTest.*
 * @endcode
 *
 * ## Expected Outcomes
 *
 * **Success Criteria**:
 * - All issued requests receive corresponding responses
 * - Transaction IDs match between requests and responses
 * - No deadlocks or livelocks occur
 * - Backpressure is correctly propagated and released
 * - Simulation completes within expected time bounds
 *
 * **Failure Modes**:
 * - Missing responses (finished_requests < num_requests)
 * - Transaction ID mismatches
 * - Simulation hangs (indicates deadlock)
 * - Assertion failures in arbitration logic
 *
 * ## Performance Measurement
 *
 * The test framework measures:
 * - **Total Simulation Time**: Wall-clock time for test execution
 * - **Per-Transaction Latency**: Request issuance to response reception
 * - **Throughput**: Transactions completed per simulated cycle
 * - **Arbitration Fairness**: Distribution of grants across masters
 *
 * Timing data is collected using high-resolution clocks and reported in seconds.
 *
 * ## Chrome Trace Integration
 *
 * The test generates Chrome trace events for visualization:
 * - Opens `chrome://tracing` in Chrome browser
 * - Load generated trace file to visualize:
 *   - Request/response path timelines
 *   - Arbitration decisions
 *   - Backpressure periods
 *   - Transaction lifetimes
 *
 * ## Code Example: Basic Test Pattern
 *
 * @code{.cpp}
 * // Typical usage in a GoogleTest fixture:
 * TEST_F(CrossBarTest, MultiMasterArbitration) {
 *     // Setup: Configure 4 masters, 1 slave, 50 requests each
 *     acalsim::top = std::make_shared<testcrossbar::CrossBarTestTop>();
 *
 *     // Execute simulation
 *     auto start = std::chrono::high_resolution_clock::now();
 *     acalsim::top->run();
 *     auto stop = std::chrono::high_resolution_clock::now();
 *
 *     // Verify all transactions completed
 *     EXPECT_EQ(completed_transactions, expected_transactions);
 *
 *     // Report timing
 *     auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
 *     std::cout << "Test completed in " << duration.count() << "ms" << std::endl;
 * }
 * @endcode
 *
 * ## Best Practices for Unit Testing
 *
 * 1. **Isolation**: Each test should be independent and not rely on previous test state
 * 2. **Determinism**: Use fixed seeds for any randomization to ensure reproducibility
 * 3. **Coverage**: Test both normal operation and corner cases (e.g., full buffers)
 * 4. **Assertions**: Use EXPECT/ASSERT macros liberally to catch subtle bugs
 * 5. **Documentation**: Document the specific behavior being validated in each test
 * 6. **Performance**: Keep individual tests fast (<1 second) for rapid iteration
 *
 * ## Dependencies
 *
 * - ACALSim simulation framework
 * - CrossBarTestTop test harness
 * - GoogleTest framework (implicitly used via ACALSim)
 * - C++17 standard library (chrono, memory)
 *
 * @see CrossBarTestTop Test harness implementation
 * @see MasterTBSim Master device test bench simulator
 * @see SlaveTBSim Slave device test bench simulator
 * @see acalsim::crossbar::CrossBar CrossBar component under test
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include <memory>

#include "ACALSim.hh"
#include "CrossBarTestTop.hh"

/**
 * @brief Main entry point for CrossBar GoogleTest execution
 *
 * @details
 * This function orchestrates the complete test execution lifecycle for the CrossBar
 * unit tests. It follows the standard ACALSim simulation pattern with integrated
 * performance measurement.
 *
 * **Execution Flow**:
 * 1. **Instantiation**: Creates CrossBarTestTop simulation environment
 * 2. **Initialization**: Parses CLI arguments and configures test parameters
 * 3. **Execution**: Runs the simulation with timing measurement
 * 4. **Reporting**: Logs performance metrics
 * 5. **Cleanup**: Finalizes simulation and releases resources
 *
 * ## Performance Timing
 *
 * The function measures wall-clock execution time using C++11 high-resolution clocks,
 * providing microsecond precision for performance analysis. This timing includes:
 * - Simulation initialization overhead
 * - Event scheduling and execution
 * - Packet transmission and arbitration
 * - Logging and tracing operations
 *
 * ## Example Output
 *
 * @code
 * [INFO] MDevice:0 issue request at Master:0 to Slave:0 tid=0
 * [INFO] SDevice:0 handle a request from Master:0 tid=0
 * [INFO] MDevice:0 receive a response from Slave:0 tid=0
 * [INFO] Timer: Simulation Time: 0.00234 seconds.
 * @endcode
 *
 * @param[in] argc Command-line argument count
 * @param[in] argv Command-line argument values
 *                 - `--n_master <N>`: Number of master devices
 *                 - `--n_slave <M>`: Number of slave devices
 *                 - `--n_requests <K>`: Number of requests per master
 *                 - Additional ACALSim arguments (logging, tracing, etc.)
 *
 * @return int Exit status (0 = success, non-zero = failure)
 *
 * @note The global `acalsim::top` shared pointer is used to maintain the simulation
 *       instance throughout the test execution. This follows the ACALSim convention
 *       for accessing the top-level simulation context from any component.
 *
 * @see testcrossbar::CrossBarTestTop
 * @see acalsim::SimTop::init()
 * @see acalsim::SimTop::run()
 * @see acalsim::SimTop::finish()
 */
int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	acalsim::top = std::make_shared<testcrossbar::CrossBarTestTop>();
	acalsim::top->init(argc, argv);

	auto start = std::chrono::high_resolution_clock::now();
	acalsim::top->run();
	auto stop = std::chrono::high_resolution_clock::now();

	auto diff = duration_cast<std::chrono::nanoseconds>(stop - start);
	acalsim::LogOStream(acalsim::LoggingSeverity::L_INFO, __FILE__, __LINE__, "Timer")
	    << "Simulation Time: " << (double)diff.count() / pow(10, 9) << " seconds.";

	acalsim::top->finish();
	return 0;
}
