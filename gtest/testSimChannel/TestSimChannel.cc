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
 * @file TestSimChannel.cc
 * @brief Implementation of SimChannel test components and packet handlers
 *
 * @details
 * This file implements the core test infrastructure for validating SimChannel communication
 * functionality. It provides concrete implementations of packet handlers, traffic injection
 * methods, and simulation components used to test lock-free, asynchronous message passing
 * between simulator modules.
 *
 * # Purpose
 * The implementation serves several critical testing purposes:
 * - Implement packet visitor pattern for type-safe packet routing
 * - Provide traffic injection methods with varying latency configurations
 * - Handle packet reception and validation through bit mask mechanism
 * - Demonstrate proper packet lifecycle management (allocation, transmission, recycling)
 * - Establish bidirectional channel connections for realistic communication testing
 *
 * # Component Overview
 *
 * ## ReqPacket Class
 * A specialized packet type for SimChannel testing that:
 * - Extends SimPacket with test-specific metadata (TestMode)
 * - Implements visitor pattern for polymorphic packet handling
 * - Routes packets to appropriate handlers based on simulator type
 * - Carries test configuration through the communication pipeline
 *
 * ## SendSim Class
 * A traffic generator simulator that:
 * - Injects test packets using different communication methods
 * - Tests five different packet transmission APIs
 * - Validates latency parameter handling
 * - Demonstrates proper channel port usage
 *
 * ## RecvSim Class
 * A receiver simulator that:
 * - Accepts packets from SendSim via channel connections
 * - Validates packet arrival and content
 * - Sets bit mask indicators for test verification
 * - Recycles packets to prevent memory leaks
 *
 * ## TestSimChannelTop Class
 * A simulation coordinator that:
 * - Registers SendSim and RecvSim instances
 * - Establishes bidirectional channel connections
 * - Manages test mode configuration
 * - Provides clean simulation lifecycle
 *
 * # Architecture Diagram
 * @code
 * +--------------------------------------------------------------------------+
 * |                       TestSimChannel.cc Architecture                     |
 * +--------------------------------------------------------------------------+
 * |                                                                          |
 * |  +--------------------------------------------------------------------+  |
 * |  |                    TestSimChannelTop                               |  |
 * |  |  - registerSimulators()                                            |  |
 * |  |  - Manages SendSim and RecvSim lifecycles                          |  |
 * |  |  - Establishes channel connections                                 |  |
 * |  +--------------------------------------------------------------------+  |
 * |         |                                                        |        |
 * |         | creates                                          creates|        |
 * |         v                                                        v        |
 * |  +---------------------------+                    +---------------------+ |
 * |  |        SendSim            |                    |       RecvSim       | |
 * |  |  +--------------------+   |                    |  +---------------+  | |
 * |  |  | init()             |   |                    |  | handler()     |  | |
 * |  |  |  - Creates packet  |   |                    |  |  - Sets bit   |  | |
 * |  |  |  - Calls inject*() |   |   ReqPacket        |  |  - Recycles   |  | |
 * |  |  +--------------------+   | =================> |  +---------------+  | |
 * |  |                           |                    |                     | |
 * |  |  +--------------------+   |                    |                     | |
 * |  |  | injectTraffic_*()  |   |                    |                     | |
 * |  |  |  - Method1: push   |   |                    |                     | |
 * |  |  |  - Method2: l0,r0  |   |                    |                     | |
 * |  |  |  - Method3: l0,r1  |   |                    |                     | |
 * |  |  |  - Method4: l1,r0  |   |                    |                     | |
 * |  |  |  - Method5: l1,r1  |   |                    |                     | |
 * |  |  +--------------------+   |                    |                     | |
 * |  +---------------------------+                    +---------------------+ |
 * |                                                                          |
 * |  Channel Connections (Bidirectional):                                   |
 * |  - SendSim -> RecvSim: "RecvSim-M" <-> "SendSim-S"                      |
 * |  - RecvSim -> SendSim: "SendSim-M" <-> "RecvSim-S"                      |
 * |                                                                          |
 * +--------------------------------------------------------------------------+
 * @endcode
 *
 * # Packet Visitor Pattern
 *
 * ## Design Motivation
 * The visitor pattern enables:
 * - Type-safe packet routing without runtime type checks
 * - Extensible packet handling for different simulator types
 * - Compile-time verification of packet handler implementations
 * - Clean separation between packet definition and handling logic
 *
 * ## Implementation Flow
 * @code
 * 1. ReqPacket::visit(when, simulator) is called
 *    |
 *    v
 * 2. dynamic_cast determines simulator type
 *    |
 *    +--- If SendSim* -> calls send->handler(when, this)
 *    |
 *    +--- If RecvSim* -> calls recv->handler(when, this)
 *    |
 *    +--- Otherwise -> ERROR: "Invalid simulator type"
 *
 * 3. Specific handler processes packet with full type information
 * @endcode
 *
 * ## Code Example
 * @code{.cpp}
 * void ReqPacket::visit(acalsim::Tick when, acalsim::SimBase& simulator) {
 *     if (auto send = dynamic_cast<SendSim*>(&simulator)) {
 *         send->handler(when, this);  // SendSim-specific handling
 *     } else if (auto recv = dynamic_cast<RecvSim*>(&simulator)) {
 *         recv->handler(when, this);  // RecvSim-specific handling
 *     } else {
 *         LABELED_ERROR("ReqPacket") << "Invalid simulator type";
 *     }
 * }
 * @endcode
 *
 * # Traffic Injection Methods
 *
 * ## Method 1: pushToMasterChannelPort
 * Direct packet injection into receiver's master port.
 * @code{.cpp}
 * void SendSim::injectTraffic_Method1(ReqPacket* packet) {
 *     this->pushToMasterChannelPort(RecvSim::getSimName() + "-M", packet);
 * }
 * @endcode
 * - **Characteristics**:
 *   - Immediate delivery (no latency modeling)
 *   - Bypasses normal channel latency mechanisms
 *   - Useful for testing direct port connections
 *   - Packet appears instantly in receiver's queue
 *
 * ## Method 2: sendPacketViaChannel(l=0, r=0)
 * Standard channel send with zero latencies.
 * @code{.cpp}
 * void SendSim::injectTraffic_Method2(ReqPacket* packet) {
 *     this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 0, 0, packet);
 * }
 * @endcode
 * - **Characteristics**:
 *   - Goes through normal channel infrastructure
 *   - No sender-side delay (localLat=0)
 *   - No receiver-side delay (remoteLat=0)
 *   - Validates basic channel operation
 *
 * ## Method 3: sendPacketViaChannel(l=0, r=1)
 * Channel send with receiver-side latency only.
 * @code{.cpp}
 * void SendSim::injectTraffic_Method3(ReqPacket* packet) {
 *     this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 0, 1, packet);
 * }
 * @endcode
 * - **Characteristics**:
 *   - Sender transmits immediately (localLat=0)
 *   - Receiver delayed by 1 cycle (remoteLat=1)
 *   - Models receiver processing overhead
 *   - Tests remote latency handling
 *
 * ## Method 4: sendPacketViaChannel(l=1, r=0)
 * Channel send with sender-side latency only.
 * @code{.cpp}
 * void SendSim::injectTraffic_Method4(ReqPacket* packet) {
 *     this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 1, 0, packet);
 * }
 * @endcode
 * - **Characteristics**:
 *   - Sender delayed by 1 cycle (localLat=1)
 *   - Receiver processes immediately (remoteLat=0)
 *   - Models sender processing overhead
 *   - Tests local latency handling
 *
 * ## Method 5: sendPacketViaChannel(l=1, r=1)
 * Channel send with both sender and receiver latencies.
 * @code{.cpp}
 * void SendSim::injectTraffic_Method5(ReqPacket* packet) {
 *     this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 1, 1, packet);
 * }
 * @endcode
 * - **Characteristics**:
 *   - Sender delayed by 1 cycle (localLat=1)
 *   - Receiver delayed by 1 cycle (remoteLat=1)
 *   - Total delay: 2 cycles
 *   - Models realistic communication overhead
 *   - Tests combined latency effects
 *
 * # Packet Lifecycle Management
 *
 * ## Packet Creation
 * Packets are created in SendSim::init():
 * @code{.cpp}
 * void SendSim::init() {
 *     auto packet = new ReqPacket(this->mode);
 *     // Route to appropriate injection method
 *     switch (this->mode) {
 *         case TestMode::Method1: this->injectTraffic_Method1(packet); break;
 *         // ... other methods
 *     }
 * }
 * @endcode
 *
 * ## Packet Transmission
 * Packets traverse channel infrastructure:
 * 1. SendSim calls sendPacketViaChannel() or pushToMasterChannelPort()
 * 2. Packet enters lock-free queue
 * 3. Channel applies latency scheduling
 * 4. Packet delivered to RecvSim
 *
 * ## Packet Reception
 * RecvSim handles incoming packets:
 * @code{.cpp}
 * void RecvSim::handler(acalsim::Tick when, ReqPacket* packet) {
 *     // Validate packet mode and set corresponding bit
 *     switch (packet->getTestMode()) {
 *         case TestMode::Method1:
 *             acalsim::top->setGTestBitMask(0, 0);  // Set bit[0]
 *             break;
 *         // ... other cases
 *     }
 *     // Recycle packet
 *     const auto rc = acalsim::top->getRecycleContainer();
 *     rc->recycle(packet);
 * }
 * @endcode
 *
 * ## Packet Recycling
 * Critical for preventing memory leaks:
 * - RecycleContainer manages packet memory pool
 * - Packets returned to pool for reuse
 * - Avoids repeated allocation/deallocation
 * - Improves performance for high-traffic scenarios
 *
 * # Channel Connection Topology
 *
 * ## Bidirectional Connection Setup
 * TestSimChannelTop establishes symmetric connections:
 * @code{.cpp}
 * void TestSimChannelTop::registerSimulators() {
 *     this->send = new SendSim(this->mode);
 *     this->recv = new RecvSim(this->mode);
 *     this->addSimulator(this->send);
 *     this->addSimulator(this->recv);
 *
 *     // Forward path: SendSim -> RecvSim
 *     acalsim::ChannelPortManager::ConnectPort(
 *         this->send,                  // Source simulator
 *         this->recv,                  // Destination simulator
 *         RecvSim::getSimName() + "-M", // Destination port (master)
 *         SendSim::getSimName() + "-S"  // Source port (slave)
 *     );
 *
 *     // Reverse path: RecvSim -> SendSim (for potential responses)
 *     acalsim::ChannelPortManager::ConnectPort(
 *         this->recv,                  // Source simulator
 *         this->send,                  // Destination simulator
 *         SendSim::getSimName() + "-M", // Destination port (master)
 *         RecvSim::getSimName() + "-S"  // Source port (slave)
 *     );
 * }
 * @endcode
 *
 * ## Port Naming Convention
 * - **Master Port (-M)**: Primary receive port for incoming packets
 * - **Slave Port (-S)**: Associated transmit port for outgoing packets
 * - **Format**: "[SimulatorName]-[M|S]"
 * - **Example**: "RecvSim-M", "SendSim-S"
 *
 * ## Connection Diagram
 * @code
 *     SendSim                           RecvSim
 *  +------------+                    +------------+
 *  |            |   "RecvSim-M"      |            |
 *  | SendSim-S  | -----------------> | RecvSim-M  |
 *  |            |                    |            |
 *  | SendSim-M  | <----------------- | RecvSim-S  |
 *  |            |   "SendSim-M"      |            |
 *  +------------+                    +------------+
 * @endcode
 *
 * # Bit Mask Validation Mechanism
 *
 * ## Purpose
 * The bit mask provides a simple, efficient way to validate test success:
 * - Each test sets a unique bit when packet successfully received
 * - GoogleTest checks the bit to determine test pass/fail
 * - Supports multiple concurrent validations in complex tests
 *
 * ## Bit Assignments
 * @code
 * TestMode::Method1 -> setGTestBitMask(0, 0) -> bit[0] = 0x1
 * TestMode::Method2 -> setGTestBitMask(0, 1) -> bit[1] = 0x2
 * TestMode::Method3 -> setGTestBitMask(0, 2) -> bit[2] = 0x4
 * TestMode::Method4 -> setGTestBitMask(0, 3) -> bit[3] = 0x8
 * TestMode::Method5 -> setGTestBitMask(0, 4) -> bit[4] = 0x10
 * @endcode
 *
 * ## Validation in Tests
 * Each test checks its corresponding bit:
 * @code{.cpp}
 * TEST_F(SimChannelTest, Method1_pushToMasterChannelPort) {
 *     // ... run simulation ...
 *     EXPECT_EQ(acalsim::top->checkGTestBitMask(0, 1 << 0), true)
 *         << "Test error pushToMasterChannelPort()";
 * }
 * @endcode
 *
 * # Error Handling
 *
 * ## Invalid Test Mode
 * @code{.cpp}
 * default:
 *     LABELED_ERROR(getSimName()) << "Invalid TestMode";
 * @endcode
 * - Catches configuration errors
 * - Prevents silent failures
 * - Provides diagnostic information
 *
 * ## Invalid Simulator Type
 * @code{.cpp}
 * void ReqPacket::visit(...) {
 *     // ... dynamic_cast checks ...
 *     else {
 *         LABELED_ERROR("ReqPacket") << "Invalid simulator type";
 *     }
 * }
 * @endcode
 * - Detects packet routing errors
 * - Ensures type safety
 * - Aids debugging
 *
 * # Performance Considerations
 *
 * - **Packet Allocation**: Single packet per test minimizes overhead
 * - **Recycling**: Prevents memory leaks and improves cache locality
 * - **Lock-Free Queues**: Minimal synchronization overhead
 * - **Dynamic Cast**: Used only once per packet (acceptable for testing)
 *
 * # Extensibility
 *
 * ## Adding New Test Methods
 * 1. Add new TestMode enumeration value
 * 2. Implement injectTraffic_MethodN() in SendSim
 * 3. Add case in SendSim::init() switch
 * 4. Add case in RecvSim::handler() switch
 * 5. Create corresponding TEST_F in main.cc
 *
 * ## Example: Adding Method 6
 * @code{.cpp}
 * // In TestSimChannel.hh
 * enum class TestMode { ..., Method6 };
 *
 * // In TestSimChannel.cc (SendSim)
 * void SendSim::injectTraffic_Method6(ReqPacket* packet) {
 *     this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 5, 5, packet);
 * }
 *
 * void SendSim::init() {
 *     auto packet = new ReqPacket(this->mode);
 *     switch (this->mode) {
 *         // ... existing cases ...
 *         case TestMode::Method6: this->injectTraffic_Method6(packet); break;
 *     }
 * }
 *
 * // In TestSimChannel.cc (RecvSim)
 * void RecvSim::handler(acalsim::Tick when, ReqPacket* packet) {
 *     switch (packet->getTestMode()) {
 *         // ... existing cases ...
 *         case TestMode::Method6:
 *             acalsim::top->setGTestBitMask(0, 5);
 *             break;
 *     }
 * }
 * @endcode
 *
 * # Related Components
 *
 * @see main.cc - Test driver and GoogleTest integration
 * @see TestSimChannel.hh - Header definitions for test classes
 * @see SimChannel - Core lock-free channel implementation
 * @see CPPSimBase - Base class providing channel communication APIs
 * @see SimPacket - Base packet class with visitor pattern support
 * @see ChannelPortManager - Manages channel port creation and connection
 * @see RecycleContainer - Memory pool for packet recycling
 *
 * # Debugging Tips
 *
 * 1. **Add Logging in Handlers**:
 *    @code{.cpp}
 *    LABELED_INFO(getSimName()) << "Received packet at time " << when;
 *    @endcode
 *
 * 2. **Verify Packet Mode**:
 *    @code{.cpp}
 *    LABELED_DEBUG("RecvSim") << "TestMode = " << (int)packet->getTestMode();
 *    @endcode
 *
 * 3. **Check Channel Connections**:
 *    Enable verbose logging to see ConnectPort() calls
 *
 * 4. **Monitor Bit Mask**:
 *    @code{.cpp}
 *    LABELED_INFO(getSimName()) << "Bit mask = 0x"
 *        << std::hex << acalsim::top->getGTestBitMask(0);
 *    @endcode
 *
 * @author ACAL/Playlab Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This implementation demonstrates proper SimChannel usage patterns
 * @warning Always recycle packets to prevent memory leaks
 */

#include "TestSimChannel.hh"

void ReqPacket::visit(acalsim::Tick when, acalsim::SimBase& simulator) {
	if (auto send = dynamic_cast<SendSim*>(&simulator)) {
		send->handler(when, this);
	} else if (auto recv = dynamic_cast<RecvSim*>(&simulator)) {
		recv->handler(when, this);
	} else {
		LABELED_ERROR("ReqPacket") << "Invalid simulator type";
	}
}

void SendSim::init() {
	auto packet = new ReqPacket(this->mode);

	switch (this->mode) {
		case TestMode::Method1: this->injectTraffic_Method1(packet); break;
		case TestMode::Method2: this->injectTraffic_Method2(packet); break;
		case TestMode::Method3: this->injectTraffic_Method3(packet); break;
		case TestMode::Method4: this->injectTraffic_Method4(packet); break;
		case TestMode::Method5: this->injectTraffic_Method5(packet); break;
		default: LABELED_ERROR(getSimName()) << "Invalid TestMode";
	}
}

void SendSim::injectTraffic_Method1(ReqPacket* packet) {
	this->pushToMasterChannelPort(RecvSim::getSimName() + "-M", packet);
}
void SendSim::injectTraffic_Method2(ReqPacket* packet) {
	this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 0, 0, packet);
}
void SendSim::injectTraffic_Method3(ReqPacket* packet) {
	this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 0, 1, packet);
}
void SendSim::injectTraffic_Method4(ReqPacket* packet) {
	this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 1, 0, packet);
}
void SendSim::injectTraffic_Method5(ReqPacket* packet) {
	this->sendPacketViaChannel(RecvSim::getSimName() + "-M", 1, 1, packet);
}

void SendSim::handler(acalsim::Tick when, ReqPacket* packet) {}

void RecvSim::handler(acalsim::Tick when, ReqPacket* packet) {
	switch (packet->getTestMode()) {
		case TestMode::Method1:
			acalsim::top->setGTestBitMask(0, 0);
			LABELED_INFO(getSimName()) << "TestMode::Method1";
			break;
		case TestMode::Method2:
			acalsim::top->setGTestBitMask(0, 1);
			LABELED_INFO(getSimName()) << "TestMode::Method2";
			break;
		case TestMode::Method3:
			acalsim::top->setGTestBitMask(0, 2);
			LABELED_INFO(getSimName()) << "TestMode::Method3";
			break;
		case TestMode::Method4:
			acalsim::top->setGTestBitMask(0, 3);
			LABELED_INFO(getSimName()) << "TestMode::Method4";
			break;
		case TestMode::Method5:
			acalsim::top->setGTestBitMask(0, 4);
			LABELED_INFO(getSimName()) << "TestMode::Method5";
			break;
		default: LABELED_ERROR(getSimName()) << "Invalid TestMode"; break;
	}
	const auto rc = acalsim::top->getRecycleContainer();
	rc->recycle(packet);
}

void TestSimChannelTop::registerSimulators() {
	this->send = new SendSim(this->mode);
	this->recv = new RecvSim(this->mode);
	this->addSimulator(this->send);
	this->addSimulator(this->recv);

	acalsim::ChannelPortManager::ConnectPort(this->send, this->recv, RecvSim::getSimName() + "-M",
	                                         SendSim::getSimName() + "-S");
	acalsim::ChannelPortManager::ConnectPort(this->recv, this->send, SendSim::getSimName() + "-M",
	                                         RecvSim::getSimName() + "-S");
}
