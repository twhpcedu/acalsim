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
 * @file testCommunication.cc
 * @brief Mixed communication patterns example - Hybrid ports and channels architecture
 *
 * This example demonstrates **mixed communication patterns** combining both channel-based
 * and port-based communication in ACALSim. It showcases a simplified Processing Element (PE)
 * architecture where a TrafficGenerator injects traffic to PE simulators using channels for
 * event delivery while maintaining bidirectional channel connectivity for responses.
 *
 * **Key Learning Objectives:**
 * - Understanding when to use channels vs ports
 * - Combining asynchronous channel communication with structured port connections
 * - Event-driven traffic injection patterns
 * - Callback-based request-response protocols
 * - Bidirectional communication flow management
 *
 * **System Architecture:**
 * ```
 * ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
 * │      TrafficGenerator             │       │         PE Simulator              │
 * │   (Traffic Injection Engine)      │       │   (Processing Element)            │
 * │                                   │       │                                   │
 * │  Event-Driven Request Generation  │       │  Request Processing               │
 * │                                   │       │  & Response Handling              │
 * │  ┌─────────────────────────────┐  │       │  ┌─────────────────────────────┐  │
 * │  │   TrafficEvent Scheduler    │  │       │  │   PEReqEvent Processor      │  │
 * │  │   - Create PEReqPacket      │  │       │  │   - Compute: d = a*b + c    │  │
 * │  │   - Wrap in PEReqEvent      │  │       │  │   - Update PERespPacket     │  │
 * │  │   - Wrap in EventPacket     │  │       │  │   - Invoke callback         │  │
 * │  │   - Schedule for future     │  │       │  │                             │  │
 * │  └─────────────────────────────┘  │       │  └─────────────────────────────┘  │
 * │                                   │       │                                   │
 * │  MasterChannelPort "DSPE"         │       │  SlaveChannelPort                 │
 * │  (Send requests/events)           │       │  "USTrafficGenerator"             │
 * │                                   │       │  (Receive events)                 │
 * │  SlaveChannelPort                 │       │                                   │
 * │  "DSPE"                           │       │  MasterChannelPort                │
 * │  (Receive responses)              │       │  "USTrafficGenerator"             │
 * │                                   │       │  (Send responses)                 │
 * └───────────────────────────────────┘       └───────────────────────────────────┘
 *             │                                            ▲
 *             │        EventPacket(PEReqEvent)             │
 *             │        via Channel "DSPE"                  │
 *             └────────────────────────────────────────────┘
 *                          Bidirectional Channel
 *             ┌────────────────────────────────────────────┐
 *             │        EventPacket(PERespCallback)         │
 *             ▼        via Channel "USTrafficGenerator"    │
 * ```
 *
 * **Channel Connectivity Graph:**
 * ```
 *   TrafficGenerator.DSPE ──────────────────► PE.USTrafficGenerator (Events: TG → PE)
 *   PE.USTrafficGenerator ──────────────────► TrafficGenerator.DSPE (Callbacks: PE → TG)
 *
 * Note: Bidirectional channels established using ChannelPortManager::ConnectPort()
 * ```
 *
 * **Communication Flow Timeline:**
 * ```
 * Tick 0 (Initialization):
 *   TrafficGenerator::init():
 *     1. Create TrafficEvent with transaction ID 1
 *     2. Schedule for tick 1
 *     3. Create TrafficEvent with transaction ID 2
 *     4. Schedule for tick 2
 *
 * Tick 1:
 *   TrafficEvent #1 (TID=1):
 *     5. Get downstream PE via getDownStream("DSPE")
 *     6. Create PEReqPacket(TEST, a=200, b=2, c=400)
 *     7. Create PERespPacket to hold result
 *     8. Define callback lambda: PERespHandler(TID, PERespPacket)
 *     9. Create PEReqEvent with callback
 *     10. Wrap in EventPacket for tick 6 (current + 5)
 *     11. Push to MasterChannelPort "DSPE"
 *
 *   Framework (Phase 2 - Channel Transfer):
 *     12. ChannelPortManager transfers EventPacket
 *     13. PE.USTrafficGenerator receives EventPacket
 *     14. Framework schedules PEReqEvent for tick 6
 *
 * Tick 2:
 *   TrafficEvent #2 (TID=2):
 *     15-21. Same flow as TID=1 with different timing
 *
 * Tick 6:
 *   PE Processing (TID=1):
 *     22. PEReqEvent::process() invoked
 *     23. Extract a=200, b=2, c=400 from PEReqPacket
 *     24. Compute: d = 200 * 2 + 400 = 800
 *     25. Store result in PERespPacket
 *     26. Invoke caller's callback immediately
 *
 *   TrafficGenerator (via callback):
 *     27. PERespHandler(TID=1, PERespPacket) executed
 *     28. Extract result: 800
 *     29. Log completion of transaction
 *
 * Tick 7:
 *   PE Processing (TID=2):
 *     30-34. Same flow as TID=1
 * ```
 *
 * **Design Decisions - When to Use Channels:**
 *
 * This example uses **channels exclusively** for the following reasons:
 *
 * 1. **Asynchronous Event Delivery**
 *    - TrafficGenerator needs to inject events into PE's event queue
 *    - Channels naturally support EventPacket wrapping
 *    - Events can be scheduled for future ticks
 *
 * 2. **Decoupled Timing**
 *    - Traffic generation timing independent of PE processing
 *    - Events can be scheduled with arbitrary delays
 *    - No immediate processing requirements
 *
 * 3. **Callback-Based Responses**
 *    - Responses handled via callbacks, not return packets
 *    - No need for structured port-based response protocol
 *    - Callbacks executed within the same simulator context
 *
 * 4. **Simplified Topology**
 *    - Point-to-point communication (TG ↔ PE)
 *    - No complex routing or forwarding needed
 *    - Direct bidirectional channel sufficient
 *
 * **Contrast with testChannel Example:**
 *
 * testChannel (TG → NOC → Cache):
 * - Multi-hop communication through intermediate NOC
 * - Packet forwarding and routing logic
 * - Structured request-response protocol
 * - Multiple channel hops for request/response
 *
 * testCommunication (TG → PE):
 * - Direct communication, no intermediaries
 * - Event injection with callbacks
 * - Single channel hop
 * - Simpler architecture for testing PE behavior
 *
 * **Packet Types and Their Roles:**
 *
 * 1. **EventPacket**
 *    - Container for events sent through channels
 *    - Carries target execution tick
 *    - Transferred during Phase 2 (channel phase)
 *
 * 2. **PEReqPacket**
 *    - Request to PE: compute d = a*b + c
 *    - Contains input operands (a, b, c)
 *    - Holds reference to PERespPacket
 *    - Contains callback function
 *
 * 3. **PERespPacket**
 *    - Response from PE computation
 *    - Updated with result value
 *    - Passed to callback function
 *
 * 4. **DataReqPacket**
 *    - Alternative packet type (not used in this example)
 *    - Demonstrates packet visitor pattern
 *    - Shows extensibility for different packet types
 *
 * **Event Types:**
 *
 * 1. **TrafficEvent**
 *    - Scheduled by TrafficGenerator in init()
 *    - Creates and sends PEReqEvent to PE
 *    - One per transaction
 *
 * 2. **PEReqEvent (CallbackEvent)**
 *    - Scheduled in PE's event queue
 *    - Processes computation request
 *    - Invokes callback with response
 *
 * **Downstream/Upstream Relationships:**
 *
 * Logical flow: TrafficGenerator → PE
 * - TrafficGenerator.addDownStream(PE, "DSPE")
 * - Traffic flows "downstream" to PE
 *
 * Channel connections (bidirectional):
 * - Forward:  TG.DSPE → PE.USTrafficGenerator
 * - Reverse:  PE.USTrafficGenerator → TG.DSPE
 * - Both established via ChannelPortManager::ConnectPort()
 *
 * **Usage and Compilation:**
 *
 * Build:
 * ```bash
 * cd acalsim-workspace/projects/acalsim
 * mkdir -p build && cd build
 * cmake ..
 * make testCommunication
 * ```
 *
 * Run:
 * ```bash
 * ./testCommunication
 * ```
 *
 * Expected output:
 * ```
 * [TrafficGenerator] Issue traffic with transaction id: 1 at Tick=1
 * [TrafficGenerator] Issue traffic with transaction id: 2 at Tick=2
 * [PE] Process PEReqEvent with transaction id: 1 at Tick=6
 * [TrafficGenerator] Receive PERespPacket with transaction id: 1
 * [TrafficGenerator] Receive PE computation result : 800
 * [PE] Process PEReqEvent with transaction id: 2 at Tick=7
 * [TrafficGenerator] Receive PERespPacket with transaction id: 2
 * [TrafficGenerator] Receive PE computation result : 800
 * ```
 *
 * **Key Differences from Other Examples:**
 *
 * vs testSimChannel:
 * - No NOC intermediary
 * - Simpler event-based protocol
 * - Direct PE computation
 *
 * vs testPETile:
 * - No complex PE mesh topology
 * - Focus on communication patterns
 * - Simplified PE logic
 *
 * **Extension Points:**
 *
 * This example can be extended to:
 * - Add PE mesh with inter-PE routing
 * - Mix ports for synchronous PE-to-PE communication
 * - Add memory hierarchy (cache/DRAM)
 * - Implement realistic PE workloads
 * - Add NoC for multi-PE communication
 *
 * **Performance Considerations:**
 *
 * - Channel communication has overhead of event scheduling
 * - Suitable for loosely-coupled components
 * - Callback overhead minimal (function pointer call)
 * - EventPacket wrapping adds memory overhead
 * - Trade flexibility for slight performance cost
 *
 * @see PE.cc for Processing Element implementation
 * @see TrafficGenerator.cc for traffic injection logic
 * @see PEEvent.cc for PE event processing
 * @see TrafficEvent.cc for traffic event generation
 * @see PEReq.cc for request/response packet definitions
 * @see DataReq.cc for alternative packet types
 */

#include "ACALSim.hh"
using namespace acalsim;

/**
 * @brief Include simulator class headers
 *
 * PE.hh - Processing Element simulator with request handlers
 * TrafficGenerator.hh - Traffic injection engine with event generation
 */
#include "PE.hh"
#include "TrafficGenerator.hh"

/**
 * @class TestCommunicationTop
 * @brief Top-level simulation coordinator for mixed communication pattern example
 *
 * This class orchestrates a simple two-simulator system demonstrating channel-based
 * communication between a TrafficGenerator and Processing Element. It establishes
 * bidirectional channel connections and logical downstream relationships.
 *
 * **Architecture Setup:**
 * - Creates two simulators: TrafficGenerator and PE
 * - Establishes logical flow: TrafficGenerator → PE (downstream)
 * - Sets up bidirectional channel ports for event exchange
 *
 * **Channel Configuration:**
 * ```
 * Forward path (TG → PE):
 *   TrafficGenerator.MasterPort("DSPE") ──► PE.SlavePort("USTrafficGenerator")
 *   - TrafficGenerator pushes EventPackets containing PEReqEvents
 *   - PE receives events in its event queue
 *
 * Reverse path (PE → TG):
 *   PE.MasterPort("USTrafficGenerator") ──► TrafficGenerator.SlavePort("DSPE")
 *   - PE can send responses back (though this example uses callbacks)
 *   - Maintains bidirectional communication capability
 * ```
 *
 * **Port Naming Convention:**
 * - "DSPE" = Downstream PE (from TrafficGenerator's perspective)
 * - "USTrafficGenerator" = Upstream TrafficGenerator (from PE's perspective)
 * - Names must match on both sides of ChannelPortManager::ConnectPort()
 *
 * **Simulation Phases:**
 * 1. Construction: Create simulator instances
 * 2. Registration: Add to SimTop via addSimulator()
 * 3. Connection: Establish channels and downstream links
 * 4. Initialization: Each simulator's init() called
 * 5. Execution: run() processes events tick-by-tick
 * 6. Cleanup: finish() releases resources
 *
 * @note This is a minimal example focusing on communication patterns.
 *       For complex PE meshes, see testPETile example.
 */
class TestCommunicationTop : public SimTop {
public:
	/**
	 * @brief Constructor for TestCommunicationTop
	 *
	 * Initializes the base SimTop class. The actual simulator setup
	 * happens in registerSimulators() which is called by SimTop::init().
	 */
	TestCommunicationTop() : SimTop() {}

	/**
	 * @brief Register and connect all simulators in the system
	 *
	 * This method is called by SimTop::init() to set up the simulation topology.
	 * It performs three key tasks:
	 * 1. Instantiate simulator objects
	 * 2. Register them with the simulation framework
	 * 3. Establish communication channels and logical relationships
	 *
	 * **Simulator Creation:**
	 * - TrafficGenerator: Injects PEReqEvents into the system
	 * - PE: Processes requests and computes results
	 *
	 * **Logical Relationships:**
	 * ```cpp
	 * trafficGenerator->addDownStream(pe, "DSPE");
	 * ```
	 * - Establishes that PE is downstream from TrafficGenerator
	 * - Enables TrafficGenerator to retrieve PE via getDownStream("DSPE")
	 * - Used for directing traffic to the correct destination
	 *
	 * **Channel Connections:**
	 * ```cpp
	 * ChannelPortManager::ConnectPort(trafficGenerator, pe, "DSPE", "USTrafficGenerator");
	 * ChannelPortManager::ConnectPort(pe, trafficGenerator, "USTrafficGenerator", "DSPE");
	 * ```
	 *
	 * First connection (TG → PE):
	 * - Source: TrafficGenerator's MasterChannelPort "DSPE"
	 * - Destination: PE's SlaveChannelPort "USTrafficGenerator"
	 * - Direction: Forward (requests)
	 *
	 * Second connection (PE → TG):
	 * - Source: PE's MasterChannelPort "USTrafficGenerator"
	 * - Destination: TrafficGenerator's SlaveChannelPort "DSPE"
	 * - Direction: Reverse (responses - though this example uses callbacks)
	 *
	 * **Channel vs Port Terminology:**
	 * - "Channel": The underlying communication mechanism (lock-free queue)
	 * - "Port": The endpoint attached to a simulator (Master sends, Slave receives)
	 * - ChannelPortManager manages the mapping between ports and channels
	 *
	 * @note The order of addSimulator() calls doesn't matter for channels,
	 *       but may affect initialization order for other communication types.
	 */
	void registerSimulators() override {
		// Create simulators
		SimBase* trafficGenerator = (SimBase*)new TrafficGenerator("Traffic Generator");
		SimBase* pe               = (SimBase*)new PE("PE Simulator");

		// register Simulators
		this->addSimulator(trafficGenerator);
		this->addSimulator(pe);

		trafficGenerator->addDownStream(pe, "DSPE");

		// connect channels
		ChannelPortManager::ConnectPort(trafficGenerator, pe, "DSPE", "USTrafficGenerator");
		ChannelPortManager::ConnectPort(pe, trafficGenerator, "USTrafficGenerator", "DSPE");
	}
};

/**
 * @brief Main entry point for testCommunication example
 *
 * Demonstrates the standard three-phase simulation workflow in ACALSim:
 *
 * **Phase 1: Initialization**
 * ```cpp
 * top->init(argc, argv);
 * ```
 * - Parses command-line arguments
 * - Calls registerSimulators() to build topology
 * - Invokes each simulator's init() method
 * - TrafficGenerator schedules initial TrafficEvents
 * - Simulation ready to run
 *
 * **Phase 2: Execution**
 * ```cpp
 * top->run();
 * ```
 * - Processes events tick-by-tick in a loop
 * - Each tick consists of multiple phases:
 *   1. Process scheduled events (SimEvent::process())
 *   2. Transfer channel packets (Phase 2)
 *   3. Advance to next tick
 * - Continues until event queue empty or max tick reached
 * - In this example:
 *   - Tick 1: TrafficEvent #1 sends PEReqEvent
 *   - Tick 2: TrafficEvent #2 sends PEReqEvent
 *   - Tick 6: PEReqEvent #1 processes
 *   - Tick 7: PEReqEvent #2 processes
 *
 * **Phase 3: Cleanup**
 * ```cpp
 * top->finish();
 * ```
 * - Calls each simulator's cleanup() method
 * - Releases allocated resources
 * - Prints simulation statistics (if enabled)
 * - Ensures proper shutdown
 *
 * **Command-Line Options:**
 *
 * The framework supports various options (passed to init):
 * ```bash
 * ./testCommunication --max-tick 1000    # Limit simulation duration
 * ./testCommunication --debug            # Enable debug output
 * ./testCommunication --help             # Show all options
 * ```
 *
 * **Return Value:**
 * - 0: Simulation completed successfully
 * - Non-zero: Error occurred during simulation
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, non-zero on error
 *
 * @note The global variable 'top' is declared in ACALSim.hh and used
 *       throughout the framework for accessing simulation state.
 */
int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestCommunicationTop>();
	top->init(argc, argv);
	top->run();
	top->finish();

	return 0;
}
