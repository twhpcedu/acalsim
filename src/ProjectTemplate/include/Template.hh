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
 * @file Template.hh
 * @brief Comprehensive template for creating custom SimBase simulator components
 *
 * This header file demonstrates how to create custom simulator components in ACALSim by
 * inheriting from SimBase (or CPPSimBase for C++-only simulators). It covers the complete
 * lifecycle: initialization, execution (step function), communication via ports/channels,
 * event handling, and cleanup.
 *
 * **SimBase Inheritance Hierarchy:**
 * ```
 * SimBase (Abstract base class)
 *   │
 *   ├─ CPPSimBase (C++-only simulators, no SystemC)
 *   │    │
 *   │    └─ Template (this example)
 *   │         └─ User-defined hardware models
 *   │
 *   └─ SystemCSimBase (SystemC-integrated simulators)
 *        └─ User SystemC modules
 * ```
 *
 * **Simulator Component Lifecycle:**
 * ```
 * 1. Construction:
 *    Template("MySimulator")
 *      └─ Call CPPSimBase(name) constructor
 *           └─ Allocate ports, initialize member variables
 *
 * 2. Initialization (before simulation loop):
 *    init()
 *      ├─ Retrieve configuration parameters
 *      ├─ Create and register MasterPorts/SlavePorts
 *      ├─ Create and register MasterChannelPorts/SlaveChannelPorts
 *      ├─ Register SimModules (if using hierarchical design)
 *      ├─ Schedule initial events
 *      └─ Initialize internal state
 *
 * 3. Execution (every simulation iteration):
 *    stepWrapper()  // Called by ThreadManager in Phase 1
 *      ├─ processInboundChannelRequests()
 *      ├─ processEventQueue()
 *      ├─ triggerRetryCallback()
 *      └─ step()  ◄─── USER IMPLEMENTS THIS
 *           └─ Hardware modeling logic
 *
 * 4. Cleanup (after simulation loop):
 *    cleanup()
 *      ├─ Flush event queues
 *      ├─ Generate component-specific statistics
 *      └─ Release allocated resources
 *
 * 5. Destruction:
 *    ~Template()
 *      └─ Final cleanup (automatic via RAII)
 * ```
 *
 * **Required Method Implementations:**
 *
 * 1. **Constructor:**
 *    - Pass simulator name to CPPSimBase constructor
 *    - Initialize member variables
 *    - DO NOT create ports here (do in init())
 *
 * 2. **init():**
 *    - Retrieve parameters from SimTop
 *    - Create and register ports
 *    - Schedule initial events
 *    - Setup internal data structures
 *
 * 3. **step():**
 *    - Core hardware modeling logic
 *    - Executed every iteration (if simulator is active)
 *    - Send packets via MasterPorts
 *    - Process received packets from SlavePorts
 *    - Schedule future events
 *
 * 4. **cleanup():**
 *    - Called once at simulation end
 *    - Generate final statistics
 *    - Release resources
 *
 * **Communication Patterns:**
 *
 * A. **Port-Based Communication (for intra-simulator components):**
 * ```cpp
 * class CPUSimulator : public CPPSimBase {
 *     MasterPort* req_port;  // Send requests
 *     SlavePort* resp_port;  // Receive responses
 *
 * public:
 *     void init() override {
 *         // Create ports with queue sizes
 *         req_port = new MasterPort("req");
 *         resp_port = new SlavePort("resp", 16, new RoundRobinArbiter());
 *
 *         // Register with SimBase
 *         this->addMasterPort(req_port);
 *         this->addSlavePort(resp_port);
 *     }
 *
 *     void step() override {
 *         // Send request (if port available)
 *         if (has_work()) {
 *             auto pkt = top->getRecycleContainer()->pop<RequestPacket>();
 *             pkt->addr = 0x1000;
 *             if (!req_port->push(pkt)) {
 *                 // Port full, retry next iteration
 *             }
 *         }
 *
 *         // Process responses
 *         while (!resp_port->empty()) {
 *             auto resp = resp_port->pop();
 *             processResponse(resp);
 *             top->getRecycleContainer()->recycle(resp);
 *         }
 *     }
 * };
 * ```
 *
 * B. **Channel-Based Communication (for inter-simulator messaging):**
 * ```cpp
 * class Accelerator : public CPPSimBase, public ChannelPortManager {
 *     MasterChannelPort* to_host;
 *     SlaveChannelPort* from_host;
 *
 * public:
 *     void init() override {
 *         // Channels created by ChannelPortManager::ConnectChannelPort()
 *         // in SimTop::registerSimulators()
 *     }
 *
 *     void processInboundChannelRequests() override {
 *         // Called automatically before step()
 *         while (!from_host->empty()) {
 *             auto req = from_host->pop();
 *             handleHostCommand(req);
 *         }
 *     }
 *
 *     void step() override {
 *         if (result_ready()) {
 *             auto result = std::make_shared<AccelResult>();
 *             to_host->push(result);  // Send to host CPU
 *         }
 *     }
 * };
 * ```
 *
 * **Event-Driven Programming:**
 * ```cpp
 * class TimerSimulator : public CPPSimBase {
 *     class TimerEvent : public SimEvent {
 *     public:
 *         uint32_t timer_id;
 *         TimerEvent(Tick when, uint32_t id)
 *             : SimEvent(when), timer_id(id) {}
 *     };
 *
 * public:
 *     void init() override {
 *         // Schedule initial timer events
 *         auto event = new TimerEvent(100, 0);  // Fire at tick 100
 *         this->schedule(event, &TimerSimulator::handleTimeout);
 *     }
 *
 *     void handleTimeout(SimEvent* evt) {
 *         auto timer_evt = static_cast<TimerEvent*>(evt);
 *         CLASS_INFO << "Timer " << timer_evt->timer_id << " fired!";
 *
 *         // Reschedule for next timeout
 *         auto next = new TimerEvent(getCurrentTick() + 100, timer_evt->timer_id);
 *         this->schedule(next, &TimerSimulator::handleTimeout);
 *     }
 * };
 * ```
 *
 * **SimModule Hierarchical Design:**
 * ```cpp
 * class CoreSimModule : public SimModule {
 *     // Sub-component within a SimBase
 *     void step() override {
 *         // Execute core logic
 *     }
 * };
 *
 * class CPUSimulator : public CPPSimBase {
 *     CoreSimModule* core0;
 *     CoreSimModule* core1;
 *
 * public:
 *     void registerModules() override {
 *         core0 = new CoreSimModule("Core0");
 *         core1 = new CoreSimModule("Core1");
 *
 *         this->addModule(core0);
 *         this->addModule(core1);
 *     }
 *
 *     void step() override {
 *         // Modules execute via their own step() calls
 *         // Coordinate between modules here
 *     }
 * };
 * ```
 *
 * **Parameter Access:**
 * ```cpp
 * void init() override {
 *     // Retrieve from SimTop configuration
 *     int num_cores = top->getParameter<int>("cpu", "cores");
 *     float freq = top->getParameter<float>("cpu", "frequency");
 *     std::string arch = top->getParameter<std::string>("cpu", "arch");
 *
 *     // Use in initialization
 *     for (int i = 0; i < num_cores; i++) {
 *         cores.push_back(new Core(freq, arch));
 *     }
 * }
 * ```
 *
 * **Activity Tracking:**
 * Simulators are only executed in Phase 1 if they are "active":
 * ```
 * SimBase is active if ANY of:
 *   - Event queue is non-empty
 *   - Inbound channel requests pending
 *   - Pending activity flag set (manual override)
 *   - Has SimModules that are active
 *
 * To keep simulator active:
 *   this->setPendingActivityFlag();
 * ```
 *
 * **Memory Management Best Practices:**
 * ```cpp
 * // GOOD: Use RecycleContainer for packets/events
 * auto pkt = top->getRecycleContainer()->pop<MyPacket>();
 * pkt->data = 42;
 * req_port->push(pkt);
 * // Later: recycle instead of delete
 * top->getRecycleContainer()->recycle(pkt);
 *
 * // BAD: Manual new/delete (causes memory fragmentation)
 * auto pkt = new MyPacket();
 * req_port->push(pkt);
 * delete pkt;  // Don't do this!
 * ```
 *
 * **Logging and Debugging:**
 * ```cpp
 * void step() override {
 *     CLASS_INFO << "Processing request at tick " << getCurrentTick();
 *     CLASS_WARNING << "Queue almost full: " << queue.size();
 *     CLASS_ERROR << "Invalid packet type: " << pkt->type;
 *     CLASS_ASSERT_MSG(addr < 0x10000, "Address out of range: " << addr);
 * }
 * ```
 *
 * **Complete Example: Simple CPU Model:**
 * ```cpp
 * class SimpleCPU : public CPPSimBase {
 *     MasterPort* imem_port;  // Instruction fetch
 *     MasterPort* dmem_port;  // Data access
 *     SlavePort* resp_port;   // Responses from memory
 *
 *     uint64_t pc;            // Program counter
 *     bool waiting_fetch;
 *
 * public:
 *     SimpleCPU(const std::string& name) : CPPSimBase(name), pc(0), waiting_fetch(false) {}
 *
 *     void init() override {
 *         // Create ports
 *         imem_port = new MasterPort("imem_req");
 *         dmem_port = new MasterPort("dmem_req");
 *         resp_port = new SlavePort("resp", 8, new RoundRobinArbiter());
 *
 *         this->addMasterPort(imem_port);
 *         this->addMasterPort(dmem_port);
 *         this->addSlavePort(resp_port);
 *
 *         // Schedule initial fetch
 *         pc = top->getParameter<uint64_t>("cpu", "entry_point");
 *         fetchInstruction();
 *     }
 *
 *     void fetchInstruction() {
 *         auto req = top->getRecycleContainer()->pop<MemoryRequest>();
 *         req->addr = pc;
 *         req->type = REQ_IFETCH;
 *
 *         if (imem_port->push(req)) {
 *             waiting_fetch = true;
 *         } else {
 *             // Retry next cycle
 *         }
 *     }
 *
 *     void step() override {
 *         // Process responses
 *         while (!resp_port->empty()) {
 *             auto resp = resp_port->pop();
 *             if (resp->type == RESP_IFETCH) {
 *                 executeInstruction(resp->data);
 *                 pc += 4;
 *                 waiting_fetch = false;
 *             }
 *             top->getRecycleContainer()->recycle(resp);
 *         }
 *
 *         // Fetch next instruction if ready
 *         if (!waiting_fetch) {
 *             fetchInstruction();
 *         }
 *     }
 *
 *     void executeInstruction(uint32_t inst) {
 *         // Decode and execute...
 *         CLASS_INFO << "Executed instruction at PC=" << (pc);
 *     }
 *
 *     void cleanup() override {
 *         CLASS_INFO << "Final PC: " << pc;
 *         CLASS_INFO << "Total instructions: " << (pc / 4);
 *     }
 * };
 * ```
 *
 * @see SimBase For base class interface documentation
 * @see CPPSimBase For C++-specific simulator base
 * @see MasterPort For sending packets
 * @see SlavePort For receiving packets
 * @see SimEvent For event-driven programming
 * @see RecycleContainer For memory management
 */

#pragma once

#include <string>

#include "ACALSim.hh"
using namespace acalsim;

/**
 * @brief Template simulator component for demonstration purposes.
 *
 * This class demonstrates the minimal structure required for a custom simulator
 * component in ACALSim. It inherits from CPPSimBase and implements the required
 * lifecycle methods: init(), step(), and cleanup().
 *
 * **Usage in SimTop:**
 * ```cpp
 * void registerSimulators() override {
 *     auto template_sim = new Template("TemplateSimulator");
 *     this->addSimulator(template_sim);
 * }
 * ```
 *
 * **Customization Workflow:**
 * 1. Rename Template to your component name (e.g., CPUCore, CacheController)
 * 2. Add member variables for state (registers, queues, counters)
 * 3. Implement init() to create ports and load configuration
 * 4. Implement step() for hardware modeling logic
 * 5. Implement cleanup() for statistics and resource release
 *
 * @note This template operates independently without SimModules. For hierarchical
 *       designs with sub-components, override registerModules() to add SimModule instances.
 */
class Template : public CPPSimBase {
public:
	/**
	 * @brief Constructor for the Template simulator.
	 *
	 * @param name Unique identifier for this simulator instance (used in logging and tracing)
	 *
	 * @note Constructor should only initialize basic member variables. Port creation
	 *       and complex initialization should be done in init().
	 */
	Template(const std::string& name) : CPPSimBase(name) {}

	/**
	 * @brief Destructor for the Template simulator.
	 *
	 * @note Cleanup is automatically handled by RAII. Explicit cleanup should be done
	 *       in cleanup() method instead of destructor.
	 */
	~Template() {}

	/**
	 * @brief Simulator-level initialization called before simulation loop starts.
	 *
	 * This method is called once during SimTop::init() after all simulators are
	 * registered and before the simulation loop begins.
	 *
	 * **Typical Initialization Tasks:**
	 * 1. Retrieve configuration parameters:
	 *    ```cpp
	 *    int buffer_size = top->getParameter<int>("config_name", "buffer_size");
	 *    ```
	 *
	 * 2. Create and register communication ports:
	 *    ```cpp
	 *    auto req_port = new MasterPort("req");
	 *    this->addMasterPort(req_port);
	 *
	 *    auto resp_port = new SlavePort("resp", 16, new RoundRobinArbiter());
	 *    this->addSlavePort(resp_port);
	 *    ```
	 *
	 * 3. Create and register channel ports (if using ChannelPortManager):
	 *    ```cpp
	 *    // Channels are created in SimTop::registerSimulators()
	 *    // This method can access them after ConnectChannelPort() is called
	 *    ```
	 *
	 * 4. Schedule initial events:
	 *    ```cpp
	 *    auto init_event = new MyEvent(getCurrentTick() + 10);
	 *    this->schedule(init_event, &Template::handleEvent);
	 *    ```
	 *
	 * 5. Initialize internal state:
	 *    ```cpp
	 *    state = IDLE;
	 *    counter = 0;
	 *    queue.clear();
	 *    ```
	 *
	 * 6. (Optional) Register SimModules for hierarchical design:
	 *    ```cpp
	 *    // Override registerModules() instead of adding modules here
	 *    ```
	 *
	 * @note This method is called in the main thread before parallel execution starts.
	 * @note Do NOT perform heavy computation here; keep initialization lightweight.
	 */
	void init() {
		// Example: Retrieve configuration parameters
		// int param = top->getParameter<int>("template", "param_name");

		// Example: Create and register ports
		// auto req_port = new MasterPort("req");
		// this->addMasterPort(req_port);

		// Example: Schedule initial events
		// auto event = new MyEvent(getCurrentTick() + 100);
		// this->schedule(event, &Template::handleEvent);

		// Note: SimModules can be added by overriding registerModules()
		// A SimBase can operate independently without any SimModules
	}

	/**
	 * @brief Core hardware modeling logic executed every simulation iteration.
	 *
	 * This method is called by the ThreadManager during Phase 1 of each simulation
	 * iteration IF the simulator is active (has pending events, channel requests,
	 * or manually set activity flag).
	 *
	 * **Execution Context:**
	 * - Called in parallel with other SimBase::step() methods
	 * - DO NOT access shared state without synchronization
	 * - Communication with other simulators must use ports or channels
	 *
	 * **Common step() Patterns:**
	 *
	 * 1. State machine progression:
	 *    ```cpp
	 *    switch (state) {
	 *        case IDLE:
	 *            if (work_available()) state = PROCESSING;
	 *            break;
	 *        case PROCESSING:
	 *            doWork();
	 *            state = IDLE;
	 *            break;
	 *    }
	 *    ```
	 *
	 * 2. Packet transmission:
	 *    ```cpp
	 *    if (has_request()) {
	 *        auto pkt = top->getRecycleContainer()->pop<RequestPacket>();
	 *        pkt->data = generateRequest();
	 *        if (!req_port->push(pkt)) {
	 *            // Port full, retry next iteration
	 *        }
	 *    }
	 *    ```
	 *
	 * 3. Packet reception:
	 *    ```cpp
	 *    while (!resp_port->empty()) {
	 *        auto resp = resp_port->pop();
	 *        processResponse(resp);
	 *        top->getRecycleContainer()->recycle(resp);
	 *    }
	 *    ```
	 *
	 * 4. Event-driven delays:
	 *    ```cpp
	 *    auto delayed_action = new ActionEvent(getCurrentTick() + latency);
	 *    this->schedule(delayed_action, &Template::handleAction);
	 *    ```
	 *
	 * 5. Conditional activity:
	 *    ```cpp
	 *    if (long_running_operation()) {
	 *        this->setPendingActivityFlag();  // Stay active next iteration
	 *    }
	 *    ```
	 *
	 * @note This method is called ONLY when the simulator is active.
	 * @note If there are no pending events/requests, step() will NOT be called.
	 * @note Avoid busy-waiting; use events for timed actions instead.
	 */
	void step() {
		// Example: Process incoming packets
		// while (!resp_port->empty()) {
		//     auto pkt = resp_port->pop();
		//     handlePacket(pkt);
		//     top->getRecycleContainer()->recycle(pkt);
		// }

		// Example: Send outgoing packets
		// if (has_work()) {
		//     auto req = top->getRecycleContainer()->pop<MyPacket>();
		//     req->data = 42;
		//     req_port->push(req);
		// }

		// Example: State machine logic
		// updateState();
	}

	/**
	 * @brief Cleanup function called once after simulation loop ends.
	 *
	 * This method is called during SimTop::finish() after the simulation loop
	 * terminates (either by reaching max_tick or global termination condition).
	 *
	 * **Typical Cleanup Tasks:**
	 * 1. Generate component-specific statistics:
	 *    ```cpp
	 *    CLASS_INFO << "Total packets processed: " << packet_count;
	 *    CLASS_INFO << "Average latency: " << (total_latency / packet_count);
	 *    ```
	 *
	 * 2. Flush pending events (if needed):
	 *    ```cpp
	 *    while (!event_queue.empty()) {
	 *        auto evt = event_queue.pop();
	 *        delete evt;
	 *    }
	 *    ```
	 *
	 * 3. Write trace files or logs:
	 *    ```cpp
	 *    trace_file.close();
	 *    ```
	 *
	 * 4. Release manually allocated resources:
	 *    ```cpp
	 *    delete[] buffer;
	 *    ```
	 *
	 * @note Event queues and ports are automatically cleaned up by SimBase.
	 * @note Do NOT delete recycled packets here; RecycleContainer handles cleanup.
	 * @note This runs in the main thread after parallel execution completes.
	 */
	void cleanup() {
		// Example: Report final statistics
		// CLASS_INFO << "Simulation completed for " << getName();
		// CLASS_INFO << "Total operations: " << operation_count;

		// Example: Release resources
		// delete[] allocated_buffer;

		// Note: Event queues and ports are cleaned up automatically
	}
};
