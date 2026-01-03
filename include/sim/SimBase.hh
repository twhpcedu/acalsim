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
 * @file SimBase.hh
 * @brief Core simulator base classes for discrete-event parallel simulation
 *
 * SimBase provides the foundation for all simulator instances in ACALSim, implementing
 * discrete-event simulation with thread-safe communication, modular architecture,
 * and two-phase parallel execution.
 *
 * **Simulator Architecture:**
 * ```
 * ┌────────────────────────────────────────────────────────────┐
 * │                       SimBase                              │
 * │  (Abstract Base for All Simulators)                        │
 * ├────────────────────────────────────────────────────────────┤
 * │  • Event Queue (Priority Queue of SimEvents)               │
 * │  • Channel Ports (Thread-Safe Inter-Simulator Comm)        │
 * │  • Simulation Ports (Master/Slave for Modules)             │
 * │  • Module Registry (Hierarchical Component Management)     │
 * │  • Task Association (For Parallel Execution)               │
 * └────────┬───────────────────────┬───────────────────────────┘
 *          │                       │
 *    ┌─────▼─────┐           ┌─────▼──────┐
 *    │ CPPSimBase│           │ SystemC... │
 *    │ (C++ Impl)│           │ (SC Impl)  │
 *    └─────┬─────┘           └────────────┘
 *          │
 *    ┌─────▼──────────────────────────────┐
 *    │ User Simulator (e.g., CPUSimulator)│
 *    │   init()     - Setup hardware      │
 *    │   step()     - Execute iteration   │
 *    │   cleanup()  - Release resources   │
 *    └────────────────────────────────────┘
 * ```
 *
 * **Simulator Lifecycle:**
 * ```
 * SimTop                   SimBase (Worker Thread)              Modules
 *    |                            |                                |
 *    |-- init() ----------------->|                                |
 *    |                            |-- registerModules() ---------->|
 *    |                            |-- init() (user code) --------->|
 *    |                            |-- initSimPort() -------------->|
 *    |                            |                                |
 *    |== Simulation Loop ======   |                                |
 *    |-- startPhase1() ---------->|                                |
 *    |                            |-- stepWrapper() [Main Loop]    |
 *    |                            |    if (hasWork) {              |
 *    |                            |      step() (user code) ------>|
 *    |                            |      process events             |
 *    |                            |    }                           |
 *    |-- finishPhase1() <---------|                                |
 *    |-- startPhase2() ---------->|                                |
 *    |                            |-- processInBoundChannelReq()   |
 *    |                            |-- interIterationUpdate()       |
 *    |-- finishPhase2() <---------|                                |
 *    |                            |                                |
 *    |-- cleanup() --------------->|                                |
 *    |                            |-- cleanup() (user code) ------>|
 * ```
 *
 * **Event-Driven Execution Model:**
 * ```
 * Time: 0     10    20    30    40    50    60
 *       |     |     |     |     |     |     |
 * Event Queue:
 *   [Event1@10] → [Event2@20] → [Event3@50]
 *
 * Iteration 1 (Tick 10):
 *   step() called → Event1 processed → schedules Event4@30
 *
 * Iteration 2 (Tick 20):
 *   step() called → Event2 processed → no new events
 *
 * Iteration 3 (Tick 30):
 *   step() called → Event4 processed → schedules Event5@60
 *
 * Iteration 4 (Tick 50):
 *   step() called → Event3 processed → done
 * ```
 *
 * **Module Hierarchy:**
 * ```
 * CPUSimulator (SimBase)
 *    ├─ FetchModule (SimModule)
 *    │    └─ InstructionCache (SimModule)
 *    ├─ DecodeModule (SimModule)
 *    ├─ ExecuteModule (SimModule)
 *    │    ├─ ALU (SimModule)
 *    │    └─ FPU (SimModule)
 *    └─ MemoryModule (SimModule)
 *         └─ DataCache (SimModule)
 * ```
 *
 * **Key Features:**
 * - **Discrete-Event Simulation**: Priority queue-based event scheduling
 * - **Modular Architecture**: Hierarchical component organization with SimModule
 * - **Thread-Safe Communication**: Channel ports for inter-simulator messages
 * - **Port-Based Connectivity**: Master/slave ports for module-to-module data flow
 * - **Two-Phase Execution**: Parallel computation + synchronized communication
 * - **Event Recycling**: Memory-efficient event object reuse
 * - **Flexible Initialization**: User-defined init/step/cleanup hooks
 *
 * **Use Cases:**
 *
 * | Use Case                    | Description                                              | Example |
 * |-----------------------------|----------------------------------------------------------|---------------------------------------|
 * | CPU Core Simulation         | Model instruction pipeline with fetch/decode/execute     | RISC-V 5-stage pipeline |
 * | GPU Compute Simulation      | Parallel kernel execution with memory hierarchy          | CUDA-style compute
 * simulation        | | NoC Router Simulation       | Packet routing with buffer management                    | Mesh
 * NoC with virtual channels       | | Memory Controller           | DRAM timing and scheduling | DDR4 controller with
 * page policies   | | Accelerator Modeling        | Custom compute units with DMA                            | NPU/TPU
 * matrix multiplication        |
 *
 * **Thread Safety:**
 * - Each SimBase runs on dedicated worker thread (managed by ThreadManager)
 * - Channel ports provide lock-free inter-simulator communication
 * - Event queue is single-threaded (no synchronization needed)
 * - Port synchronization occurs at Phase 2 barriers
 *
 * **Performance Characteristics:**
 *
 * | Operation                   | Complexity       | Notes                                  |
 * |-----------------------------|------------------|----------------------------------------|
 * | scheduleEvent()             | O(log N)         | N = events in queue (heap insert)      |
 * | drainEventQueue()           | O(log N)         | Extract min from heap                  |
 * | addModule()                 | O(1) amortized   | HashMap insert                         |
 * | sendPacketViaChannel()      | O(1)             | Lock-free queue push                   |
 * | processInBoundChannelReq()  | O(M)             | M = inbound channel packets            |
 *
 * @see SimTop For top-level orchestration
 * @see SimModule For modular components within simulators
 * @see ChannelPort For inter-simulator communication
 * @see SimPort For module-to-module data flow
 * @see ThreadManager For parallel execution coordination
 *
 * @example
 * @code
 * // Example 1: Simple CPU simulator with pipeline stages
 * class SimpleCPUSimulator : public CPPSimBase {
 * public:
 *     SimpleCPUSimulator(const std::string& name, SimTopBase* top)
 *         : CPPSimBase(name), top(top) {}
 *
 *     void init() override {
 *         // Register pipeline stage modules
 *         fetchModule = new FetchModule("Fetch", this);
 *         addModule(fetchModule);
 *
 *         decodeModule = new DecodeModule("Decode", this);
 *         addModule(decodeModule);
 *
 *         executeModule = new ExecuteModule("Execute", this);
 *         addModule(executeModule);
 *
 *         // Schedule first instruction fetch
 *         auto* fetchEvent = new FetchInstructionEvent();
 *         scheduleEvent(fetchEvent, top->getGlobalTick() + 1);
 *     }
 *
 *     void step() override {
 *         // Process all events at current tick
 *         Tick curTick = top->getGlobalTick();
 *         while (!eventQueueEmpty() && getEventNextTick() == curTick) {
 *             SimEvent* event = drainEventQueue(curTick);
 *             event->process();
 *
 *             // Recycle event if managed
 *             if (event->isManaged()) {
 *                 top->getRecycleContainer()->recycle(event);
 *             }
 *         }
 *     }
 *
 *     void cleanup() override {
 *         LOG(INFO) << getName() << " completed " << insnsExecuted << " instructions";
 *     }
 *
 * private:
 *     SimTopBase* top;
 *     FetchModule* fetchModule;
 *     DecodeModule* decodeModule;
 *     ExecuteModule* executeModule;
 *     uint64_t insnsExecuted = 0;
 * };
 * @endcode
 *
 * @code
 * // Example 2: Memory controller with channel communication
 * class MemoryControllerSimulator : public CPPSimBase {
 * public:
 *     MemoryControllerSimulator(const std::string& name, SimTopBase* top)
 *         : CPPSimBase(name), top(top) {}
 *
 *     void init() override {
 *         // Create slave channel port to receive memory requests
 *         auto slavePort = createSlaveChannelPort("mem_req");
 *         registerSlaveChannelPort("mem_req", slavePort);
 *
 *         // Create master channel port to send responses
 *         auto masterPort = createMasterChannelPort("mem_resp");
 *         registerMasterChannelPort("mem_resp", masterPort);
 *     }
 *
 *     void step() override {
 *         // Process DRAM events (refresh, row buffer hits, etc.)
 *         Tick curTick = top->getGlobalTick();
 *         while (!eventQueueEmpty() && getEventNextTick() == curTick) {
 *             SimEvent* event = drainEventQueue(curTick);
 *             event->process();
 *         }
 *
 *         // Check for inbound memory requests via channels
 *         if (hasInboundChannelReq()) {
 *             processInBoundChannelRequest();
 *         }
 *     }
 *
 *     void accept(Tick when, SimPacket& pkt) override {
 *         if (auto* memReq = dynamic_cast<MemoryRequest*>(&pkt)) {
 *             // Schedule memory access with DRAM timing
 *             Tick latency = calculateDRAMLatency(memReq->addr);
 *             auto* respEvent = new MemoryResponseEvent(memReq);
 *             scheduleEvent(respEvent, when + latency);
 *         }
 *     }
 *
 *     void cleanup() override {
 *         LOG(INFO) << getName() << " handled " << requestsServed << " memory requests";
 *         LOG(INFO) << "  Row buffer hit rate: " << (rowBufferHits * 100.0 / requestsServed) << "%";
 *     }
 *
 * private:
 *     SimTopBase* top;
 *     uint64_t requestsServed = 0;
 *     uint64_t rowBufferHits = 0;
 *
 *     Tick calculateDRAMLatency(uint64_t addr) {
 *         // Simplified DRAM timing model
 *         if (isRowBufferHit(addr)) {
 *             rowBufferHits++;
 *             return CL_LATENCY; // ~15ns
 *         } else {
 *             return RAS_LATENCY + CL_LATENCY; // ~45ns
 *         }
 *     }
 * };
 * @endcode
 *
 * @code
 * // Example 3: NoC router with multiple ports
 * class NoCRouterSimulator : public CPPSimBase {
 * public:
 *     NoCRouterSimulator(const std::string& name, SimTopBase* top, int x, int y)
 *         : CPPSimBase(name), top(top), xPos(x), yPos(y) {}
 *
 *     void init() override {
 *         // Create master/slave channel ports for 4 directions (N, S, E, W)
 *         for (const auto& dir : {"north", "south", "east", "west"}) {
 *             auto inPort = createSlaveChannelPort(std::string(dir) + "_in");
 *             registerSlaveChannelPort(std::string(dir) + "_in", inPort);
 *
 *             auto outPort = createMasterChannelPort(std::string(dir) + "_out");
 *             registerMasterChannelPort(std::string(dir) + "_out", outPort);
 *         }
 *
 *         // Local port for attached compute node
 *         auto localPort = createSlaveChannelPort("local");
 *         registerSlaveChannelPort("local", localPort);
 *     }
 *
 *     void step() override {
 *         // Process routing events (packet forwarding)
 *         Tick curTick = top->getGlobalTick();
 *         while (!eventQueueEmpty() && getEventNextTick() == curTick) {
 *             SimEvent* event = drainEventQueue(curTick);
 *             event->process();
 *         }
 *     }
 *
 *     void accept(Tick when, SimPacket& pkt) override {
 *         if (auto* nocPkt = dynamic_cast<NoCPacket*>(&pkt)) {
 *             // XY routing algorithm
 *             std::string nextPort = routePacket(nocPkt->destX, nocPkt->destY);
 *
 *             // Schedule forwarding with router delay
 *             Tick routerDelay = ROUTER_PIPELINE_STAGES;
 *             sendPacketViaChannel(nextPort + "_out", routerDelay, 1, nocPkt);
 *             packetsRouted++;
 *         }
 *     }
 *
 *     void cleanup() override {
 *         LOG(INFO) << getName() << " at (" << xPos << "," << yPos << ")";
 *         LOG(INFO) << "  Routed " << packetsRouted << " packets";
 *     }
 *
 * private:
 *     SimTopBase* top;
 *     int xPos, yPos;
 *     uint64_t packetsRouted = 0;
 *
 *     std::string routePacket(int destX, int destY) {
 *         // XY routing: route in X dimension first, then Y
 *         if (destX < xPos) return "west";
 *         if (destX > xPos) return "east";
 *         if (destY < yPos) return "south";
 *         if (destY > yPos) return "north";
 *         return "local"; // Reached destination
 *     }
 * };
 * @endcode
 */

#pragma once

#include <string>

#include "channel/ChannelPortManager.hh"
#include "common/HashVector.hh"
#include "common/LinkManager.hh"
#include "event/SimEvent.hh"
#include "external/gem5/EventManager.hh"
#include "packet/EventPacket.hh"
#include "port/SimPortManager.hh"
#include "sim/SimTop.hh"
#include "sim/Task.hh"
#include "utils/HashableType.hh"
#include "utils/Logging.hh"

/**
 * @defgroup Simulator Base class
 * @brief Base class for a single-thread simulator
 */

namespace acalsim {

// Forward declarations
class SimModule;
class SimBase;

/**
 * @class ExitEvent
 * @brief Special event for simulator termination
 *
 * ExitEvent signals a simulator to exit its main loop and terminate worker thread execution.
 * Automatically sets exit flag and clears managed status to prevent recycling.
 *
 * **Usage Pattern:**
 * ```cpp
 * // Schedule simulator to exit at tick 10000
 * auto* exitEvent = new ExitEvent("CPU0");
 * scheduleEvent(exitEvent, 10000);
 * ```
 *
 * @see SimBase::issueExitEvent()
 * @see SimEvent
 */
class ExitEvent : public SimEvent {
private:
	std::string _name;  // Name of the exit event

public:
	/**
	 * @brief Default constructor.
	 *
	 * Sets the exit flag and clears the Managed flag.
	 */
	ExitEvent() : SimEvent() {
		this->setExitFlag();
		this->clearFlags(Managed);
	}

	/**
	 * @brief Named constructor.
	 * @param name The name to assign to this exit event.
	 *
	 * Sets the exit flag and clears the Managed flag.
	 */
	ExitEvent(std::string name) : SimEvent(), _name("ExitEvent_" + name) {
		this->setExitFlag();
		this->clearFlags(Managed);
	}

	/**
	 * @brief Destructor.
	 */
	~ExitEvent() {}

	/**
	 * @brief Renews the event with a new name.
	 * @param name The new name to assign.
	 *
	 * Resets the event state and updates the name.
	 */
	void renew(const std::string& name) {
		this->SimEvent::renew();

		this->_name = "ExitEvent_" + name;
		this->setExitFlag();
		this->clearFlags(Managed);
	}

	/**
	 * @brief Returns the name of this exit event.
	 * @return The name of the exit event.
	 */
	const std::string name() const { return this->_name; }

	/**
	 * @brief Process function called when the event is executed.
	 *
	 * Does nothing for exit events as the exit is handled separately.
	 */
	void process() {
		// Do nothing
	}
};

/**
 * @class DelayEvent
 * @brief Event for handling delayed operations, particularly for channel communications.
 *
 * Used for delayed packet processing, either as channel requests or general packet operations.
 */
class DelayEvent : public SimEvent {
private:
	SimPacket*  pkt    = nullptr;   // The packet associated with this event
	SimBase*    callee = nullptr;   // The simulation object that will handle this event
	std::string dsChannelPortName;  // Downstream channel port name for routing
	bool        isChannelReq;       // Whether this is a channel request

public:
	/**
	 * @brief Constructor for channel requests.
	 * @param _callee The simulator that will handle this event.
	 * @param _dsChannelPortName The downstream channel port name.
	 * @param _pkt The packet to process.
	 */
	DelayEvent(SimBase* _callee, std::string _dsChannelPortName, SimPacket* _pkt)
	    : SimEvent(), callee(_callee), dsChannelPortName(_dsChannelPortName), pkt(_pkt) {
		this->clearFlags(Managed);
		this->isChannelReq = true;
	}

	/**
	 * @brief Constructor for packet-only events.
	 * @param _pkt The packet to process.
	 */
	DelayEvent(SimPacket* _pkt) : SimEvent(), pkt(_pkt) {
		this->clearFlags(Managed);
		this->isChannelReq = false;
	}

	/**
	 * @brief Default constructor.
	 */
	DelayEvent() : SimEvent(), pkt(nullptr), callee(nullptr), dsChannelPortName(""), isChannelReq(false) {}

	/**
	 * @brief Destructor.
	 */
	~DelayEvent() {}

	/**
	 * @brief Renews the event with new channel request parameters.
	 * @param _callee The simulator that will handle this event.
	 * @param _dsChannelPortName The downstream channel port name.
	 * @param _pkt The packet to process.
	 */
	void renew(SimBase* _callee, std::string _dsChannelPortName, SimPacket* _pkt) {
		this->callee            = _callee;
		this->dsChannelPortName = _dsChannelPortName;
		this->pkt               = _pkt;
		this->isChannelReq      = true;
	}

	/**
	 * @brief Renews the event with new packet.
	 * @param _pkt The packet to process.
	 */
	void renew(SimPacket* _pkt) {
		this->SimEvent::renew();
		this->pkt          = _pkt;
		this->isChannelReq = false;
	}

	/**
	 * @brief Sets the simulator that will handle this event.
	 * @param callee The target simulator.
	 */
	void setCallee(SimBase* callee) { this->callee = callee; }

	/**
	 * @brief Gets the packet associated with this event.
	 * @return The packet.
	 */
	const SimPacket* getPacket() const { return this->pkt; }

	/**
	 * @brief Gets the channel port name.
	 * @return The channel port name.
	 */
	const std::string getChannelPortName() { return this->dsChannelPortName; }

	/**
	 * @brief Process function called when the event is executed.
	 *
	 * Implementation defined in source file.
	 */
	void process();
};

/**
 * @class ChannelEventPacket
 * @brief A packet specifically for events sent through channels.
 */
class ChannelEventPacket : public EventPacket {
public:
	/**
	 * @brief Constructor.
	 * @param event The event associated with this packet.
	 * @param when The time when the event should be processed.
	 */
	ChannelEventPacket(SimEvent* event = nullptr, Tick when = 0) : EventPacket(event, when) {}

	/**
	 * @brief Destructor.
	 */
	~ChannelEventPacket() {}

	/**
	 * @brief Renews the packet with a new event and time.
	 * @param event The new event.
	 * @param when The new time.
	 */
	void renew(SimEvent* event, Tick when) { this->EventPacket::renew(event, when); }

	/**
	 * @brief Visits the simulator to process this packet.
	 * @param when The time when the visit occurs.
	 * @param simulator The simulator to visit.
	 *
	 * Implementation defined in source file.
	 */
	void visit(Tick when, SimBase& simulator) override;
};

/**
 * @class SimBase
 * @brief Base class for simulation objects.
 *
 * Provides the foundation for discrete event-driven simulation, handling event queues,
 * channel communications, port management, and thread synchronization.
 */
class SimBase : protected EventManager,
                public LinkManager<SimBase*>,
                public ChannelPortManager,
                public SimPortManager,
                virtual public HashableType {
	friend class EventPacket;  // Allow EventPacket to access private members

public:
	/**
	 * @brief Constructor.
	 * @param _name The name of this simulator.
	 */
	SimBase(std::string _name);

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~SimBase();

	/**
	 * @brief Pure virtual function for step wrapper implementation in derived classes.
	 */
	virtual void stepWrapperBase() = 0;

	/**
	 * @brief The main callable function passed to std::thread for simulator execution.
	 *
	 * Controls the execution flow of the simulator as a child thread.
	 */
	void stepWrapper();

	/**
	 * @brief Updates inter-iteration state variables.
	 * @return True if there's pending activity, false otherwise.
	 *
	 * Updates bPendingActivity and hasPendingActivityLastIteration right before
	 * the 2nd barrier synchronization in each iteration.
	 */
	virtual bool interIterationUpdate() = 0;

	/**
	 * @brief Processes all inbound channel requests.
	 *
	 * Pops packets from each inbound channel and executes accept/scheduleEvent functions.
	 */
	void processInBoundChannelRequest();

	/**
	 * @brief Drains requests from a specific inbound channel.
	 * @param channel_port The channel port to drain.
	 */
	void drainInBoundChannelRequest(SlaveChannelPort::SharedPtr channel_port);

	/**
	 * @brief Processes a single inbound packet.
	 * @param ptr Pointer to the packet.
	 *
	 * Executes accept/scheduleEvent function for an inbound channel packet.
	 */
	void processInBoundPacket(void* ptr);

	/**
	 * @brief Gets the simulator name.
	 * @return The name of this simulator.
	 */
	std::string getName() const { return this->name; }

	/**
	 * @brief Sets the corresponding task for the simulator.
	 * @param _task The task to set.
	 */
	void setTask(std::shared_ptr<Task> _task) { this->pTask = _task; }

	/**
	 * @brief Gets the corresponding task for the simulator.
	 * @return The task associated with this simulator.
	 */
	std::shared_ptr<Task> getTask() const { return this->pTask; }

	/**
	 * @brief Sets the simulator unique ID.
	 * @param i The ID to set.
	 */
	void setID(int i) { this->id = i; }

	/**
	 * @brief Gets the simulator unique ID.
	 * @return The unique ID of this simulator.
	 */
	int getID() const { return this->id; }

	/**
	 * @brief Registers a single module with this simulator.
	 * @param module The module to add.
	 */
	void addModule(SimModule* module);

	/**
	 * @brief Gets a module by name.
	 * @param name The name of the module to retrieve.
	 * @return Pointer to the requested module.
	 */
	SimModule* getModule(std::string name) const;

	/**
	 * @ingroup PortManagement
	 * @brief Initializes all slave ports.
	 *
	 * Final implementation that cannot be overridden.
	 */
	void initSimPort() override final;

	/**
	 * @ingroup PortManagement
	 * @brief Synchronizes all slave ports with master ports.
	 *
	 * Synchronizes all slave ports with the MasterPort and calls
	 * syncSimPort on all modules. Final implementation that cannot be overridden.
	 */
	void syncSimPort() override final;

	/**
	 * @brief Forces the step function to be called in the next iteration.
	 */
	void forceStepInNextIteration() { this->bForceStepInNextIteration = true; }

	/**
	 * @brief Checks if step is forced in the next iteration.
	 * @return True if step is forced, false otherwise.
	 */
	bool isForceStepInNextIteration() const { return this->bForceStepInNextIteration; }

	/**
	 * @brief Checks if the step should be executed due to inbound packets or users' intention
	 * @return True if step is forced, false otherwise.
	 */
	bool getStepInCurrIteration() const { return this->bStepInCurrIteration; }

	/**
	 * @brief Triggers retry callbacks.
	 *
	 * Final implementation that cannot be overridden.
	 */
	void triggerRetryCallback();

	/**
	 * @ingroup PortManagement
	 * @brief Checks if any SlavePort has a packet.
	 * @param t Current simulation time.
	 * @return True if any port has pending activity, false otherwise.
	 *
	 * Checks if any SlavePort in simulator or its modules has a packet
	 * in its entry. Final implementation that cannot be overridden.
	 */
	bool hasPendingActivityInSimPort(bool pipeRegisterDump) const override;

	/**
	 * @brief Clears the hasPacketInSlavePort flag.
	 *
	 * Final implementation that cannot be overridden.
	 */
	void clearHasPendingActivityInSimPortFlag() final;

	/**
	 * @brief Sets the channel connection to the SimTop thread.
	 * @param channelPort The channel port to set.
	 */
	void setToTopChannelPort(MasterChannelPort::SharedPtr channelPort) { this->toTopChannelPort = channelPort; }

	/**
	 * @brief Sets the channel connection from the SimTop thread.
	 * @param channelPort The channel port to set.
	 */
	void setFromTopChannelPort(SlaveChannelPort::SharedPtr channelPort) { this->fromTopChannelPort = channelPort; }

	/**
	 * @brief Pushes data pointer to the SimTop thread.
	 * @param ptr The pointer to push.
	 */
	void pushToTopChannelPort(SimPacket* const& ptr) { this->toTopChannelPort->push(ptr); }

	/**
	 * @brief Pops data pointer from the SimTop thread.
	 * @return The popped packet.
	 */
	SimPacket* popFromTopChannelPort() { return this->fromTopChannelPort->pop(); }

	/**
	 * @brief Gets the next tick for the entire simulator.
	 * @return The next tick when an event will occur.
	 *
	 * Virtual method that can be overridden in derived classes.
	 */
	virtual Tick getSimNextTick();

	/**
	 * @brief Gets the next tick from the event queue.
	 * @return The next tick when an event will occur.
	 */
	Tick getEventNextTick() const { return this->eventq->nextTick(); }

	/**
	 * @brief Drains all events at the current tick.
	 * @param curTick The current simulation time.
	 * @return The first event drained from the queue.
	 */
	SimEvent* drainEventQueue(Tick curTick) { return static_cast<SimEvent*>(this->eventq->serviceOne(curTick)); }

	/**
	 * @brief Checks if the event queue is empty.
	 * @return True if empty, false otherwise.
	 */
	bool eventQueueEmpty() const { return this->eventq->empty(); }

	/**
	 * @brief Checks if there was pending activity in the last iteration.
	 * @return True if there was pending activity, false otherwise.
	 */
	bool hasPendingActivityLastIteration() const { return bPendingActivityLastIteration; }

	/**
	 * @brief Sets the pending activity flag.
	 * @param _flag The value to set.
	 */
	void setPendingActivityLastIteration(bool _flag) { bPendingActivityLastIteration = _flag; }

	/**
	 * @brief Checks if there are pending requests in inbound channels.
	 * @return True if there are pending requests, false otherwise.
	 */
	bool hasInboundChannelReq() const { return bInboundChannelReq; }

	/**
	 * @brief Checks if there were inbound channel requests in the last iteration.
	 * @return True if there were requests, false otherwise.
	 */
	bool hasInboundChannelReqLastIteration() const { return bInboundChannelReqLastIteration; }

	/**
	 * @brief Sets the inbound channel request flag.
	 * @param _flag The value to set.
	 */
	void setInboundChannelReq(bool _flag = true) { bInboundChannelReq = _flag; }

	/**
	 * @brief Sets the inbound channel request last iteration flag.
	 * @param _flag The value to set.
	 */
	void setInboundChannelReqLastIteration(bool _flag) { bInboundChannelReqLastIteration = _flag; }

	/**
	 * @brief Handles notification of incoming channel data.
	 *
	 * Sets the inbound channel request flag when notified.
	 */
	void handleInboundNotification() override { this->setInboundChannelReq(); }

	/**
	 * @brief Issues an ExitEvent to terminate the simulation.
	 * @param when The time when the exit should occur.
	 */
	void issueExitEvent(Tick when);

	/**
	 * @brief Schedules an event.
	 * @param event The event to schedule.
	 * @param when The time when the event should occur.
	 *
	 * A wrapper to schedule an event, ensuring the scheduled time is valid.
	 */
	void scheduleEvent(SimEvent* event, Tick when) {
		CLASS_ASSERT_MSG(when > top->getGlobalTick() || !top->isRunning(),
		                 "An event is scheduled at Tick " + std::to_string(when) + " when Tick " +
		                     std::to_string(top->getGlobalTick()) + ".");
		this->eventq->schedule(event, when);
	}

	/**
	 * @brief Initializes the simulator and its ports.
	 *
	 * Wrapper that calls init() and initSimPort().
	 */
	void initWrapper() {
		this->init();
		this->initSimPort();
	}

	/**
	 * @brief Sends a packet via a channel with delays.
	 * @param dsChannelPortName The destination channel port name.
	 * @param localDelay The delay at the sender side.
	 * @param remoteDelay The delay at the receiver side.
	 * @param pkt The packet to send.
	 */
	void sendPacketViaChannel(std::string dsChannelPortName, Tick localDelay, Tick remoteDelay, SimPacket* pkt);

	/**
	 * @brief Initializes the simulator.
	 *
	 * Pure virtual function to be implemented by derived classes for simulator initialization.
	 */
	virtual void init() = 0;

	/**
	 * @brief Executes one step of the simulation.
	 *
	 * Pure virtual function to be implemented by derived classes for simulation stepping.
	 */
	virtual void step() = 0;

	/**
	 * @brief Cleans up after simulation ends.
	 *
	 * Pure virtual function to be implemented by derived classes for cleanup.
	 */
	virtual void cleanup() = 0;

	/**
	 * @brief Registers all the SimModule components.
	 *
	 * Virtual function that can be overridden to register modules.
	 */
	virtual void registerModules() {}

	/**
	 * @brief Checks if the simulator is SystemC-based.
	 * @return True if SystemC, false otherwise.
	 *
	 * Pure virtual function to be implemented by derived classes.
	 */
	virtual bool isSystemC() const = 0;

	/**
	 * @brief Accepts a packet at the specified time.
	 * @param when The time of acceptance.
	 * @param pkt The packet to accept.
	 *
	 * Virtual function that visits the packet on this simulator.
	 */
	virtual void accept(Tick when, SimPacket& pkt) { pkt.visit(when, *this); }

	/**
	 * @brief Handles ChannelEventPacket.
	 * @param when The time of handling.
	 * @param packet The packet to handle.
	 */
	void handler(int when, ChannelEventPacket* packet);

protected:
	/**
	 * @brief Sets the current iteration execution flag.
	 * @param _value The value to set.
	 */
	void setStepInCurrIteration(const bool& _value) { this->bStepInCurrIteration = _value; }

protected:
	const std::string name;  // Simulator name

	// A thread-safe channel to the SimTop thread
	MasterChannelPort::SharedPtr toTopChannelPort = nullptr;

	// A thread-safe channel from the SimTop thread
	SlaveChannelPort::SharedPtr fromTopChannelPort = nullptr;

	// A list of all the modules
	HashVector<std::string, SimModule*> modules;

	// Pointer to the corresponding Task
	std::shared_ptr<Task> pTask = nullptr;

private:
	/**
	 * @brief Internal method to schedule an event.
	 * @param event The event to schedule.
	 * @param when The time when the event should occur.
	 *
	 * For internal use only, allows scheduling events to the current tick.
	 */
	void _scheduleEvent(SimEvent* event, Tick when) {
		CLASS_ASSERT_MSG(when >= top->getGlobalTick() || !top->isRunning(),
		                 "An event is scheduled at Tick " + std::to_string(when) + " when Tick " +
		                     std::to_string(top->getGlobalTick()) + "." + "event id " + event->getIDStr());
		this->eventq->schedule(event, when);
	}

	int  id;                             // A unique simulator ID in the whole system
	bool bPendingActivityLastIteration;  // Flag recording the status of pending activity in the last iteration
	bool bInboundChannelReq;             // Flag indicating whether there is pending activity in the inbound channel
	bool
	    bInboundChannelReqLastIteration;  // Flag recording the status of inbound channel requests in the last iteration
	bool bForceStepInNextIteration;       // Flag to force step() in the next iteration
	bool bStepInCurrIteration;  // Flag indicating the current iteration should be executed due to inbound packets or
	                            // users' intention
};

/**
 * @class CPPSimBase
 * @brief Base class for C++ (non-SystemC) simulators.
 *
 * Extends SimBase to provide specific functionality for C++ simulations.
 */
class CPPSimBase : public SimBase {
public:
	/**
	 * @brief Constructor.
	 * @param name The name of this simulator.
	 */
	CPPSimBase(const std::string& name) : SimBase(name) {}

	/**
	 * @brief Virtual destructor.
	 */
	virtual ~CPPSimBase() {}

	/**
	 * @brief Indicates this is not a SystemC simulator.
	 * @return Always false for CPPSimBase.
	 */
	bool isSystemC() const final { return false; }

	/**
	 * @brief Updates inter-iteration state variables.
	 * @return Result of the update.
	 *
	 * Final implementation for C++ simulators.
	 */
	bool interIterationUpdate() final;

	/**
	 * @brief Calls the stepWrapper method.
	 *
	 * Final implementation for the stepWrapperBase method.
	 */
	void stepWrapperBase() final { this->stepWrapper(); }

	/**
	 * @brief Initializes the simulator.
	 *
	 * Virtual method with empty default implementation.
	 */
	virtual void init() {}

	/**
	 * @brief Executes one step of the simulation.
	 *
	 * Virtual method with empty default implementation.
	 */
	virtual void step() {}

	/**
	 * @brief Cleans up after simulation ends.
	 *
	 * Virtual method with empty default implementation.
	 */
	virtual void cleanup() {}

	/**
	 * @brief Callback for master port retry on arbitration win.
	 * @param portName The name of the port that won arbitration.
	 *
	 * Virtual method that can be overridden by derived classes.
	 */
	virtual void masterPortRetry(MasterPort* port) {}
};

}  // end of namespace acalsim
