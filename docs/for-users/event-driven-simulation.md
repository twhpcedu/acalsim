# Event-Driven Simulation

<!--
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


---

- Author: Chia-Pao Chiang \<<daniel100373@gmail.com>\>
- Date: 2024/08/05

([Back To Documentation Portal](/docs/README.md))

## Overview

ACALSim is an event-driven, multi-threaded simulation framework designed for high-speed simulation. Each `SimBase`-derived simulator in ACALSim maintains its own private event queue. The control thread (`SimTop`-drived object) monitors pending events in each simulator to determine the number of cycles to fast-forward, optimizing simulation speed. This document introduces the fundamental concepts of event-driven simulation in ACALSim.

## What is Event-Driven Programming?

Event-driven programming is a paradigm that manages asynchronous operations using events and callbacks. Events represent occurrences or changes in the system's state, such as user input, network requests, timers, or errors. Callbacks are functions executed in response to these events, often passed as arguments to other functions or methods.
In event-driven programming, the program's flow is determined by events and their associated callbacks rather than following a predefined sequence of instructions. The program reacts to events as they occur. In our simulator framework, the main thread manages the progress of the time clock. When no events occur within a given time interval, the main thread can advance the time clock to the next scheduled event, enhancing simulation efficiency.

## The `SimEvent` Class

Our simulator framework implements `SimEvent`, which inherits the design of Event from gem5, and adopts EventQueue from gem5. `SimEvent` can simulate real hardware behavior, such as sending memory requests or performing computations. It allows customization of the gem5 Event design without altering its original implementation.

```cpp
class SimEvent : public Event, public RecyclableObject {
public:
	SimEvent(std::string _name = "SimEvent");
	virtual ~SimEvent() {}
	void        renew(std::string _name = "SimEvent");
	std::string getName() { return name; }

	system_id_t getID() const { return this->id; }
	std::string getIDStr() const { return std::to_string(this->id); }

protected:
	void releaseImpl() override;

private:
	system_id_t                     id = 0;
	std::string                     name;
	static std::atomic<system_id_t> uniqueEventId;
};
```

In the `SimEvent` prototype, we've incorporated specific members and member functions to enhance performance and functionality. Each `SimEvent` is assigned a unique ID, facilitating simulation tracing and debugging. Two key functions, `renew` and `releaseImpl`, are features of our simulator framework designed to minimize repeated allocation and deallocation operations. For a more in-depth explanation of these features, please refer to the [Recycle Container]() documentation.

With a foundational understanding of `SimEvent`, you can create custom events by inheriting from this class. Let's examine the `TrafficEvent` as an illustrative example. When developing a customized event, the most crucial aspect is overriding the process function. This function defines the specific behavior to be simulated when your event is processed.

```cpp
class TrafficEvent : public SimEvent {
public:
	TrafficEvent(SimBase* _sim, std::string name, int _tID)
	    : SimEvent(), sim(_sim), _name("TrafficEvent_" + name), tID(_tID) {
		this->clearFlags(this->Managed);
	}
	~TrafficEvent() {}

	const int         getTID() { return tID; }
	const std::string name() const override { return _name; }
	void              process() override;

private:
	std::string _name;

	// Transaction id
	int tID;

	// Simulator pointer
	SimBase* sim;
};
```

> **Note**: When initiating an event, the event's `flags` third bit is set to 1 by default, indicating that the event will be automatically released once it has been processed. You can find more details in `include/external/gem5/Event.hh`

## How to Schedule a Event

After events are initiated, they are scheduled into an event queue to simulate the resulting latency by specifying the time at which the event should be processed. There are two scheduling possibilities:

1. Schedule an event to the local event queue of the current simulator:
    ```cpp
    scheduleEvent((Event*)TrafficEvent, top->getGlobalTick() + 1);
    ```
2. Schedule an event to the event queue of another simulator:
    ```cpp
    PEReqEvent*  peReqEvent = new PEReqEvent(...);
    EventPacket* eventPkt   = new EventPacket(peReqEvent, top->getGlobalTick() + 5);

    this->pushToMasterChannelPort("DSPE", (void*)eventPkt);
    ```
	> **Note**: `top->getGlobalTick` retrieves the absolute time clock for the current iteration.

When scheduling an event, we determine the specific time at which the event will occur. This is the core mechanism of event-driven latency modeling. For instance, if a request transmission takes 5 cycles, setting `top->getGlobalTick() + 5` ensures that the event will be processed at that future time point, indicating the completion of the action.

## The `CallbackEvent` Class

In addition to regular events, we can define a CallbackEvent class, which incorporates a function pointer known as a callback function. In hardware simulation, the callback function serves to simulate the return path. For instance, as previously mentioned, an event can be used to simulate sending a memory request. When the downstream component has the data ready, the callback function can be executed immediately, effectively simulating the return path and completing the transaction cycle.

This approach allows for more flexible and realistic modeling of complex hardware interactions, particularly in scenarios where asynchronous operations and responses are common. By using callback events, we can accurately represent the bidirectional flow of data and control signals in a simulated hardware environment, enhancing the overall fidelity of the simulation.

```cpp
template <typename T>
class CallbackEvent : public SimEvent {
protected:
	uint64_t         tid;                       // transacation ID
	void*            callee         = nullptr;  // pointer of the callee
	std::function<T> callerCallback = nullptr;  // callback function for the caller

public:
	CallbackEvent(uint64_t _tid = 0, void* _callee = nullptr, std::function<T> _callback = nullptr)
	    : SimEvent(), tid(_tid), callee(_callee), callerCallback(_callback) {
		this->clearFlags(this->Managed);
	}
	virtual ~CallbackEvent() {}

	void renew(uint64_t _tid = 0, void* _callee = nullptr, std::function<T> _callback = nullptr) {
		this->tid            = _tid;
		this->callee         = _callee;
		this->callerCallback = _callback;
	}

	// Set the event as the ExitEvent
	void setExitFlag() {
		this->setFlags(IsExitEvent);
		this->clearFlags(Managed);
	}

	virtual void process() = 0;
};
```

A `CallbackEvent` will not be deleted after being processed because it needs to retain the callback function until the function is executed. Therefore, the third bit of the previously mentioned flags needs to be cleared to ensure it is not released after being processed.

```cpp
this->clearFlags(this->Managed);
```

The CallbackEvent class includes a pointer to the callee simulator. However, users should exercise caution when using the callee pointer, as improper use may lead to unexpected errors. It's crucial to understand that the callback function is invoked when its host simulator drains the event from the event queue in a future cycle. Users must be aware of the following key points:

1. Thread safety is only guaranteed when accessing local resources, such as the local event queue or modifying local states within the host simulator. Attempting to access any members or member functions of the callee simulator within the callback function may result in race conditions.

2. If a callback function needs to schedule an event or access any member functions of the callee simulator, it can only do so safely by pushing an EventPacket to the corresponding outbound channel.

3. An alternative thread-safe communication method available in a callback function is the SimPort mechanism. This feature, provided by the ACALSim framework, enables safe hardware modeling across threads.

These guidelines ensure proper synchronization and prevent potential concurrency issues in multi-threaded simulations. Adhering to these practices will help maintain the integrity and reliability of the simulated hardware environment.

## Example - `src/testCommunication`

In this section, we use `src/testCommunication` as an example to demonstrate the communication process. We've created two simulators (`SimBase`): `TrafficGenerator` and `PE`. In `TrafficGenerator`, we schedule the `TrafficEvent`.
Whe `TrafficEvent` is processed, it will create a peReqPkt (`SimPacket`) carrying computation data and models latency through an Event. Since this involves inter-simulator communication, an `EventPacket` is created and sent to the `PE` via a `SimChannel`. In the ACALSim framework, the `EventPacket` is unpacked, and the event is scheduled into the corresponding event queue of the receiver simulator.

```cpp
void TrafficGenerator::init() {
    // insert event in PE's event queue through channel
    int           _tID      = 1;
    TrafficEvent* tfEvent_1 = new TrafficEvent(this, "TestEventFromTG2PE_1", _tID);
    this->scheduleEvent(tfEvent_1, top->getGlobalTick() + 1);
    _tID                    = 2;
    TrafficEvent* tfEvent_2 = new TrafficEvent(this, "TestEventFromTG2PE_2", _tID);
    this->scheduleEvent(tfEvent_2, top->getGlobalTick() + 2);
}
```

```cpp
void TrafficEvent::process() {
    SimBase*                                pe        = this->sim->getDownStream("DSPE");
    int*                                    _d        = new int;
    Tick                                    t         = top->getGlobalTick() + 5;
    PERespPacket*                           peRespPkt = new PERespPacket(PEReqTypeEnum::TEST, _d);
    std::function<void(int, PERespPacket*)> callback  = [this, peRespPkt](int id, PERespPacket* pkt) {
        dynamic_cast<TrafficGenerator*>(this->sim)->PERespHandler(this->getTID(), peRespPkt);
    };
    PEReqPacket* peReqPkt   = new PEReqPacket(PEReqTypeEnum::TEST, 200, 2, 400, peRespPkt);
    PEReqEvent*  peReqEvent = new PEReqEvent(this->getTID(), pe, callback, peReqPkt);
    EventPacket* eventPkt   = new EventPacket(peReqEvent, t);

    CLASS_INFO << "Issue traffic with transaction id: " << this->getTID() << " at Tick=" << top->getGlobalTick();
    this->sim->pushToMasterChannelPort("DSPE", (void*)eventPkt);
}
```

When an event is processed, the data carried by the `SimPacket` is retrieved and computed. The result is then stored in another `SimPacket`, and the callback function is executed:

```cpp
void PEReqEvent::process() {
    INFO("Process PEReqEvent with transaction id: " + std::to_string(this->id) +
         " at Tick=" + std::to_string(top->getGlobalTick()));

    int a = this->peReqPkt->getA();
    int b = this->peReqPkt->getB();
    int c = this->peReqPkt->getC();

    int d = a * b + c;

    this->peReqPkt->getPERespPkt()->setResult(d);

    callerCallback(this->id, this->peReqPkt->getPERespPkt());
}
```

In this example, the callback function simply prints out the value. The result is as follows:

```text
	    _   ___   _   _    ___ ___ __  __
	   /_\ / __| /_\ | |  / __|_ _|  \/  |
	  / _ \ (__ / _ \| |__\__ \| || |\/| |
	 /_/ \_\___/_/ \_\____|___/___|_|  |_|

Tick=0 Info: [16TrafficGenerator] Constructing Traffic Generator... [./src/testCommunication/include/TrafficGenerator.hh:29]
Tick=0 Info: [2PE] Constructing Proccessing Element Simulator [./src/testCommunication/include/PE.hh:28]
Tick=0 Info: [2PE] PE Simulator PE::init()! [./src/testCommunication/include/PE.hh:36]
Tick=0 Info: [2PE] PE::step() PE Simulator [./src/testCommunication/include/PE.hh:38]
Tick=1 Info: [12TrafficEvent] Issue traffic with transaction id: 1 at Tick=1 [./src/testCommunication/libs/TrafficEvent.cc:34]
Tick=2 Info: [12TrafficEvent] Issue traffic with transaction id: 2 at Tick=2 [./src/testCommunication/libs/TrafficEvent.cc:34]
Tick=2 Info: [2PE] PE::step() PE Simulator [./src/testCommunication/include/PE.hh:38]
Tick=3 Info: [2PE] PE::step() PE Simulator [./src/testCommunication/include/PE.hh:38]
Tick=6 Info: [10PEReqEvent] Process PEReqEvent with transaction id: 1 at Tick=6 [./src/testCommunication/libs/PEEvent.cc:20]
Tick=6 Info: [16TrafficGenerator] Receive PERespPacket with transaction id: 1 [./src/testCommunication/libs/TrafficGenerator.cc:35]
Tick=6 Info: [16TrafficGenerator] Receive PE computation result : 800 [./src/testCommunication/libs/TrafficGenerator.cc:36]
Tick=6 Info: [2PE] PE::step() PE Simulator [./src/testCommunication/include/PE.hh:38]
Tick=7 Info: [10PEReqEvent] Process PEReqEvent with transaction id: 2 at Tick=7 [./src/testCommunication/libs/PEEvent.cc:20]
Tick=7 Info: [16TrafficGenerator] Receive PERespPacket with transaction id: 2 [./src/testCommunication/libs/TrafficGenerator.cc:35]
Tick=7 Info: [16TrafficGenerator] Receive PE computation result : 800 [./src/testCommunication/libs/TrafficGenerator.cc:36]
Tick=7 Info: [2PE] PE::step() PE Simulator [./src/testCommunication/include/PE.hh:38]
Tick=8 Info: [2PE] PE::cleanup() PE Simulator [./src/testCommunication/include/PE.hh:40]
Tick=8 Info: [20TestCommunicationTop] There is no trace to be saved. [./libs/sim/SimTop.cc:236]
Tick=8 Info: [20TestCommunicationTop] Simulation complete. [./libs/sim/SimTop.cc:239]
```

---

([Back To Documentation Portal](/docs/README.md))
