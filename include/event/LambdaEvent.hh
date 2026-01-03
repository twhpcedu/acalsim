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

#pragma once

#include <functional>

#include "event/SimEvent.hh"

namespace acalsim {

/**
 * @file LambdaEvent.hh
 * @brief Template event class for inline lambda/function execution
 *
 * @details
 * LambdaEvent provides a lightweight event type that wraps a lambda function
 * or std::function for inline event creation. This is ideal for one-off events,
 * quick prototyping, and situations where creating a custom event class would
 * be overkill.
 *
 * **Lambda Event Model:**
 * ```
 * // Instead of creating a custom event class:
 * class MyCustomEvent : public SimEvent {
 *     void process() override { doSomething(); }
 * };
 *
 * // Use LambdaEvent for inline definition:
 * auto* evt = new LambdaEvent<void()>([](){ doSomething(); });
 * scheduler.schedule(evt, tick);
 * ```
 *
 * **Key Features:**
 *
 * - **Inline Definition**: Create events without defining custom classes
 * - **Lambda Support**: Wrap C++11 lambdas with captures
 * - **std::function**: Compatible with any callable object
 * - **Template-based**: Generic function signature via template parameter
 * - **Lightweight**: Minimal overhead beyond std::function storage
 *
 * **Use Cases:**
 *
 * | Scenario | Example | Benefits |
 * |----------|---------|----------|
 * | **Quick Prototyping** | Test event-driven logic | Fast iteration |
 * | **State Updates** | Update counter at future tick | No boilerplate |
 * | **Delayed Actions** | Delayed logging or printing | Simple one-liners |
 * | **Timed Callbacks** | Periodic status checks | Inline lambda |
 * | **Test Harnesses** | Inject test events | Easy setup |
 *
 * **Comparison with Other Event Types:**
 *
 * | Feature | LambdaEvent | CallbackEvent | Custom SimEvent |
 * |---------|-------------|---------------|-----------------|
 * | **Boilerplate** | None | Low | High (new class) |
 * | **Flexibility** | Medium | High | Very High |
 * | **Transaction ID** | ✗ | ✓ | ✓ (optional) |
 * | **Best for** | Simple tasks | Async ops | Complex logic |
 *
 * **Lambda Capture Best Practices:**
 * ```cpp
 * // ✓ Good: Capture by value for simple types
 * int value = 42;
 * auto* evt = new LambdaEvent<void()>([value]() {
 *     LOG_INFO << "Value: " << value;
 * });
 *
 * // ✓ Good: Capture 'this' for member access
 * auto* evt = new LambdaEvent<void()>([this]() {
 *     this->processData();
 * });
 *
 * // ✗ Bad: Dangling reference if object dies before event fires
 * SomeObject obj;
 * auto* evt = new LambdaEvent<void()>([&obj]() {  // DANGER!
 *     obj.doSomething();  // obj may not exist anymore
 * });
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | Constructor | O(1) | std::function assignment |
 * | renew() | O(1) | Update function pointer |
 * | process() | Varies | User-defined lambda logic |
 *
 * **Memory:** sizeof(LambdaEvent<T>) ≈ sizeof(SimEvent) + sizeof(std::function<T>) ≈ 80 bytes
 *
 * **Thread Safety:**
 * - **Lambda Execution**: Not thread-safe - ensure single-threaded access
 * - **Captures**: User must ensure captured variables are thread-safe
 * - **Function Assignment**: Not atomic - avoid concurrent modification
 *
 * @tparam T Function signature (typically void() or similar)
 *
 * @code{.cpp}
 * // Example 1: Simple delayed logging
 * void scheduleLog(const std::string& message, Tick delay) {
 *     auto* evt = new LambdaEvent<void()>([message]() {
 *         LOG_INFO << "Delayed message: " << message;
 *     });
 *     scheduler.schedule(evt, currentTick() + delay);
 * }
 *
 * // Example 2: Periodic status printing
 * class StatusMonitor {
 * public:
 *     void startMonitoring() {
 *         schedulePrint();
 *     }
 *
 * private:
 *     void schedulePrint() {
 *         auto* evt = new LambdaEvent<void()>([this]() {
 *             LOG_INFO << "Cycles: " << currentTick();
 *             schedulePrint();  // Schedule next print
 *         });
 *         scheduler.schedule(evt, currentTick() + 1000);
 *     }
 * };
 *
 * // Example 3: Cleanup event at simulation end
 * void scheduleCleanup(Tick endTick) {
 *     auto* evt = new LambdaEvent<void()>([]() {
 *         LOG_INFO << "Simulation ending - cleanup started";
 *         // Perform cleanup operations
 *         dumpStatistics();
 *         closeFiles();
 *     });
 *     scheduler.schedule(evt, endTick);
 * }
 *
 * // Example 4: Conditional state update
 * class Counter {
 * public:
 *     void scheduleIncrement(int value, Tick when) {
 *         auto* evt = new LambdaEvent<void()>([this, value]() {
 *             if (this->enabled) {
 *                 this->count += value;
 *                 LOG_DEBUG << "Counter now: " << this->count;
 *             }
 *         });
 *         scheduler.schedule(evt, when);
 *     }
 *
 * private:
 *     int count = 0;
 *     bool enabled = true;
 * };
 *
 * // Example 5: Test harness event injection
 * TEST(EventTest, LambdaEventExecution) {
 *     bool executed = false;
 *
 *     auto* evt = new LambdaEvent<void()>([&executed]() {
 *         executed = true;
 *     });
 *
 *     scheduler.schedule(evt, 100);
 *     scheduler.run();
 *
 *     ASSERT_TRUE(executed);
 * }
 * @endcode
 *
 * @note Lambda captures must outlive the event execution
 * @note Function pointer is checked for nullptr before invocation
 * @note Error logged if process() called with nullptr function
 *
 * @warning Capturing by reference [&] can lead to dangling references
 * @warning Large captures increase memory usage
 *
 * @see SimEvent for base event class
 * @see CallbackEvent for transaction-aware callback events
 * @since ACALSim 0.1.0
 */
template <typename T>
class LambdaEvent : public SimEvent {
public:
	/**
	 * @brief Default constructor - creates event with null function
	 *
	 * @note Must call renew() with valid function before scheduling
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * LambdaEvent<void()>* evt = new LambdaEvent<void()>();
	 * evt->renew([]() { LOG_INFO << "Hello"; });
	 * @endcode
	 */
	LambdaEvent() : SimEvent("LambdaEvent"), func(nullptr) {}

	/**
	 * @brief Construct event with lambda/function
	 *
	 * @param _func Lambda or std::function to execute
	 *
	 * @note Stores copy of the function object
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * auto* evt = new LambdaEvent<void()>([]() {
	 *     LOG_INFO << "Event fired!";
	 * });
	 * scheduler.schedule(evt, tick);
	 * @endcode
	 */
	LambdaEvent(const std::function<T>& _func) : SimEvent("LambdaEvent"), func(_func) { ; }

	/**
	 * @brief Reset event with new function for recycling
	 *
	 * @param _func New lambda/function to execute
	 *
	 * @note Called by RecycleContainer before reuse
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * LambdaEvent<void()>* evt = pool.get<LambdaEvent<void()>>();
	 * evt->renew([this]() { this->doWork(); });
	 * scheduler.schedule(evt, tick);
	 * @endcode
	 */
	void renew(const std::function<T>& _func) {
		this->SimEvent::renew("LambdaEvent");
		this->func = _func;
	}

	/**
	 * @brief Process the event by invoking the stored function
	 *
	 * @note Called by scheduler at scheduled tick
	 * @note Logs error if func is nullptr
	 * @note Executes the lambda with captured variables
	 *
	 * @code{.cpp}
	 * // Automatically called by scheduler:
	 * // event->process()  →  func()  →  lambda body executes
	 * @endcode
	 */
	void process() override {
		if (this->func) {
			this->func();
		} else {
			LABELED_ERROR(this->getName()) << "The function pointer `func` is nullptr";
		}
	}

private:
	/** @brief Lambda/function to execute when event is processed */
	std::function<T> func = nullptr;
};

}  // namespace acalsim
