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
 * @file TaskManager.cc
 * @brief TaskManager base class implementation (interface-only)
 *
 * This file provides the base TaskManager interface for task scheduling in ACALSim's
 * parallel simulation framework. TaskManager is an abstract base class that defines
 * the interface for managing task creation, queuing, and distribution to worker threads.
 *
 * **Architecture Overview:**
 * TaskManager is part of ACALSim's two-phase synchronization model, working in
 * conjunction with ThreadManager to enable parallel execution of simulator instances.
 *
 * **Design Pattern:**
 * - TaskManager is a template-heavy abstract interface
 * - All implementation logic resides in versioned subclasses
 * - This file intentionally contains minimal code
 * - Actual implementations: TaskManagerV1.cc, TaskManagerV2.cc, TaskManagerV3.cc, etc.
 *
 * **Task Scheduling Hierarchy:**
 * ```
 *                     TaskManager (Abstract Base)
 *                            │
 *       ┌────────────────────┼────────────────────┬────────────────┐
 *       │                    │                    │                │
 * TaskManagerV1      TaskManagerV2       TaskManagerV3      TaskManagerV6
 * (Basic queue)    (Work-stealing)   (Lock-free queue)  (Optimized dist)
 *       │                    │                    │                │
 *       │                    │                    │                │
 *  Experimental:       Experimental:        Experimental:    Experimental:
 * TaskManagerV4       TaskManagerV5       TaskManagerV7      TaskManagerV8
 * ```
 *
 * **Key Responsibilities (Implemented in Subclasses):**
 * 1. **Task Creation**: Convert active SimBase instances into TaskFunctor objects
 * 2. **Task Queuing**: Maintain per-thread or global task queues
 * 3. **Load Balancing**: Distribute tasks across worker threads efficiently
 * 4. **Work Stealing**: Enable idle threads to steal tasks (V2, V5, V7)
 * 5. **Priority Scheduling**: Support task prioritization based on dependencies
 * 6. **Statistics Collection**: Track task execution time and thread utilization
 *
 * **Interaction with ThreadManager:**
 * ```
 * SimTop Main Loop (Phase 1):
 *   │
 *   ├─ SimTop calls: threadManager->startPhase1()
 *   │
 *   ├─ ThreadManager queries: taskManager->createTasks()
 *   │    └─ TaskManager creates TaskFunctors for active SimBase instances
 *   │
 *   ├─ ThreadManager calls: taskManager->distributeTasks()
 *   │    └─ TaskManager assigns tasks to worker thread queues
 *   │
 *   ├─ Worker threads: while (task = getNextTask()) task->execute()
 *   │    └─ TaskManager handles task retrieval (queue/steal/priority)
 *   │
 *   └─ ThreadManager calls: threadManager->finishPhase1()
 *        └─ Wait for all tasks to complete
 * ```
 *
 * **Version Selection Trade-offs:**
 * - **V1 (Basic)**: Simple FIFO queue, low overhead, predictable performance
 * - **V2 (Work-Stealing)**: Better load balancing, higher overhead for lock contention
 * - **V3 (Lock-Free)**: Minimal synchronization overhead, best for many short tasks
 * - **V6 (Optimized)**: Production-ready with best overall performance
 * - **V4-V5, V7-V8**: Experimental features, subject to change
 *
 * **Template Design:**
 * TaskManager uses CRTP (Curiously Recurring Template Pattern) to enable
 * static polymorphism and avoid virtual function call overhead in hot paths.
 *
 * Example:
 * ```cpp
 * template <typename BaseType>
 * class TaskManagerV3 : public TaskManager<BaseType> {
 *     // Compile-time polymorphism via templates
 *     void distributeTasks() override { ... }
 * };
 * ```
 *
 * @see TaskManager.hh For complete interface documentation
 * @see ThreadManager.cc For thread pool coordination
 * @see TaskManagerV3.cc For recommended production implementation
 * @see TaskFunctor.hh For task wrapper details
 */

#include "sim/TaskManager.hh"

namespace acalsim {}  // namespace acalsim
