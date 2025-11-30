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
 * @file SimTop.cc
 * @brief SimTop implementation - top-level simulation coordinator and main loop
 *
 * This file implements SimTop, the central coordinator managing all SimBase instances
 * in a parallel discrete-event simulation. It orchestrates thread pool execution,
 * global clock advancement, configuration management, and performance statistics.
 *
 * **Top-Level Coordination Architecture:**
 * ```
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │                            SimTop (Control Thread)                       │
 * │                                                                          │
 * │  Global Tick: 1000    Pending Activity Bitmask: 0b1010 (2 active)      │
 * │  Thread Pool: 8 threads    ThreadManager: V3                            │
 * └─────────────────┬────────────────────────────────────────────────────────┘
 *                   │
 *                   ▼
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │                        Main Simulation Loop (run())                      │
 * │                                                                          │
 * │  while (true) {                                                          │
 * │    ┌──────────────────────────────────────────────────────────────────┐ │
 * │    │ Phase 1: Parallel Step Execution                                 │ │
 * │    ├──────────────────────────────────────────────────────────────────┤ │
 * │    │ threadManager->startPhase1()                                     │ │
 * │    │   - Signal worker threads to start                               │ │
 * │    │   - Each active SimBase runs stepWrapper() in parallel           │ │
 * │    │   - TaskManager schedules tasks to thread pool workers           │ │
 * │    │                                                                  │ │
 * │    │ control_thread_step()                                            │ │
 * │    │   - Control thread processing (user override)                    │ │
 * │    │                                                                  │ │
 * │    │ threadManager->finishPhase1()                                    │ │
 * │    │   - Wait for all worker threads to complete                      │ │
 * │    │   - Collect task execution statistics                            │ │
 * │    └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                          │
 * │    ┌──────────────────────────────────────────────────────────────────┐ │
 * │    │ Phase 2: Synchronization & Coordination                          │ │
 * │    ├──────────────────────────────────────────────────────────────────┤ │
 * │    │ threadManager->startPhase2()                                     │ │
 * │    │                                                                  │ │
 * │    │ 1. Check termination condition                                   │ │
 * │    │    if (readyToTerminate) break;                                  │ │
 * │    │                                                                  │ │
 * │    │ 2. Synchronize pipe registers                                    │ │
 * │    │    pipeRegisterManager->runSyncPipeRegister()                    │ │
 * │    │                                                                  │ │
 * │    │ 3. Synchronize SimPorts                                          │ │
 * │    │    runSyncSimPort() - All simulators sync ports                  │ │
 * │    │                                                                  │ │
 * │    │ 4. Update activity tracking                                      │ │
 * │    │    threadManager->runInterIterationUpdate()                      │ │
 * │    │    - Each SimBase updates pendingEventBitMask                    │ │
 * │    │                                                                  │ │
 * │    │ 5. Advance global clock                                          │ │
 * │    │    if (isAllSimulatorDone()) {                                   │ │
 * │    │        issueExitEvent(t);                                        │ │
 * │    │        readyToTerminate = true;                                  │ │
 * │    │    } else {                                                      │ │
 * │    │        t = getFastForwardCycles();  // Optimize clock advance    │ │
 * │    │        fastForwardGlobalTick(t);                                 │ │
 * │    │    }                                                             │ │
 * │    │                                                                  │ │
 * │    │ 6. Toggle channel buffers                                        │ │
 * │    │    SimChannelGlobal::toggleChannelDualQueueStatus()              │ │
 * │    │    - Swap ping-pong buffers for next iteration                   │ │
 * │    │                                                                  │ │
 * │    │ threadManager->finishPhase2()                                    │ │
 * │    └──────────────────────────────────────────────────────────────────┘ │
 * │  }                                                                       │
 * └──────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Initialization Flow (init()):**
 * ```
 * SimTop::init(argc, argv)
 *   │
 *   ├─ 1. Configuration Initialization (initConfig)
 *   │     ├─ Create ACALSimConfig (thread manager version, etc.)
 *   │     ├─ registerConfigs() - User configs with defaults
 *   │     ├─ registerCLIArguments() - Map CLI to configs
 *   │     ├─ parseCLIArguments(argc, argv)
 *   │     ├─ parseConfigFiles(configFilePaths) - JSON configs
 *   │     └─ setCLIParametersToSimConfig() - CLI overrides
 *   │
 *   ├─ 2. Thread Pool Initialization (initThreadManager)
 *   │     ├─ Detect hardware_concurrency()
 *   │     ├─ Create ThreadManager (V1-V8 based on config)
 *   │     ├─ Create TaskManager (matching version)
 *   │     └─ Link ThreadManager ↔ TaskManager
 *   │
 *   ├─ 3. Simulator Registration
 *   │     ├─ registerSimulators() - User creates SimBase instances
 *   │     ├─ registerPipeRegisters() - User creates SimPipeRegisters
 *   │     └─ Each SimBase assigned unique ID
 *   │
 *   ├─ 4. Channel Connection
 *   │     ├─ connectTopChannelPorts()
 *   │     ├─ For each SimBase: create bidirectional channels
 *   │     │   - SimBase → SimTop (toTopChannelPort)
 *   │     │   - SimTop → SimBase (fromTopChannelPort)
 *   │     └─ Lock-free dual-queue channels for thread safety
 *   │
 *   ├─ 5. User Initialization Hooks
 *   │     ├─ preSimInitSetup() - User override
 *   │     ├─ threadManager->preSimInitWrapper() - Framework pre-init
 *   │     ├─ threadManager->simInit() - Initialize worker threads
 *   │     ├─ postSimInitSetup() - User override
 *   │     └─ threadManager->postSimInit() - Framework post-init
 *   │
 *   └─ 6. Launch Thread Pool
 *         ├─ threadManager->startSimThreads()
 *         └─ Toggle dual-queue channel status
 * ```
 *
 * **Fast-Forward Clock Optimization:**
 * ```
 * Problem: If all simulators have events at tick 1000, 1050, 2000
 *          Don't iterate ticks 1001-1049 and 1051-1999 (no activity)
 *
 * Solution: getFastForwardCycles()
 *   - Query each active SimBase for next event tick
 *   - Return minimum across all simulators
 *   - fastForwardGlobalTick(min_tick) jumps directly to next activity
 *
 * Example:
 *   Tick 1000: SimBase[0] next=1050, SimBase[1] next=2000, SimBase[2] inactive
 *   Result: Fast-forward to tick 1050 (skip 1001-1049)
 * ```
 *
 * **ThreadManager Version Selection:**
 * ACALSim supports multiple ThreadManager implementations optimized for different
 * workloads and parallelism patterns:
 *
 * | Version | Status       | Optimization Focus                              |
 * |---------|--------------|------------------------------------------------|
 * | V1      | Production   | Basic task-based parallelism                   |
 * | V2      | Production   | Work-stealing scheduler                        |
 * | V3      | Production   | Optimized task queue, reduced locking          |
 * | V4      | Experimental | Fine-grained dependency tracking               |
 * | V5      | Experimental | Adaptive load balancing                        |
 * | V6      | Production   | Lock-free task distribution                    |
 * | V7      | Experimental | NUMA-aware scheduling                          |
 * | V8      | Experimental | Speculative execution                          |
 *
 * Selection via config: `thread_manager_version` in ACALSimConfig or CLI
 *
 * **Activity Tracking with Bitmask:**
 * ```
 * Each SimBase updates its bit in pendingEventBitMask:
 *   - Bit set: Has pending events/channel requests → schedule in Phase 1
 *   - Bit clear: Idle → skip in Phase 1
 *
 * Example (4 simulators):
 *   Bitmask: 0b1010 → Only SimBase[1] and SimBase[3] active
 *   ThreadManager schedules only 2 tasks instead of 4
 * ```
 *
 * **Performance Statistics (ACALSIM_STATISTICS):**
 * - Phase 1/2 execution time breakdown
 * - Per-thread task execution vs idle time
 * - Scheduling overhead measurement
 * - Parallelism degree distribution
 * - SimPort/SimChannel connection counts and costs
 * - Average activation ratio across simulation
 *
 * **Global State Management:**
 * - `top` global variable provides singleton access to SimTop
 * - RecycleContainer for event/packet pooling across all simulators
 * - SimChannelGlobal for dual-queue synchronization
 * - Trace containers for performance/debug logging
 *
 * @see SimTop.hh For detailed API documentation
 * @see SimBase.cc For individual simulator implementation
 * @see ThreadManager.hh For thread pool interface
 * @see TaskManager.hh For task scheduling interface
 */

#include "sim/SimTop.hh"

#include <sys/resource.h>

#include <atomic>
#include <exception>
#include <sstream>
#include <string>
#include <syncstream>
#include <thread>
#include <vector>

#include "channel/SimChannel.hh"
#include "config/ACALSimConfig.hh"
#include "container/RecycleContainer/RecycleContainer.hh"
#include "profiling/Utils.hh"
#include "sim/SimBase.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
// Production ThreadManager versions
#include "sim/ThreadManagerV1/TaskManagerV1.hh"
#include "sim/ThreadManagerV1/ThreadManagerV1.hh"
#include "sim/ThreadManagerV2/TaskManagerV2.hh"
#include "sim/ThreadManagerV2/ThreadManagerV2.hh"
#include "sim/ThreadManagerV3/TaskManagerV3.hh"
#include "sim/ThreadManagerV3/ThreadManagerV3.hh"
#include "sim/ThreadManagerV6/TaskManagerV6.hh"
#include "sim/ThreadManagerV6/ThreadManagerV6.hh"

// Experimental ThreadManager versions (not for production)
#include "sim/experimental/ThreadManagerV4/TaskManagerV4.hh"
#include "sim/experimental/ThreadManagerV4/ThreadManagerV4.hh"
#include "sim/experimental/ThreadManagerV5/TaskManagerV5.hh"
#include "sim/experimental/ThreadManagerV5/ThreadManagerV5.hh"
#include "sim/experimental/ThreadManagerV7/TaskManagerV7.hh"
#include "sim/experimental/ThreadManagerV7/ThreadManagerV7.hh"
#include "sim/experimental/ThreadManagerV8/TaskManagerV8.hh"
#include "sim/experimental/ThreadManagerV8/ThreadManagerV8.hh"
#include "utils/Logging.hh"

namespace acalsim {

// Context switch measurement
static struct rusage g_rusage_start;

// A global variable for others to access the common variable
std::shared_ptr<SimTopBase> top = nullptr;

SimTopBase::SimTopBase(const std::vector<std::string>& _configFilePaths, const std::string& _tracingJsonFileName)
    : CLIManager("CLIManager", _configFilePaths, _tracingJsonFileName),
      DeviceManager("DeviceManager"),
      globalTick(0),
      readyToTerminate(false),
      recycleContainer(new RecycleContainer()) {
	this->traceCntr.add("trace", "trace");
	this->traceCntr.add("chrome-trace", "trace");

	std::stringstream ss;
	ss << "	    _   ___   _   _    ___ ___ __  __ 			" << std::endl;
	ss << "	   /_\\ / __| /_\\ | |  / __|_ _|  \\/  |		" << std::endl;
	ss << "	  / _ \\ (__ / _ \\| |__\\__ \\| || |\\/| |		" << std::endl;
	ss << "	 /_/ \\_\\___/_/ \\_\\____|___/___|_|  |_|		" << std::endl << std::endl;

	std::osyncstream(std::cout) << ss.str();

	// Register custom terminate function
	// Ref: https://en.cppreference.com/w/cpp/error/set_terminate
	std::set_terminate(&LogOStream::handleTerminate);
}

SimTopBase::~SimTopBase() {
	if (this->threadManager) { delete this->threadManager; }
	if (this->taskManager) { delete this->taskManager; }
	if (this->pipeRegisterManager) { delete this->pipeRegisterManager; }
}

void SimTopBase::init(int argc, char** argv) {
	// Parse SimConfig and CLI Arguments.
	this->initConfig(argc, argv);

	// Create TaskManager / ThreadManager
	auto hw_nthreads            = std::thread::hardware_concurrency();
	auto thread_manager_version = this->getParameter<ThreadManagerVersion>("ACALSim", "thread_manager_version");
	this->initThreadManager(thread_manager_version, hw_nthreads);

	// link ThreadManager && TaskManager
	this->threadManager->linkTaskManager(this->taskManager);
	this->taskManager->linkThreadManager(this->threadManager);

	// Create & register Simulators
	this->registerSimulators();

	// Create & register pipe registers
	this->registerPipeRegisters();

	int nSimulators = this->getNumSimulators();

	// set the googleTest Flag
	this->pGTestBitMask = std::make_shared<SharedContainer<BitVector>>();
	for (int i = 0; i < nSimulators; i++) {
		// one 64-bit BitMask per simulator
		this->pGTestBitMask->add(64, false);
	}

	// create Json event containers
	this->pMsgPktJsonContainer = std::make_shared<JsonContainer<MessagePacketJsonEvent>>();

	// Connect SimChannel between SimBase and SimTop
	this->connectTopChannelPorts();

	// [User override method] Pre-Simulation Initialization Setup
	this->preSimInitSetup();

	// Pre-Simulation Initialization
	this->threadManager->preSimInitWrapper();

	// Thread Initialization
	this->threadManager->simInit();

	// [User override method] Post-Simulation Initialization Setup
	this->postSimInitSetup();

	// Post-Simulation Initialization
	this->threadManager->postSimInit();

	// configure the mapping between the simulators and the available threads
	this->threadManager->startSimThreads();

	// toggle all dual channel status
	SimChannelGlobal::toggleChannelDualQueueStatus();

	// Record context switches at simulation start
	getrusage(RUSAGE_SELF, &g_rusage_start);
}

void SimTopBase::run() {
	// Simulation main loop
	// Each iteration, each simulator's step() function get called

	if (this->getNumSimulators() == 0) {
		CLASS_INFO << "No simulator is registered. Terminate simulation.";
		this->finish();
		exit(0);
	}

	bool* hasPendingActivityLastIteration = new bool[this->threadManager->getNumSimulators()];

	// Set task scheduler to the running state
	this->threadManager->startRunning();

	while (true) {
		{  // [Phase #1] : Step Phase.
#ifdef ACALSIM_STATISTICS
			auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

			this->threadManager->startPhase1();
			MT_DEBUG_CLASS_INFO << "Control thread Phase 1 starts ";

			/* ------------------------------------------------------------
			 *  [Phase #1 Per-Iteration framework Update Phase ]
			 *  control thread step function
			 * ------------------------------------------------------------
			 */

			// While all the simulators are processing the events for this cycle,
			// does the main control thread has anything to do here?
			// e.g. void control_thread_step();
			this->control_thread_step();

			/* ------------------------------------------------------------
			 *  [Phase #1 Per-Iteration framework Update Phase end ]
			 * ------------------------------------------------------------
			 */

			this->threadManager->finishPhase1();
#ifdef ACALSIM_STATISTICS
			auto stop      = std::chrono::high_resolution_clock::now();
			auto curr_time = (double)(stop - start).count() / pow(10, 3);
			this->timer_phase1_us += curr_time;

			tasks_time_dist_statistics.getEntry(this->threadManager->getNTasksPerIter())->push(curr_time);

			this->addTraceRecord(/* trace */ std::make_shared<ParallelismRecord>(
			                         /* tick */ top->getGlobalTick(),
			                         /* degree */ this->threadManager->getNTasksPerIter()),
			                     /* category */ "SimTopStatistic");
#endif  // ACALSIM_STATISTICS
		}

		{  // [Phase #2] Status Phase
#ifdef ACALSIM_STATISTICS
			auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

			this->threadManager->startPhase2();
			MT_DEBUG_CLASS_INFO << "Control thread Phase 2 starts. nThreads=" << this->threadManager->getNumThreads()
			                    << " readyToTerminate=" << this->readyToTerminate;

			/* -------------------------------------------------------------------
			 *  [Phase #2 Inter-iteration framework Update Phase]
			 *  1. The Control thread terminate the simulation
			 *  if readyToTerminate is set in the 2nd phase of the last iteration
			 * --------------------------------------------------------------------
			 */

			if (this->readyToTerminate) {
				this->threadManager->terminateAllThreadsWrapper();

#ifdef ACALSIM_STATISTICS
				auto stop = std::chrono::high_resolution_clock::now();
				this->timer_phase2_us += (double)(stop - start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS

				break;
			}

			/* ------------------------------------------------------------------------------------------
			 * [Phase #2 Intra-iteration Framework Update phase start]
			 * 1. The runSyncSimPort() function updates
			 *     - The syncSimPort() function sync the Entry in SlavePort with SlavePort.
			 *	   - The syncSimPort() function may call user-defined callback function if SimPacket in
			 *						   MasterPort::Entry has been moved into SlavePort::Entry
			 * ------------------------------------------------------------------------------------------
			 */
			MEASURE_TIME_MICROSECONDS(/*var_name*/ pipereg,
			                          /*code_block*/ { this->pipeRegisterManager->runSyncPipeRegister(); })

			/* ------------------------------------------------------------------------------------------
			 * [Phase #2 Intra-iteration Framework Update phase start]
			 * 1. The runSyncSimPort() function updates
			 *     - The syncSimPort() function sync the Entry in SlavePort with SlavePort.
			 *	   - The syncSimPort() function may call user-defined callback function if SimPacket in
			 *						   MasterPort::Entry has been moved into SlavePort::Entry
			 * ------------------------------------------------------------------------------------------
			 */
			MEASURE_TIME_MICROSECONDS(/*var_name*/ simport,
			                          /*code_block*/ { this->runSyncSimPort(); });

			/* ------------------------------------------------------------------------------------------
			 * [Phase #3 Inter-iteration Framework Update phase start]
			 * 1. The interIterationUpdate() function updates
			 *     - the simulator's bPendingActivity flag based on the eventQ status and channel status
			 *     - the global PendingEventBitMask  based on all simulator's bPendingActivity flag
			 *       - pass in hasPendingActivityLastIteration[i] from the previous iteration to
			 *         capture negative edge for the global pendingEventBitMask update
			 * ------------------------------------------------------------------------------------------
			 */
			MEASURE_TIME_MICROSECONDS(/*var_name*/ inter_iter_update,
			                          /*code_block*/ { this->threadManager->runInterIterationUpdate(); });

			/* -----------------------------------------------------------------------------------------
			 *  [Phase #2 Inter-iteration framework Update Phase]
			 *  1. calculate fast-forward cycles
			 *  2. isAllSimulatorDone() check the global pendingEventBitMask
			 *  3. toggle all dual channel status
			 * ------------------------------------------------------------------------------------------
			 */
			Tick t = getGlobalTick() + 1;

			MEASURE_TIME_MICROSECONDS(
			    /*var_name*/ get_next_tick,
			    /*code_block*/ {  // check the termination condition and send out the Exit event if all done
				    if (this->threadManager->isAllSimulatorDone()) {  // no events to process, exit the main loop
					    MT_DEBUG_CLASS_INFO << "Child threads should be going to terminate themselves.";
					    this->threadManager->issueExitEvent(t);
					    this->readyToTerminate = true;
				    } else {
					    t = this->threadManager->getFastForwardCycles();
					    ASSERT(t > getGlobalTick());
				    }
			    });

			MT_DEBUG_CLASS_INFO << "Most recent event occurs at Tick: " << t;

			// Advance to the next event across all the simualtors
			this->fastForwardGlobalTick(t);

			// toggle all dual channel status
			SimChannelGlobal::toggleChannelDualQueueStatus();

			MT_DEBUG_CLASS_INFO << "Control Thread Phase 2 ends ";

			// Flush all logs from this tick to ensure proper ordering in multi-threaded output
			std::cout.flush();

			this->threadManager->finishPhase2();

#ifdef ACALSIM_STATISTICS
			auto stop = std::chrono::high_resolution_clock::now();
			this->timer_phase2_us += (double)(stop - start).count() / pow(10, 3);
			this->simpipereg_cost_us += pipereg_lat;
			this->simport_cost_us += simport_lat;
			this->inter_iter_update_cost_us += inter_iter_update_lat;
			this->get_next_tick_cost += get_next_tick_lat;
#endif  // ACALSIM_STATISTICS
		}
	}

	delete[] hasPendingActivityLastIteration;
}

void SimTopBase::finish() {
	// Report context switches during simulation
	struct rusage rusage_end;
	getrusage(RUSAGE_SELF, &rusage_end);
	long voluntary_cs   = rusage_end.ru_nvcsw - g_rusage_start.ru_nvcsw;
	long involuntary_cs = rusage_end.ru_nivcsw - g_rusage_start.ru_nivcsw;
	std::cout << "\n[SimTopBase] Context Switches during simulation:\n";
	std::cout << "[SimTopBase]   Voluntary context switches: " << voluntary_cs << "\n";
	std::cout << "[SimTopBase]   Involuntary context switches: " << involuntary_cs << "\n";

	MT_DEBUG_CLASS_INFO << "Simulation finish.";

	// Post-Simulation clean up
	this->threadManager->postSimCleanupWrapper();

	// Dump traces from SimTraceContainer
	this->traceCntr.run(0, &SimTraceContainer::writeToFile);
	this->traceCntr.run(1, &SimTraceContainer::writeToFile);

	LABELED_INFO("SimTopBase") << "Simulation complete.";

#ifdef ACALSIM_STATISTICS
	// Average activation ratio of SimBase instances across all simulated cycles
	size_t act_cnt  = 0;
	size_t iter_cnt = 0;

	for (auto& [parallel_degree, statistic] : this->tasks_time_dist_statistics) {
		act_cnt += parallel_degree * statistic->size();
		iter_cnt += statistic->size();
	}

	LABELED_STATISTICS("SimTopBase") << "Execution Time Distribution: "
	                                 << "Phase 1: " << this->timer_phase1_us << " us, "
	                                 << "Phase 2: " << this->timer_phase2_us << " us.";
	LABELED_STATISTICS("SimTopBase") << "Execution Time Distribution (per iteration): "
	                                 << "Phase 1: " << this->timer_phase1_us / iter_cnt << " us, "
	                                 << "Phase 2: " << this->timer_phase2_us / iter_cnt << " us.";

	LABELED_STATISTICS("SimTopBase") << "Avg Activation Ratio: "
	                                 << (double)act_cnt / (this->getNumSimulators() * iter_cnt) * 100 << "%.";
	LABELED_STATISTICS("SimTopBase") << "Avg Parallelism Degree: " << (double)act_cnt / iter_cnt << ".";

	// Connection between components
	LABELED_STATISTICS("SimTopBase") << "Total SimPipeRegister Connection Count: "
	                                 << SimPipeRegister::getConnectionCnt() << ".";
	LABELED_STATISTICS("SimTopBase") << "Total SimPort Connection Count: " << SimPortManager::getConnectionCnt() << ".";

	// Execution time breakdown of each worker thread
	double task_exec_proportion = this->threadManager->getAvgTaskExecTimePerThread() / this->timer_phase1_us;
	double idle_proportion      = this->threadManager->getAvgThreadIdleTime() / this->timer_phase1_us;

	LABELED_STATISTICS("SimTopBase") << "Execution Time Breakdown (proportion): "
	                                 << "(1) Task Execution: " << task_exec_proportion * 100 << "%, "
	                                 << "(2) Idle: " << idle_proportion * 100 << "%, "
	                                 << "(3) Scheduling Overheads: "
	                                 << (1 - task_exec_proportion - idle_proportion) * 100 << "%.";
	LABELED_STATISTICS("SimTopBase") << "Execution Time Breakdown (all threads): "
	                                 << "(1) Task Execution: " << this->threadManager->getTotalTaskExecTime() << " us, "
	                                 << "(2) Idle: " << this->threadManager->getTotalThreadIdleTime() << " us, "
	                                 << "(3) Scheduling Overheads: "
	                                 << (this->timer_phase1_us * this->getNumThreads() -
	                                     this->threadManager->getTotalTaskExecTime() -
	                                     this->threadManager->getTotalThreadIdleTime())
	                                 << " us.";
	LABELED_STATISTICS("SimTopBase") << "Phase 2 Breakdown (total): "
	                                 << "(1) Sync SimPipeRegister: " << this->simpipereg_cost_us << " us, "
	                                 << "(2) Sync SimPort: " << this->simport_cost_us << " us, "
	                                 << "(3) Inter-Iteration Update: " << this->inter_iter_update_cost_us << " us, "
	                                 << "(4) Determine Next Tick: " << this->get_next_tick_cost << " us.";
	LABELED_STATISTICS("SimTopBase") << "Phase 2 Breakdown (per iteration): "
	                                 << "(1) Sync SimPipeRegister: " << this->simpipereg_cost_us / iter_cnt << " us, "
	                                 << "(2) Sync SimPort: " << this->simport_cost_us / iter_cnt << " us, "
	                                 << "(3) Inter-Iteration Update: " << this->inter_iter_update_cost_us / iter_cnt
	                                 << " us, "
	                                 << "(4) Determine Next Tick: " << this->get_next_tick_cost / iter_cnt << " us.";

	this->threadManager->printSchedulingOverheads(this->timer_phase1_us -
	                                              this->threadManager->getAvgTaskExecTimePerThread() -
	                                              this->threadManager->getAvgThreadIdleTime());

	// Cost of SimChannel
	LABELED_STATISTICS("SimTopBase") << "SimChannel Cost (total): " << ChannelPortManager::getCostStat() << " us.";
	LABELED_STATISTICS("SimTopBase") << "SimChannel Cost (per iteration): "
	                                 << ChannelPortManager::getCostStat() / iter_cnt << " us.";

	// Parallelism distribution of all simulated cycles
	auto n_tasks_prop_dist  = this->tasks_time_dist_statistics.sizeDistribution();
	auto n_tasks_times_dist = this->tasks_time_dist_statistics.sumDistribution();

	for (auto& [parallel_degree, statistic] : this->tasks_time_dist_statistics) {
		LABELED_STATISTICS("SimTopBase") << "Phase 1 Proportion of Parallelism Degree " << parallel_degree << ": "
		                                 << n_tasks_prop_dist[parallel_degree] * 100
		                                 << "% | Exec Time: " << n_tasks_times_dist[parallel_degree] * 100 << "% -> "
		                                 << statistic->sum() << " us (sum), " << statistic->avg() << " us (avg).";
	}
#endif  // ACALSIM_STATISTICS
}

void SimTopBase::addSimulator(SimBase* sim) { this->threadManager->addSimulator(sim); }

void SimTopBase::setTopChannelPort(const std::string& _name, SlaveChannelPort::SharedPtr _toTopChannelPort,
                                   MasterChannelPort::SharedPtr _fromTopChannelPort) {
	this->toTopChannelPorts.insert(std::make_pair(_name, _toTopChannelPort));
	this->fromTopChannelPorts.insert(std::make_pair(_name, _fromTopChannelPort));

	this->slaveChannelPorts.insert(std::make_pair(_name, _toTopChannelPort));
	this->masterChannelPorts.insert(std::make_pair(_name, _fromTopChannelPort));
}

void SimTopBase::initConfig(int argc, char** argv) {
	// Create and Register Framework Configuration
	auto acalsim_config = new ACALSimConfig("ACALSim Configuration");
	this->addConfig("ACALSim", acalsim_config);

	// [Priority #3 : SimConfig Default Value] Register user SimConfig. (virtual function)
	this->registerConfigs();

	// Register framework CLI aruments
	this->registerACALSimCLIArguments();

	// Register user-defined CLI aruments (virtual function)
	this->registerCLIArguments();

	// Parse the arguments and get the configuration file path(s)
	this->parseCLIArguments(argc, argv);

	// [Priority #2.2 : JSON Configuration File] Parse the confiuration file path(s) from SimTop Constructor.
	this->parseConfigFiles(this->configFilePaths);

	// [Priority #2.1 : JSON Configuration File] Parse the confiuration file path(s) from CLI
	this->parseConfigFiles(this->configFilePathsFromCLI);

	// [Priority #1 : CLI Arguments] Set User-defined Arguments
	this->setCLIParametersToSimConfig();

	VERBOSE_CLASS_INFO << "[ACALSim] Command Line Arguments:";
	VERBOSE_CLASS_INFO << "	- Google Test (1:Enabled / 0:Disabled): " + std::to_string(this->gTestMode);
	VERBOSE_CLASS_INFO << "	- Trace Json File Name: " + this->tracingJsonFileName;
	VERBOSE_CLASS_INFO
	    << "	- Thread Manager Type: " +
	           ThreadManagerVersionReMap[this->getParameter<ThreadManagerVersion>("ACALSim", "thread_manager_version")];
}

void SimTopBase::runSyncSimPort() {
	for (auto& sim : this->threadManager->getAllSimulators()) { sim->syncSimPort(); }
}

void SimTopBase::addTraceRecord(const std::shared_ptr<SimTraceRecord>& _trace, const std::string& _category) {
	this->addTraceRecord(std::move(_trace), std::move(_category), this->getGlobalTick());
}

void SimTopBase::addTraceRecord(const std::shared_ptr<SimTraceRecord>& _trace, const std::string& _category,
                                const Tick& _tick) {
	this->traceCntr.run(0, &SimTraceContainer::addTraceRecord, std::move(_trace), std::move(_category),
	                    std::move(_tick));
}

void SimTopBase::addChromeTraceRecord(const std::shared_ptr<ChromeTraceRecord>& _trace) {
	this->traceCntr.run(1, &SimTraceContainer::addTraceRecord, std::move(_trace), "traceEvents", this->getGlobalTick());
}

void SimTop::connectTopChannelPorts() {
	for (auto& sim : this->threadManager->getAllSimulators()) {
		// create thread-safe channel for sim -> top
		auto to_top_channel_ptr  = std::make_shared<SimChannel<SimPacket*>>();
		auto sim_to_top_out_port = std::make_shared<MasterChannelPort>(this, to_top_channel_ptr);
		auto sim_to_top_in_port  = std::make_shared<SlaveChannelPort>(sim, to_top_channel_ptr);

		// create thread-safe channel for top -> sim
		auto from_top_channel_ptr = std::make_shared<SimChannel<SimPacket*>>();
		auto top_to_sim_out_port  = std::make_shared<MasterChannelPort>(sim, from_top_channel_ptr);
		auto top_to_sim_in_port   = std::make_shared<SlaveChannelPort>(this, from_top_channel_ptr);

		this->setTopChannelPort(sim->getName(), sim_to_top_in_port, top_to_sim_out_port);

		sim->setToTopChannelPort(sim_to_top_out_port);
		sim->setFromTopChannelPort(top_to_sim_in_port);
	}
}

void SimTop::initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) {
	unsigned n_threads            = this->nCustomThreads > 0 ? this->nCustomThreads : hw_nthreads;
	bool     n_threads_adjustable = this->nCustomThreads <= 0;

	switch (version) {
		case ThreadManagerVersion::V1:
			this->threadManager =
			    new ThreadManagerV1<ThreadManager>("ThreadManagerV1", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV1<ThreadManager>("TaskManagerV1");
			break;
		case ThreadManagerVersion::V2:
			this->threadManager =
			    new ThreadManagerV2<ThreadManager>("ThreadManagerV2", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV2<ThreadManager>("TaskManagerV2");
			break;
		case ThreadManagerVersion::V3:
			this->threadManager =
			    new ThreadManagerV3<ThreadManager>("ThreadManagerV3", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV3<ThreadManager>("TaskManagerV3");
			break;
		case ThreadManagerVersion::V4:
			this->threadManager =
			    new ThreadManagerV4<ThreadManager>("ThreadManagerV4", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV4<ThreadManager>("TaskManagerV4");
			break;
		case ThreadManagerVersion::V5:
			this->threadManager =
			    new ThreadManagerV5<ThreadManager>("ThreadManagerV5", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV5<ThreadManager>("TaskManagerV5");
			break;
		case ThreadManagerVersion::V6:
			this->threadManager =
			    new ThreadManagerV6<ThreadManager>("ThreadManagerV6", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV6<ThreadManager>("TaskManagerV6");
			break;
		case ThreadManagerVersion::V7:
			this->threadManager =
			    new ThreadManagerV7<ThreadManager>("ThreadManagerV7", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV7<ThreadManager>("TaskManagerV7");
			break;
		case ThreadManagerVersion::V8:
			this->threadManager =
			    new ThreadManagerV8<ThreadManager>("ThreadManagerV8", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV8<ThreadManager>("TaskManagerV8");
			break;
		default: CLASS_ERROR << "Invalid Thread/Task Manager Type!"; break;
	}
}

}  // end of namespace acalsim
