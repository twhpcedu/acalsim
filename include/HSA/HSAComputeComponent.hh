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

#ifndef _HSA_COMPUTE_COMPONENT_HH
#define _HSA_COMPUTE_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <queue>

#include "HSAEvents.hh"

namespace SST {
namespace ACALSim {
namespace HSA {

/**
 * HSA Compute Component (GPU/Accelerator Agent)
 *
 * Represents a compute agent (GPU/accelerator) that executes kernels
 * submitted via HSA AQL packets. Models kernel execution latency and
 * sends completion signals back to the host.
 */
class HSAComputeComponent : public SST::Component {
public:
	SST_ELI_REGISTER_COMPONENT(HSAComputeComponent, "acalsim", "HSACompute", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "HSA Compute Agent Component for kernel execution", COMPONENT_CATEGORY_PROCESSOR)

	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"}, {"verbose", "Verbosity level (0-3)", "0"},
	                        {"queue_depth", "Maximum AQL queue depth", "256"},
	                        {"cycles_per_workitem", "Simulated cycles per work-item", "100"},
	                        {"kernel_launch_overhead", "Kernel launch overhead (cycles)", "1000"},
	                        {"memory_latency", "Memory access latency (cycles)", "100"})

	SST_ELI_DOCUMENT_PORTS({"aql_port", "AQL packet reception port", {"ACALSim.HSA.HSAAQLPacketEvent"}},
	                       {"signal_port", "Signal/completion port", {"ACALSim.HSA.HSASignalEvent"}},
	                       {"doorbell_port", "Doorbell notification port (optional)", {"ACALSim.HSA.HSADoorbellEvent"}})

	SST_ELI_DOCUMENT_STATISTICS({"kernels_executed", "Total kernels executed", "kernels", 1},
	                            {"total_workitems", "Total work-items executed", "workitems", 1},
	                            {"avg_kernel_latency", "Average kernel execution latency", "cycles", 2},
	                            {"queue_occupancy", "Average queue occupancy", "packets", 2})

	HSAComputeComponent(SST::ComponentId_t id, SST::Params& params);
	~HSAComputeComponent();

	void setup() override;
	void finish() override;

private:
	// Clock handler
	bool clockTick(SST::Cycle_t currentCycle);

	// Event handlers
	void handleAQLPacket(SST::Event* ev);
	void handleDoorbellRing(SST::Event* ev);

	// Kernel execution
	void     executeKernel(HSAAQLPacketEvent* packet);
	uint64_t calculateKernelLatency(HSAAQLPacketEvent* packet);
	void     sendCompletionSignal(uint64_t signal_handle, int64_t value, uint32_t dispatch_id);

	// Ports
	SST::Link* aql_port_;
	SST::Link* signal_port_;
	SST::Link* doorbell_port_;

	// Configuration
	uint32_t verbose_;
	uint32_t queue_depth_;
	uint64_t cycles_per_workitem_;
	uint64_t kernel_launch_overhead_;
	uint64_t memory_latency_;

	// AQL Queue (HSA user mode queue model)
	struct QueuedKernel {
		HSAAQLPacketEvent* packet;
		uint64_t           start_time;
		uint64_t           completion_time;
	};
	std::queue<QueuedKernel> aql_queue_;
	QueuedKernel*            current_kernel_;  // Currently executing kernel

	// Statistics
	uint64_t             kernels_executed_;
	uint64_t             total_workitems_;
	Statistic<uint64_t>* stat_kernels_executed_;
	Statistic<uint64_t>* stat_total_workitems_;
	Statistic<uint64_t>* stat_avg_kernel_latency_;
	Statistic<uint64_t>* stat_queue_occupancy_;

	// Output
	SST::Output output_;
};

}  // namespace HSA
}  // namespace ACALSim
}  // namespace SST

#endif  // _HSA_COMPUTE_COMPONENT_HH
