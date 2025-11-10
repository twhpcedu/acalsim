/*
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
*/

#ifndef _HSA_HOST_COMPONENT_HH
#define _HSA_HOST_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <queue>
#include <map>
#include "HSAEvents.hh"

namespace SST {
namespace ACALSim {
namespace HSA {

/**
 * HSA Host Component (CPU Agent)
 *
 * Represents the host CPU agent that submits jobs to compute agents
 * using the HSA protocol. Implements AQL packet submission and signal
 * monitoring for completion notification.
 */
class HSAHostComponent : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        HSAHostComponent,
        "acalsim",
        "HSAHost",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "HSA Host Agent Component for job submission",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level (0-3)", "0"},
        {"num_dispatches", "Number of kernel dispatches to submit", "10"},
        {"workgroup_size_x", "Work-group size in X dimension", "256"},
        {"workgroup_size_y", "Work-group size in Y dimension", "1"},
        {"workgroup_size_z", "Work-group size in Z dimension", "1"},
        {"grid_size_x", "Grid size in X dimension", "1024"},
        {"grid_size_y", "Grid size in Y dimension", "1"},
        {"grid_size_z", "Grid size in Z dimension", "1"},
        {"dispatch_interval", "Cycles between dispatches", "1000"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"aql_port", "AQL packet submission port", {"ACALSim.HSA.HSAAQLPacketEvent"}},
        {"signal_port", "Signal/completion port", {"ACALSim.HSA.HSASignalEvent"}},
        {"doorbell_port", "Doorbell notification port (optional)", {"ACALSim.HSA.HSADoorbellEvent"}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"dispatches_submitted", "Total kernel dispatches submitted", "dispatches", 1},
        {"dispatches_completed", "Total kernel dispatches completed", "dispatches", 1},
        {"avg_latency", "Average kernel execution latency", "cycles", 2}
    )

    HSAHostComponent(SST::ComponentId_t id, SST::Params& params);
    ~HSAHostComponent();

    void setup() override;
    void finish() override;

private:
    // Clock handler
    bool clockTick(SST::Cycle_t currentCycle);

    // Event handlers
    void handleCompletionSignal(SST::Event* ev);
    void handleDoorbellAck(SST::Event* ev);

    // Job submission
    void submitKernelDispatch();
    HSAAQLPacketEvent* createAQLPacket(uint32_t dispatch_id);

    // Ports
    SST::Link* aql_port_;
    SST::Link* signal_port_;
    SST::Link* doorbell_port_;

    // Configuration
    uint32_t verbose_;
    uint32_t num_dispatches_;
    uint32_t workgroup_size_x_;
    uint32_t workgroup_size_y_;
    uint32_t workgroup_size_z_;
    uint32_t grid_size_x_;
    uint32_t grid_size_y_;
    uint32_t grid_size_z_;
    uint64_t dispatch_interval_;

    // State tracking
    uint32_t next_dispatch_id_;
    uint32_t dispatches_submitted_;
    uint32_t dispatches_completed_;
    uint64_t next_dispatch_cycle_;

    // Signal tracking
    std::map<uint64_t, uint64_t> signal_to_dispatch_;  // signal_handle -> dispatch_id
    std::map<uint32_t, uint64_t> dispatch_submit_time_;  // dispatch_id -> submit_time

    // Statistics
    Statistic<uint64_t>* stat_dispatches_submitted_;
    Statistic<uint64_t>* stat_dispatches_completed_;
    Statistic<uint64_t>* stat_avg_latency_;

    // Output
    SST::Output output_;
};

} // namespace HSA
} // namespace ACALSim
} // namespace SST

#endif // _HSA_HOST_COMPONENT_HH
