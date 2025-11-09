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

#ifndef __RISCV_SOC_STANDALONE_HH__
#define __RISCV_SOC_STANDALONE_HH__

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

// Forward declarations
class SOCTop;

namespace ACALSim {
namespace SSTIntegration {

/**
 * @brief Standalone RISC-V SoC SST Component
 *
 * This component wraps the complete SOCTop infrastructure in a self-contained
 * SST component. It manages its own SimTop instance and runs the RISC-V
 * simulator independently within SST's event-driven framework.
 *
 * This approach encapsulates the entire ACALSim SimTop (with all its
 * simulators, channels, and thread management) as a single SST component.
 */
class RISCVSoCStandalone : public ::SST::Component {
public:
	SST_ELI_REGISTER_COMPONENT(RISCVSoCStandalone, "acalsim", "RISCVSoCStandalone", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "Standalone RISC-V RV32I SoC (includes full SimTop)", COMPONENT_CATEGORY_PROCESSOR)

	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"}, {"asm_file", "Path to RISC-V assembly file", ""},
	                        {"config_file", "Path to config JSON (optional)", ""},
	                        {"memory_size", "Memory size in bytes", "65536"},
	                        {"text_offset", "Text segment offset", "0"}, {"data_offset", "Data segment offset", "8192"},
	                        {"max_cycles", "Maximum cycles (0=unlimited)", "0"}, {"verbose", "Verbosity level", "1"})

	/**
	 * @brief Constructor
	 */
	RISCVSoCStandalone(::SST::ComponentId_t id, ::SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~RISCVSoCStandalone() override;

	void setup() override;
	void init(unsigned int phase) override;
	void finish() override;

	/**
	 * @brief Clock tick handler
	 */
	bool clockTick(::SST::Cycle_t cycle);

private:
	::SST::Output         out_;
	::SST::TimeConverter* tc_;

	std::shared_ptr<SOCTop> soc_top_;  ///< Complete RISC-V simulator

	uint64_t    current_cycle_;
	uint64_t    max_cycles_;
	bool        simulation_done_;
	bool        ready_to_terminate_;  ///< ExitEvent issued, need one more iteration
	std::string asm_file_;
	std::string config_file_;
};

}  // namespace SSTIntegration
}  // namespace ACALSim

#endif  // __RISCV_SOC_STANDALONE_HH__
