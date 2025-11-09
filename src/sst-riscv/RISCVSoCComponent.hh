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

#ifndef __RISCV_SOC_COMPONENT_HH__
#define __RISCV_SOC_COMPONENT_HH__

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>

// Forward declare RISC-V classes to avoid including full headers
class SOC;
class IFStage;
class EXEStage;
class WBStage;

namespace ACALSim {
namespace SSTIntegration {

/**
 * @brief SST Component wrapper for RISC-V RV32I System-on-Chip
 *
 * This component wraps the complete RISC-V RV32I ISA simulator from ACALSim,
 * enabling RISC-V processor simulations within the SST framework.
 *
 * Features:
 * - Complete RV32I ISA implementation (32 instructions)
 * - Event-driven timing model
 * - Pipeline visualization (IF, EXE, WB stages)
 * - Data and control hazard handling
 * - Assembly program execution from .s files
 * - Register file and memory state tracking
 *
 * System Architecture:
 * - SOC: Main system-on-chip containing CPU, memory, emulator
 * - IFStage: Instruction fetch with hazard detection
 * - EXEStage: Execution stage
 * - WBStage: Write-back stage
 * - Pipeline registers for inter-stage communication
 *
 * Supported Instructions:
 * - R-Type: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
 * - I-Type: ADDI, ANDI, ORI, XORI, SLLI, SRLI, SRAI, SLTI, SLTIU
 * - Load: LB, LBU, LH, LHU, LW
 * - Store: SB, SH, SW
 * - Branch: BEQ, BNE, BLT, BLTU, BGE, BGEU
 * - Jump: JAL, JALR
 * - Upper Immediate: LUI, AUIPC
 * - Special: HCF (Halt and Catch Fire)
 */
class RISCVSoCComponent : public ::SST::Component {
public:
	/**
	 * @brief SST ELI registration for RISC-V SoC component
	 */
	SST_ELI_REGISTER_COMPONENT(RISCVSoCComponent, "acalsim", "RISCVSoC", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "RISC-V RV32I System-on-Chip Simulator", COMPONENT_CATEGORY_PROCESSOR)

	/**
	 * @brief Parameter documentation
	 */
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency (e.g., '1GHz')", "1GHz"},
	                        {"asm_file", "Path to RISC-V assembly file (.s or .txt)", ""},
	                        {"memory_size", "Data memory size in bytes", "65536"},
	                        {"text_offset", "Text segment offset in memory", "0"},
	                        {"data_offset", "Data segment offset in memory", "8192"},
	                        {"verbose", "Verbosity level (0-5)", "1"},
	                        {"max_cycles", "Maximum simulation cycles (0 = unlimited)", "0"},
	                        {"dump_registers", "Dump register file on finish", "true"},
	                        {"dump_memory", "Dump memory contents on finish", "false"})

	/**
	 * @brief Port documentation
	 */
	SST_ELI_DOCUMENT_PORTS({"mem_port", "External memory interface port (optional)", {}})

	/**
	 * @brief Statistics documentation
	 */
	SST_ELI_DOCUMENT_STATISTICS({"instructions_executed", "Total instructions executed", "instructions", 1},
	                            {"cycles", "Total simulation cycles", "cycles", 1},
	                            {"branches_taken", "Number of branches taken", "branches", 2},
	                            {"loads", "Number of load instructions", "loads", 2},
	                            {"stores", "Number of store instructions", "stores", 2},
	                            {"pipeline_stalls", "Number of pipeline stalls", "stalls", 2})

	/**
	 * @brief Constructor
	 */
	RISCVSoCComponent(::SST::ComponentId_t id, ::SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~RISCVSoCComponent() override;

	/**
	 * @brief SST setup phase
	 */
	void setup() override;

	/**
	 * @brief SST init phase
	 */
	void init(unsigned int phase) override;

	/**
	 * @brief SST finish phase
	 */
	void finish() override;

	/**
	 * @brief Clock handler - executes one simulation cycle
	 */
	bool clockTick(::SST::Cycle_t cycle);

private:
	/**
	 * @brief Initialize RISC-V simulator components
	 */
	void initRISCVSimulator(::SST::Params& params);

	/**
	 * @brief Load assembly program
	 */
	void loadAssemblyProgram(const std::string& asm_file);

	/**
	 * @brief Check if simulation should end
	 */
	bool shouldEndSimulation();

	/**
	 * @brief Dump register file state
	 */
	void dumpRegisterFile();

	/**
	 * @brief Dump memory contents
	 */
	void dumpMemory();

	// SST infrastructure
	::SST::Output         out_;         ///< SST output for logging
	::SST::TimeConverter* tc_;          ///< Time converter
	std::string           clock_freq_;  ///< Clock frequency

	// RISC-V simulator components
	std::unique_ptr<SOC>      soc_;        ///< System-on-Chip
	std::unique_ptr<IFStage>  if_stage_;   ///< Instruction Fetch stage
	std::unique_ptr<EXEStage> exe_stage_;  ///< Execute stage
	std::unique_ptr<WBStage>  wb_stage_;   ///< Write-Back stage

	// Simulation state
	uint64_t current_cycle_;    ///< Current simulation cycle
	uint64_t max_cycles_;       ///< Maximum cycles (0 = unlimited)
	bool     simulation_done_;  ///< Simulation complete flag

	// Configuration
	std::string asm_file_path_;   ///< Assembly file path
	uint32_t    memory_size_;     ///< Memory size
	uint32_t    text_offset_;     ///< Text segment offset
	uint32_t    data_offset_;     ///< Data segment offset
	bool        dump_registers_;  ///< Dump registers on finish
	bool        dump_memory_;     ///< Dump memory on finish

	// Statistics (SST will manage these)
	::SST::Statistic<uint64_t>* stat_instructions_;     ///< Instructions executed
	::SST::Statistic<uint64_t>* stat_cycles_;           ///< Cycles
	::SST::Statistic<uint64_t>* stat_branches_;         ///< Branches taken
	::SST::Statistic<uint64_t>* stat_loads_;            ///< Load instructions
	::SST::Statistic<uint64_t>* stat_stores_;           ///< Store instructions
	::SST::Statistic<uint64_t>* stat_pipeline_stalls_;  ///< Pipeline stalls

	// External links (optional)
	::SST::Link* mem_link_;  ///< External memory link
};

}  // namespace SSTIntegration
}  // namespace ACALSim

#endif  // __RISCV_SOC_COMPONENT_HH__
