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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// ACALSim
#include "hw/SimPipeRegister.hh"
#include "utils/HashableType.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**
 * @file PipeRegisterManager.hh
 * @brief Infrastructure for managing pipeline registers in pipelined processor models
 *
 * @details
 * PipeRegisterManager provides centralized management of pipeline registers,
 * enabling coordinated stalling, flushing, and synchronization across pipeline
 * stages. Essential for modeling pipelined processors with hazard detection and
 * control.
 *
 * **Pipeline Register Model:**
 * ```
 * Fetch Stage → [IF/ID Reg] → Decode Stage → [ID/EX Reg] → Execute Stage
 *                   ↓                              ↓
 *            PipeRegisterManager         PipeRegisterManager
 *                   ↓                              ↓
 *            Stall Control              Stall Control
 *            Flush Control              Flush Control
 *            Synchronization            Synchronization
 * ```
 *
 * **Key Features:**
 *
 * - **Centralized Management**: Single point for all pipeline register control
 * - **Stall Control**: Coordinate pipeline stalls for hazards
 * - **Register Synchronization**: Atomic update of all pipeline registers
 * - **Name-based Access**: Retrieve registers by pipeline stage name
 * - **Batch Operations**: Stall/clear multiple registers simultaneously
 * - **Virtual Inheritance**: Supports complex inheritance patterns
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Operation |
 * |----------|-------------|-----------|
 * | **Data Hazards** | RAW hazard detected | Stall IF/ID, ID/EX registers |
 * | **Control Hazards** | Branch misprediction | Flush IF/ID, ID/EX registers |
 * | **Memory Stalls** | Cache miss | Stall all upstream registers |
 * | **Pipeline Flush** | Exception/interrupt | Clear all pipeline registers |
 * | **Clock Update** | End of cycle | Synchronize all register updates |
 *
 * **Typical Pipeline with Registers:**
 * ```
 * IF Stage → IF/ID Reg → ID Stage → ID/EX Reg → EX Stage → EX/MEM Reg → MEM Stage → MEM/WB Reg → WB Stage
 *               ↓           ↓          ↓           ↓          ↓              ↓           ↓
 *             [Inst]     [Decode]   [ALU Op]   [ALU Out]   [Mem Data]   [Write Data]
 *             [PC]       [Rs, Rt]   [Rs, Rt]   [Dest]      [Dest]       [Dest]
 * ```
 *
 * **Stall Propagation:**
 * ```
 * Time T:   IF  | ID  | EX  | MEM | WB
 * Time T+1: IF  | ID  | EX  | MEM | WB  (normal)
 *
 * Data Hazard Detected!
 * Time T+2: IF  | ID  | STALL | EX  | MEM  (IF/ID, ID/EX stalled)
 * Time T+3: IF  | ID  | EX    | MEM | WB   (resume)
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | addPipeRegister() | O(1) average | Hash map insert |
 * | getPipeRegister() | O(1) average | Hash map lookup |
 * | setPipeStallControl() | O(n) | n = num registers to stall |
 * | clearPipeStallControl() | O(n) | n = num registers to clear |
 * | runSyncPipeRegister() | O(m) | m = total registers |
 *
 * @code{.cpp}
 * // Example: 5-stage pipelined processor with hazard handling
 * class PipelinedCPU {
 * public:
 *     PipelinedCPU() : pipeRegMgr("CPUPipeRegs") {
 *         // Create pipeline registers for each stage boundary
 *         ifidReg = new SimPipeRegister("IF/ID");
 *         idexReg = new SimPipeRegister("ID/EX");
 *         exmemReg = new SimPipeRegister("EX/MEM");
 *         memwbReg = new SimPipeRegister("MEM/WB");
 *
 *         // Register with manager
 *         pipeRegMgr.addPipeRegister(ifidReg);
 *         pipeRegMgr.addPipeRegister(idexReg);
 *         pipeRegMgr.addPipeRegister(exmemReg);
 *         pipeRegMgr.addPipeRegister(memwbReg);
 *     }
 *
 *     void executeInstruction() {
 *         // Fetch stage
 *         Instruction inst = fetch(pc);
 *
 *         // Check for hazards
 *         if (hasDataHazard(inst)) {
 *             // Stall pipeline stages
 *             std::vector<std::string> stallRegs = {"IF/ID", "ID/EX"};
 *             pipeRegMgr.setPipeStallControl(stallRegs);
 *             LOG_INFO << "Data hazard - stalling pipeline";
 *             return;  // Don't advance this cycle
 *         }
 *
 *         // Decode stage
 *         DecodeResult decode = decodeInstruction(inst);
 *
 *         // Execute stage
 *         ExecuteResult exec = execute(decode);
 *
 *         // Memory stage
 *         MemoryResult mem = accessMemory(exec);
 *
 *         // Writeback stage
 *         writeback(mem);
 *
 *         // End of cycle: synchronize all pipeline registers
 *         pipeRegMgr.runSyncPipeRegister();
 *     }
 *
 * private:
 *     PipeRegisterManager pipeRegMgr;
 *     SimPipeRegister* ifidReg;
 *     SimPipeRegister* idexReg;
 *     SimPipeRegister* exmemReg;
 *     SimPipeRegister* memwbReg;
 * };
 *
 * // Example: Branch misprediction flush
 * class BranchUnit {
 * public:
 *     BranchUnit(PipeRegisterManager& mgr) : pipeRegMgr(mgr) {}
 *
 *     void handleBranchMisprediction(uint64_t correctPC) {
 *         LOG_INFO << "Branch mispredicted - flushing pipeline";
 *
 *         // Flush incorrectly fetched instructions
 *         auto* ifidReg = pipeRegMgr.getPipeRegister("IF/ID");
 *         auto* idexReg = pipeRegMgr.getPipeRegister("ID/EX");
 *
 *         ifidReg->flush();  // Clear IF/ID register
 *         idexReg->flush();  // Clear ID/EX register
 *
 *         // Restart fetch from correct PC
 *         pc = correctPC;
 *     }
 *
 * private:
 *     PipeRegisterManager& pipeRegMgr;
 *     uint64_t pc;
 * };
 *
 * // Example: Cache miss stalling
 * class CacheController {
 * public:
 *     CacheController(PipeRegisterManager& mgr) : pipeRegMgr(mgr) {}
 *
 *     bool accessCache(uint64_t addr) {
 *         if (!cache.contains(addr)) {
 *             // Cache miss - stall entire pipeline
 *             LOG_INFO << "Cache miss - stalling all stages";
 *
 *             std::vector<std::string> allRegs = {
 *                 "IF/ID", "ID/EX", "EX/MEM", "MEM/WB"
 *             };
 *             pipeRegMgr.setPipeStallControl(allRegs);
 *
 *             // Initiate memory fetch
 *             fetchFromMemory(addr);
 *             return false;  // Miss
 *         }
 *         return true;  // Hit
 *     }
 *
 *     void onMemoryFetchComplete(uint64_t addr, uint64_t data) {
 *         // Update cache
 *         cache.insert(addr, data);
 *
 *         // Resume pipeline
 *         LOG_INFO << "Memory fetch complete - resuming pipeline";
 *         std::vector<std::string> allRegs = {
 *             "IF/ID", "ID/EX", "EX/MEM", "MEM/WB"
 *         };
 *         pipeRegMgr.clearPipeStallControl(allRegs);
 *     }
 *
 * private:
 *     PipeRegisterManager& pipeRegMgr;
 *     Cache cache;
 * };
 * @endcode
 *
 * @note Pipeline registers are owned by PipeRegisterManager in concrete class
 * @note Stall control is non-atomic - use within single-threaded pipeline
 * @note Virtual inheritance supports diamond inheritance patterns
 *
 * @warning Do not delete registers manually - manager handles cleanup
 * @warning Ensure register names are unique within manager
 * @warning runSyncPipeRegister() must be called at cycle boundaries
 *
 * @see SimPipeRegister for pipeline register implementation
 * @since ACALSim 0.1.0
 */

/**
 * @class PipeRegisterManagerBase
 * @brief Abstract base class for pipeline register management
 *
 * @details
 * Provides interface and common functionality for managing collections of
 * pipeline registers. Derived classes implement the synchronization logic
 * for updating registers at cycle boundaries.
 *
 * **Design Pattern:**
 * - **Template Method**: runSyncPipeRegister() is pure virtual
 * - **Registry Pattern**: Maintains named register collection
 * - **Virtual Inheritance**: Prevents diamond inheritance issues
 *
 * @note Abstract class - must derive to implement runSyncPipeRegister()
 */
class PipeRegisterManagerBase : virtual public HashableType {
protected:
	/** @brief Manager name for identification */
	const std::string name;

public:
	/**
	 * @brief Construct pipeline register manager with name
	 *
	 * @param name Manager name for identification
	 *
	 * @note Initializes with empty register collection
	 * @note Complexity: O(1)
	 */
	PipeRegisterManagerBase(const std::string& name) : name(name) {}

	/**
	 * @brief Virtual destructor for safe polymorphic deletion
	 *
	 * @note Derived classes handle register cleanup
	 */
	virtual ~PipeRegisterManagerBase() {}

	/**
	 * @brief Registry of all managed pipeline registers
	 * @details Map from register name to register pointer
	 * @note Public for direct access if needed
	 */
	std::unordered_map<std::string, SimPipeRegister*> registers;

	/**
	 * @brief Add a pipeline register to the manager
	 *
	 * @param _reg Pointer to register to add
	 *
	 * @note Register must have unique name
	 * @note Manager does not take ownership (base class)
	 * @note Implementation in source file
	 * @note Complexity: O(1) average
	 *
	 * @code{.cpp}
	 * auto* ifidReg = new SimPipeRegister("IF/ID");
	 * pipeRegMgr.addPipeRegister(ifidReg);
	 * @endcode
	 */
	void addPipeRegister(SimPipeRegister* _reg);

	/**
	 * @brief Retrieve pipeline register by name
	 *
	 * @param _name Name of register to retrieve
	 * @return SimPipeRegister* Pointer to register, or nullptr if not found
	 *
	 * @note Returns nullptr for unknown names
	 * @note Implementation in source file
	 * @note Complexity: O(1) average
	 *
	 * @code{.cpp}
	 * auto* reg = pipeRegMgr.getPipeRegister("IF/ID");
	 * if (reg) {
	 *     reg->setStallFlag();
	 * }
	 * @endcode
	 */
	SimPipeRegister* getPipeRegister(const std::string& _name) const;

	/**
	 * @brief Set stall flags on specified pipeline registers
	 *
	 * @param stalled_pipe_names Vector of register names to stall
	 *
	 * @details
	 * Activates stall flag on each specified register. Stalled registers
	 * will not update on next clock cycle. Unknown names log warning
	 * and are skipped.
	 *
	 * @note Non-atomic - use in single-threaded context
	 * @note Complexity: O(n) where n = size of stalled_pipe_names
	 *
	 * @code{.cpp}
	 * // Data hazard detected - stall IF/ID and ID/EX
	 * std::vector<std::string> stallRegs = {"IF/ID", "ID/EX"};
	 * pipeRegMgr.setPipeStallControl(stallRegs);
	 * @endcode
	 */
	void setPipeStallControl(std::vector<std::string> stalled_pipe_names) {
		for (const auto& name : stalled_pipe_names) {
			// Using find() method first to check if key exists
			auto it = registers.find(name);
			if (it != registers.end()) {
				// Key exists, safe to set stall flag
				it->second->setStallFlag();
			} else {
				// Log warning about missing register
				CLASS_ASSERT_MSG("Pipeline register '%s' not found when attempting to set stall", name.c_str());
			}
		}
	}

	/**
	 * @brief Clear stall flags on specified pipeline registers
	 *
	 * @param stalled_pipe_names Vector of register names to resume
	 *
	 * @details
	 * Deactivates stall flag on each specified register, allowing them
	 * to update normally on next clock cycle. Unknown names log warning
	 * and are skipped.
	 *
	 * @note Non-atomic - use in single-threaded context
	 * @note Complexity: O(n) where n = size of stalled_pipe_names
	 *
	 * @code{.cpp}
	 * // Hazard resolved - resume pipeline
	 * std::vector<std::string> resumeRegs = {"IF/ID", "ID/EX"};
	 * pipeRegMgr.clearPipeStallControl(resumeRegs);
	 * @endcode
	 */
	void clearPipeStallControl(std::vector<std::string> stalled_pipe_names) {
		for (const auto& name : stalled_pipe_names) {
			// Using find() method first to check if key exists
			auto it = registers.find(name);
			if (it != registers.end()) {
				// Key exists, safe to set stall flag
				it->second->clearStallFlag();
			} else {
				// Log warning about missing register
				CLASS_ASSERT_MSG("Pipeline register '%s' not found when attempting to set stall", name.c_str());
			}
		}
	}

	/**
	 * @brief Get all managed pipeline registers
	 *
	 * @return const unordered_map& Reference to register collection
	 *
	 * @note Returns const reference - no modification
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * const auto& allRegs = pipeRegMgr.getAllPipeRegisters();
	 * for (const auto& [name, reg] : allRegs) {
	 *     LOG_INFO << "Register: " << name;
	 * }
	 * @endcode
	 */
	const std::unordered_map<std::string, SimPipeRegister*>& getAllPipeRegisters() const { return this->registers; }

	/**
	 * @brief Synchronize all pipeline registers (pure virtual)
	 *
	 * @details
	 * Called at end of clock cycle to update all pipeline registers.
	 * Non-stalled registers advance to next value, stalled registers
	 * maintain current value.
	 *
	 * @note Pure virtual - must be implemented by derived classes
	 * @note Should be called at cycle boundaries
	 * @note Complexity: O(m) where m = total registers
	 *
	 * @code{.cpp}
	 * // At end of every cycle
	 * pipeRegMgr.runSyncPipeRegister();
	 * @endcode
	 */
	virtual void runSyncPipeRegister() = 0;
};

/**
 * @class PipeRegisterManager
 * @brief Concrete implementation of pipeline register manager
 *
 * @details
 * Provides complete implementation including register ownership and
 * synchronization logic. Manages register lifetime and implements
 * the synchronization method.
 *
 * **Ownership:**
 * - Takes ownership of added registers
 * - Deletes all registers in destructor
 * - Registers should not be deleted externally
 */
class PipeRegisterManager : public PipeRegisterManagerBase {
public:
	/**
	 * @brief Construct concrete pipeline register manager
	 *
	 * @param name Manager name for identification
	 *
	 * @note Delegates to base class constructor
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * PipeRegisterManager pipeRegMgr("CPUPipeline");
	 * @endcode
	 */
	PipeRegisterManager(const std::string& name) : PipeRegisterManagerBase(name) {}

	/**
	 * @brief Destructor - cleans up all pipeline registers
	 *
	 * @details
	 * Iterates through all registered pipeline registers and deletes them.
	 * Ensures no memory leaks from managed registers.
	 *
	 * @note Automatically deletes all registered pipeline registers
	 * @note Registers should not be deleted externally
	 * @note Complexity: O(m) where m = number of registers
	 *
	 * @code{.cpp}
	 * {
	 *     PipeRegisterManager mgr("Pipeline");
	 *     mgr.addPipeRegister(new SimPipeRegister("IF/ID"));
	 *     // ... use manager
	 * }  // Destructor automatically cleans up IF/ID register
	 * @endcode
	 */
	virtual ~PipeRegisterManager() {
		// Clean up all registered pipe registers
		for (auto& pair : registers) { delete pair.second; }
		registers.clear();
	}

	/**
	 * @brief Implementation of pipeline register synchronization
	 *
	 * @details
	 * Iterates through all pipeline registers and calls their sync() method.
	 * Registers that are not stalled will update to their next values.
	 * Stalled registers maintain their current values.
	 *
	 * @note Implementation in source file
	 * @note Should be called at end of each clock cycle
	 * @note Complexity: O(m) where m = number of registers
	 *
	 * @code{.cpp}
	 * void CPUPipeline::clockTick() {
	 *     // Execute all pipeline stages
	 *     fetchStage();
	 *     decodeStage();
	 *     executeStage();
	 *     memoryStage();
	 *     writebackStage();
	 *
	 *     // Synchronize all pipeline registers at end of cycle
	 *     pipeRegMgr.runSyncPipeRegister();
	 * }
	 * @endcode
	 */
	virtual void runSyncPipeRegister();
};

}  // namespace acalsim
