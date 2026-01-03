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
 * @file SimPort.hh
 * @brief Base class for simulation port connectivity between modules and simulators
 *
 * SimPort provides the abstract interface for port-based communication in ACALSim.
 * Ports enable structured data flow between simulation components using a master-slave
 * paradigm similar to hardware interconnects (AXI, TileLink, etc.).
 *
 * **Port Architecture:**
 * ```
 *                 SimPort (Abstract Base)
 *                      |
 *        ┌─────────────┴─────────────┐
 *        │                           │
 *   MasterPort                  SlavePort
 *   (Initiator)                 (Target)
 *        │                           │
 *        └────────► Packet ►─────────┘
 *                 (SimPacket)
 * ```
 *
 * **Port Connection Model:**
 * ```
 * CPU Module                           Cache Module
 *   ┌──────────┐                        ┌──────────┐
 *   │          │                        │          │
 *   │  Master  ├──────► Request ──────►│  Slave   │
 *   │  Port    │                        │  Port    │
 *   │          │◄───── Response ◄──────┤          │
 *   └──────────┘                        └──────────┘
 * ```
 *
 * **Key Features:**
 * - **Master-Slave Paradigm**: Clear initiator/target roles
 * - **Named Ports**: Human-readable identifiers for debugging
 * - **Owner Tracking**: Each port knows its managing SimPortManager
 * - **Polymorphic Design**: Virtual base enables MasterPort/SlavePort specialization
 * - **Lightweight**: Minimal overhead base class
 *
 * **Use Cases:**
 *
 * | Use Case                  | Description                                       | Example                           |
 * |---------------------------|---------------------------------------------------|-----------------------------------|
 * | CPU-Cache Connection      | Memory requests from CPU to cache                 | IFetch → L1-I Cache              |
 * | Cache Hierarchy           | Victim buffer, miss handling between cache levels | L1 → L2 → L3 → Memory            |
 * | DMA Transfers             | Direct memory access from accelerators            | NPU → Memory Controller          |
 * | Interconnect Modeling     | Bus or crossbar arbitration                       | Multiple masters → Shared bus    |
 * | Pipeline Stage Comm       | Data forwarding between pipeline stages           | Decode → Execute → Memory        |
 *
 * @see MasterPort For initiator port implementation
 * @see SlavePort For target port implementation
 * @see SimPortManager For port lifecycle and connection management
 * @see SimModule For port-based modular components
 */

#pragma once

#include <string>

#include "utils/HashableType.hh"

namespace acalsim {

// Forward declaration
class SimPortManager;

/**
 * @class SimPort
 * @brief Abstract base class for simulation ports
 *
 * SimPort provides the foundation for port-based connectivity between simulation
 * components. Derived classes (MasterPort, SlavePort) implement specific initiator
 * and target behaviors.
 *
 * **Design Pattern:**
 * - Uses virtual inheritance from HashableType to prevent diamond inheritance issues
 * - Lightweight base with only name and owner tracking
 * - All port operations implemented in derived classes
 *
 * **Port Ownership:**
 * ```
 * SimPortManager (owner)
 *      │
 *      ├─ MasterPort "cpu_to_cache"
 *      ├─ SlavePort "cache_interface"
 *      └─ MasterPort "cache_to_mem"
 * ```
 *
 * @note Port names should be descriptive and unique within a module
 * @note Virtual inheritance ensures single HashableType base in hierarchy
 *
 * @see MasterPort, SlavePort, SimPortManager
 */
class SimPort : virtual public HashableType {
public:
	/**
	 * @brief Construct named simulation port
	 *
	 * @param name Port identifier (e.g., "cpu_mem_req", "cache_response")
	 *
	 * @note Default name "anonymous" provided for convenience, but named ports
	 *       recommended for debugging and visualization
	 */
	SimPort(const std::string& name = "anonymous") : name_(name) {}

	/**
	 * @brief Virtual destructor for polymorphic deletion
	 */
	virtual ~SimPort() {}

	/**
	 * @brief Get port name
	 *
	 * @return std::string Port identifier
	 *
	 * @note Port name does NOT represent the owning simulator's name
	 * @note Port names are scoped within their SimPortManager
	 */
	std::string getName() const { return this->name_; }

	/**
	 * @brief Set owning port manager
	 *
	 * Called automatically by SimPortManager during port registration.
	 * Enables port to access manager services (e.g., retry callbacks).
	 *
	 * @param owner Pointer to managing SimPortManager
	 *
	 * @note Typically called once during port creation, not by users
	 */
	void setOwner(SimPortManager* owner) { this->owner_ = owner; }

protected:
	/// @brief Port name for identification and debugging
	std::string name_;

	/// @brief Owning SimPortManager (set during registration)
	SimPortManager* owner_ = nullptr;
};

}  // namespace acalsim
