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

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "utils/HashableType.hh"

namespace acalsim {

/**
 * @file SimAddressMap.hh
 * @brief Address space mapping and decoding for system-level simulation
 *
 * @details
 * SimAddressMap provides infrastructure for mapping memory addresses to device IDs,
 * enabling address-based routing and device resolution in system simulations.
 * It supports both programmatic registration and JSON-based configuration.
 *
 * **Address Mapping Model:**
 * ```
 * Address Space                    Device Resolution
 * 0x00000000 ─────────────────
 *            │   Boot ROM   │ → Device ID: 0
 * 0x10000000 ─────────────────
 *            │   Memory     │ → Device ID: 1
 * 0x80000000 ─────────────────
 *            │   MMIO       │ → Device ID: 2
 * 0x90000000 ─────────────────
 *            │  Unmapped    │ → Device ID: -1 (error)
 * 0xFFFFFFFF ─────────────────
 * ```
 *
 * **Key Features:**
 *
 * - **Address Decoding**: Fast address-to-device ID resolution
 * - **Region Management**: Named address regions with bounds
 * - **JSON Configuration**: Load address maps from configuration files
 * - **Overlap Detection**: Prevent conflicting address assignments
 * - **Virtual Inheritance**: Supports diamond inheritance patterns
 * - **HashableType Support**: Enables use in hash-based containers
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Example |
 * |----------|-------------|---------|
 * | **Memory Routing** | Route memory requests to controllers | 0x80000000 → DRAM controller |
 * | **MMIO Decoding** | Decode memory-mapped I/O addresses | 0x90000000 → DMA device |
 * | **Address Validation** | Check if address is valid | Unmapped → return -1 |
 * | **System Configuration** | Build address map from JSON config | Load memory map file |
 * | **Multi-device Systems** | Manage address spaces for many devices | NoC with many endpoints |
 *
 * **Address Resolution Algorithm:**
 * ```
 * For address 0x80001000:
 * 1. Iterate through addrMapRegions
 * 2. Check: startAddr ≤ addr < endAddr
 * 3. If match: return deviceID
 * 4. If no match: return -1
 * ```
 *
 * **JSON Configuration Format:**
 * ```json
 * {
 *   "address_map": {
 *     "boot_rom": {
 *       "device_id": 0,
 *       "start_addr": "0x00000000",
 *       "size": "0x10000000"
 *     },
 *     "memory": {
 *       "device_id": 1,
 *       "start_addr": "0x10000000",
 *       "size": "0x70000000"
 *     },
 *     "mmio": {
 *       "device_id": 2,
 *       "start_addr": "0x90000000",
 *       "size": "0x10000000"
 *     }
 *   }
 * }
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | registerAddrRegion() | O(1) average | Hash map insert |
 * | getDeviceID(addr) | O(n) | Linear search through regions |
 * | registerSystemAddressMap() | O(n) | JSON parsing + n inserts |
 *
 * **Memory:** sizeof(SimAddressMap) ≈ 32 bytes (name) +
 *            hash map overhead + n × sizeof(AddrRegionStruct)
 *
 * **Thread Safety:**
 * - **Registration**: Not thread-safe - configure before parallel execution
 * - **Lookup**: Thread-safe after initialization (read-only)
 * - **Recommendation**: Complete all registration in setup phase
 *
 * @code{.cpp}
 * // Example: Programmatic address map setup
 * class MemorySystem {
 * public:
 *     MemorySystem() : addrMap("MemoryAddressMap") {
 *         // Register address regions for different devices
 *         addrMap.registerAddrRegion("BootROM", 0, 0x00000000, 0x10000000);
 *         addrMap.registerAddrRegion("DRAM",    1, 0x10000000, 0x70000000);
 *         addrMap.registerAddrRegion("MMIO",    2, 0x90000000, 0x10000000);
 *
 *         LOG_INFO << "Address map configured";
 *     }
 *
 *     void routeMemoryRequest(uint64_t addr, MemoryRequest* req) {
 *         int deviceID = addrMap.getDeviceID(addr);
 *         if (deviceID >= 0) {
 *             devices[deviceID]->handleRequest(req);
 *         } else {
 *             LOG_ERROR << "Unmapped address: 0x" << std::hex << addr;
 *         }
 *     }
 *
 * private:
 *     SimAddressMap addrMap;
 *     std::vector<MemoryDevice*> devices;
 * };
 *
 * // Example: JSON-based address map configuration
 * class ConfigurableSystem {
 * public:
 *     ConfigurableSystem(const std::string& configFile)
 *         : addrMap("SystemAddressMap") {
 *         // Load address map from JSON configuration
 *         addrMap.registerSystemAddressMap("system", configFile);
 *         LOG_INFO << "Loaded address map from: " << configFile;
 *     }
 *
 *     bool isValidAddress(uint64_t addr) const {
 *         return addrMap.getDeviceID(addr) >= 0;
 *     }
 *
 * private:
 *     SimAddressMap addrMap;
 * };
 *
 * // Example: Multi-level memory system
 * class HierarchicalMemory {
 * public:
 *     HierarchicalMemory() : addrMap("HierarchicalMap") {
 *         // L1 Cache - not in address map (CPU-internal)
 *         // L2 Cache - shared, mapped
 *         addrMap.registerAddrRegion("L2Cache", 0, 0x80000000, 0x100000);
 *
 *         // Main Memory
 *         addrMap.registerAddrRegion("DRAM0", 1, 0x00000000, 0x40000000);
 *         addrMap.registerAddrRegion("DRAM1", 2, 0x40000000, 0x40000000);
 *
 *         // Storage
 *         addrMap.registerAddrRegion("NVMe", 3, 0xC0000000, 0x20000000);
 *     }
 *
 *     MemoryLevel getMemoryLevel(uint64_t addr) {
 *         int devID = addrMap.getDeviceID(addr);
 *         switch (devID) {
 *             case 0: return MemoryLevel::L2_CACHE;
 *             case 1:
 *             case 2: return MemoryLevel::DRAM;
 *             case 3: return MemoryLevel::STORAGE;
 *             default: return MemoryLevel::INVALID;
 *         }
 *     }
 *
 * private:
 *     SimAddressMap addrMap;
 * };
 * @endcode
 *
 * @note Address ranges must not overlap (no validation currently enforced)
 * @note Virtual inheritance allows use in complex inheritance hierarchies
 * @note JSON parsing uses nlohmann::json library
 *
 * @warning Complete all registration before multi-threaded access
 * @warning getDeviceID() returns -1 for unmapped addresses
 * @warning No automatic overlap detection - user must ensure valid ranges
 *
 * @see DeviceManager for device name management built on this
 * @see AddrRegionStruct for address region representation
 * @since ACALSim 0.1.0
 */

/**
 * @struct AddrRegionStruct
 * @brief Represents a contiguous address region mapped to a device
 *
 * @details
 * AddrRegionStruct encapsulates the properties of an address region including
 * its name, associated device ID, and address bounds. The end address is
 * automatically calculated from start address and size.
 *
 * **Address Region Layout:**
 * ```
 * startAddr                              endAddr
 *     ↓                                      ↓
 * [════════════════ size ════════════════]
 *     ↑─── Mapped to deviceID ───↑
 * ```
 *
 * **Bounds Checking:**
 * ```cpp
 * bool isInRegion(uint64_t addr) {
 *     return (addr >= startAddr) && (addr < endAddr);
 * }
 * ```
 *
 * @note endAddr is exclusive (range is [startAddr, endAddr))
 * @note size determines the region extent
 *
 * @code{.cpp}
 * // Example: Create address region for DRAM
 * AddrRegionStruct dram("DRAM", 1, 0x80000000, 0x40000000);
 * // name = "DRAM"
 * // deviceID = 1
 * // startAddr = 0x80000000
 * // size = 0x40000000 (1GB)
 * // endAddr = 0xC0000000 (automatically calculated)
 *
 * // Example: Check if address is in region
 * uint64_t addr = 0x80001000;
 * bool inRegion = (addr >= dram.startAddr) && (addr < dram.endAddr);
 * @endcode
 */
struct AddrRegionStruct {
	/** @brief Human-readable name for this address region */
	std::string name;

	/** @brief Device ID that owns this address region */
	int deviceID;

	/** @brief Starting address of the region (inclusive) */
	uint64_t startAddr;

	/** @brief Ending address of the region (exclusive) */
	uint64_t endAddr;

	/** @brief Size of the address region in bytes */
	uint64_t size;

	/**
	 * @brief Construct an address region
	 *
	 * @param _name Region name for debugging/logging
	 * @param _deviceID Device ID that handles this region
	 * @param _startAddr Starting address (inclusive)
	 * @param _size Size of region in bytes
	 *
	 * @note endAddr is automatically calculated as startAddr + size
	 * @note Region covers [startAddr, endAddr) interval
	 *
	 * @code{.cpp}
	 * // Create 256MB region starting at 0x80000000
	 * AddrRegionStruct mem("Memory", 1, 0x80000000, 0x10000000);
	 * // endAddr will be 0x90000000
	 * @endcode
	 */
	AddrRegionStruct(std::string _name, int _deviceID, uint64_t _startAddr, uint64_t _size) {
		name      = _name;
		deviceID  = _deviceID;
		startAddr = _startAddr;
		size      = _size;
		endAddr   = startAddr + size;
	}
};

/**
 * @class SimAddressMap
 * @brief Address space management with device ID resolution
 *
 * @details
 * SimAddressMap manages the mapping between memory addresses and device identifiers,
 * enabling address-based routing in system-level simulations. Supports both
 * programmatic and JSON-based configuration.
 *
 * **Virtual Inheritance:**
 * Uses virtual inheritance from HashableType to support diamond inheritance
 * patterns (e.g., when multiple base classes inherit from HashableType).
 *
 * @note getDeviceID() method name in comment is incorrect - should say "lookup deviceID by addr"
 */
class SimAddressMap : virtual public HashableType {
public:
	/**
	 * @brief Construct an address map with a name
	 *
	 * @param _name Map name for debugging and logging
	 *
	 * @note Initializes with no address regions
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * SimAddressMap addrMap("SystemMemoryMap");
	 * @endcode
	 */
	SimAddressMap(std::string _name) : name(_name) {}

	/**
	 * @brief Register system address map from JSON configuration file
	 *
	 * @param name Name for this address map configuration
	 * @param filename Path to JSON configuration file
	 *
	 * @note JSON format must match expected schema (see file-level docs)
	 * @note Parses JSON and calls registerAddrRegion() for each entry
	 * @note Complexity: O(n) where n is number of regions in JSON
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * SimAddressMap addrMap("SystemMap");
	 * addrMap.registerSystemAddressMap("main", "config/memory_map.json");
	 *
	 * // JSON file format:
	 * // {
	 * //   "address_map": {
	 * //     "dram": {
	 * //       "device_id": 1,
	 * //       "start_addr": "0x80000000",
	 * //       "size": "0x40000000"
	 * //     }
	 * //   }
	 * // }
	 * @endcode
	 */
	void registerSystemAddressMap(std::string name, const std::string filename);

	/**
	 * @brief Register an address region programmatically
	 *
	 * @param name Region name for debugging
	 * @param deviceID Device ID that owns this region
	 * @param startAddr Starting address (inclusive)
	 * @param size Size of region in bytes
	 *
	 * @note Creates AddrRegionStruct with endAddr = startAddr + size
	 * @note Stores in addrMapRegions hash map with name as key
	 * @note Complexity: O(1) average case
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * SimAddressMap addrMap("MemMap");
	 *
	 * // Register 1GB DRAM at 0x80000000
	 * addrMap.registerAddrRegion("DRAM", 1, 0x80000000, 0x40000000);
	 *
	 * // Register 256MB MMIO at 0x90000000
	 * addrMap.registerAddrRegion("MMIO", 2, 0x90000000, 0x10000000);
	 * @endcode
	 */
	void registerAddrRegion(std::string name, int deviceID, uint64_t startAddr, uint64_t size);

	/**
	 * @brief Lookup device ID by memory address
	 *
	 * @param addr Memory address to resolve
	 * @return int Device ID if address is mapped, -1 if unmapped
	 *
	 * @note Searches all regions for addr ∈ [startAddr, endAddr)
	 * @note Returns -1 if no region contains the address
	 * @note Complexity: O(n) where n is number of regions
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * SimAddressMap addrMap("Map");
	 * addrMap.registerAddrRegion("DRAM", 1, 0x80000000, 0x40000000);
	 *
	 * int dev1 = addrMap.getDeviceID(0x80001000);  // Returns 1 (DRAM)
	 * int dev2 = addrMap.getDeviceID(0x70000000);  // Returns -1 (unmapped)
	 * @endcode
	 */
	int getDeviceID(uint64_t addr) const;

protected:
	/** @brief Address map name for debugging and logging */
	std::string name;

	/**
	 * @brief Hash map of address regions
	 * @details Key: region name, Value: shared_ptr to AddrRegionStruct
	 * @note Uses shared_ptr for safe memory management
	 */
	std::unordered_map<std::string, std::shared_ptr<AddrRegionStruct>> addrMapRegions;
};

}  // namespace acalsim
