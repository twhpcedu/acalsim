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

#include "sim/SimAddressMap.hh"

namespace acalsim {

/**
 * @file DeviceManager.hh
 * @brief Device registration and identifier management for simulation components
 *
 * @details
 * DeviceManager extends SimAddressMap to provide device registration, naming,
 * and identifier lookup capabilities. It maintains bidirectional mappings between
 * device names, IDs, and address ranges, enabling efficient routing and debugging.
 *
 * **Device Management Model:**
 * ```
 * DeviceManager
 *     |
 *     |--- registerDevice("CPU0") → ID=0
 *     |--- registerDevice("CPU1") → ID=1
 *     |--- registerDevice("Memory") → ID=2
 *     |
 * Lookup Operations:
 *     |--- getDeviceID("CPU0") → 0
 *     |--- getDeviceID("CPU1") → 1
 *     |--- getDeviceID(0x80000000) → 2 (via address map)
 * ```
 *
 * **Key Features:**
 *
 * - **Name-to-ID Mapping**: Fast lookup of device IDs by name
 * - **Address-to-ID Mapping**: Inherited address-based device resolution
 * - **Automatic ID Assignment**: Sequential ID allocation on registration
 * - **Centralized Registry**: Single point for device management
 * - **SimAddressMap Integration**: Leverages address map for routing
 *
 * **Use Cases:**
 *
 * | Scenario | Description | Example |
 * |----------|-------------|---------|
 * | **Device Registration** | Register simulation components | CPU, memory, accelerators |
 * | **Name-based Routing** | Route messages by device name | Send packet to "CPU0" |
 * | **Address Decoding** | Resolve address to device ID | 0x80000000 → Memory controller |
 * | **Debug/Logging** | Map IDs back to readable names | "Device 2 (Memory) accessed" |
 * | **Configuration** | Build device topology | Parse config, register devices |
 *
 * **Integration with SimAddressMap:**
 * ```
 * SimAddressMap (base)
 *     ↓
 * - Address range mapping (addr → device ID)
 * - Range overlap detection
 *
 * DeviceManager (derived)
 *     ↓
 * + Device name registry (name → device ID)
 * + Automatic ID assignment
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Data Structure |
 * |-----------|-----------|----------------|
 * | registerDevice() | O(1) average | unordered_map insert |
 * | getDeviceID(name) | O(1) average | unordered_map lookup |
 * | getDeviceID(addr) | O(log n) | Address map (inherited) |
 *
 * **Memory:** sizeof(DeviceManager) ≈ sizeof(SimAddressMap) + 32 bytes (name) +
 *            8 bytes (numDevices) + hash map overhead
 *
 * **Thread Safety:**
 * - **Registration**: Not thread-safe - register all devices before parallel execution
 * - **Lookup**: Read-only after registration - thread-safe for queries
 * - **Recommendation**: Complete registration in initialization phase
 *
 * @code{.cpp}
 * // Example: Device registration and lookup
 * class System {
 * public:
 *     System() : deviceMgr("SystemDevices") {
 *         // Register all system devices
 *         int cpu0ID = deviceMgr.registerDevice("CPU0");
 *         int cpu1ID = deviceMgr.registerDevice("CPU1");
 *         int memID = deviceMgr.registerDevice("Memory");
 *         int dmaID = deviceMgr.registerDevice("DMA");
 *
 *         LOG_INFO << "CPU0 ID: " << cpu0ID;      // 0
 *         LOG_INFO << "CPU1 ID: " << cpu1ID;      // 1
 *         LOG_INFO << "Memory ID: " << memID;     // 2
 *         LOG_INFO << "DMA ID: " << dmaID;        // 3
 *
 *         // Set up address mappings (inherited from SimAddressMap)
 *         deviceMgr.addAddressRange(memID, 0x80000000, 0x80000000 + 0x10000000);
 *         deviceMgr.addAddressRange(dmaID, 0x90000000, 0x90000000 + 0x1000);
 *     }
 *
 *     void routeByName(const std::string& targetName, Packet* pkt) {
 *         int deviceID = deviceMgr.getDeviceID(targetName);
 *         if (deviceID >= 0) {
 *             devices[deviceID]->receivePacket(pkt);
 *         } else {
 *             LOG_ERROR << "Unknown device: " << targetName;
 *         }
 *     }
 *
 *     void routeByAddress(uint64_t addr, Packet* pkt) {
 *         int deviceID = deviceMgr.getDeviceID(addr);
 *         if (deviceID >= 0) {
 *             devices[deviceID]->receivePacket(pkt);
 *         } else {
 *             LOG_ERROR << "No device at address: " << std::hex << addr;
 *         }
 *     }
 *
 * private:
 *     DeviceManager deviceMgr;
 *     std::vector<Device*> devices;
 * };
 *
 * // Example: Configuration-driven device setup
 * class SystemBuilder {
 * public:
 *     void buildFromConfig(const Config& cfg) {
 *         DeviceManager devMgr("ConfiguredSystem");
 *
 *         // Parse device list from config
 *         for (const auto& devCfg : cfg.devices) {
 *             int id = devMgr.registerDevice(devCfg.name);
 *             LOG_INFO << "Registered " << devCfg.name << " with ID " << id;
 *
 *             // Add address mappings if specified
 *             if (devCfg.hasAddressRange) {
 *                 devMgr.addAddressRange(id, devCfg.baseAddr, devCfg.endAddr);
 *             }
 *         }
 *     }
 * };
 *
 * // Example: Debug logging with device names
 * class DebugLogger {
 * public:
 *     DebugLogger(DeviceManager& mgr) : devMgr(mgr) {}
 *
 *     void logAccess(uint64_t addr) {
 *         int deviceID = devMgr.getDeviceID(addr);
 *         if (deviceID >= 0) {
 *             // In real code, you'd maintain reverse mapping (ID → name)
 *             LOG_DEBUG << "Access to device " << deviceID
 *                      << " at address 0x" << std::hex << addr;
 *         }
 *     }
 *
 * private:
 *     DeviceManager& devMgr;
 * };
 * @endcode
 *
 * @note Device IDs are assigned sequentially starting from 0
 * @note Device names must be unique - duplicate registration returns existing ID
 * @note Inherits address mapping from SimAddressMap
 *
 * @warning Complete all device registration before multi-threaded execution
 * @warning Do not register devices during simulation - not thread-safe
 *
 * @see SimAddressMap for address-to-device mapping
 * @since ACALSim 0.1.0
 */
class DeviceManager : public SimAddressMap {
public:
	/**
	 * @brief Construct a device manager with a name
	 *
	 * @param _name Manager name for debugging/logging
	 *
	 * @note Initializes with zero devices registered
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * DeviceManager devMgr("SystemDevices");
	 * @endcode
	 */
	DeviceManager(std::string _name) : SimAddressMap("SimAddressMap"), name(_name), numDevices(0) {}

	/**
	 * @brief Register a device by name and assign unique ID
	 *
	 * @param deviceName Name of device to register
	 * @return int Assigned device ID (sequential: 0, 1, 2, ...)
	 *
	 * @note If device already registered, returns existing ID
	 * @note IDs assigned sequentially starting from 0
	 * @note Complexity: O(1) average case
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * DeviceManager mgr("System");
	 * int cpu0 = mgr.registerDevice("CPU0");      // Returns 0
	 * int cpu1 = mgr.registerDevice("CPU1");      // Returns 1
	 * int dup  = mgr.registerDevice("CPU0");      // Returns 0 (duplicate)
	 * @endcode
	 */
	int registerDevice(std::string deviceName);

	/**
	 * @brief Lookup device ID by device name
	 *
	 * @param name Device name to lookup
	 * @return int Device ID if found, -1 if not registered
	 *
	 * @note Case-sensitive name matching
	 * @note Complexity: O(1) average case
	 * @note Implementation defined in source file
	 *
	 * @code{.cpp}
	 * DeviceManager mgr("System");
	 * mgr.registerDevice("CPU0");
	 * mgr.registerDevice("Memory");
	 *
	 * int id1 = mgr.getDeviceID("CPU0");      // Returns 0
	 * int id2 = mgr.getDeviceID("Memory");    // Returns 1
	 * int id3 = mgr.getDeviceID("Unknown");   // Returns -1
	 * @endcode
	 */
	int getDeviceID(std::string name) const;

	/**
	 * @brief Lookup device ID by address (inherited from SimAddressMap)
	 *
	 * @param addr Memory address to lookup
	 * @return int Device ID if address mapped, -1 if not found
	 *
	 * @note Uses SimAddressMap's address range lookup
	 * @note Requires prior address range registration
	 * @note Complexity: O(log n) where n is number of address ranges
	 *
	 * @code{.cpp}
	 * DeviceManager mgr("System");
	 * int memID = mgr.registerDevice("Memory");
	 * mgr.addAddressRange(memID, 0x80000000, 0x90000000);
	 *
	 * int id = mgr.getDeviceID(0x80001000);  // Returns memID
	 * int no = mgr.getDeviceID(0x70000000);  // Returns -1 (unmapped)
	 * @endcode
	 */
	int getDeviceID(uint64_t addr) const { return SimAddressMap::getDeviceID(addr); }

protected:
	/** @brief Manager name for debugging and logging */
	std::string name;

	/** @brief Count of registered devices (also next ID to assign) */
	int numDevices;

	/**
	 * @brief Hash map for fast device name to ID lookup
	 * @details Key: device name, Value: assigned device ID
	 */
	std::unordered_map<std::string, int> deviceIDMap;
};

}  // namespace acalsim
