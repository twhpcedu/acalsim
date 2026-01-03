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

#include "sim/SimAddressMap.hh"

#include "utils/Logging.hh"

namespace acalsim {

void SimAddressMap::registerAddrRegion(std::string name, int deviceID, uint64_t startAddr, uint64_t size) {
	auto iter = this->addrMapRegions.find(name);

	CLASS_ASSERT_MSG(iter == this->addrMapRegions.end(), "Device :`" + name + "` Already Exist !");
	std::shared_ptr<AddrRegionStruct> ptr = std::make_shared<AddrRegionStruct>(name, deviceID, startAddr, size);
	this->addrMapRegions.insert(std::make_pair(name, ptr));
}

int SimAddressMap::getDeviceID(uint64_t addr) const {
	auto iter = addrMapRegions.begin();
	for (; iter != addrMapRegions.end(); iter++) {
		// check if the addr falls within the region
		if (addr >= iter->second->startAddr && addr < iter->second->endAddr) return iter->second->deviceID;
	}

	CLASS_ASSERT_MSG(iter != this->addrMapRegions.end(), "Addr :" + std::to_string(addr) + " is out of bound !");

	return -1;  // out of bound
}

void SimAddressMap::registerSystemAddressMap(std::string name, const std::string filename) {
	CLASS_ASSERT_MSG(false, "SimAddressMap::registerSystemAddressMap() is not implemented yet!\n");
}

}  // namespace acalsim
