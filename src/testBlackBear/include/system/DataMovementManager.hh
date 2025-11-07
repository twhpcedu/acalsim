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

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "container/RecycleContainer/RecycleContainer.hh"
#include "event/SimEvent.hh"
#include "port/MasterPort.hh"
#include "port/SimPortManager.hh"
#include "sim/SimBase.hh"
#include "sim/SimModule.hh"
#include "system/TensorPacket.hh"
#include "utils/Logging.hh"
#include "workloads/tensor/SimTensor.hh"
#include "workloads/tensor/SimTensorManager.hh"

namespace acalsim {

struct Transaction {
	int upStreamDeviceID;   // 16 bits upstream devcie ID
	int downStreamDevicID;  // 16 bits downstream device ID
	int packetID;           // 32 bits

	Transaction(int upDID, int dsDID, int pID) : upStreamDeviceID(upDID), downStreamDevicID(dsDID), packetID(pID) {}
	uint64_t getID() {
		return (((uint64_t)upStreamDeviceID) << 48 | ((uint64_t)downStreamDevicID) << 32 | (uint64_t)packetID);
	}
};

class TensorReqEvent : public SimEvent {
public:
	TensorReqEvent(uint64_t _tid, const std::string& _name, TensorReqPacket* _pkt, SimPortManager* _callee = nullptr,
	               std::function<void(MasterPort*)> _callback = nullptr)
	    : SimEvent("TensorReqEvent_" + _name),
	      tid(_tid),
	      callee(_callee),
	      callback(_callback),
	      name(_name),
	      pkt(_pkt) {}

	~TensorReqEvent() {}

	TensorReqPacket* getPacket() { return pkt; }
	virtual void     process() {}

private:
	uint64_t                         tid;
	std::string                      name;
	TensorReqPacket*                 pkt;
	SimPortManager*                  callee;
	std::function<void(MasterPort*)> callback;
};

class TensorDataEvent : public SimEvent {
public:
	TensorDataEvent(uint64_t _tid, const std::string& _name, TensorDataPacket* _pkt, SimPortManager* _callee = nullptr,
	                std::function<void(MasterPort*)> _callback = nullptr)
	    : SimEvent("TensorReqEvent_" + _name),
	      tid(_tid),
	      callee(_callee),
	      callback(_callback),
	      name(_name),
	      pkt(_pkt) {}

	~TensorDataEvent() {}

	TensorDataPacket* getPacket() { return pkt; }
	virtual void      process() {}

private:
	uint64_t                         tid;
	std::string                      name;
	TensorDataPacket*                pkt;
	SimPortManager*                  callee;
	std::function<void(MasterPort*)> callback;
};

class DataMovementManager {
private:
	std::shared_ptr<SimTensorManager> pTensorManager;
	static uint64_t                   dataMovementSequenceID;

public:
	/**
	 * @brief Construct a new DataMovementManager interface.
	 *
	 */
	DataMovementManager(const std::string& name, std::shared_ptr<SimTensorManager> _pTensorManager)
	    : pTensorManager(_pTensorManager) {}

	SimTensor* aquaireTensor(const std::string& name, uint64_t _addr, uint32_t _width, uint32_t _height,
	                         uint32_t _srcStride = 0, uint32_t _destStride = 0,
	                         SimTensor::TENSORTYPE _type = SimTensor::TENSORTYPE::WEIGHT);
	void       recycleTensor(SimTensor* pTensor);

	/**
	 * @brief create a TensorReqPacket and send it to the downstream simulator
	 * @param _name : tensor name
	 */
	bool sendTensorReq(SimBase* srcSim, std::string dsSimName, std::string dsPortName,
	                   SimPortEvent::PushToEntryNotifyFnc  pushToEntryCallback,
	                   SimPortEvent::PopFromEntryNotifyFnc popFromEntryCallback, uint32_t srcID, uint32_t destID,
	                   uint64_t seqID,  // sequence ID, requests issued from the source
	                   int type, uint64_t addr, uint64_t size, SimTensor* pTensor);

	/**
	 * @brief create a TensorDataPacket and send it to the downstream simulator
	 * @param _name : tensor name
	 */
	bool sendTensorData(SimBase* srcSim, std::string downstream, std::string dsPortName,
	                    SimPortEvent::PushToEntryNotifyFnc  pushToEntryCallback,
	                    SimPortEvent::PopFromEntryNotifyFnc popFromEntryCallback, uint32_t srcID, uint32_t destID,
	                    uint64_t seqID,  // sequence ID, data issued from the source
	                    int type, uint64_t addr, uint64_t size, SimTensor* pTensor);
};

}  // end of namespace acalsim
