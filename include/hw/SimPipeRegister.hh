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

#include "packet/SimPacket.hh"
#include "port/MasterPort.hh"
#include "port/SlavePort.hh"

namespace acalsim {

/**
 * @class SimPipeRegister
 * @brief A pipeline register implementation for RISC-V simulator
 *
 * This class implements a pipeline register using master-slave port architecture
 * to facilitate data transfer between pipeline stages in a thread-safe manner.
 */
class SimPipeRegister : virtual public HashableType {
private:
	MasterPort*       mp;                // Master port for sending data
	SlavePort*        sp;                // Slave port for receiving data
	bool              flagStall;         // Flag for pipeline stall control for consumer to set in phase 1
	bool              stallStatus;       // read in phase 1, update in phase 2
	bool              clearStallAlways;  // true in normal node.  false for CrossBar design
	const std::string name;              // pipe register name

#ifdef ACALSIM_STATISTICS
public:
	inline static size_t getConnectionCnt() { return SimPipeRegister::connection_cnt_; }

private:
	inline static std::atomic<size_t> connection_cnt_ = 0;
#endif  // ACALSIM_STATISTICS

public:
	/**
	 * @brief Constructs a pipeline register with connected ports
	 * @param name Base name for the register (ports will be named name+"-S" and name+"-M")
	 */
	SimPipeRegister(const std::string& _name)
	    : name(_name), stallStatus(false), flagStall(false), clearStallAlways(true) {
		mp = new MasterPort(_name + "-S");
		sp = new SlavePort(_name + "-M", 1, new RoundRobin());
		sp->addMasterPort(mp);
		mp->registerSlavePort(sp);

#ifdef ACALSIM_STATISTICS
		SimPipeRegister::connection_cnt_++;
#endif  // ACALSIM_STATISTICS
	}

	/**
	 * @brief Destructor to clean up allocated resources
	 */
	virtual ~SimPipeRegister() {
		// Although master and slave ports are allocated by the constructor. They will be managed by the SimPortManager
		// and delted by the SimPortManager
	}

	const std::string getName() { return name; }

	/**
	 * @brief Set the stall flag to stop pipeline flow in phase 1
	 * @param flag Value to set the stall flag (default: true)
	 */
	void setStallFlag(bool flag = true) { flagStall = flag; }
	void clearStallFlag() { flagStall = false; }
	void setClearStallAlways(bool flag = true) { clearStallAlways = flag; }

	/**
	 * @brief update the stall status in phase 2
	 * @param flag Value to set the stall flag (default: true)
	 */
	void setStallStatus(bool flag = true) { stallStatus = flag; }

	/**
	 * @brief Clear the stall flag to resume pipeline flow
	 */
	void clearStallStatus() { stallStatus = false; }

	/**
	 * @brief Get the master port instance
	 * @return Pointer to the master port
	 */
	MasterPort* getMasterPort() { return mp; }

	/**
	 * @brief Get the slave port instance
	 * @return Pointer to the slave port
	 */
	SlavePort* getSlavePort() { return sp; }

	/**
	 * @brief Push a packet into the pipeline register
	 * @param packet The packet to be pushed
	 * @param callback Function to call after successful push
	 * @param strict Whether to enforce strict pushing rules
	 * @return true if push operation succeeded, false otherwise
	 */
	bool push(SimPacket* packet) {
		if (stallStatus) return false;
		return mp->push(packet);
	}

	/**
	 * @brief Check if the pipeline stage is stalled
	 * @param t Current simulation tick
	 * @return true if the pipeline is stalled, false otherwise
	 */
	bool isStalled() const { return stallStatus; }

	/**
	 * @brief Remove and return the front packet from the pipeline register
	 * @return Pointer to the removed packet
	 */
	SimPacket* pop() { return sp->pop(); }

	/**
	 * @brief Check if a pop operation would be valid
	 * @return true if pop operation can be performed
	 */
	bool isValid() const { return sp->isPopValid(); }

	/**
	 * @brief Get the front packet without removing it
	 * @return Pointer to the front packet
	 */
	SimPacket* value() const { return sp->front(); }

	/**
	 * @brief Synchronize the pipeline register at a given tick
	 *
	 * This function is part of the pipeline control phase #2.
	 *
	 * @param t Current simulation tick
	 */
	void sync() {
		bool retry = false;

		// stall condition is released
		if (stallStatus && !this->flagStall) { retry = true; }

		this->setStallStatus(this->flagStall);

		if (clearStallAlways) { this->clearStallFlag(); }  // set to false in normal mode

		SimPacket* packet = mp->value();

		if (packet && !this->stallStatus) {
			mp->pop(retry);
			mp->setPendingActivityFlag();
			if (sp->isPopValid()) {
				LABELED_WARNING(name) << "overwrite a valid packet in the slave port " << sp->getName();
				sp->pop();
			}
			sp->push(packet);
			// Log new data arrival time
			MT_DEBUG_CLASS_INFO << name << "push new packet form " << mp->getName() << " to " << sp->getName();
		} else if (retry) {
			this->mp->setRetryFlag();
		}
	}

	void dump() {
		LABELED_INFO(name) << "mp: isPushReady(): " << (mp->isPushReady() ? "Yes" : "No")
		                   << " isRetry(): " << (mp->isRetry() ? "Yes" : "No")
		                   << " sp: isValid(): " << (sp->isPopValid() ? "Yes" : "No")
		                   << " flagStall: " << (flagStall ? "Yes" : "No")
		                   << " stallStatus: " << (stallStatus ? "Yes" : "No");
	}
};

}  // namespace acalsim
