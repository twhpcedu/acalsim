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

#pragma once

#include "packet/SimPacket.hh"
#include "sim/SimBase.hh"

namespace acalsim {
namespace crossbar {

class CrossBarPacket : public SimPacket {
	friend class CrossBar;

public:
	explicit CrossBarPacket(size_t _src_idx = 0, size_t _dst_idx = 0)
	    : SimPacket(), dst_idx_(_dst_idx), src_idx_(_src_idx), tid(0) {}

	virtual ~CrossBarPacket() override = default;

	void renew(size_t _src_idx, size_t _dst_idx) {
		this->SimPacket::renew();
		this->src_idx_ = _src_idx;
		this->dst_idx_ = _dst_idx;
		this->tid      = 0;
	}

	size_t getSrcIdx() const { return this->src_idx_; }
	size_t getDstIdx() const { return this->dst_idx_; }
	int    getTransactionId() const { return this->tid; }

	void setSrcIdx(const size_t& _id) { this->src_idx_ = _id; }
	void setDstIdx(const size_t& _id) { this->dst_idx_ = _id; }
	void setTransactionId(const int& _id) { this->tid = _id; }

private:
	size_t dst_idx_, src_idx_;
	int    tid;  // Transaction ID
};

class XBarChannel : virtual public HashableType {
	std::string                           name;
	int                                   nMasters;  // number of masters
	int                                   nSlaves;   // number of slaves
	std::vector<SimPipeRegister*>         prRegs;    // nMasters pipeline registers at master side
	std::vector<std::vector<MasterPort*>> mps;       // nSlaves * nMasters MasterPorts inside BarChannel

public:
	XBarChannel(std::string _name, int _nMasters, int _nSlaves) : name(_name), nMasters(_nMasters), nSlaves(_nSlaves) {
		// Initialize prRegs vector with nMasters pipeline registers
		for (int m = 0; m < nMasters; m++) {
			std::string regName = _name + "-PR-" + std::to_string(m);
			auto        reg     = new SimPipeRegister(regName);
			reg->setClearStallAlways(
			    false);  // use the special mode in the pipeline register to keep stalling until retry
			prRegs.push_back(reg);
		}

		// Initialize mps vector with nSlaves * nMasters MasterPorts
		mps.resize(nSlaves);
		for (int s = 0; s < nSlaves; s++) {
			for (int m = 0; m < nMasters; m++) {
				std::string portName = _name + "-MP-M" + std::to_string(m) + "-S" + std::to_string(s);
				auto        mp       = new MasterPort(portName);
				mps[s].push_back(mp);
			}
		}
	}

	~XBarChannel() override = default;

	int                           getNMasters() { return nMasters; }
	int                           getNSlaves() { return nSlaves; }
	SimPipeRegister*              getPipeRegister(int mIdx) { return prRegs[mIdx]; }
	MasterPort*                   getPRMasterPort(int mIdx) { return prRegs[mIdx]->getMasterPort(); }
	SlavePort*                    getPRSlavePort(int mIdx) { return prRegs[mIdx]->getSlavePort(); }
	std::vector<SimPipeRegister*> getAllPipeRegisters() { return prRegs; }
	std::vector<MasterPort*>&     getMasterPortsBySlave(int sIdx) { return mps[sIdx]; }
};

class CrossBar : public CPPSimBase {
	std::string name;
	int         nMasters;  // number of masters
	int         nSlaves;   // number of slaves
	bool        isRetry;

	XBarChannel*                                  reqChannel;   // Channel for request transactions
	XBarChannel*                                  respChannel;  // Channel for response transactions
	std::unordered_map<std::string, XBarChannel*> channels;     // Map of channel names to channel objects

	void registerXBarChannel(SimBase* sim, XBarChannel* c) {
		// Register all request pipeline register slave ports to the SimBase Object
		for (int i = 0; i < c->getNMasters(); i++) {
			// CLASS_INFO << "PRSlavePortName: " << c->getPRSlavePort(i)->getName();
			sim->addPRSlavePort(c->getPRSlavePort(i)->getName(), c->getPipeRegister(i));
		}
		// Register all MasterPorts
		for (int s = 0; s < c->getNSlaves(); s++) {
			auto mps = c->getMasterPortsBySlave(s);
			for (auto mp : mps) sim->addMasterPort(mp->getName(), mp);
		}
	}

public:
	CrossBar(std::string _name, int _nMasters, int _nSlaves)
	    : CPPSimBase(_name), nMasters(_nMasters), nSlaves(_nSlaves), isRetry(false) {
		// Set up request and response channels
		reqChannel  = new XBarChannel(_name + ":Req", nMasters, nSlaves);
		respChannel = new XBarChannel(_name + ":Resp", nSlaves, nMasters);  // Response channel has reverse direction

		// Store channels in the map for easy access
		channels["Req"]  = reqChannel;
		channels["Resp"] = respChannel;

		// Register components to the SimBase Object
		this->registerXBarChannel(this, reqChannel);
		this->registerXBarChannel(this, respChannel);
	}

	virtual ~CrossBar() {
		// Clean up allocated resources
		delete reqChannel;
		delete respChannel;
	}

	/**
	 * Get a pipe register from a specific channel by master index
	 * @param cname Channel name ("Req" or "Resp")
	 * @param mIdx Master index
	 * @return Pointer to the pipe register
	 */
	SimPipeRegister* getPipeRegister(std::string cname, int mIdx) { return channels[cname]->getPipeRegister(mIdx); }

	/**
	 * Get the master port of a pipe register from a channel
	 * @param cname Channel name ("Req" or "Resp")
	 * @param mIdx Master index
	 * @return Pointer to the master port
	 */
	MasterPort* getPRMasterPort(std::string cname, int mIdx) { return channels[cname]->getPRMasterPort(mIdx); }

	/**
	 * Get the slave port of a pipe register from a channel
	 * @param cname Channel name ("Req" or "Resp")
	 * @param mIdx Master index
	 * @return Pointer to the slave port
	 */
	SlavePort* getPRSlavePort(std::string cname, int mIdx) { return channels[cname]->getPRSlavePort(mIdx); }

	/**
	 * Get all master ports connected to a specific slave
	 * @param cname Channel name ("Req" or "Resp")
	 * @param sIdx Slave index
	 * @return Vector of master ports connected to the slave
	 */
	std::vector<MasterPort*>& getMasterPortsBySlave(std::string cname, int sIdx) {
		return channels[cname]->getMasterPortsBySlave(sIdx);
	}

	/**
	 * Get all pipe registers from a channel
	 * @param cname Channel name ("Req" or "Resp")
	 * @return Vector of all pipe registers in the channel
	 */
	std::vector<SimPipeRegister*> getAllPipeRegisters(std::string cname) {
		return channels[cname]->getAllPipeRegisters();
	}

	/**
	 * Demultiplex transactions from master to slave in a specific channel
	 * @param cname Channel name to demultiplex
	 */
	void demux(const std::string& port_name, bool release_pressure);

	/**
	 * Handle retry mechanism for master ports
	 * @param port_name Name of the port to retry
	 */
	void masterPortRetry(MasterPort* port) override;

	/**
	 * Initialize crossbar components
	 */
	void init() override {}

	void step() override;

	void cleanup() override {}
};

}  // namespace crossbar
}  // namespace acalsim
