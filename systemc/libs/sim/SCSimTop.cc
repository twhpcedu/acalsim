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

#include "sim/SCSimTop.hh"

// ACALSim SystemC Library
#include "sim/SCThreadManager.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

void SCSimTop::connectTopChannelPorts() {
	for (auto& sim : this->threadManager->getAllSimulators()) {
		// create thread-safe channel for sim -> top
		auto to_top_channel_ptr  = std::make_shared<SimChannel<SimPacket*>>();
		auto sim_to_top_out_port = std::make_shared<MasterChannelPort>(this, to_top_channel_ptr);
		auto sim_to_top_in_port  = std::make_shared<SlaveChannelPort>(sim, to_top_channel_ptr);

		// create thread-safe channel for top -> sim
		auto from_top_channel_ptr = std::make_shared<SimChannel<SimPacket*>>();
		auto top_to_sim_out_port  = std::make_shared<MasterChannelPort>(sim, from_top_channel_ptr);
		auto top_to_sim_in_port   = std::make_shared<SlaveChannelPort>(this, from_top_channel_ptr);

		this->setTopChannelPort(sim->getName(), sim_to_top_in_port, top_to_sim_out_port);

		if (sim->isSystemC()) [[unlikely]] { this->setSCTopChannelPort(sim_to_top_in_port, top_to_sim_out_port); }

		sim->setToTopChannelPort(sim_to_top_out_port);
		sim->setFromTopChannelPort(top_to_sim_in_port);
	}
}

void SCSimTop::setSCTopChannelPort(SlaveChannelPort::SharedPtr  _toTopChannelPort,
                                   MasterChannelPort::SharedPtr _fromTopChannelPort) {
	this->toSCTopChannelPort   = _toTopChannelPort;
	this->fromSCTopChannelPort = _fromTopChannelPort;
}

void SCSimTop::initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) {
	unsigned n_threads            = this->nCustomThreads > 0 ? this->nCustomThreads : hw_nthreads;
	bool     n_threads_adjustable = this->nCustomThreads <= 0;

	switch (version) {
		case ThreadManagerVersion::V1:
			this->threadManager =
			    new ThreadManagerV1<SCThreadManager>("ThreadManagerV1", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV1<SCThreadManager>("TaskManagerV1");
			break;
		case ThreadManagerVersion::V2:
			this->threadManager =
			    new ThreadManagerV2<SCThreadManager>("ThreadManagerV2", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV2<SCThreadManager>("TaskManagerV2");
			break;
		case ThreadManagerVersion::V3:
			this->threadManager =
			    new ThreadManagerV3<SCThreadManager>("ThreadManagerV3", n_threads, n_threads_adjustable);
			this->taskManager = new TaskManagerV3<SCThreadManager>("TaskManagerV3");
			break;
		default: CLASS_ERROR << "Invalid Thread/Task Manager Type!"; break;
	}
}

}  // namespace acalsim
