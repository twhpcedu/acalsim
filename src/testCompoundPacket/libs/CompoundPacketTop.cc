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

#include "CompoundPacketTop.hh"

namespace test_compound {

void CompoundPacketTop::registerCLIArguments() {
	this->getCLIApp()
	    ->add_option("--total-packets", totalPackets_, "Total number of logical packets to transfer")
	    ->default_str(std::to_string(totalPackets_));

	this->getCLIApp()
	    ->add_option("--packets-per-cycle", packetsPerCycle_,
	                 "Number of packets per CompoundPacket (bandwidth modeling)")
	    ->default_str(std::to_string(packetsPerCycle_));

	this->getCLIApp()
	    ->add_option("--queue-size", queueSize_, "Consumer queue size")
	    ->default_str(std::to_string(queueSize_));
}

void CompoundPacketTop::preSimInitSetup() {
	CLASS_INFO << "========================================";
	CLASS_INFO << "CompoundPacket Bandwidth Modeling Example";
	CLASS_INFO << "========================================";
	CLASS_INFO << "Configuration:";
	CLASS_INFO << "  Total packets:      " << totalPackets_;
	CLASS_INFO << "  Packets per cycle:  " << packetsPerCycle_;
	CLASS_INFO << "  Bytes per packet:   " << DataPacket::PACKET_SIZE_BYTES;
	CLASS_INFO << "  Effective bandwidth: " << (packetsPerCycle_ * DataPacket::PACKET_SIZE_BYTES) << " bytes/cycle";
	CLASS_INFO << "  Queue size:         " << queueSize_;
	CLASS_INFO << "========================================";
}

void CompoundPacketTop::registerSimulators() {
	// Create producer and consumer
	producer_ = new Producer("producer", 0, totalPackets_, packetsPerCycle_);
	consumer_ = new Consumer("consumer", totalPackets_, queueSize_);

	// Register with the framework
	this->addSimulator(producer_);
	this->addSimulator(consumer_);

	// Connect producer's output to consumer's input
	acalsim::SimPortManager::ConnectPort(producer_, consumer_, "out", "in");
}

void CompoundPacketTop::postSimInitSetup() {
	CLASS_INFO << "System initialized. Starting simulation...";
	CLASS_INFO << "----------------------------------------";
}

}  // namespace test_compound
