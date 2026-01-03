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

/**
 * @file main.cc
 * @brief CompoundPacket bandwidth modeling example
 *
 * This example demonstrates how to use CompoundPacket to model different bandwidth
 * requirements in ACALSim. The key insight is that SimPort (MasterPort/SlavePort)
 * has a throughput limit of 1 packet per cycle. By packing multiple logical packets
 * into a single CompoundPacket, you can model higher bandwidths.
 *
 * **The Problem:**
 * ```
 * SimPort throughput = 1 packet/cycle
 * If each packet = 64 bytes, then max bandwidth = 64 bytes/cycle
 * ```
 *
 * **The Solution: CompoundPacket**
 * ```
 * Pack N packets into 1 CompoundPacket
 * Send 1 CompoundPacket/cycle through SimPort
 * Effective bandwidth = N × 64 bytes/cycle
 * ```
 *
 * **Example: Modeling 256 bytes/cycle bandwidth**
 * ```cpp
 * // Create compound with 4 packets (4 × 64 = 256 bytes)
 * auto* compound = new CompoundPacket<DataPacket>(sourceId);
 * compound->addPacket(new DataPacket(seq++, srcId));
 * compound->addPacket(new DataPacket(seq++, srcId));
 * compound->addPacket(new DataPacket(seq++, srcId));
 * compound->addPacket(new DataPacket(seq++, srcId));
 *
 * // Send through MasterPort (1 cycle for 256 bytes)
 * masterPort->push(compound);
 *
 * // Receiver unpacks
 * auto packets = compound->extractPackets();
 * for (auto* pkt : packets) { process(pkt); delete pkt; }
 * delete compound;
 * ```
 *
 * **System Architecture:**
 * ```
 * ┌──────────────────┐                    ┌──────────────────┐
 * │    Producer      │                    │    Consumer      │
 * │                  │                    │                  │
 * │  Creates N       │   CompoundPacket   │  Unpacks N       │
 * │  DataPackets     │ ────────────────►  │  DataPackets     │
 * │  per cycle       │   (1 per cycle)    │  per cycle       │
 * │                  │                    │                  │
 * │  Bandwidth:      │                    │  Receives:       │
 * │  N × 64 B/cycle  │                    │  N × 64 B/cycle  │
 * └──────────────────┘                    └──────────────────┘
 *        │                                       │
 *        └──── MasterPort ──── SlavePort ───────┘
 *                  (1 packet/cycle limit)
 * ```
 *
 * **Command-Line Options:**
 * ```bash
 * # Default: 100 packets, 4 packets/compound (256 B/cycle)
 * ./testCompoundPacket
 *
 * # High bandwidth: 8 packets/compound (512 B/cycle)
 * ./testCompoundPacket --packets-per-cycle 8
 *
 * # Low bandwidth: 1 packet/compound (64 B/cycle, no batching)
 * ./testCompoundPacket --packets-per-cycle 1
 *
 * # Large transfer with high bandwidth
 * ./testCompoundPacket --total-packets 1000 --packets-per-cycle 16
 * ```
 *
 * **Expected Output:**
 * ```
 * ========================================
 * CompoundPacket Bandwidth Modeling Example
 * ========================================
 * Configuration:
 *   Total packets:      100
 *   Packets per cycle:  4
 *   Bytes per packet:   64
 *   Effective bandwidth: 256 bytes/cycle
 * ========================================
 * [Producer] Sent CompoundPacket with 4 packets (seq 0-3) at tick 1
 * [Consumer] Received CompoundPacket from source 0 with 4 packets at tick 2
 * [Consumer]   - Unpacked DataPacket[seq=0, producer=0]
 * [Consumer]   - Unpacked DataPacket[seq=1, producer=0]
 * [Consumer]   - Unpacked DataPacket[seq=2, producer=0]
 * [Consumer]   - Unpacked DataPacket[seq=3, producer=0]
 * ...
 * [Consumer] All 100 packets received!
 * [Consumer] Average packets per CompoundPacket: 4 (effective bandwidth: 256 bytes/cycle)
 * ```
 *
 * @see CompoundPacket For the compound packet implementation
 * @see Producer For how to create and send compound packets
 * @see Consumer For how to receive and unpack compound packets
 */

#include "ACALSim.hh"
#include "CompoundPacketTop.hh"

int main(int argc, char** argv) {
	acalsim::top = std::make_shared<test_compound::CompoundPacketTop>();
	acalsim::top->init(argc, argv);
	acalsim::top->run();
	acalsim::top->finish();
	return 0;
}
