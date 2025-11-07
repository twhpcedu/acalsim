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

#include "channel/ChannelPortManager.hh"

namespace acalsim {

void ChannelPortManager::ConnectPort(ChannelPortManager* _sender, ChannelPortManager* _receiver,
                                     std::string _sender_port_name, std::string _receiver_port_name) {
	auto channel_ptr = std::make_shared<ChannelPort::TSimChannel>();
	auto out_port    = std::make_shared<MasterChannelPort>(_receiver, channel_ptr);
	auto in_port     = std::make_shared<SlaveChannelPort>(_sender, channel_ptr);

	static_cast<ChannelPortManager*>(_sender)->addMasterChannelPort(_sender_port_name, out_port);
	static_cast<ChannelPortManager*>(_receiver)->addSlaveChannelPort(_receiver_port_name, in_port);
}

}  // namespace acalsim
