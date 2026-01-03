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

#include <memory>
#include <string>

enum FlitTypeEnum { HEAD, BODY, TAIL };
enum TrafficTypeEnum { UNICAST, BROADCAST, MULTICAST };
enum DestDMTypeEnum { TRAFFIC_GENERATOR, CACHE, PE, BLACKHOLE };

class DLLRoutingInfo {
public:
	DLLRoutingInfo() {}
	DLLRoutingInfo(FlitTypeEnum _flitType, TrafficTypeEnum _trafficType, uint32_t _destRId, DestDMTypeEnum _destDMType)
	    : flitType(_flitType), trafficType(_trafficType), destRId(_destRId), destDMType(_destDMType) {}

	void set(FlitTypeEnum _flitType, TrafficTypeEnum _trafficType, uint32_t _destRId, DestDMTypeEnum _destDMType) {
		this->flitType    = _flitType;
		this->trafficType = _trafficType;
		this->destRId     = _destRId;
		this->destDMType  = _destDMType;
	}

	FlitTypeEnum getFlitType() const { return this->flitType; }

	TrafficTypeEnum getTrafficType() const { return this->trafficType; }

	uint32_t getDestRId() const { return this->destRId; }

	DestDMTypeEnum getDestDMType() const { return this->destDMType; }

protected:
	FlitTypeEnum    flitType;
	TrafficTypeEnum trafficType;
	uint32_t        destRId;
	DestDMTypeEnum  destDMType;
};
