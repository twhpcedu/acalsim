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

#include "ACALSim.hh"
using namespace acalsim;

#include "dataLinkLayer/DLLHeader.hh"
#include "dataLinkLayer/DLLPayload.hh"
#include "dataLinkLayer/DLLRoutingInfo.hh"

template <typename THeader, typename TPayload>
class DLLFrame {
public:
	DLLFrame() {}
	DLLFrame(int _transactionID) : transactionID(_transactionID) {}
	DLLFrame(int _transactionID, DLLRoutingInfo* rInfo) : transactionID(_transactionID), routingInfo(rInfo) {}

	~DLLFrame() {
		delete this->header;
		delete this->payload;
	}

	void setHeader(THeader* _header) { this->header = _header; }
	void setPayload(TPayload* _payload) { this->payload = _payload; }
	void setTransactionId(int _transactionID) { this->transactionID = _transactionID; }
	void setRoutingInfo(DLLRoutingInfo* rInfo) { this->routingInfo = rInfo; }

	THeader*        getHeader() { return this->header; }
	TPayload*       getPayload() { return this->payload; }
	DLLRoutingInfo* getRoutingInfo() { return this->routingInfo; }
	int             getTransactionId() { return this->transactionID; }

protected:
	int             transactionID;
	THeader*        header;
	TPayload*       payload;
	DLLRoutingInfo* routingInfo;
};

using DLLRNocFrame = DLLFrame<DLLRNocHeader, DLLRNocPayload>;
using DLLDNocFrame = DLLFrame<DLLDNocHeader, DLLDNocPayload>;
