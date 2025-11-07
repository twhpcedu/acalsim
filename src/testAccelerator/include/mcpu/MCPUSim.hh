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

#include "ACALSim.hh"
using namespace acalsim;

#include "dataLinkLayer/DLLRoutingInfo.hh"

class MCPUSim : public CPPSimBase {
public:
	MCPUSim(std::string name) : CPPSimBase(name) { CLASS_INFO << "Contructing MCPUSim..."; }
	~MCPUSim() {}
	void init() override;
	void cleanup() override;
	void injectTraffic();
	void catchResponse() {
		CLASS_INFO << "catchResponse()" << this->transactionID;
		injectTraffic();
	}
	void genTraffic(uint32_t destRId, DestDMTypeEnum destDMType);

private:
	int transactionID = 0;
};
