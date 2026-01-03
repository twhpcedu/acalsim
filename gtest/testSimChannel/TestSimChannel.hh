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

#include "ACALSim.hh"

enum class TestMode { Invalid = -1, Method1, Method2, Method3, Method4, Method5 };

class ReqPacket : public acalsim::SimPacket {
public:
	ReqPacket(TestMode mode = TestMode::Invalid) : SimPacket(PTYPE::PEREQ), mode(mode) {}
	void visit(acalsim::Tick when, acalsim::SimModule& module) final {}
	void visit(acalsim::Tick when, acalsim::SimBase& simulator) final;

	TestMode getTestMode() const { return this->mode; }

private:
	TestMode mode;
};

class SendSim : public acalsim::CPPSimBase {
public:
	SendSim(TestMode mode) : acalsim::CPPSimBase(getSimName()), mode(mode) {}
	~SendSim() {}

	void init() final;

	void injectTraffic_Method1(ReqPacket* packet);
	void injectTraffic_Method2(ReqPacket* packet);
	void injectTraffic_Method3(ReqPacket* packet);
	void injectTraffic_Method4(ReqPacket* packet);
	void injectTraffic_Method5(ReqPacket* packet);

	void handler(acalsim::Tick when, ReqPacket* packet);

	static std::string getSimName() { return "SendSim"; }

private:
	TestMode mode;
};

class RecvSim : public acalsim::CPPSimBase {
public:
	RecvSim(TestMode mode) : acalsim::CPPSimBase(getSimName()), mode(mode) {}
	~RecvSim() {}

	void handler(acalsim::Tick when, ReqPacket* packet);

	static std::string getSimName() { return "RecvSim"; }

private:
	TestMode mode;
};

class TestSimChannelTop : public acalsim::SimTop {
public:
	TestSimChannelTop(TestMode mode) : acalsim::SimTop(), mode(mode) {}
	~TestSimChannelTop() {}

	void registerSimulators() final;

	void registerCLIArguments() final {}

private:
	TestMode mode;

	SendSim* send;
	RecvSim* recv;
};
