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
#include "TBSim.hh"

namespace testcrossbar {

class CrossBarTestTop : public acalsim::SimTop {
public:
	explicit CrossBarTestTop(const std::vector<std::string>& _configFilePaths = {},
	                         const std::string&              _tracingFileName = "");
	virtual ~CrossBarTestTop() override = default;

	void registerConfigs() final;
	void registerPipeRegisters() final;
	void registerCLIArguments() final;
	void registerSimulators() final;
	void postSimInitSetup() final;

private:
	acalsim::crossbar::CrossBar* bus;
	std::vector<MasterTBSim*>    m_devices;
	std::vector<SlaveTBSim*>     s_devices;
};

}  // namespace testcrossbar
