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

#include "sim/SCThreadManager.hh"

// ACALSim Library
#include "ACALSim.hh"

namespace acalsim {

class SCSimTop : public SimTopBase {
public:
	/**
	 * @brief Constructor for SimTop.
	 *
	 * @param _configFilePath Path of the configuration file. Default is an empty string.
	 * @param _tracingFileName Name of the tracing file. Default is "trace".
	 */
	SCSimTop(const std::string& _configFilePath = "", const std::string& _tracingFileName = "trace")
	    : SimTopBase(_configFilePath == "" ? std::vector<std::string>({}) : std::vector<std::string>({_configFilePath}),
	                 _tracingFileName) {}

	SCSimTop(const std::vector<std::string>& _configFilePaths, const std::string& _tracingFileName = "trace")
	    : SimTopBase(_configFilePaths, _tracingFileName) {}

	~SCSimTop() {}

	virtual void control_thread_step() override {}

	virtual void registerSimulators() override {}

	void connectTopChannelPorts() final;

	/**
	 * @brief Set the top channel for SystemC communication.
	 *
	 * @param name Name of the channel.
	 * @param _toTopChannel Pointer to the channel for sending data to SystemC.
	 * @param _fromTopChannel Pointer to the channel for receiving data from SystemC.
	 */
	void setSCTopChannelPort(SlaveChannelPort::SharedPtr _toTopChannel, MasterChannelPort::SharedPtr _fromTopChannel);

	/**
	 * @brief Pop data pointer from the SystemC simulator channel.
	 *
	 * @return The data pointer.
	 */
	SimPacket* popFromSCChannelPort() { return this->toSCTopChannelPort->pop(); }

	/**
	 * @brief Push a data pointer from the SystemC simulator channel.
	 *
	 * @param ptr The data pointer.
	 */
	void pushToSCChannelPort(SimPacket* const& ptr) { this->fromSCTopChannelPort->push(ptr); }

	/**
	 * @brief Performs system initial setup before the simulators are launched.
	 *
	 * This function is responsible for configuring any necessary parameters
	 * and performing setup tasks required before launching the simulators.
	 * It ensures that the environment is prepared and all prerequisites are met.
	 */
	void preSimInitSetup() override {}

	/**
	 * @brief Performs system initial setup after the simulators have been launched.
	 *
	 * This function handles tasks that need to be completed once the simulators
	 * are up and running. It may involve post-initialization adjustments,
	 * resource allocation, or other necessary operations to ensure the simulators
	 * function correctly.
	 */
	void postSimInitSetup() override {}

	void initThreadManager(ThreadManagerVersion version, unsigned int hw_nthreads) final;

protected:
	/**
	 * @brief get SystemC simulator ID
	 *
	 * @return int : SystemC simulator ID
	 */
	int getSCSimId() const { return this->getThreadManager()->getSCSimId(); }

	/**
	 * @brief get SystemC simulator name
	 *
	 * @return std::string : SystemC simulator name
	 */
	std::string getSCSimName() const { return this->getThreadManager()->getSCSimName(); }

private:
	SCThreadManager* getThreadManager() const { return static_cast<SCThreadManager*>(this->threadManager); }

	/// @brief Channel for sending data to SystemC.
	SlaveChannelPort::SharedPtr toSCTopChannelPort = nullptr;

	/// @brief Channel for receiving data from SystemC.
	MasterChannelPort::SharedPtr fromSCTopChannelPort = nullptr;
};

}  // namespace acalsim
