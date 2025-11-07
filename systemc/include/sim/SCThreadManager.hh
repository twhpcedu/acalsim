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

#include <barrier>
#include <string>
#include <systemc>
#include <thread>

#include "sim/ThreadManager.hh"

namespace acalsim {

class SCSimBase;

/**
 * @brief Plugin class for managing SystemC threads.
 */
class SCThreadManager : public ThreadManagerBase {
public:
	/**
	 * @brief Constructor.
	 *
	 * @param name Name of the thread manager plugin.
	 */
	SCThreadManager(const std::string& name, unsigned int _nThreads, bool _nThreadsAdjustable = true);

	~SCThreadManager() {}

	void addSimulator(SimBase* _sim) final;

	void startSimThreads() override;

	void terminateAllThreadsWrapper() final {
		this->terminateAllThreads();
		if (this->isSystemcExist()) this->terminateSCThread();
	}

	/**
	 * @brief get SystemC simulator ID
	 *
	 * @return int : SystemC simulator ID
	 */
	int getSCSimId() const;

	/**
	 * @brief Get the name of the SystemC simulator.
	 *
	 * @return std::string SystemC simulator name.
	 */
	const std::string getSCSimName() const;

	/**
	 * @brief Register the SystemC simulator instance.
	 *
	 * @param sim Pointer to the SystemC simulator instance.
	 */
	void registerSCSimulator(SCSimBase* sim);

	/**
	 * @brief Check if SystemC exists.
	 *
	 * @return true if SystemC exists, false otherwise.
	 */
	bool isSystemcExist() const { return this->systemcExist; }

	/**
	 * @brief Start the SystemC simulation thread.
	 */
	void startSCSimThread();

	/**
	 * @brief Terminate the SystemC simulation thread.
	 */
	void terminateSCThread() { this->scThread->join(); }

private:
	/// @brief Pointer to the SystemC simulator instance.
	SCSimBase* scSim = nullptr;

	/// @brief Pointer to the SystemC simulation thread.
	std::thread* scThread = nullptr;

	/// @brief Flag indicating if SystemC exists.
	bool systemcExist = false;

	bool nThreadsAdjustable = true;
};

}  // end of namespace acalsim
