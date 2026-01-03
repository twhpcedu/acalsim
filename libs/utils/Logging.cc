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

#include "utils/Logging.hh"

#include <iostream>
#include <syncstream>
#include <thread>

#include "sim/SimTop.hh"

namespace acalsim {

void LogOStream::setPrefix() {
	if (top) {
		this->ss << "Tick=" << top->getGlobalTick() << " ";
	} else {
		this->ss << "Tick=N/A ";
	}

	switch (this->level) {
		case LoggingSeverity::L_STATISTICS:
			this->ss << ANSI_SGR(ANSI_SGR::PARAMETER::FG_GREEN).getCode() + "Stats: ";
			break;
		case LoggingSeverity::L_INFO: this->ss << ANSI_SGR(ANSI_SGR::PARAMETER::FG_BLUE).getCode() + "Info: "; break;
		case LoggingSeverity::L_WARNING:
			this->ss << ANSI_SGR(ANSI_SGR::PARAMETER::FG_YELLOW).getCode() + "Warning: ";
			break;
		case LoggingSeverity::L_ERROR: this->ss << ANSI_SGR(ANSI_SGR::PARAMETER::FG_RED).getCode() + "Error: "; break;
	}

	this->ss << ANSI_SGR(ANSI_SGR::PARAMETER::RESET).getCode();
}

void LogOStream::handleTerminate() {
	bool expected_false = false;

	// Ref: https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange
	if (LogOStream::hasCalledTerminate.compare_exchange_strong(expected_false, true)) {
		try {
			// Ref: https://en.cppreference.com/w/cpp/error/rethrow_exception
			std::rethrow_exception(std::current_exception());
		} catch (const std::exception& e) { std::osyncstream(std::cerr) << e.what() << std::endl; } catch (...) {
			std::stringstream ss;
			ss << ANSI_SGR(ANSI_SGR::PARAMETER::FG_RED).getCode();
			ss << "An uncaught unknown exception happened.";
			ss << ANSI_SGR(ANSI_SGR::PARAMETER::RESET).getCode();
			std::osyncstream(std::cerr) << ss.str() << std::endl;
		}
	} else {
		while (true) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
	}

	// Ref: https://en.cppreference.com/w/cpp/utility/program/abort
	std::abort();
}

}  // namespace acalsim
