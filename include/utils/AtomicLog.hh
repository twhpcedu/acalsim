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

#include <iostream>
#include <mutex>
#include <sstream>

namespace acalsim {

/**
 * @brief Thread-safe atomic logging utility
 *
 * This class provides atomic (non-interruptible) logging to prevent
 * log message corruption when multiple components log concurrently.
 *
 * Usage:
 *   ATOMIC_LOG() << "[TAG] message " << value << " more text" << std::endl;
 *
 * The entire message is buffered and written atomically in one operation.
 */
class AtomicLog {
public:
	AtomicLog() = default;
	~AtomicLog() {
		// Flush buffered message atomically on destruction
		static std::mutex           logMutex;
		std::lock_guard<std::mutex> lock(logMutex);
		std::cout << buffer.str();
		std::cout.flush();
	}

	// Enable stream-style usage: atomicLog << "text" << value
	template <typename T>
	AtomicLog& operator<<(const T& value) {
		buffer << value;
		return *this;
	}

	// Handle std::endl and other manipulators
	AtomicLog& operator<<(std::ostream& (*manip)(std::ostream&)) {
		buffer << manip;
		return *this;
	}

	// Handle hex/dec manipulators
	AtomicLog& operator<<(std::ios_base& (*manip)(std::ios_base&)) {
		buffer << manip;
		return *this;
	}

private:
	std::ostringstream buffer;
};

}  // namespace acalsim

// Convenience macro for atomic logging
#define ATOMIC_LOG() acalsim::AtomicLog()
