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

#include <atomic>
#include <iostream>
#include <sstream>
#include <string>
#include <syncstream>

#include "utils/HashableType.hh"

/**********************************
 *                                *
 *    Basic Macro Definitions     *
 *                                *
 **********************************/

#ifndef NO_LOGS
#define INFO                    acalsim::LogOStream(acalsim::LoggingSeverity::L_INFO, __FILE__, __LINE__)
#define WARNING                 acalsim::LogOStream(acalsim::LoggingSeverity::L_WARNING, __FILE__, __LINE__)
#define LABELED_INFO(_label)    acalsim::LogOStream(acalsim::LoggingSeverity::L_INFO, __FILE__, __LINE__, _label)
#define LABELED_WARNING(_label) acalsim::LogOStream(acalsim::LoggingSeverity::L_WARNING, __FILE__, __LINE__, _label)
#else
#define INFO                    acalsim::FakeLogOStream()
#define WARNING                 acalsim::FakeLogOStream()
#define LABELED_INFO(_label)    acalsim::FakeLogOStream()
#define LABELED_WARNING(_label) acalsim::FakeLogOStream()
#endif  // #ifndef NO_LOGS

#define ERROR                 acalsim::LogOStream(acalsim::LoggingSeverity::L_ERROR, __FILE__, __LINE__)
#define LABELED_ERROR(_label) acalsim::LogOStream(acalsim::LoggingSeverity::L_ERROR, __FILE__, __LINE__, _label)

#ifndef NDEBUG
// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The `message` will be dumped as the error message.
#define ASSERT_MSG(_cond, _message) \
	if (!(_cond)) [[unlikely]]      \
	ERROR << _message

// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The `message` will be dumped as the error message.
#define LABELED_ASSERT_MSG(_cond, _label, _message) \
	if (!(_cond)) [[unlikely]]                      \
	LABELED_ERROR(_label) << _message
#else
#define ASSERT_MSG(_cond, _message)                 ((void)0)
#define LABELED_ASSERT_MSG(_cond, _label, _message) ((void)0)
#endif  // NDEBUG

// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The error message will be generated automatically.
#define ASSERT(_cond) ASSERT_MSG(_cond, std::string("Condition \"") + #_cond + "\" failed.")

// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The error message will be generated automatically.
#define LABELED_ASSERT(_cond, _label) \
	LABELED_ASSERT_MSG(_cond, _label, std::string("Condition \"") + #_cond + "\" failed.")

/**********************************
 *                                *
 *     Object Logging Macros      *
 *                                *
 **********************************/

#ifndef NO_LOGS
#define CLASS_INFO    acalsim::LogOStream(acalsim::LoggingSeverity::L_INFO, __FILE__, __LINE__, this)
#define CLASS_WARNING acalsim::LogOStream(acalsim::LoggingSeverity::L_WARNING, __FILE__, __LINE__, this)
#else
#define CLASS_INFO    acalsim::FakeLogOStream()
#define CLASS_WARNING acalsim::FakeLogOStream()
#endif  // #ifndef NO_LOGS

#define CLASS_ERROR acalsim::LogOStream(acalsim::LoggingSeverity::L_ERROR, __FILE__, __LINE__, this)

#ifndef NDEBUG
// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The `message` will be dumped as the error message.
#define CLASS_ASSERT_MSG(_cond, _message) \
	if (!(_cond)) [[unlikely]]            \
	CLASS_ERROR << _message
#else
#define CLASS_ASSERT_MSG(_cond, _message) ((void)0)
#endif

// Exit if the `cond` is unsatisfied (i.e. can be converted to false).
// The error message will be generated automatically.
#define CLASS_ASSERT(_cond) CLASS_ASSERT_MSG(_cond, std::string("Condition \"") + #_cond + "\" failed.")

/**********************************
 *                                *
 *    ACALSim Internal Logging    *
 *                                *
 **********************************/

#ifdef ACALSIM_VERBOSE
#define VERBOSE_CLASS_INFO              CLASS_INFO
#define VERBOSE_CLASS_WARNING           CLASS_WARNING
#define VERBOSE_LABELED_INFO(_label)    LABELED_INFO(_label)
#define VERBOSE_LABELED_WARNING(_label) LABELED_WARNING(_label)
#else
#define VERBOSE_CLASS_INFO              acalsim::FakeLogOStream()
#define VERBOSE_CLASS_WARNING           acalsim::FakeLogOStream()
#define VERBOSE_LABELED_INFO(_label)    acalsim::FakeLogOStream()
#define VERBOSE_LABELED_WARNING(_label) acalsim::FakeLogOStream()
#endif  // ACALSIM_VERBOSE

/******************************************************************
 *                                                                *
 *   ACALSim Internal Logging (Multi-Threading Infrastructure)    *
 *                                                                *
 ******************************************************************/

#ifdef MT_DEBUG
#define MT_DEBUG_CLASS_INFO    CLASS_INFO
#define MT_DEBUG_CLASS_WARNING CLASS_WARNING
#else
#define MT_DEBUG_CLASS_INFO    acalsim::FakeLogOStream()
#define MT_DEBUG_CLASS_WARNING acalsim::FakeLogOStream()
#endif

/******************************************
 *                                        *
 *      ACALSim Internal Statistics       *
 *                                        *
 ******************************************/

#define LABELED_STATISTICS(_label) \
	acalsim::LogOStream(acalsim::LoggingSeverity::L_STATISTICS, __FILE__, __LINE__, _label)

/**********************************
 *                                *
 *       Logging Utilities        *
 *                                *
 **********************************/

namespace acalsim {

// ref: https://stackoverflow.com/a/17469726
class ANSI_SGR {
public:
	// ref: https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
	enum class PARAMETER {
		RESET      = 0,
		BOLD       = 1,
		FG_RED     = 31,
		FG_GRAY    = 90,
		FG_BLUE    = 94,
		FG_GREEN   = 32,
		FG_YELLOW  = 33,
		FG_DEFAULT = 39
	};

	inline ANSI_SGR(PARAMETER _code) : code(_code) { ; }

	inline std::string getCode() { return "\033[" + std::to_string(static_cast<int>(code)) + "m"; }

private:
	const PARAMETER code;
};

enum class LoggingSeverity { L_STATISTICS, L_INFO, L_WARNING, L_ERROR };

class LogOStream {
public:
	LogOStream(const LoggingSeverity& _level, const std::string& _file, const size_t& _line)
	    : level(std::move(_level)) {
		this->setPrefix();
		this->setPostfix(std::move(_file), std::move(_line));
	}

	LogOStream(const LoggingSeverity& _level, const std::string& _file, const size_t& _line,
	           const HashableType* const& _ptr)
	    : level(std::move(_level)) {
		this->setPrefix();
		this->setPostfix(std::move(_file), std::move(_line));

		this->ss << "[" + _ptr->getTypeName() + "] ";
	}

	LogOStream(const LoggingSeverity& _level, const std::string& _file, const size_t& _line, const std::string& _label)
	    : level(std::move(_level)) {
		this->setPrefix();
		this->setPostfix(std::move(_file), std::move(_line));

		this->ss << "[" + _label + "] ";
	}

	~LogOStream() noexcept(false) {
		if (this->level != LoggingSeverity::L_ERROR) {
			std::string msg = this->ss.str() + this->postfix + "\n";
			std::osyncstream(std::cout) << msg;
		} else {
			if (!LogOStream::hasCalledTerminate) throw std::runtime_error(this->ss.str() + this->postfix);
		}
	}

	template <typename T>
	LogOStream& operator<<(const T& _val) {
		this->ss << _val;
		return *this;
	}

	// To handle std::endl and other manipulators
	LogOStream& operator<<(std::ostream& (*_manip)(std::ostream&)) {
		this->ss << _manip;
		return *this;
	}

public:
	static void handleTerminate();

	inline static std::atomic<bool> hasCalledTerminate = false;

private:
	void setPrefix();

	void setPostfix(const std::string& _file, const size_t& _line) {
#ifndef NDEBUG
		this->postfix += ANSI_SGR(ANSI_SGR::PARAMETER::FG_GRAY).getCode();
		this->postfix += " [" + _file + ":" + std::to_string(_line) + "]";
		this->postfix += ANSI_SGR(ANSI_SGR::PARAMETER::RESET).getCode();
#endif  // #ifndef NDEBUG
	}

private:
	const LoggingSeverity level;
	std::stringstream     ss;
	std::string           postfix;
};

class FakeLogOStream {
public:
	FakeLogOStream() = default;

	~FakeLogOStream() = default;

	template <typename T>
	inline constexpr FakeLogOStream& operator<<(const T& _val) noexcept {
		return *this;
	}

	// To handle std::endl and other manipulators
	virtual inline constexpr FakeLogOStream& operator<<(std::ostream& (*_manip)(std::ostream&)) noexcept {
		return *this;
	}
};

}  // end of namespace acalsim
