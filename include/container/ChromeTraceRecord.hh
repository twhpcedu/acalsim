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

#include <optional>

#include "container/SimTraceContainer.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**
 * @brief A single-class version of ChromeTraceRecord, handling various event types in one place.
 *
 * This class uses an enum (EventType) to differentiate between different types
 * of Chrome trace events (e.g., Complete, Instant). It also supports optional fields
 * that are relevant to some event types but not others.
 */
class ChromeTraceRecord : public SimTraceRecord {
public:
	/**
	 * @brief Enumerations for different possible event types.
	 */
	enum class EventType { Duration, Complete, Instant, Counter };

	ChromeTraceRecord(EventType evtType, const std::string& ph, const std::string& pid, const std::string& name,
	                  acalsim::Tick ts, const std::string& cat = "", const std::string& tid = "",
	                  acalsim::SimTraceRecord* args = nullptr)
	    : eventType(evtType), m_ph(ph), m_pid(pid), m_name(name), m_ts(ts), m_cat(cat), m_tid(tid), m_args(args) {}

	/**
	 * @brief Creates a Complete Event.
	 *
	 * @param _pid   Process ID.
	 * @param _name  Event name.
	 * @param _ts    Timestamp.
	 * @param _dur   Duration (used only for Complete events).
	 * @param _cat   Category (optional).
	 * @param _tid   Thread ID (optional).
	 * @param _args  Additional arguments (JsonSerializable).
	 * @return ChromeTraceRecord shared pointer
	 */
	static std::shared_ptr<ChromeTraceRecord> createCompleteEvent(const std::string& _pid, const std::string& _name,
	                                                              Tick _ts, Tick _dur, const std::string& _cat = "",
	                                                              const std::string& _tid  = "",
	                                                              SimTraceRecord*    _args = nullptr) {
		// Phase "X" is used for Complete events
		auto record = std::make_shared<ChromeTraceRecord>(EventType::Complete,
		                                                  /* ph */ "X", _pid, _name, _ts, _cat, _tid, _args);
		record->dur = _dur;  // Field only for Complete events
		return record;
	}

	/**
	 * @brief Creates an Instant Event.
	 *
	 * @param _pid   Process ID.
	 * @param _name  Event name.
	 * @param _ts    Timestamp.
	 * @param _scope The scope of the event.
	 * @param _cat   Category (optional).
	 * @param _tid   Thread ID (optional).
	 * @param _args  Additional arguments (JsonSerializable).
	 * @return ChromeTraceRecord shared pointer
	 */
	static std::shared_ptr<ChromeTraceRecord> createInstantEvent(const std::string& _pid, const std::string& _name,
	                                                             Tick _ts, const std::string& _scope,
	                                                             const std::string& _cat  = "",
	                                                             const std::string& _tid  = "",
	                                                             SimTraceRecord*    _args = nullptr) {
		// Phase "I" is used for Instant events
		auto record = std::make_shared<ChromeTraceRecord>(EventType::Instant, "i", _pid, _name, _ts, _cat, _tid, _args);
		record->scope = _scope;
		return record;
	}

	/**
	 * @brief Creates an Counter Event.
	 *
	 * @param _pid   Process ID.
	 * @param _name  Event name.
	 * @param _ts    Timestamp.
	 * @param _cat   Category (optional).
	 * @param _tid   Thread ID (optional).
	 * @param _args  Additional arguments (JsonSerializable).
	 * @return ChromeTraceRecord shared pointer
	 */
	static std::shared_ptr<ChromeTraceRecord> createCounterEvent(const std::string& _pid, const std::string& _name,
	                                                             Tick _ts, const std::string& _cat = "",
	                                                             const std::string& _tid  = "",
	                                                             SimTraceRecord*    _args = nullptr) {
		// Phase "C" is used for Counter events
		return std::make_shared<ChromeTraceRecord>(EventType::Instant, "C", _pid, _name, _ts, _cat, _tid, _args);
	}

	/**
	 * @brief Creates an Duration Event.
	 *
	 * @param _ph    Phase
	 * @param _pid   Process ID.
	 * @param _name  Event name.
	 * @param _ts    Timestamp.
	 * @param _cat   Category (optional).
	 * @param _tid   Thread ID (optional).
	 * @param _args  Additional arguments (JsonSerializable).
	 * @return ChromeTraceRecord shared pointer
	 */
	static std::shared_ptr<ChromeTraceRecord> createDurationEvent(const std::string& _ph, const std::string& _pid,
	                                                              const std::string& _name, acalsim::Tick _ts,
	                                                              const std::string&       _cat  = "",
	                                                              const std::string&       _tid  = "",
	                                                              acalsim::SimTraceRecord* _args = nullptr) {
		return std::make_shared<ChromeTraceRecord>(EventType::Duration, _ph, _pid, _name, _ts, _cat, _tid, _args);
	}

	/**
	 * @brief Serializes the fields common to all event types into a JSON object.
	 *
	 * @return nlohmann::json
	 */
	nlohmann::json serializeBaseFields() const {
		nlohmann::json j = nlohmann::json::object();

		j["ph"]   = m_ph;
		j["pid"]  = m_pid;
		j["name"] = m_name;
		j["ts"]   = m_ts;

		if (!m_cat.empty()) { j["cat"] = m_cat; }
		if (!m_tid.empty()) { j["tid"] = m_tid; }
		if (m_args) { j["args"] = m_args->toJson(); }

		return j;
	}

	/**
	 * @brief Converts the record to a JSON object following the Chrome Trace format.
	 *
	 * @return nlohmann::json
	 */
	nlohmann::json toJson() const override {
		// Start with common fields
		nlohmann::json j = serializeBaseFields();

		// Depending on the event type, add specific fields
		switch (eventType) {
			case EventType::Complete:
				if (dur.has_value())
					j["dur"] = dur.value();
				else
					WARNING << "No duration set; this might cause issues in the visualization.";
				break;

			case EventType::Duration:
				// No specific field for Instant event
				break;

			case EventType::Instant:
				if (scope.has_value())
					j["s"] = scope.value();
				else
					WARNING << "No scope set; this might cause issues in the visualization.";
				break;

			case EventType::Counter:
				// No specific field for Counter event
				if (m_args == nullptr) WARNING << "No args set; this might cause issues in the visualization.";
				break;
		}
		return j;
	}

	/**
	 * @brief Returns the current event type (Complete, Instant, etc.).
	 */
	EventType getEventType() const { return eventType; }

private:
	//--------------------------------------------------------------------------
	// Event type
	//--------------------------------------------------------------------------
	EventType eventType;

	//--------------------------------------------------------------------------
	// Common fields
	//--------------------------------------------------------------------------
	std::string     m_ph;    // Phase
	std::string     m_pid;   // Process ID
	std::string     m_name;  // Event name
	Tick            m_ts;    // Timestamp
	std::string     m_cat;   // Category
	std::string     m_tid;   // Thread ID
	SimTraceRecord* m_args;  // Additional JSON-serializable arguments

	//--------------------------------------------------------------------------
	// Optional fields for specific event types
	//--------------------------------------------------------------------------
	std::optional<Tick>        dur;    // Used only for Complete events.
	std::optional<std::string> scope;  // Used only for Instant events.
};

}  // namespace acalsim
