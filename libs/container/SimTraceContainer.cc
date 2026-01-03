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

#include "container/SimTraceContainer.hh"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "utils/Logging.hh"

using json = nlohmann::json;

namespace acalsim {

void SimTraceContainer::setFilePath(const std::string& _filename_prefix, const std::string& _folder) {
	this->filename_prefix = _filename_prefix;
	this->folder          = _folder;
}

void SimTraceContainer::addTraceRecord(std::shared_ptr<SimTraceRecord> _trace, const std::string& _category,
                                       Tick _tick) {
	if (!this->cntr.contains(_category)) [[unlikely]] {
		this->cntr[_category] = std::multiset<Elem, Elem::Comparator>();
	}

	this->cntr[_category].insert(Elem{.tick = std::move(_tick), .trace = std::move(_trace)});
}

void SimTraceContainer::writeToFile() const {
	if (this->cntr.empty()) [[unlikely]] {
		LABELED_INFO("SimTraceContainer") << "There is no trace to be saved.";
		return;
	}

	json file = json::object();

	for (const auto& [category, elem_set] : this->cntr) {
		file[category] = json::array();
		for (const auto& elem : elem_set) { file[category].push_back(elem.trace->toJson()); }
	}

	std::filesystem::path folder_path = this->folder;
	if (this->folder != "" && !std::filesystem::exists(folder_path)) {
		std::filesystem::create_directories(folder_path);
	}

	std::string filename = (this->filename_prefix != "") ? this->filename_prefix + "-" : "";
	filename += SimTraceContainer::getCurrentDateTime() + ".json";
	std::ofstream ofs(folder_path / filename);
	ofs << std::setw(2) << file;

	LABELED_INFO("SimTraceContainer") << "The tracing file has been saved as "
	                                  << std::filesystem::absolute(folder_path / filename);
}

std::string SimTraceContainer::getCurrentDateTime() {
	// Get the current time
	std::time_t now       = std::time(nullptr);
	std::tm*    localTime = std::localtime(&now);  // Convert to local time

	// Create a string stream to format the output
	std::ostringstream dateTimeStream;
	dateTimeStream << std::put_time(localTime, "%Y%m%d-%H%M%S");

	return dateTimeStream.str();
}

}  // namespace acalsim
