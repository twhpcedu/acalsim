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

#include "common/LinkManager.hh"
#include "utils/Logging.hh"

namespace acalsim {

template <typename T>
void LinkManager<T>::addUpStream(T _mate, std::string _upStreamName) {
	bool is_present = this->upStreams.contains(_upStreamName);
	CLASS_ASSERT_MSG(!is_present, "LinkManager `" + _upStreamName + "` is present in upstream list!");

	this->upStreams.insert(std::make_pair(_upStreamName, _mate));
}

template <typename T>
T LinkManager<T>::getUpStream(std::string _name) const {
	auto iter = this->upStreams.find(_name);
	CLASS_ASSERT_MSG(iter != this->upStreams.end(), "The upstream device " + _name + " does not exist.");
	return iter->second;
}

template <typename T>
void LinkManager<T>::addDownStream(T _mate, std::string _downStreamName) {
	bool is_present = this->downStreams.contains(_downStreamName);
	CLASS_ASSERT_MSG(!is_present, "LinkManager `" + _downStreamName + "` is present in downstream list!");

	this->downStreams.insert(std::make_pair(_downStreamName, _mate));
}

template <typename T>
T LinkManager<T>::getDownStream(std::string _name) const {
	auto iter = this->downStreams.find(_name);
	CLASS_ASSERT_MSG(iter != this->downStreams.end(), "The downstream device " + _name + " does not exist.");
	return iter->second;
}

}  // namespace acalsim
