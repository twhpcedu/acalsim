
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

#include "config/SimConfig.hh"
#include "utils/Logging.hh"

namespace acalsim {

template <typename T>
void SimConfig::addParameter(const std::string& _name, const T& _value, ParamType _type) {
	// Check if the parameter already exists
	LABELED_ASSERT_MSG(!this->parameters.contains(_name), this->name,
	                   "Parameter \'" + _name + "\' already exists in `SimConfig::parameters`.");

	// Create a new Parameter and add it to the map
	auto param = new Parameter<T>(_name, _value, _type);
	VERBOSE_LABELED_INFO(this->name) << "Adding parameter: \'" << _name << "\' to SimConfig.";

	this->parameters.emplace(_name, param);
}

template <typename T>
void SimConfig::setParameter(const std::string& _name, const T& _value) {
	VERBOSE_LABELED_INFO(this->name) << "Parameter \'" + _name + "\' is updated";
	this->getParameterPtr<T>(_name)->template setValue<T>("", _value);
}

template <typename T>
T SimConfig::getParameter(const std::string& _name) const {
	return this->getParameterPtr<T>(_name)->template getValue<T>("");
}

template <typename TStruct, typename T>
void SimConfig::setParameterMemberData(const std::string& _name, const std::string& _member_name, const T& _value) {
	VERBOSE_LABELED_INFO(this->name) << "Parameter \'" + _name + "::" + _member_name + "\' is updated";
	return this->getParameterPtr<TStruct>(_name)->template setValue<T>(_member_name, _value);
}

template <typename TStruct, typename T>
T SimConfig::getParameterMemberData(const std::string& _name, const std::string& _member_name) const {
	return this->getParameterPtr<TStruct>(_name)->template getValue<T>(_member_name);
}

template <typename T>
Parameter<T>* SimConfig::getParameterPtr(const std::string& _name) const {
	auto iter = this->parameters.find(_name);
	LABELED_ASSERT_MSG(iter != this->parameters.end(), this->name, "The parameter \'" + _name + "\' does not exist.");

	auto param = dynamic_cast<Parameter<T>*>(iter->second);
	LABELED_ASSERT_MSG(param, this->name, "Type mismatch for parameter \'" + _name + "\'.");

	return param;
}

}  // namespace acalsim
