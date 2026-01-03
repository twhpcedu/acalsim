
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

#include <CLI/CLI.hpp>
#include <string>

#include "config/CLIManager.hh"

namespace acalsim {

template <typename T, typename TStruct>
inline CLI::Option* CLIManager::addCLIOption(const std::string& _optionName, const std::string& _optionDescription,
                                             const std::string& _configName, const std::string& _paramName,
                                             const std::string& _memberName, const bool& _defaultValue) {
	// Create a callback function for CLI::App
	std::function<void(const T&)> callback = [this, _configName, _paramName, _memberName](const T& value) {
		// Create a lambda function that calls updateParameter
		auto updateFunc = [this, _configName, _paramName, _memberName, value]() {
			if constexpr (std::is_same_v<TStruct, void>) {
				this->updateParameter<T>(_configName, _paramName, _memberName, value);
			} else {
				this->updateParameter<T, TStruct>(_configName, _paramName, _memberName, value);
			}
		};

		// Add the CLI parameter with the lambda function and value
		this->addCLIParameter(_configName, _paramName, updateFunc);
	};

	// register the option with callback function into CLI Application.
	auto option = this->app.add_option_function(_optionName, callback, _optionDescription);

	// set the default value when _defaultValue flag is true. user can check the value from "--help"
	if (_defaultValue) {
		if constexpr (std::is_same_v<TStruct, void>) {
			option->default_val(this->getParameter<T>(_configName, _paramName));
		} else {
			option->default_val(this->getParameterMemberData<TStruct, T>(_configName, _paramName, _memberName));
		}
	}

	return option;
}

}  // namespace acalsim
