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

#include <cstdlib>
#include <string>
#include <typeinfo>

namespace acalsim {

/**
 * @class HashableType
 * @brief Provides functionality to retrieve and cache the type name and hash code of the most derived object.
 *
 * This class allows derived classes to obtain a string representation of their type name and a hash code
 * representing their type. These values are computed on the first call and cached for subsequent calls.
 */
class HashableType {
public:
	HashableType()          = default;
	virtual ~HashableType() = default;

	/**
	 * @brief Retrieves the name of the most derived type.
	 *
	 * This method returns a string representing the type name of the most derived object.
	 * The name is computed and cached on the first call to optimize performance.
	 *
	 * @return A string representing the type name.
	 */
	inline virtual const std::string getTypeName() const { return typeid(*this).name(); }

	/**
	 * @brief Retrieves the hash code of the most derived type.
	 *
	 * This method returns a hash code that uniquely represents the type of the most derived object.
	 * The hash code is computed and cached on the first call to optimize performance.
	 *
	 * @return A size_t representing the hash code of the type.
	 */
	inline virtual size_t getTypeHash() const { return typeid(*this).hash_code(); }
};

}  // namespace acalsim
