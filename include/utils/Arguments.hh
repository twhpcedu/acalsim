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

#include <cstring>
#include <vector>

namespace acalsim {

inline const bool isOption(const char* arg) {
	return arg[0] == '-' && (arg[1] == '-' || (arg[1] != '\0' && arg[2] == '\0'));
}
std::vector<char*> getGoogleTestArguments(int argc, char** argv);
std::vector<char*> getACALSimArguments(int argc, char** argv);
std::vector<char*> getExclusiveArguments(int argc, char** argv, const char* prefix);
std::vector<char*> getInclusiveArguments(int argc, char** argv, const char* prefix);
std::vector<char*> getExclusiveArguments(std::vector<char*> args, const char* prefix);
std::vector<char*> getInclusiveArguments(std::vector<char*> args, const char* prefix);
std::vector<char*> getArguments(int argc, char** argv, const char* prefix, const bool inclusive);

}  // namespace acalsim
