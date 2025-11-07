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

#include "utils/Arguments.hh"

namespace acalsim {

std::vector<char*> getGoogleTestArguments(int argc, char** argv) {
	return getInclusiveArguments(argc, argv, "--gtest");
}

std::vector<char*> getACALSimArguments(int argc, char** argv) { return getExclusiveArguments(argc, argv, "--gtest"); }

std::vector<char*> getExclusiveArguments(int argc, char** argv, const char* prefix) {
	return getArguments(argc, argv, prefix, false);
}

std::vector<char*> getInclusiveArguments(int argc, char** argv, const char* prefix) {
	return getArguments(argc, argv, prefix, true);
}

std::vector<char*> getExclusiveArguments(std::vector<char*> args, const char* prefix) {
	return getArguments(args.size(), args.data(), prefix, false);
}

std::vector<char*> getInclusiveArguments(std::vector<char*> args, const char* prefix) {
	return getArguments(args.size(), args.data(), prefix, true);
}

std::vector<char*> getArguments(int argc, char** argv, const char* prefix, const bool inclusive) {
	std::vector<char*> args;
	args.push_back(argv[0]);

	for (int i = 1; i < argc;) {
		// find the option flag index
		bool status = false;
		if (isOption(argv[i])) {
			status = (std::strncmp(argv[i], prefix, std::strlen(prefix)) == 0) == inclusive;
			if (status) args.push_back(argv[i]);

			for (i++; i < argc && !isOption(argv[i]); ++i) {
				if (status) args.push_back(argv[i]);
			}
		}
	}
	return args;
}

}  // namespace acalsim
