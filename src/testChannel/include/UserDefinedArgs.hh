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

#include "ACALSim.hh"

struct UserDefinedArgs : public acalsim::SimTraceRecord {
	int         value1;
	std::string value2;

	UserDefinedArgs(int v1, const std::string& v2) : value1(v1), value2(v2) {}

	nlohmann::json toJson() const override {
		nlohmann::json data;
		data["value1"] = value1;

		nlohmann::json nested;
		nested["key1"] = 10;
		nested["key2"] = 20;
		data["nested"] = nested;

		nlohmann::json subTest;
		subTest["value2"] = 30;
		subTest["array"]  = std::vector<int>{1, 2, 3};

		nlohmann::json test;
		test["subTest"] = subTest;

		return {{"data", data}, {"test", test}};
	}
};
