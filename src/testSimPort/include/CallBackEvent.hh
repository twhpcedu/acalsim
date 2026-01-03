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

#include "ACALSim.hh"

namespace test_port {

template <typename T>
class CallBackEvent : public acalsim::SimEvent {
public:
	CallBackEvent(std::function<T> _callback = nullptr) : SimEvent(), callback_(_callback) {}
	~CallBackEvent() override = default;

	void renew(std::function<T> _callback) { this->callback_ = _callback; }
	void process() final { callback_(); }

private:
	std::function<T> callback_ = nullptr;
};

}  // namespace test_port
