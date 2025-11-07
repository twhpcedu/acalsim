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

#include "profiling/Utils.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**********************************
 *                                *
 *          ConstexprStr          *
 *                                *
 **********************************/

template <size_t N>
consteval ConstexprStr<N>::ConstexprStr(const char (&_str)[N]) {
	std::copy_n(_str, N, data_);
}

template <size_t N>
consteval bool ConstexprStr<N>::operator==(const ConstexprStr<N>& _str) const {
	return std::equal(_str.data_, _str.data_ + N, data_);
}

template <size_t N>
template <size_t N2>
consteval bool ConstexprStr<N>::operator==(const ConstexprStr<N2>& _str) const {
	return false;
}

template <size_t N>
consteval char ConstexprStr<N>::operator[](size_t _n) const {
	return data_[_n];
}

template <size_t N>
consteval size_t ConstexprStr<N>::size() const {
	return N - 1;
}

template <size_t N1, size_t N2>
consteval ConstexprStr<N1 + N2 - 1> operator+(const ConstexprStr<N1>& _a, const ConstexprStr<N2>& _b) {
	char newchar[N1 + N2 - 1]{};
	std::copy_n(_a.data_, N1 - 1, newchar);
	std::copy_n(_b.data_, N2, newchar + N1 - 1);
	return newchar;
}

template <size_t N1, size_t N2>
consteval ConstexprStr<N1 + N2 - 1> operator+(const char (&_a)[N1], const ConstexprStr<N2>& _b) {
	return ConstexprStr(_a) + _b;
}

template <size_t N1, size_t N2>
consteval ConstexprStr<N1 + N2 - 1> operator+(const ConstexprStr<N1>& _a, const char (&_b)[N2]) {
	return _a + ConstexprStr(_b);
}

/**********************************
 *                                *
 *           NamedTimer           *
 *                                *
 **********************************/

template <ConstexprStr TimerName>
void NamedTimer<TimerName>::start() {
	this->setUsed();
	this->begin_tp_.emplace(std::chrono::high_resolution_clock::now());
}

template <ConstexprStr TimerName>
void NamedTimer<TimerName>::stop() {
	auto end = std::chrono::high_resolution_clock::now();

	LABELED_ASSERT_MSG(this->begin_tp_.has_value(), std::string("NamedTimer<") + TimerName.data_ + ">",
	                   "The start() has not been invoked!");

	NamedTimer<TimerName>::timer_ +=
	    std::chrono::duration_cast<std::chrono::microseconds>(end - this->begin_tp_.value()).count();
	this->begin_tp_.reset();
}

template <ConstexprStr TimerName>
double NamedTimer<TimerName>::getTimerVal() {
	if (!NamedTimer<TimerName>::has_used_) {
		LABELED_WARNING("NamedTimer") << std::string("Timer \"") + TimerName.data_ + "\" is unused during simulation.";
	}

	return NamedTimer<TimerName>::timer_;
}

template <ConstexprStr TimerName>
void NamedTimer<TimerName>::setUsed() {
	if (!this->getUsed()) [[unlikely]] { NamedTimer<TimerName>::has_used_.store(true, std::memory_order_relaxed); }
}

template <ConstexprStr TimerName>
bool NamedTimer<TimerName>::getUsed() const {
	return NamedTimer<TimerName>::has_used_.load(std::memory_order_relaxed);
}

}  // namespace acalsim
