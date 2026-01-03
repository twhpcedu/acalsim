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

#include <atomic>
#include <chrono>
#include <optional>

#ifdef ACALSIM_STATISTICS
/**
 * @brief Measures the execution time of a code block in a specified time precision.
 *
 * This macro measures the time taken by the given code block (`_code_block`) using high-resolution
 * clocks and computes the duration in the specified time precision (`_precision`). It automatically
 * generates unique variable names based on `_var_name` to store the start time and the measured
 * duration. This macro can be used to measure time in any `std::chrono` duration type, such as
 * `std::chrono::milliseconds`, `std::chrono::microseconds`, etc.
 *
 * @param _var_name A unique name for the timer variables. It will be used to create unique names
 *                  for the timer start time and the resulting duration variable (e.g., `_var_name##_start`
 *                  and `_var_name##_lat`).
 * @param _code_block A block of code whose execution time is to be measured.
 * @param _precision The precision for the time measurement. Typically, this would be a type from the
 *                   `std::chrono` library such as `std::chrono::milliseconds` or `std::chrono::microseconds`.
 *
 * @note This macro uses `std::chrono::high_resolution_clock` for precise time measurement.
 *
 * Example usage:
 * @code
 * MEASURE_TIME(task1, {
 *     // Code block to measure time for
 *     for (int i = 0; i < 1000000; ++i);
 * }, std::chrono::microseconds);
 * std::cout << "Task time in microseconds: " << task1_lat << std::endl;
 * @endcode
 */
#define MEASURE_TIME(_var_name, _code_block, _precision)                                                         \
	auto _##_var_name##_start = std::chrono::high_resolution_clock::now();                                       \
	_code_block;                                                                                                 \
	double _var_name##_lat =                                                                                     \
	    std::chrono::duration_cast<_precision>(std::chrono::high_resolution_clock::now() - _##_var_name##_start) \
	        .count();

/**
 * @brief Measures the execution time of a code block in milliseconds.
 *
 * This macro measures the time taken by the given code block (`_code_block`) using high-resolution
 * clocks and computes the duration in milliseconds. It automatically generates unique variable names
 * based on `_var_name` to store the start time and the measured duration. This macro is a convenience
 * function specifically for measuring time in milliseconds, without requiring the user to specify the
 * precision.
 *
 * @param _var_name A unique name for the timer variables. It will be used to create unique names
 *                  for the timer start time and the resulting duration variable (e.g., `_var_name##_start`
 *                  and `_var_name##_lat`).
 * @param _code_block A block of code whose execution time is to be measured.
 *
 * @note This macro uses `std::chrono::high_resolution_clock` for precise time measurement and assumes
 *       that the time is measured in milliseconds.
 *
 * Example usage:
 * @code
 * MEASURE_TIME_MILLISECONDS(task1, {
 *     // Code block to measure time for
 *     for (int i = 0; i < 1000000; ++i);
 * });
 * std::cout << "Task time in milliseconds: " << task1_lat << std::endl;
 * @endcode
 */
#define MEASURE_TIME_MILLISECONDS(_var_name, _code_block)                                          \
	auto _##_var_name##_start = std::chrono::high_resolution_clock::now();                         \
	_code_block;                                                                                   \
	double _var_name##_lat = std::chrono::duration_cast<std::chrono::milliseconds>(                \
	                             std::chrono::high_resolution_clock::now() - _##_var_name##_start) \
	                             .count();

/**
 * @brief Measures the execution time of a code block in microseconds.
 *
 * This macro measures the time taken by the given code block (`_code_block`) using high-resolution
 * clocks and computes the duration in microseconds. It automatically generates unique variable names
 * based on `_var_name` to store the start time and the measured duration. This macro is a convenience
 * function specifically for measuring time in microseconds, without requiring the user to specify the
 * precision.
 *
 * @param _var_name A unique name for the timer variables. It will be used to create unique names
 *                  for the timer start time and the resulting duration variable (e.g., `_var_name##_start`
 *                  and `_var_name##_lat`).
 * @param _code_block A block of code whose execution time is to be measured.
 *
 * @note This macro uses `std::chrono::high_resolution_clock` for precise time measurement and assumes
 *       that the time is measured in microseconds.
 *
 * Example usage:
 * @code
 * MEASURE_TIME_MILLISECONDS(task1, {
 *     // Code block to measure time for
 *     for (int i = 0; i < 1000000; ++i);
 * });
 * std::cout << "Task time in microseconds: " << task1_lat << std::endl;
 * @endcode
 */
#define MEASURE_TIME_MICROSECONDS(_var_name, _code_block)                                          \
	auto _##_var_name##_start = std::chrono::high_resolution_clock::now();                         \
	_code_block;                                                                                   \
	double _var_name##_lat = std::chrono::duration_cast<std::chrono::microseconds>(                \
	                             std::chrono::high_resolution_clock::now() - _##_var_name##_start) \
	                             .count();
#else  // ACALSIM_STATISTICS
#define MEASURE_TIME(_var_name, _code_block, _precision)  _code_block;
#define MEASURE_TIME_MILLISECONDS(_var_name, _code_block) _code_block;
#define MEASURE_TIME_MICROSECONDS(_var_name, _code_block) _code_block;
#endif  // ACALSIM_STATISTICS

namespace acalsim {

enum class ProfileMode { ALWAYS, ACALSIM_STATISTICS_FLAG };

/**
 * @brief A compile-time string wrapper for use as a non-type template parameter.
 *
 * This implementation is based on the approach described in the article:
 * "Strings as Template Parameters (C++20)" by sgf4.
 *
 * @see https://dev.to/sgf4/strings-as-template-parameters-c20-4joh
 */
template <size_t N>
class ConstexprStr {
public:
	consteval ConstexprStr(const char (&_str)[N]);

	consteval bool operator==(const ConstexprStr<N>& _str) const;

	template <size_t N2>
	consteval bool operator==(const ConstexprStr<N2>& _str) const;

	consteval char operator[](size_t _n) const;

	consteval size_t size() const;

	template <size_t N1, size_t N2>
	friend consteval ConstexprStr<N1 + N2 - 1> operator+(const ConstexprStr<N1>& _a, const ConstexprStr<N2>& _b);

	template <size_t N1, size_t N2>
	friend consteval ConstexprStr<N1 + N2 - 1> operator+(const char (&_a)[N1], const ConstexprStr<N2>& _b);

	template <size_t N1, size_t N2>
	friend consteval ConstexprStr<N1 + N2 - 1> operator+(const ConstexprStr<N1>& _a, const char (&_b)[N2]);

public:
	char data_[N]{};
};

/**
 * @brief A compile-time named timer utility for profiling cumulative durations.
 *
 * This class provides a static timer associated with a compile-time constant string identifier (`TimerName`).
 * It is primarily used as a base class for profiling scenarios (e.g., lock acquisition or other timed operations),
 * allowing accumulation of timing data across all instances sharing the same tag.
 *
 * @tparam TimerName A `ConstexprStr` used to uniquely identify and label the timer instance.
 *
 * Usage:
 * - Call `getTimerRef()` to accumulate durations.
 * - Use `setUsed()` in derived classes to mark the timer as actively used.
 * - `getTimerVal()` retrieves the accumulated time in microseconds and emits a warning if unused.
 */
template <ConstexprStr TimerName>
class NamedTimer {
public:
	static double getTimerVal();

	inline void start();
	inline void stop();

protected:
	inline void setUsed();
	inline bool getUsed() const;

private:
	std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>> begin_tp_;

	inline static std::atomic<double> timer_    = 0.0;
	inline static std::atomic<bool>   has_used_ = false;
};

}  // namespace acalsim

#include "profiling/Utils.inl"
