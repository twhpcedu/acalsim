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

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "profiling/Utils.hh"
#include "utils/Logging.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename T>
class SharedContainer {
public:
	SharedContainer() = default;

	// Thread-safe add method
	template <typename... Args>
	void add(Args&&... args) {
		std::unique_lock<std::shared_mutex> lock(container_mutex);
		var.push_back(std::make_shared<T>(std::forward<Args>(args)...));
	}

	// Thread-safe get method with shared access
	std::shared_ptr<T> get(int which) {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::shared_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

#ifndef ACALSIM_STATISTICS
		return var[which];
#else
		auto target = var[which];
		MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
		                          /* code_block */ lock.unlock());
		this->lock_cost_statistics.push(lock_lat);
		this->lock_cost_statistics.push(unlock_lat);
		return target;
#endif  // ACALSIM_STATISTICS
	}

	// Thread-safe run method
	template <typename Func, typename... Args>
	auto run(int which, Func&& func, Args&&... args) -> std::invoke_result_t<Func, T, Args...> {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::unique_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

#ifndef ACALSIM_STATISTICS
		return std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
#else
		if constexpr (std::is_void_v<std::invoke_result_t<Func, T, Args...>>) {
			std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
		} else {
			auto result = std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
			return result;
		}
#endif  // ACALSIM_STATISTICS
	}

	// Const overload of run
	template <typename Func, typename... Args>
	auto run(int which, Func&& func, Args&&... args) const -> std::invoke_result_t<Func, T, Args...> {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::unique_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

#ifndef ACALSIM_STATISTICS
		return std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
#else
		if constexpr (std::is_void_v<std::invoke_result_t<Func, T, Args...>>) {
			std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
		} else {
			auto result = std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
			return result;
		}
#endif  // ACALSIM_STATISTICS
	}

	// Thread-safe atomic multi-operation method
	template <typename Func1, typename Func2, typename... Args1, typename... Args2>
	auto atomic_run(int which, Func1&& func1, Args1&&... args1, Func2&& func2, Args2&&... args2)
	    -> std::pair<std::invoke_result_t<Func1, T, Args1...>, std::invoke_result_t<Func2, T, Args2...>> {
		// Use unique_lock to ensure exclusive access during both operations
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::unique_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

		auto& obj = *var[which];

		// Perform both operations in sequence
		auto result1 = std::invoke(std::forward<Func1>(func1), obj, std::forward<Args1>(args1)...);
		auto result2 = std::invoke(std::forward<Func2>(func2), obj, std::forward<Args2>(args2)...);

#ifdef ACALSIM_STATISTICS
		MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
		                          /* code_block */ lock.unlock());
		this->lock_cost_statistics.push(lock_lat);
		this->lock_cost_statistics.push(unlock_lat);
#endif  // ACALSIM_STATISTICS

		return {result1, result2};
	}

	// Const overload of atomic_run
	template <typename Func1, typename Func2, typename... Args1, typename... Args2>
	auto atomic_run(int which, Func1&& func1, Args1&&... args1, Func2&& func2, Args2&&... args2) const
	    -> std::pair<std::invoke_result_t<Func1, const T, Args1...>, std::invoke_result_t<Func2, const T, Args2...>> {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::unique_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

		const auto& obj = *var[which];

		auto result1 = std::invoke(std::forward<Func1>(func1), obj, std::forward<Args1>(args1)...);
		auto result2 = std::invoke(std::forward<Func2>(func2), obj, std::forward<Args2>(args2)...);

#ifdef ACALSIM_STATISTICS
		MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
		                          /* code_block */ lock.unlock());
		this->lock_cost_statistics.push(lock_lat);
		this->lock_cost_statistics.push(unlock_lat);
#endif  // ACALSIM_STATISTICS

		return {result1, result2};
	}

	// Thread-safe run method
	template <typename Func, typename... Args>
	auto shared_run(int which, Func&& func, Args&&... args) -> std::invoke_result_t<Func, T, Args...> {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::shared_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

#ifndef ACALSIM_STATISTICS
		return std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
#else
		if constexpr (std::is_void_v<std::invoke_result_t<Func, T, Args...>>) {
			std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
		} else {
			auto result = std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
			return result;
		}
#endif  // ACALSIM_STATISTICS
	}

	// Const overload of run
	template <typename Func, typename... Args>
	auto shared_run(int which, Func&& func, Args&&... args) const -> std::invoke_result_t<Func, T, Args...> {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::shared_lock<std::shared_mutex> lock(container_mutex));

		LABELED_ASSERT_MSG(0 <= which && which < var.size(), "SharedContainer", "Invalid container index");

#ifndef ACALSIM_STATISTICS
		return std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
#else
		if constexpr (std::is_void_v<std::invoke_result_t<Func, T, Args...>>) {
			std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
		} else {
			auto result = std::invoke(std::forward<Func>(func), *var[which], std::forward<Args>(args)...);
			MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
			                          /* code_block */ lock.unlock());
			this->lock_cost_statistics.push(lock_lat);
			this->lock_cost_statistics.push(unlock_lat);
			return result;
		}
#endif  // ACALSIM_STATISTICS
	}

	// Thread-safe size method
	int size() const {
		MEASURE_TIME_MICROSECONDS(/* var_name */ lock,
		                          /* code_block */ std::shared_lock<std::shared_mutex> lock(container_mutex));

#ifndef ACALSIM_STATISTICS
		return var.size();
#else
		auto result = var.size();
		MEASURE_TIME_MICROSECONDS(/* var_name */ unlock,
		                          /* code_block */ lock.unlock());
		this->lock_cost_statistics.push(lock_lat);
		this->lock_cost_statistics.push(unlock_lat);
		return result;
#endif  // ACALSIM_STATISTICS
	}

private:
	mutable std::shared_mutex       container_mutex;
	std::vector<std::shared_ptr<T>> var;

#ifdef ACALSIM_STATISTICS
public:
	double getLockCost() const { return this->lock_cost_statistics.sum(); }

private:
	mutable Statistics<double, acalsim::StatisticsMode::Accumulator, true> lock_cost_statistics;
#endif  // ACALSIM_STATISTICS
};

}  //  namespace acalsim
