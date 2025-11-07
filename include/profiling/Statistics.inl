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

#include <memory>
#include <mutex>
#include <numeric>

#include "profiling/Statistics.hh"

namespace acalsim {

/**************************
 *                        *
 *       Statistics       *
 *                        *
 **************************/

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
Statistics<TValue, Mode, EnableMutex>& Statistics<TValue, Mode, EnableMutex>::operator=(Statistics<TValue>&& _other) {
	if (this != &_other) {
		// Do not enclose the lock in brackets to keep its lifetime until exiting the method
		if constexpr (EnableMutex) std::unique_lock<std::shared_mutex> lock(this->container_mu_);
		if constexpr (EnableMutex) std::unique_lock<std::shared_mutex> lock(_other.container_mu_);

		this->container_ = std::move(_other.container_);
	}
	return *this;
}

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
void Statistics<TValue, Mode, EnableMutex>::push(const TValue& _val) {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (EnableMutex) std::unique_lock<std::shared_mutex> lock(this->container_mu_);
	this->container_.push_back(_val);
}

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
TValue Statistics<TValue, Mode, EnableMutex>::sum() const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (EnableMutex) std::shared_lock<std::shared_mutex> lock(this->container_mu_);
	return std::accumulate(this->container_.begin(), this->container_.end(), 0);
}

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
TValue Statistics<TValue, Mode, EnableMutex>::avg() const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (EnableMutex) std::shared_lock<std::shared_mutex> lock(this->container_mu_);
	return (this->container_.size() != 0) ? this->sum() / this->container_.size() : 0;
}

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
TValue Statistics<TValue, Mode, EnableMutex>::max() const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (EnableMutex) std::shared_lock<std::shared_mutex> lock(this->container_mu_);
	return (this->container_.size() > 0) ? *std::max_element(this->container_.begin(), this->container_.end()) : 0;
}

template <typename TValue, StatisticsMode Mode, bool EnableMutex>
size_t Statistics<TValue, Mode, EnableMutex>::size() const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (EnableMutex) std::shared_lock<std::shared_mutex> lock(this->container_mu_);
	return this->container_.size();
}

/**********************************
 *                                *
 *    Statistics - Accumulator    *
 *                                *
 **********************************/

template <typename TValue, bool EnableMutex>
Statistics<TValue, StatisticsMode::Accumulator, EnableMutex>&
Statistics<TValue, StatisticsMode::Accumulator, EnableMutex>::operator=(Statistics<TValue>&& _other) {
	if (this != &_other) { this->value_ = std::move(_other.value_); }
	return *this;
}

template <typename TValue, bool EnableMutex>
void Statistics<TValue, StatisticsMode::Accumulator, EnableMutex>::push(const TValue& _val) {
	this->value_ += _val;
}

template <typename TValue, bool EnableMutex>
TValue Statistics<TValue, StatisticsMode::Accumulator, EnableMutex>::sum() const {
	return this->value_;
}

/******************************************
 *                                        *
 *    Statistics - AccumulatorWithSize    *
 *                                        *
 ******************************************/

template <typename TValue, bool EnableMutex>
Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>&
Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>::operator=(Statistics<TValue>&& _other) {
	if (this != &_other) {
		this->value_ = std::move(_other.value_);
		this->size_  = std::move(_other.size_);
	}
	return *this;
}

template <typename TValue, bool EnableMutex>
void Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>::push(const TValue& _val) {
	this->value_ += _val;
	this->size_ += 1;
}

template <typename TValue, bool EnableMutex>
TValue Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>::sum() const {
	return this->value_;
}

template <typename TValue, bool EnableMutex>
TValue Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>::avg() const {
	return this->value_ / this->size_;
}

template <typename TValue, bool EnableMutex>
size_t Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex>::size() const {
	return this->size_;
}

/**********************************
 *                                *
 *     CategorizedStatistics      *
 *                                *
 **********************************/

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
void CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                           CategoryCompar>::addEntry(const TCategory& _cat) {
	if (!this->map_container_.contains(_cat)) {
		// Do not enclose the lock in brackets to keep its lifetime until exiting the method
		if constexpr (ThreadSafeMap) std::unique_lock<std::shared_mutex> lock(this->map_container_mu_);
		if (!this->map_container_.contains(_cat))
			this->map_container_[_cat] = std::make_shared<Statistics<TValue, Mode, ThreadSafeStatistics>>();
	}
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
std::shared_ptr<Statistics<TValue, Mode, ThreadSafeStatistics>>
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics, CategoryCompar>::getEntry(
    const TCategory& _cat) {
	if (!this->map_container_.contains(_cat)) [[unlikely]] { this->addEntry(_cat); }

	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	return this->map_container_[_cat];
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
TValue CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                             CategoryCompar>::sum() const {
	TValue sum = 0;

	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	for (auto& [cat, val] : this->map_container_) { sum += val->sum(); }

	return sum;
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
std::conditional_t<Sorted, std::map<TCategory, TValue, CategoryCompar>, std::unordered_map<TCategory, TValue>>
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                      CategoryCompar>::sumDistribution() const {
	std::map<TCategory, TValue, CategoryCompar> distribution_map;

	{
		// Do not enclose the lock in brackets to keep its lifetime until exiting the method
		if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
		for (auto& [cat, statistic] : this->map_container_) { distribution_map[cat] = statistic->sum(); }
	}

	TValue sum = std::accumulate(distribution_map.begin(), distribution_map.end(), 0,
	                             [](const TValue& _prev, const auto& _elem) { return _prev + _elem.second; });

	if (sum != 0) {
		for (auto& [cat, val] : distribution_map) { val = val / sum; }
	}

	return distribution_map;
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
std::conditional_t<Sorted, std::map<TCategory, double, CategoryCompar>, std::unordered_map<TCategory, double>>
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                      CategoryCompar>::sizeDistribution() const {
	std::map<TCategory, double, CategoryCompar> distribution_map;

	{
		// Do not enclose the lock in brackets to keep its lifetime until exiting the method
		if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
		for (auto& [cat, statistic] : this->map_container_) { distribution_map[cat] = (double)statistic->size(); }
	}

	double sum = std::accumulate(distribution_map.begin(), distribution_map.end(), 0,
	                             [](const TValue& _prev, const auto& _elem) { return _prev + _elem.second; });

	if (sum != 0) {
		for (auto& [cat, val] : distribution_map) { val = val / sum; }
	}

	return distribution_map;
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
typename CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                               CategoryCompar>::MapType::iterator
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics, CategoryCompar>::begin() {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	return this->map_container_.begin();
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
typename CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                               CategoryCompar>::MapType::const_iterator
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics, CategoryCompar>::begin()
    const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	return this->map_container_.begin();
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
typename CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                               CategoryCompar>::MapType::iterator
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics, CategoryCompar>::end() {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	return this->map_container_.end();
}

template <typename TCategory, typename TValue, StatisticsMode Mode, bool Sorted, bool ThreadSafeMap,
          bool ThreadSafeStatistics, class CategoryCompar>
typename CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics,
                               CategoryCompar>::MapType::const_iterator
CategorizedStatistics<TCategory, TValue, Mode, Sorted, ThreadSafeMap, ThreadSafeStatistics, CategoryCompar>::end()
    const {
	// Do not enclose the lock in brackets to keep its lifetime until exiting the method
	if constexpr (ThreadSafeMap) std::shared_lock<std::shared_mutex> lock(this->map_container_mu_);
	return this->map_container_.end();
}

}  // namespace acalsim
