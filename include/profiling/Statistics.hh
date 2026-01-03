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
#include <map>
#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace acalsim {

enum class StatisticsMode { Default, Accumulator, AccumulatorWithSize };

/**
 * @brief A class for computing basic statistical values.
 *
 * This template class stores a collection of values and provides methods to compute
 * their sum, average, and maximum. Optionally, thread-safety can be enabled using a mutex.
 *
 * @tparam TValue The type of the values stored in the statistic.
 * @tparam EnableMutex If true, enables thread-safety using a shared mutex.
 */
template <typename TValue, StatisticsMode Mode = StatisticsMode::Default, bool EnableMutex = false>
class Statistics {
public:
	/**
	 * @brief Default constructor.
	 */
	Statistics() = default;

	/**
	 * @brief Constructs a Statistic object and adds an initial value.
	 * @param _val The initial value to be added to the statistic.
	 */
	Statistics(const TValue& _val) { this->push(_val); }

	/**
	 * @brief Move assignment operator.
	 * @param _other Another Statistic object to move data from.
	 * @return Reference to the current object.
	 */
	Statistics& operator=(Statistics<TValue>&& _other);

	/**
	 * @brief Adds a new value to the statistic.
	 * @param _val The value to be added.
	 */
	void push(const TValue& _val);

	/**
	 * @brief Computes the sum of all stored values.
	 * @return The sum of all values.
	 */
	TValue sum() const;

	/**
	 * @brief Computes the average of all stored values.
	 * @return The average value.
	 */
	TValue avg() const;

	/**
	 * @brief Finds the maximum value stored.
	 * @return The maximum value in the container.
	 */
	TValue max() const;

	/**
	 * @brief Gets the number of stored values.
	 * @return The size of the container.
	 */
	size_t size() const;

private:
	std::vector<TValue>       container_;     ///< Container holding the values.
	mutable std::shared_mutex container_mu_;  ///< Mutex for thread safety if enabled.
};

template <typename TValue, bool EnableMutex>
class Statistics<TValue, StatisticsMode::Accumulator, EnableMutex> {
public:
	/**
	 * @brief Default constructor.
	 */
	Statistics() = default;

	/**
	 * @brief Constructs a Statistic object and adds an initial value.
	 * @param _val The initial value to be added to the statistic.
	 */
	Statistics(const TValue& _val) { this->push(_val); }

	/**
	 * @brief Move assignment operator.
	 * @param _other Another Statistic object to move data from.
	 * @return Reference to the current object.
	 */
	Statistics& operator=(Statistics<TValue>&& _other);

	/**
	 * @brief Adds a new value to the statistic.
	 * @param _val The value to be added.
	 */
	void push(const TValue& _val);

	/**
	 * @brief Computes the sum of all stored values.
	 * @return The sum of all values.
	 */
	TValue sum() const;

private:
	std::atomic<TValue> value_ = 0;
};

template <typename TValue, bool EnableMutex>
class Statistics<TValue, StatisticsMode::AccumulatorWithSize, EnableMutex> {
public:
	/**
	 * @brief Default constructor.
	 */
	Statistics() = default;

	/**
	 * @brief Constructs a Statistic object and adds an initial value.
	 * @param _val The initial value to be added to the statistic.
	 */
	Statistics(const TValue& _val) { this->push(_val); }

	/**
	 * @brief Move assignment operator.
	 * @param _other Another Statistic object to move data from.
	 * @return Reference to the current object.
	 */
	Statistics& operator=(Statistics<TValue>&& _other);

	/**
	 * @brief Adds a new value to the statistic.
	 * @param _val The value to be added.
	 */
	void push(const TValue& _val);

	/**
	 * @brief Computes the sum of all stored values.
	 * @return The sum of all values.
	 */
	TValue sum() const;

	/**
	 * @brief Computes the average of all stored values.
	 * @return The average value.
	 */
	TValue avg() const;

	/**
	 * @brief Gets the number of stored values.
	 * @return The size of the container.
	 */
	size_t size() const;

private:
	std::atomic<TValue> value_ = 0;
	std::atomic<size_t> size_  = 0;
};

/**
 * @brief A class for maintaining categorized statistics with optional sorting and thread safety.
 *
 * @tparam TCategory The type used for category keys.
 * @tparam TValue The type of values stored in the statistics.
 * @tparam Sorted If true, sorts categories using the specified comparison function; otherwise, uses an unordered map.
 * @tparam ThreadSafeMap If true, enables thread safety for map operations.
 * @tparam ThreadSafeStatistics If true, enables thread safety for statistic operations.
 * @tparam CategoryCompar The comparison function used for sorting categories if Sorted is true.
 */
template <typename TCategory, typename TValue, StatisticsMode Mode = StatisticsMode::Default, bool Sorted = false,
          bool ThreadSafeMap = false, bool ThreadSafeStatistics = false,
          class CategoryCompar = std::conditional_t<Sorted, std::less<TCategory>, void>>
class CategorizedStatistics {
	using MapType = std::conditional_t<
	    /* B */ Sorted,
	    /* T */ std::map<TCategory, std::shared_ptr<Statistics<TValue, Mode, ThreadSafeStatistics>>, CategoryCompar>,
	    /* F */ std::unordered_map<TCategory, std::shared_ptr<Statistics<TValue, Mode, ThreadSafeStatistics>>>>;

public:
	/**
	 * @brief Default constructor.
	 */
	CategorizedStatistics() = default;

	/**
	 * @brief Constructs a CategorizedStatistic with an initial category and value.
	 * @param _cat The category key.
	 * @param _val The initial value to be added to the category.
	 */
	CategorizedStatistics(const TCategory& _cat, const TValue& _val) { this->getEntry(_cat)->push(_val); }

	/**
	 * @brief Adds a new category entry if it does not already exist.
	 * @param _cat The category key to be added.
	 */
	void addEntry(const TCategory& _cat);

	/**
	 * @brief Get the pointer of a category entry. The entry will be constructed if it does not exist.
	 * @param _cat The category key to be retrieved.
	 */
	std::shared_ptr<Statistics<TValue, Mode, ThreadSafeStatistics>> getEntry(const TCategory& _cat);

	/**
	 * @brief Get the summation of all values across all category entries.
	 */
	TValue sum() const;

	/**
	 * @brief Computes the sum distribution across all categories.
	 * @return A map containing each category and its corresponding sum.
	 */
	std::conditional_t<Sorted, std::map<TCategory, TValue, CategoryCompar>, std::unordered_map<TCategory, TValue>>
	sumDistribution() const;

	/**
	 * @brief Computes the size distribution across all categories.
	 * @return A map containing each category and its corresponding proportion based on size.
	 */
	std::conditional_t<Sorted, std::map<TCategory, double, CategoryCompar>, std::unordered_map<TCategory, double>>
	sizeDistribution() const;

	/**
	 * @brief Gets an iterator to the beginning of the map.
	 * @return An iterator pointing to the first category.
	 */
	typename MapType::iterator begin();

	/**
	 * @brief Gets a constant iterator to the beginning of the map.
	 * @return A constant iterator pointing to the first category.
	 */
	typename MapType::const_iterator begin() const;

	/**
	 * @brief Gets an iterator to the end of the map.
	 * @return An iterator pointing to the end of the map.
	 */
	typename MapType::iterator end();

	/**
	 * @brief Gets a constant iterator to the end of the map.
	 * @return A constant iterator pointing to the end of the map.
	 */
	typename MapType::const_iterator end() const;

private:
	MapType                   map_container_;     ///< Container holding categorized statistics.
	mutable std::shared_mutex map_container_mu_;  ///< Mutex for thread safety if enabled.
};

}  // namespace acalsim

#include "profiling/Statistics.inl"
