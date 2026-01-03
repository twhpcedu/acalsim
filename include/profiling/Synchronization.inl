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

#include <numeric>
#include <vector>

#include "profiling/Synchronization.hh"

namespace acalsim {

/**********************************
 *                                *
 *         ProfiledMutex          *
 *                                *
 **********************************/

template <typename TMutex, ProfileMode PMode>
void ProfiledMutex<TMutex, PMode>::lock() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		this->mutex_.lock();
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		this->mutex_.lock();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <typename TMutex, ProfileMode PMode>
void ProfiledMutex<TMutex, PMode>::unlock() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		this->mutex_.unlock();
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		this->mutex_.unlock();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <typename TMutex, ProfileMode PMode>
bool ProfiledMutex<TMutex, PMode>::try_lock() {
	bool result;

	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		result     = this->mutex_.try_lock();
		auto end   = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		result = this->mutex_.try_lock();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}

	return result;
}

template <typename TMutex, ProfileMode PMode>
void ProfiledMutex<TMutex, PMode>::lock_shared() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		this->mutex_.lock_shared();
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		this->mutex_.lock_shared();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <typename TMutex, ProfileMode PMode>
void ProfiledMutex<TMutex, PMode>::unlock_shared() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		this->mutex_.unlock_shared();
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		this->mutex_.unlock_shared();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <typename TMutex, ProfileMode PMode>
bool ProfiledMutex<TMutex, PMode>::try_lock_shared() {
	bool result;

	if constexpr (PMode == ProfileMode::ALWAYS) {
		auto start = std::chrono::high_resolution_clock::now();
		result     = this->mutex_.try_lock_shared();
		auto end   = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		auto start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		result = this->mutex_.try_lock_shared();

#ifdef ACALSIM_STATISTICS
		auto end = std::chrono::high_resolution_clock::now();
		ProfiledMutex<TMutex, PMode>::timer_ +=
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}

	return result;
}

/**********************************
 *                                *
 *          ProfiledLock          *
 *                                *
 **********************************/

template <ConstexprStr TimerName, typename TLock, ProfileMode PMode>
template <typename... Args>
ProfiledLock<TimerName, TLock, PMode>::ProfiledLock(Args&&... _args) {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		this->NamedTimer<TimerName>::start();
		this->lock_.emplace(std::forward<Args>(_args)...);
		this->NamedTimer<TimerName>::stop();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::start();
#endif  // ACALSIM_STATISTICS

		this->lock_.emplace(std::forward<Args>(_args)...);

#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::stop();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <ConstexprStr TimerName, typename TLock, ProfileMode PMode>
ProfiledLock<TimerName, TLock, PMode>::~ProfiledLock() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		this->NamedTimer<TimerName>::start();
		this->lock_.reset();
		this->NamedTimer<TimerName>::stop();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::start();
#endif  // ACALSIM_STATISTICS

		this->lock_.reset();

#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::stop();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <ConstexprStr TimerName, typename TLock, ProfileMode PMode>
void ProfiledLock<TimerName, TLock, PMode>::lock() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		this->NamedTimer<TimerName>::start();
		this->lock_->lock();
		this->NamedTimer<TimerName>::stop();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::start();
#endif  // ACALSIM_STATISTICS

		this->lock_->lock();

#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::stop();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

template <ConstexprStr TimerName, typename TLock, ProfileMode PMode>
void ProfiledLock<TimerName, TLock, PMode>::unlock() {
	if constexpr (PMode == ProfileMode::ALWAYS) {
		this->NamedTimer<TimerName>::start();
		this->lock_->unlock();
		this->NamedTimer<TimerName>::stop();
	} else if constexpr (PMode == ProfileMode::ACALSIM_STATISTICS_FLAG) {
#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::start();
#endif  // ACALSIM_STATISTICS

		this->lock_->unlock();

#ifdef ACALSIM_STATISTICS
		this->NamedTimer<TimerName>::stop();
#endif  // ACALSIM_STATISTICS
	} else {
		static_assert(PMode == ProfileMode::ALWAYS || PMode == ProfileMode::ACALSIM_STATISTICS_FLAG);
	}
}

ThreadIdleTimer::ThreadIdleTimer(const size_t& _n_threads)
    : thread_sleep_time_(_n_threads, std::chrono::high_resolution_clock::now()), thread_timer_(_n_threads, 0.0) {}

void ThreadIdleTimer::resize(const size_t& _n_threads) {
	this->thread_sleep_time_ = std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>(
	    _n_threads, std::chrono::high_resolution_clock::now());
	this->thread_timer_ = std::vector<double>(_n_threads, 0.0);
}

void ThreadIdleTimer::endStage() {
	auto time = std::chrono::high_resolution_clock::now();

	LABELED_ASSERT(this->stage_begin_tp_.has_value(), "ThreadIdleTimer");

	for (size_t tid = 0; tid < this->thread_timer_.size(); ++tid) {
		// Calculate the idle time of each thread in current iteration
		auto& begin_tp = (this->thread_sleep_time_[tid] > this->stage_begin_tp_.value())
		                     ? this->thread_sleep_time_[tid]
		                     : this->stage_begin_tp_.value();
		this->thread_timer_[tid] += (double)(time - begin_tp).count() / pow(10, 3);
	}

	this->stage_begin_tp_.reset();
}

void ThreadIdleTimer::enterSyncPoint(const size_t& _tid, const bool& _has_tasks) {
	if (_has_tasks) { this->thread_sleep_time_[_tid] = std::chrono::high_resolution_clock::now(); }
}

double ThreadIdleTimer::getSum() const {
	return std::accumulate(this->thread_timer_.begin(), this->thread_timer_.end(), 0.0);
}

}  // namespace acalsim
