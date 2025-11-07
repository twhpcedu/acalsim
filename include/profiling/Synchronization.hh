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

#include <atomic>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

#include "profiling/Utils.hh"

namespace acalsim {

template <typename TMutex = std::mutex, ProfileMode PMode = ProfileMode::ACALSIM_STATISTICS_FLAG>
class ProfiledMutex {
public:
	inline constexpr ProfiledMutex() noexcept = default;
	ProfiledMutex(const ProfiledMutex&)       = delete;

	~ProfiledMutex() = default;

	ProfiledMutex& operator=(const ProfiledMutex&) = delete;

	inline void lock();
	inline void unlock();
	inline bool try_lock();

	inline void lock_shared();
	inline void unlock_shared();
	inline bool try_lock_shared();

	double getTimerVal() const { return this->timer_; }

private:
	TMutex mutex_;

	std::atomic<double> timer_ = 0.0;
};

/**
 * @brief A lock wrapper that profiles the time spent acquiring and releasing a lock.
 *
 * @tparam TimerName A compile-time constant string identifier for the timer.
 * @tparam TLock The type of lock to manage (e.g., std::lock_guard, std::unique_lock).
 * @tparam PMode Profiling mode: always active or conditional (based on compile-time flags).
 *
 * @details
 * ProfiledLock automatically measures and accumulates the time taken for lock
 * acquisition and release, optionally controlled by compile-time flags.
 * It supports manual locking/unlocking for locks that allow it (like unique_lock).
 *
 * The accumulated time can be retrieved using getTimerVal().
 */
template <ConstexprStr TimerName, typename TLock, ProfileMode PMode = ProfileMode::ACALSIM_STATISTICS_FLAG>
class ProfiledLock : public NamedTimer<TimerName> {
public:
	template <typename... Args>
	inline ProfiledLock(Args&&... _args);
	ProfiledLock(const ProfiledLock&) = delete;

	inline ~ProfiledLock();

	ProfiledLock& operator=(const ProfiledLock&) = delete;

	inline void lock();
	inline void unlock();

private:
	std::optional<TLock> lock_;
};

class ThreadIdleTimer {
public:
	inline ThreadIdleTimer(const size_t& _n_threads = 0);

	inline void resize(const size_t& _n_threads);
	inline void startStage() { this->stage_begin_tp_.emplace(std::chrono::high_resolution_clock::now()); }
	inline void endStage();
	inline void enterSyncPoint(const size_t& _tid, const bool& _has_tasks = true);

	inline double getSum() const;

private:
	std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>> stage_begin_tp_;
	std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>   thread_sleep_time_;
	std::vector<double>                                                        thread_timer_;
};

}  // namespace acalsim

#include "profiling/Synchronization.inl"
