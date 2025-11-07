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

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "container/RecycleContainer/ObjectPool.hh"
#include "container/RecycleContainer/RecyclableObject.hh"

namespace acalsim {

/**
 * @class RecycleContainer
 * @brief Manages object recycling pools for different types.
 *
 * @details This class provides a mechanism for recycling objects of various types to minimize
 * the overhead of frequent allocations and deallocations. It maintains a pool for each
 * type of object and allows objects to be acquired and released back to their respective pools.
 */
class RecycleContainer {
	class Deleter {
	public:
		Deleter(RecycleContainer* _cntr, std::shared_ptr<bool> _hasCntrDestructed)
		    : cntr(_cntr), hasCntrDestructed(_hasCntrDestructed) {}
		void operator()(RecyclableObject* _obj) const noexcept;

	private:
		RecycleContainer*     cntr              = nullptr;
		std::shared_ptr<bool> hasCntrDestructed = nullptr;
	};

public:
	/**
	 * @brief Constructs a RecycleContainer with an initial pool size.
	 *
	 * @param _init_size The initial number of objects to allocate in each pool.
	 */
	RecycleContainer(size_t _defaultInitSegments  = std::thread::hardware_concurrency(),
	                 size_t _defaultSegmentLength = 128);

	/**
	 * @brief Destroys the RecycleContainer, cleaning up all object pools.
	 */
	~RecycleContainer();

	/**
	 * @brief Acquires an object from the object pool without reinitialization. It is the caller's
	 * responsibility to setup the object for further use.
	 *
	 * @tparam T The type of object to acquire.
	 * @return A pointer to the acquired object.
	 */
	template <typename T>
	inline T* acquire() {
		if (!this->isObjectPoolExist<T>()) { this->createObjectPool<T>(); }

		T* ptr = static_cast<T*>(this->getObjectPool<T>()->pop());

#ifndef NDEBUG
		static_cast<RecyclableObject*>(ptr)->is_recycled_ = false;
#endif  // NDEBUG

		return ptr;
	}

	/**
	 * @brief Acquires an object from the object pool without reinitialization. It is the caller's
	 * responsibility to setup the object for further use.
	 *
	 * @tparam T The type of object to acquire.
	 * @return A shared pointer to the acquired object.
	 */
	template <typename T>
	inline std::shared_ptr<T> acquireSharedPtr() {
		std::shared_ptr<T> ptr(
		    /* ptr */ this->acquire<T>(),
		    /* deleter */ this->deleter);
		return ptr;
	}

	/**
	 * @brief Acquires an object from the object pool and resets it using the given method.
	 *
	 * @details This method picks up an object from the object pool, invokes the specified reset
	 * method on the object with the provided arguments, and returns the reset object.
	 *
	 * @tparam T The type of object to acquire.
	 * @tparam RType The return type of the reset method.
	 * @tparam Args The types of arguments passed to the reset method.
	 *
	 * @param _func Pointer to the member function of the object to reset it.
	 * @param _args Arguments to be forwarded to the reset method.
	 *
	 * @return A pointer to the acquired object.
	 */
	template <typename T, typename RType, typename... Args1, typename... Args2>
	inline T* acquire(RType (T::*_func)(Args1...), Args2&&... _args) {
		T* ptr = this->acquire<T>();
		((*ptr).*_func)(std::forward<Args2>(_args)...);
		return ptr;
	}

	/**
	 * @brief Acquires an object from the object pool and resets it using the given method.
	 *
	 * @details This method picks up an object from the object pool, invokes the specified reset
	 * method on the object with the provided arguments, and returns the reset object.
	 *
	 * @tparam T The type of object to acquire.
	 * @tparam RType The return type of the reset method.
	 * @tparam Args The types of arguments passed to the reset method.
	 *
	 * @param _func Pointer to the member function of the object to reset it.
	 * @param _args Arguments to be forwarded to the reset method.
	 *
	 * @return A pointer to the acquired object.
	 */
	template <typename T, typename RType, typename... Args1, typename... Args2>
	inline std::shared_ptr<T> acquireSharedPtr(RType (T::*_func)(Args1...), Args2&&... _args) {
		std::shared_ptr<T> ptr(
		    /* ptr */ this->acquire<T>(),
		    /* deleter */ this->deleter);
		((*ptr).*_func)(std::forward<Args2>(_args)...);
		return ptr;
	}

	/**
	 * @brief Recycles a given object for reuse.
	 *
	 * This method allows users to recycle objects, enabling them to be reused
	 * instead of discarded. It takes a pointer to a `RecyclableObject` and
	 * returns it to an object pool.
	 *
	 * @param _ptr A pointer to the `RecyclableObject` that should be recycled.
	 */
	void recycle(RecyclableObject* _ptr) {
#ifndef NDEBUG
		LABELED_ASSERT_MSG(_ptr, "RecycleContainer", "The object has been released.");
		LABELED_ASSERT_MSG(!_ptr->is_recycled_, "RecycleContainer",
		                   "The " << _ptr->getTypeName() << " is recycled twice.");
		_ptr->is_recycled_ = true;
#endif  // NDEBUG

		_ptr->preRecycle();

		auto it = this->objectPools.find(_ptr->getTypeHash());
		if (it != this->objectPools.end()) {
			(it->second)->push(_ptr);
		} else {
			delete _ptr;
		}
	}

	/**
	 * @brief Sets the initial size of the object pool for the specified type.
	 *
	 * @tparam T The type of objects for which to set the initial pool size.
	 * @param _n The initial number of objects to allocate in the pool.
	 */
	template <typename T>
	inline void setInitSize(size_t _initSegments, size_t _segmentLength) {
		this->createObjectPool<T>(_initSegments, _segmentLength);
	}

protected:
	/**
	 * @brief Creates a new object pool for the specified type if it does not already exist.
	 *
	 * @details This method creates a new object pool for the specified type if it does not already exist.
	 * It is used internally to ensure that there is a object pool available for a given type
	 * before attempting to allocate or release objects of that type.
	 *
	 * @param _type The type index of the objects for which a object pool is to be created.
	 */
	template <typename T>
	inline void createObjectPool() {
		static const size_t hash_code = typeid(T).hash_code();

		std::lock_guard<std::mutex> lock(this->poolMutex);
		if (this->isObjectPoolExist<T>()) { return; }

		this->objectPools[hash_code] = new ObjectPool<T>(this->defaultInitSegments, this->defaultSegmentLength);
	}

	/**
	 * @brief Creates a new object pool for the specified type with a specified initial size.
	 *
	 * @details This method creates a new object pool for the specified type with a specified initial size.
	 * It is used internally to ensure that there is a object pool available for a given type
	 * before attempting to allocate or release objects of that type.
	 *
	 * @tparam T The type of objects for which to create a object pool.
	 * @param _init_size The initial number of objects to allocate in the pool.
	 */
	template <typename T>
	inline void createObjectPool(size_t _initSegments, size_t _segmentLength) {
		static const size_t hash_code = typeid(T).hash_code();

		std::lock_guard<std::mutex> lock(this->poolMutex);
		if (this->isObjectPoolExist<T>()) { return; }

		this->objectPools[hash_code] = new ObjectPool<T>(_initSegments, _segmentLength);
	}

	/**
	 * @brief Gets the object pool for the specified type.
	 *
	 * @tparam T The type of objects for which to get the object pool.
	 * @return A pointer to the object pool for the specified type.
	 */
	template <typename T>
	inline ObjectPool<T>* getObjectPool() const {
		static const size_t hash_code = typeid(T).hash_code();
		return static_cast<ObjectPool<T>*>(this->objectPools.at(hash_code));
	}

	/**
	 * @brief Checks if an object pool exists for the specified type.
	 *
	 * @tparam T The type of objects for which to check the existence of a object pool.
	 * @return True if the object pool exists, false otherwise.
	 */
	template <typename T>
	inline bool isObjectPoolExist() const {
		static const size_t hash_code = typeid(T).hash_code();
		return this->objectPools.contains(hash_code);
	}

private:
	size_t                defaultInitSegments  = 32;
	size_t                defaultSegmentLength = 128;
	std::shared_ptr<bool> hasDestructed        = std::make_shared<bool>(false);
	Deleter               deleter              = Deleter(this, this->hasDestructed);

	std::mutex                        poolMutex;    // Mutex for accessing the pool
	std::map<size_t, ObjectPoolBase*> objectPools;  // Map of type index to ObjectPool<T> pointers

#ifdef ACALSIM_STATISTICS
public:
	/**
	 * @brief Returns the total count of objects of type TDerived created by this class.
	 *
	 * @details This method returns the total count of objects of type TDerived that have been created by
	 * this class. It is only available when the NDEBUG flag is not defined, indicating that the
	 * program is compiled in debug mode. This method is useful for tracking the number of objects
	 * generated during debugging sessions.
	 *
	 * @tparam TDerived The type of objects for which the count is requested.
	 * @return The total count of objects of type TDerived created by this class.
	 * @note This method is only available in debug mode.
	 */
	template <typename T>
	size_t getGenObjectCnt() const {
		return this->isObjectPoolExist<T>() ? this->getObjectPool<T>()->getAllocCnt() : 0;
	}

	size_t getTotalGenObjectCnt() const {
		size_t total = 0;
		for (const auto& [type, pool] : this->objectPools) { total += pool->getAllocCnt(); }
		return total;
	}

	size_t getTotalGenObjectSize() const {
		size_t total = 0;
		for (const auto& [type, pool] : this->objectPools) { total += pool->getAllocMemSize(); }
		return total;
	}
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim
