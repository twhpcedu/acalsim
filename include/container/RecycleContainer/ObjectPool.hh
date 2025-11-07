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
#include <cstdlib>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_set>

#include "container/RecycleContainer/LinkedListArray.hh"
#include "container/RecycleContainer/RecyclableObject.hh"
#include "container/SharedContainer.hh"
#include "utils/Logging.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

/**
 * @brief Base class for managing pools of recyclable objects with a unified interface.
 *
 * @details The ObjectPoolBase class provides a common interface for pooling objects, allowing
 * for various internal logic implementations in derived classes. It enables different template types
 * to be managed using a shared pointer type derived from the base class. All objects maintained by
 * this class and its derived implementations must be derived from RecyclableObject, ensuring
 * type-safe management of pooled objects.
 */
class ObjectPoolBase {
public:
	ObjectPoolBase() { ; }
	virtual ~ObjectPoolBase() { ; }

	/**
	 * @brief Adds an object to the pool.
	 *
	 * @details This pure virtual method must be implemented by derived classes.
	 * It is responsible for casting the given pointer to the correct type before
	 * adding the object to the pool, allowing for type-safe management of pooled objects.
	 *
	 * @param _ptr Pointer to the object to be added to the pool.
	 */
	virtual void push(RecyclableObject*& _ptr) { this->deleteObject(_ptr); }

	/**
	 * @brief Retrieves an object from the pool.
	 *
	 * @details This method returns an object from the pool, creating a new one if necessary.
	 *
	 * @return Pointer to the object retrieved from the pool.
	 */
	virtual RecyclableObject* pop() { return this->newObject(); }

protected:
	/**
	 * @brief Creates a new object instance.
	 *
	 * @details Allocates and returns a new instance of RecyclableObject,
	 * enabling derived classes to customize object creation logic if needed.
	 *
	 * @return Pointer to the newly created object.
	 */
	virtual RecyclableObject* newObject() { return new RecyclableObject(); }

	/**
	 * @brief Removes and deallocates an object.
	 *
	 * @details Deletes the specified object, freeing its memory back to the system.
	 * This method allows derived classes to override deletion logic if required.
	 *
	 * @param _ptr Pointer to the object to be deleted.
	 */
	virtual void deleteObject(RecyclableObject* _ptr) { delete _ptr; };

#ifdef ACALSIM_STATISTICS
public:
	/**
	 * @brief Gets the count of allocated objects.
	 *
	 * @return The number of allocated objects.
	 */
	size_t getAllocCnt() const { return this->allocCnt; }

	virtual size_t getAllocMemSize() const = 0;

	size_t getMaxOutstandingCnt() const {
		std::lock_guard<std::mutex> lock(this->debugMutex);
		return this->maxOutstandingCnt;
	}

	virtual double getPreallocCost() const { return this->prealloc_cost.sum(); }
	virtual double getBalancingPoolCost() const { return this->balancing_cost.sum(); }
	virtual double getGlobalPoolMutexCost() const = 0;

	virtual void dumpStatistics() const = 0;

protected:
	mutable std::mutex  debugMutex;             // Mutex for accessing debugging variables
	std::atomic<size_t> allocCnt          = 0;  // Counter for object allocations
	std::atomic<size_t> popCnt            = 0;
	std::atomic<size_t> pushCnt           = 0;
	size_t              maxOutstandingCnt = 0;
	Statistics<double, acalsim::StatisticsMode::Accumulator, true> prealloc_cost;
	Statistics<double, acalsim::StatisticsMode::Accumulator, true> balancing_cost;
#endif  // ACALSIM_STATISTICS
};

/**
 * @brief Manages a pool of recyclable objects with type-specific and thread-local management.
 *
 * @tparam T The type of objects managed by this pool, which must derive from RecyclableObject.
 *
 * @details The ObjectPool class provides a thread-safe mechanism for managing pools of objects
 * of type T. It supports type-specific and thread-local object management, ensuring efficient
 * object allocation and recycling. Derived from ObjectPoolBase, this class allows various internal
 * logic implementations while providing a consistent interface for object pooling.
 */
template <typename T>
class ObjectPool : public ObjectPoolBase, virtual public HashableType {
public:
	/**
	 * @brief Constructs an ObjectPool with specified initial segments and segment length.
	 *
	 * @param _initSegments Number of initial segments to create in the pool.
	 * @param _segmentLength Length of each segment in the pool.
	 */
	ObjectPool(size_t _initSegments = 4, size_t _segmentLength = 128);

	/**
	 * @brief Destructs the ObjectPool, cleaning up all allocated resources.
	 *
	 * @details Deletes all objects in both the global pool and per-thread pools, and then
	 * deallocates the pools themselves. Also performs a memory recycling check if debugging
	 * is enabled.
	 */
	~ObjectPool();

	/**
	 * @brief Adds an object to the thread-local pool.
	 *
	 * @details Inserts the given object into the thread-local pool and balances the
	 * thread-local and global pools as needed.
	 *
	 * @param _ptr Pointer to the object to be added to the pool.
	 */
	inline void push(RecyclableObject*& _ptr) override;

	/**
	 * @brief Retrieves an object from the thread-local pool.
	 *
	 * @details Removes and returns an object from the thread-local pool. If necessary,
	 * balances the thread-local pool with the global pool.
	 *
	 * @return Pointer to the object retrieved from the pool.
	 */
	inline T* pop() override;

protected:
	/**
	 * @brief Creates a thread-local object pool if it does not exist.
	 *
	 * @details Initializes a thread-local pool with the specified segment length.
	 * This method is called automatically if the current thread does not have a pool.
	 */
	inline void createThreadObjectPool();

	/**
	 * @brief Retrieves the thread-local object pool.
	 *
	 * @details Returns the thread-local pool for the current thread. If the pool does
	 * not exist, it is created.
	 *
	 * @return Pointer to the thread-local object pool.
	 */
	inline LinkedListArray* getThreadObjectPool();

	/**
	 * @brief Balances the thread-local and global object pools.
	 *
	 * @details Rebalances the pools based on their states. If the thread-local pool is empty,
	 * it attempts to retrieve a list from the global pool. If the thread-local pool has
	 * multiple full lists, it moves one list to the global pool.
	 */
	inline void balanceThreadObjectPools();

	/**
	 * @brief Creates a new object of type T.
	 *
	 * @details Allocates and returns a new instance of type T. Updates debugging information
	 * if debugging is enabled.
	 *
	 * @return Pointer to the newly created object.
	 */
	inline T* newObject() override;

	/**
	 * @brief Deletes an object of type RecyclableObject.
	 *
	 * @details Deallocates the specified object and updates debugging information if
	 * debugging is enabled.
	 *
	 * @param _ptr Pointer to the object to be deleted.
	 */
	inline void deleteObject(RecyclableObject* _ptr) override;

private:
	size_t segmentLength = 0;

	std::shared_mutex                           threadPoolMapMutex;
	std::map<std::thread::id, LinkedListArray*> threadPools;

	SharedContainer<LinkedListArray> globalPools;

#ifdef ACALSIM_STATISTICS
public:
	size_t getAllocMemSize() const override { return sizeof(T) * this->getAllocCnt(); }

	double getGlobalPoolMutexCost() const override { return this->globalPools.getLockCost(); }

	void dumpStatistics() const override;
#endif  // ACALSIM_STATISTICS

#if !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
private:
	/**
	 * @brief Prints a warning if there are unrecycled objects.
	 *
	 * @details Displays a warning message if there are allocated objects that have not
	 * been recycled into the object pool. This function is only available in debug mode.
	 */
	void printMemRecyclingCheck() const {
		std::lock_guard<std::mutex> lock(this->allocPtrSetMutex);
		if (this->allocPtrSet.empty()) return;

#ifdef ACALSIM_STATISTICS
		std::string label = std::string("ObjectPool<") + typeid(T).name() + ">";
		LABELED_STATISTICS(label) << "There are still " << this->allocPtrSet.size()
		                          << " objects that have not been recycled into the object pool.";
#else   // ACALSIM_STATISTICS
		std::string label = std::string("ObjectPool<") + typeid(T).name() + ">";
		LABELED_WARNING(label) << "There are still " << this->allocPtrSet.size()
		                       << " objects that have not been recycled into the object pool.";
#endif  // ACALSIM_STATISTICS
	}

private:
	std::unordered_set<T*> allocPtrSet;  // Set of all allocated and non-released pointers
	mutable std::mutex     allocPtrSetMutex;
#endif  // !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
};

}  // namespace acalsim

#include "container/RecycleContainer/ObjectPool.inl"
