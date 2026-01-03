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

#include "container/RecycleContainer/LinkedListArray.hh"
#include "container/RecycleContainer/ObjectPool.hh"

#ifdef ACALSIM_STATISTICS
#include <string>
#include <typeinfo>
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename T>
ObjectPool<T>::ObjectPool(size_t _initSegments, size_t _segmentLength)
    : ObjectPoolBase(), segmentLength(_segmentLength) {
	this->globalPools.add(_segmentLength);

	MEASURE_TIME_MICROSECONDS(/* var_name */ init, /* code_block */ {
		for (size_t i = 0; i < (_segmentLength * _initSegments); ++i) {
			this->globalPools.run(0, &LinkedListArray::insert, dynamic_cast<RecyclableObject*>(this->newObject()));
		}
	});

#ifdef ACALSIM_STATISTICS
	this->prealloc_cost.push(init_lat);
#endif  // ACALSIM_STATISTICS
}

template <typename T>
ObjectPool<T>::~ObjectPool() {
	for (auto& [id, pool] : this->threadPools) {
		while (!pool->empty()) { this->deleteObject(dynamic_cast<T*>(pool->pop())); }
		delete pool;
	}

	while (auto obj = this->globalPools.run(0, &LinkedListArray::pop</* SIZE_CHECK */ true>)) {
		this->deleteObject(dynamic_cast<T*>(obj));
	}

#if !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
	this->printMemRecyclingCheck();
#endif  // !defined(NDEBUG) || defined(ACALSIM_STATISTICS)

#ifdef ACALSIM_STATISTICS
	this->dumpStatistics();
#endif
}

template <typename T>
void ObjectPool<T>::push(RecyclableObject*& _ptr) {
	this->getThreadObjectPool()->insert(_ptr);
	this->balanceThreadObjectPools();

#ifdef ACALSIM_STATISTICS
	this->pushCnt += 1;
#endif  // ACALSIM_STATISTICS
}

template <typename T>
T* ObjectPool<T>::pop() {
	T* obj = dynamic_cast<T*>(this->getThreadObjectPool()->pop());
	this->balanceThreadObjectPools();

#ifdef ACALSIM_STATISTICS
	this->popCnt += 1;
	{
		std::lock_guard<std::mutex> lock(this->debugMutex);
		this->maxOutstandingCnt = std::max(this->popCnt - this->pushCnt, this->maxOutstandingCnt);
	}
#endif  // ACALSIM_STATISTICS

	return obj;
}

template <typename T>
void ObjectPool<T>::createThreadObjectPool() {
	std::thread::id thread_id = std::this_thread::get_id();

	{
		std::unique_lock<std::shared_mutex> lock(this->threadPoolMapMutex);
		if (!this->threadPools.contains(thread_id)) {
			this->threadPools[thread_id] = new LinkedListArray(this->segmentLength);
		}
	}

	this->balanceThreadObjectPools();
}

template <typename T>
LinkedListArray* ObjectPool<T>::getThreadObjectPool() {
	std::thread::id thread_id = std::this_thread::get_id();

	if (!this->threadPools.contains(thread_id)) { this->createThreadObjectPool(); }

	{
		std::shared_lock<std::shared_mutex> lock(this->threadPoolMapMutex);
		return this->threadPools[thread_id];
	}
}

template <typename T>
void ObjectPool<T>::balanceThreadObjectPools() {
	/**
	 * Rebalance policy:
	 * - If there are 2 full lists in the local pool -> move 1 list to the global pool.
	 * - If there are no list in the local pool -> insert 1 list from the global pool.
	 */

	LinkedListArray* local_pool = this->getThreadObjectPool();

	if (local_pool->empty()) {
		std::shared_ptr<LinkedList> list = nullptr;

		MEASURE_TIME_MICROSECONDS(
		    /* var_name */ pull,
		    /* code_block */ { list = this->globalPools.run(0, &LinkedListArray::popList</* SIZE_CHECK */ true>); });

#ifdef ACALSIM_STATISTICS
		this->balancing_cost.push(pull_lat);
#endif  // ACALSIM_STATISTICS

		if (!list) {
			list = std::make_shared<LinkedList>();
			MEASURE_TIME_MICROSECONDS(
			    /* var_name */ prealloc,
			    /* code_block */ {
				    for (size_t i = 0; i < this->segmentLength; ++i) {
					    list->insertBack(dynamic_cast<RecyclableObject*>(this->newObject()));
				    }
			    });

#ifdef ACALSIM_STATISTICS
			this->prealloc_cost.push(prealloc_lat);
#endif  // ACALSIM_STATISTICS
		}

		local_pool->insertList(list);

	} else if (local_pool->getNumFullLists() >= 2) {
		MEASURE_TIME_MICROSECONDS(
		    /* var_name */ push,
		    /* code_block */ {
			    do {
				    std::shared_ptr<LinkedList> list = local_pool->popList();
				    this->globalPools.run(0, &LinkedListArray::insertList, list);
			    } while (local_pool->getNumFullLists() >= 2);
		    });

#ifdef ACALSIM_STATISTICS
		this->balancing_cost.push(push_lat);
#endif  // ACALSIM_STATISTICS
	}
}

template <typename T>
T* ObjectPool<T>::newObject() {
	T* ptr = new T();

#if !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
	{
		std::lock_guard<std::mutex> lock(this->allocPtrSetMutex);
		this->allocPtrSet.insert(ptr);
	}
#endif  // !defined(NDEBUG) || defined(ACALSIM_STATISTICS)

#ifdef ACALSIM_STATISTICS
	this->allocCnt += 1;
#endif  // ACALSIM_STATISTICS

	return ptr;
}

template <typename T>
void ObjectPool<T>::deleteObject(RecyclableObject* _ptr) {
#if !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
	{
		std::lock_guard<std::mutex> lock(this->allocPtrSetMutex);
		if (this->allocPtrSet.contains(dynamic_cast<T*>(_ptr))) { this->allocPtrSet.erase(dynamic_cast<T*>(_ptr)); }
	}
#endif  // !defined(NDEBUG) || defined(ACALSIM_STATISTICS)
	delete _ptr;
}

#ifdef ACALSIM_STATISTICS
template <typename T>
void ObjectPool<T>::dumpStatistics() const {
	std::string label = std::string("ObjectPool<") + typeid(T).name() + ">";

	LABELED_STATISTICS(label) << "Max Outstanding Usage: " << this->maxOutstandingCnt;
	LABELED_STATISTICS(label) << "Pop Count: " << this->popCnt << ", Push Count: " << this->pushCnt;
	LABELED_STATISTICS(label) << "Allocated Count: " << this->getAllocCnt() << ", "
	                          << "Allocated Size: " << ((double)this->getAllocMemSize() / 1024) << " KB";
}
#endif  // ACALSIM_STATISTICS

}  // namespace acalsim
