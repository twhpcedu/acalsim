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

#include <unordered_map>

#include "utils/Logging.hh"

namespace acalsim {

/**
 * @file UnorderedRequestQueue.hh
 * @brief Out-of-order request tracking using integer transaction IDs
 *
 * @details
 * UnorderedRequestQueue manages outstanding requests that may complete out of order.
 * Each request is assigned a unique integer ID when issued, and can be retrieved
 * or removed using that ID when the response arrives.
 *
 * **Key Characteristics:**
 *
 * - **Out-of-Order Completion**: Requests can be removed in any order, not FIFO
 * - **ID-Based Lookup**: Fast O(1) access using transaction IDs
 * - **Outstanding Tracking**: Maintains all in-flight requests
 * - **Assertion on Missing**: get() asserts if ID doesn't exist (debug safety)
 *
 * **Use Cases:**
 *
 * | Scenario | Description |
 * |----------|-------------|
 * | **Memory Controller** | Track outstanding memory requests with variable latencies |
 * | **NoC Router** | Track packets in-flight with different routes and latencies |
 * | **Cache Controller** | Track outstanding cache misses to DRAM |
 * | **DMA Engine** | Track pending DMA transfers with transaction IDs |
 * | **TLB Miss Handler** | Track outstanding page table walks |
 *
 * **Workflow:**
 * ```
 * 1. Issue request, assign ID
 * 2. add(id, request_data)
 * 3. ... request is in-flight (out-of-order processing) ...
 * 4. Response arrives with ID
 * 5. if (contains(id)) { request_data = get(id); remove(id); }
 * ```
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | add() | O(1) avg | Hash table insert |
 * | get() | O(1) avg | Hash table lookup |
 * | contains() | O(1) avg | Hash table find |
 * | remove() | O(1) avg | Hash table erase |
 *
 * **Memory:** O(n) where n = number of outstanding requests
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Multiple threads issuing requests need mutex protection
 *
 * @tparam T Type of request data stored (typically pointers or small objects)
 *
 * @code{.cpp}
 * // Example: Memory controller tracking outstanding reads
 * UnorderedRequestQueue<MemoryRequest*> outstandingReads;
 * int nextTransactionID = 0;
 *
 * // Issue a read request
 * void issueRead(uint64_t addr, Callback* cb) {
 *     int txnID = nextTransactionID++;
 *     MemoryRequest* req = new MemoryRequest{addr, cb, txnID};
 *
 *     outstandingReads.add(txnID, req);
 *     sendToMemory(req);  // Send to DRAM controller
 * }
 *
 * // Handle response (may arrive out-of-order)
 * void handleResponse(int txnID, uint64_t data) {
 *     if (outstandingReads.contains(txnID)) {
 *         MemoryRequest* req = outstandingReads.get(txnID);
 *         req->callback->onComplete(data);
 *
 *         outstandingReads.remove(txnID);
 *         delete req;
 *     }
 * }
 * @endcode
 *
 * @note Transaction IDs must be unique across all outstanding requests
 * @note get() will assert and terminate if ID doesn't exist (use contains() first for safety)
 * @note No automatic cleanup - caller must remove() all requests
 *
 * @warning Memory leak risk if requests are not removed
 * @warning get() on non-existent ID triggers assertion failure
 *
 * @see FifoQueue for ordered FIFO request queue
 * @since ACALSim 0.1.0
 */
template <typename T>
class UnorderedRequestQueue {
public:
	/**
	 * @brief Get the request associated with the given transaction ID
	 *
	 * @param _id Transaction ID of the request
	 * @return Copy of the request data (T is typically a pointer)
	 *
	 * @note Complexity: O(1) average case
	 * @note Does NOT remove the request (use remove() after get())
	 * @note Triggers assertion failure if ID doesn't exist
	 *
	 * @warning ASSERT fails if _id not found - use contains() first for safety
	 *
	 * @code{.cpp}
	 * UnorderedRequestQueue<Request*> queue;
	 * queue.add(42, myRequest);
	 *
	 * if (queue.contains(42)) {
	 *     Request* req = queue.get(42);
	 *     req->process();
	 *     queue.remove(42);
	 * }
	 * @endcode
	 */
	T get(int _id) {
		typename std::unordered_map<int, T>::iterator it = requests.find(_id);
		ASSERT_MSG(it != requests.end(), "The ID " + std::to_string(_id) + " does not exist.");
		return it->second;
	}

	/**
	 * @brief Check if a request with the given ID exists
	 *
	 * @param _id Transaction ID to check
	 * @return true if request exists, false otherwise
	 *
	 * @note Complexity: O(1) average case
	 * @note Safe to call for any ID
	 *
	 * @code{.cpp}
	 * UnorderedRequestQueue<Packet*> queue;
	 *
	 * if (queue.contains(txnID)) {
	 *     Packet* pkt = queue.get(txnID);
	 *     // Process packet
	 * } else {
	 *     // Handle missing ID
	 * }
	 * @endcode
	 */
	bool contains(int _id) { return requests.contains(_id); }

	/**
	 * @brief Add a request to the outstanding queue
	 *
	 * @param _id Unique transaction ID for this request
	 * @param _obj Request data to store
	 *
	 * @note Complexity: O(1) average case (amortized hash table insert)
	 * @note If _id already exists, behavior is undefined (map insert fails)
	 *
	 * @warning Caller must ensure _id is unique
	 * @warning No automatic ID generation - caller manages IDs
	 *
	 * @code{.cpp}
	 * UnorderedRequestQueue<MemReq*> queue;
	 * int txnID = generateUniqueID();
	 * MemReq* req = new MemReq{addr, data};
	 *
	 * queue.add(txnID, req);  // Now tracked as outstanding
	 * @endcode
	 */
	void add(int _id, T _obj) { requests.insert(std::make_pair(_id, _obj)); }

	/**
	 * @brief Remove a request from the outstanding queue
	 *
	 * @param _id Transaction ID of the request to remove
	 *
	 * @note Complexity: O(1) average case
	 * @note Safe to call even if _id doesn't exist (erase is idempotent)
	 * @note Does NOT delete the object (caller responsible for cleanup)
	 *
	 * @code{.cpp}
	 * UnorderedRequestQueue<Request*> queue;
	 * queue.add(42, req);
	 *
	 * Request* req = queue.get(42);
	 * req->complete();
	 *
	 * queue.remove(42);  // Remove from tracking
	 * delete req;        // Caller must clean up object
	 * @endcode
	 */
	void remove(int _id) { requests.erase(_id); }

private:
	/**
	 * @brief Hash map of outstanding requests indexed by transaction ID
	 * @details Maps int (transaction ID) -> T (request data)
	 */
	std::unordered_map<int, T> requests;
};

}  // end of namespace acalsim
