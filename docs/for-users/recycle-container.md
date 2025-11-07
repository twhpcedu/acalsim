# Recycle Container - ACALSim User Guide

<!--
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


---

- Author: Jen-Chien Chang \<jenchien@twhpcedu.org\>
- Date: 2024/12/20

([Back To Documentation Portal](/docs/README.md))

## Introduction

As a system simulation framework designed for design space exploration, ACALSim prioritizes high efficiency and scalability. However, the frequent generation of events and packets during simulations can consume significant CPU time, potentially limiting overall performance. To address this performance bottleneck caused by repeated allocations and deallocations, ACALSim introduces the RecycleContainer and several auxiliary classes. These features enable developers to reuse existing objects instead of continuously creating new ones, thereby enhancing performance. The `RecycleContainer` and its associated classes provide a mechanism for object recycling, which:

- Reduces memory allocation overhead
- Minimizes garbage collection pressure
- Improves cache utilization
- Enhances overall simulation speed

By implementing this object recycling strategy, ACALSim allows developers to create more efficient and scalable simulations, particularly beneficial for complex systems or large-scale design space exploration tasks.

## QuickStart

### A Minimal Class Can Be Collected by `RecycleContainer`

```cpp
#include "ACALSim.hh"

class ExampleObject : public RecyclableObject {
public:
    ExampleObject();
};
```

The following classes provided by ACALSim are already derived classes of `RecyclableObject`. Any derived class of them is not required to explicitly inherit from it again.

- `SimEvent`
- `SimPacket`

### Acquiring An Object

There is a special container that contains multiple object pools declared in a `SimTop` object. All operations in ACALSim applications can use the `acalsim::top` pointer to acquire objects from this pool.

```cpp
ExampleObject* obj = acalsim::top->getRecycleContainer()->acquire<ExampleObject>();
```

If the requested object isn't available in the corresponding object pool, a number of objects will be created and inserted to the pool. This provides developers with a simple way to utilize these features.

### Recycling An Object

```cpp
acalsim::top->getRecycleContainer()->recycle(obj);
```

This operation makes the object available for further use. Once recycled, developers should no longer use the object.

### Updating Object Status for a New Lifetime

1. Define a member method in the class.
    ```cpp
    #include "ACALSim.hh"

    class ExampleObject : public RecyclableObject {
    public:
        ExampleObject();

        // Update status for a new lifetime like a (re)constructor
        // The naming and contents are fully user-defined
        void renew(size_t _id) { this->id = _id; }

    private:
        size_t id;
    };
    ```
    It's recommended to define the parameters as constant references to avoid unnecessary copying.

    ```cpp
    void renew(const size_t& _id) { this->id = _id; }
    ```

    > **Caution**: If your class inherits from another class (except for RecyclableObject), make sure to check if the derived class's renew() method needs to call the base class's renew() method as well.

2. Pass the function pointer and its arguments when acquiring objects.
    ```cpp
    ExampleObject* obj = acalsim::top->getRecycleContainer()->acquire<ExampleObject>(
        &ExampleObject::renew, // Points to a member method of `ExampleObject`
        3   // All arguments for the given method
            // Note that multiple arguments are allowed
    );
    ```

	> **Note**: Passing an overloaded method name in C++ can lead to ambiguity during compilation. This issue commonly arises when using `RecycleContainer::acquire()` to reinitialize objects. To resolve this, developers can:
	>
	> 1. Explicitly call the `renew()` method after receiving the pointer, or
	> 2. Use `static_cast` to specify the desired method signature ([example](https://stackoverflow.com/a/4364646)).

### Enabling Automatic Lifetime Management For `RecyclableObject`

An alternative object acquisition API `RecycleContainer::acquireSharedPtr()` is also provided for callers to obtain object pointers wrapped by `std::shared_ptr`. This enables automatically lifetime management without explicit operations by developers.

An alternative object acquisition API, `RecycleContainer::acquireSharedPtr()`, is provided to allow callers to obtain object pointers managed by `std::shared_ptr`. This enables automatic lifetime management, eliminating the need for explicit recycling operations by developers.

```cpp
auto rc = acalsim::top->getRecycleContainer();

std::shared_ptr<ExampleObject> obj = rc->acquireSharedPtr<ExampleObject>();
```

There is also an overloaded variant of this method which allows users to invoke a custom method on the object before returning the pointer:

```cpp
auto rc = acalsim::top->getRecycleContainer();

std::shared_ptr<ExampleObject> obj = rc->acquireSharedPtr<ExampleObject>(
    &ExampleObject::renew, // Points to a member method of `ExampleObject`
    3   // All arguments for the given method
        // Note that multiple arguments are allowed
);
```

> **Note** : For detailed information on using `std::shared_ptr`, please refer to the [C++ reference](https://en.cppreference.com/w/cpp/memory/shared_ptr) documentation.

## Features

- **Efficient object management**: Replaces C++ new and delete with optimized operations for improved performance.
- **Constant time complexity**: Ensures fast performance for core operations like object insertion, retrieval, and pool merging through a singly-linked-list-based implementation.
- **Reduced overhead**: Minimizes mutex usage by assigning dedicated object pools to each thread and employing "temporary recycling stations" to avoid race conditions.
- **Optimized memory usage**: Avoids excessive memory consumption and performance degradation by recycling objects back to their original SimBase and carefully managing pool sizes.

## Public API Reference

### `RecycleContainer`

- **Constructor**: Creates a RecycleContainer with a specified initial pool size.
    ```cpp
    // include/container/RecycleContainer/RecycleContainer.hh
    RecycleContainer(size_t _init_size = 512);
    ```
    - For each type of object maintained by this container, the default size of the corresponding object pool would be set to the given number.
    - Parameter `_init_size`: The initial number of objects to allocate in each pool.
- **Acquire object**: Retrieves an object from the pool without reinitialization.
    ```cpp
	template <typename T>
    T* acquire();
    ```
    - It is the caller's responsibility to set up the object for further use.
    - Template parameter `T`: The type of object to acquire.
    - Return value: A pointer to the acquired object.
- **Acquire and reset object**: Acquires an object and invokes a specified reset method before returning it.
    ```cpp
	template <typename T, typename RType, typename... Args>
	T* acquire(RType (T::*_func)(Args...), Args&&... _args);
    ```
    - This method invokes the specified reset method on the object with the provided arguments before returning it to the caller.
    - Template parameter `T`: The type of object to acquire.
    - Template parameter `RType`: The return type of the reset method. *It can usually be deduced by compilers automatically.*
    - Template parameter `Args`: The types of arguments passed to the reset method. *It can usually be deduced by compilers automatically.*
    - Return value: A pointer to the acquired object.
- **Set initial pool size**: Specifies the initial size for a specific object pool.
    ```cpp
	template <typename T>
	void setInitSize(size_t _n);
    ```
    - This method will create the corresponding object pool and prepare initial objects immediately.
    - Template parameter `T`: The type of objects for which to set the initial pool size.
    - Parameter `_n`: The initial number of objects to allocate in the pool.
- **Move objects**: Transfers objects from temporary recycling stations to outbound stations for reuse.
    ```cpp
	void moveInboundToOutbound();
    ```
    - Callers should ensure there is no thread attempting to send objects to this container while executing this method.

### `RecyclableObject`

Base class for objects managed by `RecycleContainer`, inheriting from `LinkedList::Node` for efficient pool management.

- **Constructor**: Creates a RecyclableObject with its default constructor.
	```cpp
	// include/container/RecycleContainer/RecyclableObject.hh
	RecyclableObject();
	```

## How Does Object Pooling Work In ACALSim

![](https://codimd.playlab.tw/uploads/c4863e8ad94a11804087e96c0.png)

### Core Structures

- Global Pools and Local Pools are implemented as `LinkedListArray` instances.
- The `RecycleContainer` acts as a central repository, managing all `ObjectPool<T>` instances for various object types during simulation.

### `LinkedList`

The `LinkedList` class provides a fundamental implementation of a singly linked list with the following key features:

1. Operations:
    - Back insertion of nodes
    - Front removal of nodes
    - List splicing capability
2. Node Requirements:
    - Objects stored must inherit from the nested `LinkedList::Node` class
    - Extends functionality by inheriting from the `LoggingObject` class
3. Key Methods:
	- Constructor: `LinkedList()` - Initializes an empty list
	- Destructor: `~LinkedList()` - Cleans up all nodes, deallocating saved objects
	- `void insertBack(LinkedList::Node* _node)` - Adds a node to the list's end
	- `LinkedList::Node* front() const` - Retrieves the front node
	- `LinkedList::Node* popFront()` - Removes and returns the front node
	- `bool empty() const` - Checks for list emptiness
	- `size_t size() const` - Returns node count (O(1) time complexity)
	- `void spliceAfter(LinkedList& _other)` - Splices another list after the current tail
		> **Note**: This operation empties the _other list

### `LinkedList::Node`

- Nested class representing a node in the linked list
- Contains a pointer to the next node
- All objects in the LinkedList must inherit from this class

This structure provides a flexible and efficient foundation for object pooling in ACALSim, allowing for optimized memory management and improved performance.

### `LinkedListArray`

![](https://codimd.playlab.tw/uploads/15e4fb27232cadf9d206d3702.png)

The LinkedListArray class efficiently manages multiple LinkedList instances in an array-like structure, offering the following features:

1. Purpose: Enables efficient management and manipulation of nodes across multiple lists, supporting operations such as insertion, removal, and list transfer between instances.
2. Capacity: Each list within the array has a maximum node capacity defined by LinkedListArray::maxListSize.
3. Key Methods:
    - Constructor: `LinkedListArray(size_t _maxListSize = 128)`
        - Initializes a new instance with a specified maximum list size (default: 128 nodes)
    - Destructor: `virtual ~LinkedListArray()`
        - Properly deallocates all managed nodes and lists
    - `void insert(LinkedList::Node* _node)`
        - Inserts a node into the array, creating a new list if the last one is full
    - `LinkedList::Node* front() const`
        - Retrieves the front node of the last linked list in the array
    - `LinkedList::Node* pop()`
        - Removes and returns the front node from the last list, removing the list if it becomes empty
    - `bool empty() const`
        - Checks if the array has no elements
    - `bool hasFullLists() const`
        - Checks for the presence of full lists in the array
    - `size_t getNumFullLists() const`
        - Counts the number of full linked lists in the array
    - `void insertList(LinkedList* _list)`
        - Inserts a new LinkedList at the front of the array
        - Note: Does not verify if the list length matches maxListSize
    - `LinkedList* popList()`
        - Removes and returns the front LinkedList from the array
        - Facilitates efficient list transfer between LinkedListArray instances
        - Note: Does not verify if the list length matches maxListSize

This class provides a robust and flexible structure for managing multiple linked lists, optimizing memory usage and enhancing performance in the object pooling system.

### `ObjectPool<T>`

The `ObjectPool<T>` is a template class designed for efficient object pooling, with the following key features:

1. Purpose: Manages a pool of unused objects for future reuse, optimizing memory allocation and deallocation.
2. Thread Safety: Utilizes multiple LinkedListArray instances to achieve thread-safe operations while minimizing mutex overhead.
3. Key Methods:
    - Constructor:
        ```cpp
        // include/container/RecycleContainer/ObjectPool.hh
        ObjectPool(size_t _initSegments = 4, size_t _segmentLength = 128);
        ```
        - Initializes the pool with a specified number of objects
        - _initSegments: Number of initial segments (default: 4)
        - _segmentLength: Length of each segment (default: 128)
        - Total initial objects: _initSegments * _segmentLength
        - _segmentLength also determines the batch size for object transfers between thread-local and global lists
    - Push a [`RecyclableObject`](#RecyclableObject) into the thread's local object list.
        ```cpp
        void push(RecyclableObject* _ptr);
        ```
        - Adds a RecyclableObject to the thread's local object list
        - Creates the thread's local list if it doesn't exist
        - _ptr: Pointer to the RecyclableObject to be added
    - Pop an object from the thread's local object list.
        ```cpp
        T* pop();
        ```
        - Retrieves and removes an object from the thread's local list
        - Returns a pointer to the popped object

This class provides a thread-safe and efficient mechanism for object reuse, significantly improving performance in multi-threaded environments by reducing the overhead of frequent object creation and destruction.

---

([Back To Documentation Portal](/docs/README.md))
