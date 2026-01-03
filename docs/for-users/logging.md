# Logging Utility

<!--
Copyright 2023-2026 Playlab/ACAL

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
- Date: 2024/08/11

([Back To Documentation Portal](/docs/README.md))

## Preface

Generating logging messages is a useful and intuitive method for both debugging simulators and extracting simulation results for analysis. To simplify log creation, ACALSim offers several C++ macros that provide the following features:

- Display the time tick for each log message.
- Include labels with logging messages, which can be auto-detected or user-defined.
- Ensure each message is logged without interference from other threads.
- Automatically append a newline character to each log message to avoid repetitive code.

## Built-in Logging Macros

### Unlabeled Messages

#### Normal Logs

```cpp
#include "ACALSim.hh"

INFO << "This operation is run by thread " << std::this_thread::get_id();
WARNING << "There is probably something wrong...";
```

When compiling the simulator in debug mode, the messages will appear as follows:

```shell
Tick=0 Info: This operation is run by thread 140018445849920 [./src/YourProjectName/main.cc:23]
Tick=0 Warning: There is probably something wrong... [./src/YourProjectName/main.cc:24]
```

> **Note**: The left-shift operator `<<` can accept all data types supported by `std::stringstream`. Users can overload this operator for their own classes to ensure compatibility with these logging macros.

In release mode, all file paths will be hidden to produce more concise messages:

```shell
Tick=0 Info: This operation is run by thread 140447543739712
Tick=0 Warning: There is probably something wrong...
```

#### Error Message with Segmentation Faults

Developers can also generate an error message that triggers a segmentation fault:

```cpp
#include "ACALSim.hh"

ERROR << "This situation is impossible to happen";
```

The corresponding output will be:

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: This situation is impossible to happen [./src/YourProjectName/main.cc:25]
Aborted (core dumped)
```

The file path will be hidden as well when the simulator is compiled in release mode.

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: This situation is impossible to happen
Aborted (core dumped)
```

#### Assertion

The ACALSim framework provides two macros for developers to use for assertions:

- Assertion with default error message:
    ```cpp
    #include "ACALSim.hh"

    void foo(void* ptr_) { ASSERT(ptr_); }
    ```
    If the verification fails, an error message is produced, resulting in a segmentation fault:

    ```shell
    terminate called after throwing an instance of 'std::runtime_error'
      what():  Tick=0 Error: Condition "ptr_" failed. [./src/YourProjectName/main.cc:27]
    Aborted (core dumped)
    ```
- Assertion with a customized error message:
    ```cpp
    #include "ACALSim.hh"

    void foo(void* ptr_) { ASSERT_MSG(ptr_, "The pointer should not be NULL!"); }
    ```
    If the assertion fails, the provided message will be displayed:
    ```shell
    terminate called after throwing an instance of 'std::runtime_error'
      what():  Tick=0 Error: The pointer should not be NULL! [./src/YourProjectName/main.cc:27]
    Aborted (core dumped)
    ```

These logging macros provide developers with flexible and powerful tools for debugging and error handling in both debug and release modes of the simulator.

### Messages Labeled with Auto-detected Class Names

To enhance readability and facilitate log parsing, ACALSim provides macros that automatically label messages with the caller's class name.
it from it nor invoke its constructor.

A class using these macros must derive from `acalsim::HashableType` to ensure efficient type identification and name generation. To prevent multiple inheritance issues, it is recommended to use [virtual inheritance](https://en.wikipedia.org/wiki/Virtual_inheritance) to ensure that only a single instance of `acalsim::HashableType` exists in the derived class. Additionally, if any base class of a derived class has already inherited from `acalsim::HashableType`, the derived class does not need to explicitly inherit from it or invoke its constructor.

> **Important**: These macros can only be used within class implementations.

#### Info, Warning, and Error Messages

```cpp
#include "ACALSim.hh"

class Foo : virtual public acalsim::HashableType {
    void foo {
        CLASS_INFO << "This operation is run by thread " << std::this_thread::get_id();
        CLASS_WARNING << "There is probably something wrong...";
        CLASS_ERROR << "This situation is impossible to happen";
    }
};
```

The generated messages will be prefixed with the class name to which the statements belong:

```shell
Tick=0 Info: [7Foo] This operation is run by thread 140018445849920 [./src/YourProjectName/include/foo.hh:11]
Tick=0 Warning: [7Foo] There is probably something wrong... [./src/YourProjectName/include/foo.hh:12]
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [7Foo] This situation is impossible to happen [./src/YourProjectName/include/foo.hh:13]
Aborted (core dumped)
```

> **Note**: The class names are compiler-dependent and might differ slightly from the names in the source code.

#### Assertion

```cpp
#include "ACALSim.hh"

class Foo : virtual public acalsim::HashableType {
    void foo(void* ptr_) {
        CLASS_ASSERT(ptr_);

        // The variant with an user-defined message
        CLASS_ASSERT_MSG(ptr_, "The pointer should not be NULL!");
    }
};
```

The logging messages generated by these assertions will look like:

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [7Foo] Condition "ptr_" failed. [./src/YourProjectName/include/foo.hh:27]
Aborted (core dumped)
```

and

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [7Foo] The pointer should not be NULL! [./src/YourProjectName/include/foo.hh:27]
Aborted (core dumped)
```

These class-labeled macros provide more context in logs, making it easier to trace the origin of messages and debug issues in complex class hierarchies.

### Messages Labeled with User-defined Names

ACALSim provides a set of macros that allow developers to customize the labels of logging messages, enhancing readability and simplifying log parsing. These macros include `LABELED_INFO`, `LABELED_WARNING`, `LABELED_ERROR`, `LABELED_ASSERT`, and `LABELED_ASSERT_MSG`.

#### Info, Warning, and Error Messages

```cpp
#include "ACALSim.hh"

LABELED_INFO("CPU-0") << "This operation is run by thread " << std::this_thread::get_id();
LABELED_WARNING("CPU-0") << "There is probably something wrong...";
LABELED_ERROR("CPU-0") << "This situation is impossible to happen";
```

The generated messages will be prefixed with the developer-assigned label:

```shell
Tick=0 Info: [CPU-0] This operation is run by thread 140018445849920 [./src/YourProjectName/include/foo.hh:11]
Tick=0 Warning: [CPU-0] There is probably something wrong... [./src/YourProjectName/include/foo.hh:12]
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [CPU-0] This situation is impossible to happen [./src/YourProjectName/include/foo.hh:13]
Aborted (core dumped)
```

#### Assertion

```cpp
#include "ACALSim.hh"

void foo(void* ptr_) {
    LABELED_ASSERT(ptr_, "CPU-0");

    // The variant with an user-defined message
    LABELED_ASSERT_MSG(ptr_, "CPU-0", "The pointer should not be NULL!");
}
```

The logging messages generated by these assertions will look like:

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [CPU-0] Condition "ptr_" failed. [./src/YourProjectName/main.cc:27]
Aborted (core dumped)
```

and

```shell
terminate called after throwing an instance of 'std::runtime_error'
  what():  Tick=0 Error: [CPU-0] The pointer should not be NULL! [./src/YourProjectName/main.cc:27]
Aborted (core dumped)
```

These user-defined labels provide flexibility in categorizing log messages, making it easier to identify the source or context of each log entry. This can be particularly useful in complex systems with multiple components or when debugging specific parts of an application.

## Define Custom Macros for Fine-Grained Output Control

During the development of simulators with ACALSim, developers may need to enable and disable logging messages with precision. This section explains how to implement such mechanisms in ACALSim-based applications.

1. **Define New Macros in a Header File**: Start by defining custom macros in your preferred header file:
	```cpp
	#include "ACALSim.hh"

	#ifdef USER_DEFINED_VERBOSE
	#define USER_VERBOSE_CLASS_INFO    CLASS_INFO
	#define USER_VERBOSE_CLASS_WARNING CLASS_WARNING
	#define USER_VERBOSE_CLASS_ERROR   CLASS_ERROR
	#else
	#define USER_VERBOSE_CLASS_INFO    acalsim::FakeLogOStream()
	#define USER_VERBOSE_CLASS_WARNING acalsim::FakeLogOStream()
	#define USER_VERBOSE_CLASS_ERROR   acalsim::FakeLogOStream()
	#endif
	```
2. **Add the Corresponding Option to CMake Configurations**: Using the ProjectTemplate as an example:
	- Declare the option in src/ProjectTemplate/CMakeLists.txt:
		```cmake
		# The option is disabled by default
		option(USER_DEFINED_VERBOSE "Enable user-defined verbose mode" OFF)
		```
	- Configure the desired CMake target with this flag in `src/ProjectTemplate/libs/CMakeLists.txt`.
		```cmake
		add_target_compile_definitions(${APP_PREFIX_LIB_NAME} USER_DEFINED_VERBOSE)
		```
3. **Enable the Option During Compilation**: Use the following commands to enable the custom logging option while compiling your simulators:
	```shell
	$ cmake -B build/debug/ \
		-DCMAKE_BUILD_TYPE=Debug \
		-DUSER_DEFINED_VERBOSE=ON
	$ cmake --build build/debug/ -j $(nproc) --target <TARGET_NAME>
	```
	> **Important**: CMake caches all options for future use. You must manually disable any options that are no longer needed.

This approach allows for flexible and granular control over logging output, enabling developers to focus on specific areas of interest during debugging and development processes.

---

([Back To Documentation Portal](/docs/README.md))
