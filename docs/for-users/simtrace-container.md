# SimTraceContainer - ACALSim User Guide

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
- Date: 2024/11/03

([Back To Documentation Portal](/docs/README.md))

## Introduction

To facilitate easier trace generation and collection in JSON files, this document introduces the tracing support of ACALSim. The design aims to achieve the following criteria:

- Developers can export trace records to a JSON file with a two-level hierarchical structure (i.e., the top level as the category name, followed by an array of trace records associated with that category).
- Developers have the flexibility to define custom trace classes and corresponding JSON formats without requiring changes to ACALSim's core implementation.

The feature consists of two classes:

- `SimTraceContainer`: Collect all traces during simulation and dump them to specified filename after simulation completes. There is an instance wrapped by `SharedContainer` in `SimTop`.
- `SimTraceRecord`: The base class for all types of traces to inherit and override with customized attributes and formats.

## Usage

### SimTop

During the construction of a SimTop's derived class, the `SimTraceContainer` member wrapped by a `SharedContainer` can be configured with the output file prefix and folder name.

> **Info** : This step is **not necessary**. By default, the output trace file will be saved in the directory where users execute the simulator, with the filename format `trace-YYYYMMDD-hhmmss.json`.

```cpp
CustomSimTop::CustomSimTop() {
    this->traceCntr.run(/* which */ 0,
                        /* func */ &SimTraceContainer::setFilePath,
                        /* filename_prefix */ "trace",
                        /* folder */ "/path/to/traces");
}
```

### Customize a Trace Record Class

```cpp
class CustomTraceRecord : public acalsim::SimTraceRecord {
public:
    CustomTraceRecord(): acalsim::SimTraceRecord() {}

    nlohmann::json toJson() const override {
        // Construct a `nlohmann::json` instance based on its attributes
		// The created instance can be of any valid form, not limited to an `nlohmann::json::object`
		nlohmann::json j = nlohmann::json::object();
		return j;
    }
};
```

### SimBase / SimModule / SimEvent / SimPacket

- Create a derived class instance of `SimTraceRecord`.
    ```cpp
    std::shared_ptr<SimTraceRecord> trace_event = std::make_shared<CustomTraceRecord>(
        /* tid */ transaction_id,
        /* src_addr */ src_addr,
        /* size */ size);
    ```
- Add the trace record object to the global `SimTraceContainer` owned by `SimTop` with current time tick.
    ```cpp
    acalsim::top->addTraceRecord(trace_event, /* category */ "GlobalCache");
    ```
- Add the trace record object to the global `SimTraceContainer` owned by `SimTop` with a specified time tick.
    ```cpp
    acalsim::top->addTraceRecord(
        /* trace */ trace_event,
        /* category */ "pe2gmem",
        /* tick */ acalsim::top->getGlobalTick() + 2);
    ```

> **Warning** : The time tick parameter passed to `SimTop::addTraceRecord()` is used to sort traces before exporting them to the JSON file. It is recommended to include time tick attributes in custom `SimTraceRecord` classes if this information is intended to be recorded in the trace records.

## Example - `testPETile`

1. (optional) Configure the output file name and folder in `PETileTop::PETileTop()`.
    > `src/testPETile/include/PETileTop.hh`
    ```cpp
    class PETileTop : public STSim<PETile> {
    public:
        PETileTop(const std::string _name = "PESTSim", const std::string _configFile = "")
            : STSim<PETile>(_name, _configFile) {
            this->traceCntr.run(0, &SimTraceContainer::setFilePath, "trace", "src/testPETile/trace");
        }
    }
    ```
    > **Info** : This step is not necessary. By default, the output trace file will be saved in the directory where users execute the simulator, with the filename format `trace-YYYYMMDD-hhmmss.json`.
2. Define a customized trace `CpuTrafficTraceRecord` class.
    > `src/testPETile/include/CPUReqEvent.hh`
    ```cpp
    class CpuTrafficTraceRecord : public acalsim::SimTraceRecord {
    public:
        CpuTrafficTraceRecord(acalsim::Tick _tick, MemReqTypeEnum _req_type, int _transaction_id, uint64_t _addr, int _size)
            : acalsim::SimTraceRecord(),
              tick(_tick),
              req_type(_req_type),
              transaction_id(_transaction_id),
              addr(_addr),
              size(_size) {}

        nlohmann::json toJson() const override {
            nlohmann::json j = nlohmann::json::object();

            j["tick"]           = this->tick;
            j["transaction-id"] = this->transaction_id;
            j["addr"]           = this->addr;
            j["size"]           = this->size;

            switch (this->req_type) {
                case MemReqTypeEnum::PCU_MEM_READ: j["req-type"] = "PCU_MEM_READ"; break;
                case MemReqTypeEnum::PCU_MEM_WRITE: j["req-type"] = "PCU_MEM_WRITE"; break;
                case MemReqTypeEnum::TENSOR_MEM_READ: j["req-type"] = "TENSOR_MEM_READ"; break;
                case MemReqTypeEnum::TENSOR_MEM_WRITE: j["req-type"] = "TENSOR_MEM_WRITE"; break;
                default: j["req-type"] = "UNKNOWN"; break;
            }

            return j;
        }

    private:
        acalsim::Tick  tick = -1;
        MemReqTypeEnum req_type;
        int            transaction_id = -1;
        uint64_t       addr;
        int            size;
    };
    ```
3. Add trace records into `SimTraceContainer` when specific events happen.
    > `src/testPETile/libs/CPUReqEvent.cc`
    ```cpp
	top->addTraceRecord(/* trace */ std::make_shared<CpuTrafficTraceRecord>(
	                           /* tick */ top->getGlobalTick(),
	                           /* req_type */ this->memReqPkt->reqType,
	                           /* transaction_id */ this->tid,
	                           /* addr */ this->memReqPkt->getAddr(),
	                           /* size */ this->memReqPkt->getSize()),
	                       /* category */ "CPUReq");
    ```

The tracing file will be saved as `src/testPETile/trace/trace-YYYYMMDD-hhmmss.json` with the following contents.

```json
{
  "CPUReq": [
    {
      "addr": 0,
      "req-type": "TENSOR_MEM_READ",
      "size": 0,
      "tick": 1,
      "transaction-id": 0
    },
    {
      "addr": 4096,
      "req-type": "TENSOR_MEM_READ",
      "size": 20,
      "tick": 11,
      "transaction-id": 1
    },
    {
      "addr": 8192,
      "req-type": "TENSOR_MEM_READ",
      "size": 40,
      "tick": 21,
      "transaction-id": 2
    },
    {
      "addr": 12288,
      "req-type": "TENSOR_MEM_READ",
      "size": 60,
      "tick": 31,
      "transaction-id": 3
    },
    {
      "addr": 16384,
      "req-type": "TENSOR_MEM_READ",
      "size": 80,
      "tick": 41,
      "transaction-id": 4
    }
  ]
}
```
