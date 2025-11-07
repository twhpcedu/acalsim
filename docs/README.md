# ACALSim Documentation

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

**ACALSim** is a high-performance, multi-threaded event-driven simulation framework designed for large-scale parallel system design space exploration. It enables fast and scalable co-simulation of complex hardware architectures including GPUs, NPUs, and heterogeneous computing systems.

## Quick Navigation

### 🚀 **I'm new to ACALSim**
- <a href="https://acalsim.playlab.tw/docs/getting-started/env-setup">Installation Guide</a> - Set up your development environment
- <a href="https://acalsim.playlab.tw/docs/getting-started/for-users">Quick Start for Users</a> - Build and run your first simulation
- <a href="https://acalsim.playlab.tw/docs/getting-started/for-developers">Quick Start for Developers</a> - Understanding the build system and debugging

### 📚 **I want to learn how to use ACALSim**
- <a href="https://acalsim.playlab.tw/docs/for-users/simchannel">Event-Driven Simulation Basics</a> - Core simulation concepts
- <a href="https://acalsim.playlab.tw/docs/for-users/simchannel">Inter-Simulator Communication</a> - Connect simulators together
- <a href="https://acalsim.playlab.tw/docs/for-users/recycle-container">Memory Management</a> - RecycleContainer object pooling
- <a href="https://acalsim.playlab.tw/docs/for-users/logging">Logging and Debugging</a> - Debug your simulations

### 🔧 **I need help with a specific task**
- <a href="https://acalsim.playlab.tw/docs/for-users/simtrace-container">Performance Tracing</a> - Analyze simulation performance
- <a href="https://acalsim.playlab.tw/docs/for-users/simtensor">Tensor Workloads</a> - Model neural network workloads

### 🧑‍💻 **I want to contribute or understand internals**
- <a href="https://acalsim.playlab.tw/docs/for-developers/thread-manager">ThreadManager Architecture</a> - Multi-threading design
- <a href="https://acalsim.playlab.tw/docs/for-developers/thread-manager-v1">PriorityQueue ThreadManager</a> - Default ThreadManager implementation
- <a href="https://acalsim.playlab.tw/docs/MIGRATION_GUIDE">Migration Guide</a> - Recent API changes

---

## Documentation Structure

### Getting Started
Step-by-step guides to get you up and running:

| Document | Description |
|----------|-------------|
| [Environment Setup](https://acalsim.playlab.tw/docs/getting-started/env-setup) | Install dependencies, set up Docker, build the framework |
| [For Users](https://acalsim.playlab.tw/docs/getting-started/for-users) | Run example simulations, understand project structure |
| [For Developers](https://acalsim.playlab.tw/docs/getting-started/for-developers) | Build system, compilation modes, debugging tools |

### Core Concepts (For Users)

Essential features for building simulations:

| Topic | Document | Description |
|-------|----------|-------------|
| **Communication** | [SimChannel](https://acalsim.playlab.tw/docs/for-users/simchannel) | Message passing between simulators |
| **Memory Management** | [RecycleContainer](https://acalsim.playlab.tw/docs/for-users/recycle-container) | Object pooling for high-performance |
| **Workloads** | [SimTensor](https://acalsim.playlab.tw/docs/for-users/simtensor) | Tensor workload modeling |
| **Debugging** | [Logging](https://acalsim.playlab.tw/docs/for-users/logging) | Multi-threaded logging utilities |
| | [SimTrace](https://acalsim.playlab.tw/docs/for-users/simtrace-container) | Chrome trace format for performance analysis |

### Advanced Topics (For Developers)

Deep dives into framework internals:

| Topic | Document | Description |
|-------|----------|-------------|
| **Threading** | [ThreadManager Overview](https://acalsim.playlab.tw/docs/for-developers/thread-manager) | Multi-threading architecture and available implementations |
| | [PriorityQueue (V1)](https://acalsim.playlab.tw/docs/for-developers/thread-manager-v1) | Default ThreadManager with sparse activation optimization |
| | **Barrier (V2)** | ⏳ Coming soon - C++20 barrier-based synchronization |
| | **PrebuiltTaskList (V3)** | ⏳ Coming soon - Memory-intensive workload optimization |
| | **LocalTaskQueue (V6)** | ⏳ Coming soon - Lock-optimized ThreadManager |

### Migration Guides

- [ThreadManager Naming Refactoring](https://acalsim.playlab.tw/docs/MIGRATION_GUIDE) - November 2025 update from V# to descriptive names

---

## Popular Topics

### 🎯 **How do I...**

- **Build my first simulator?** → Start with [For Users Guide](https://acalsim.playlab.tw/docs/getting-started/for-users)
- **Connect multiple simulators?** → See [SimChannel Documentation](https://acalsim.playlab.tw/docs/for-users/simchannel)
- **Configure ThreadManager for performance?** → Check [ThreadManager Selection](https://acalsim.playlab.tw/docs/MIGRATION_GUIDE#choosing-the-right-threadmanager)
- **Debug multi-threaded simulations?** → Use [Logging](https://acalsim.playlab.tw/docs/for-users/logging) and [Tracing](https://acalsim.playlab.tw/docs/for-users/simtrace-container)
- **Profile performance?** → Enable [SimTrace](https://acalsim.playlab.tw/docs/for-users/simtrace-container) and visualize in Chrome

### ⚡ **Performance Optimization**

- **Choose the right ThreadManager**: [Selection Guide](https://acalsim.playlab.tw/docs/MIGRATION_GUIDE#choosing-the-right-threadmanager)
  - **PriorityQueue (V1)**: Default, best for sparse activation patterns (e.g., DGXSim)
  - **PrebuiltTaskList (V3)**: Best for memory-intensive workloads (e.g., GPUSim)
  - **LocalTaskQueue (V6)**: Lock-optimized version of PriorityQueue
- **Object Pooling**: Use [RecycleContainer](https://acalsim.playlab.tw/docs/for-users/recycle-container) for zero-allocation hot paths
- **Logging**: Disable in production with `--no-logs` flag

### 🐛 **Common Issues**

- **Build errors?** → Check [Environment Setup](https://acalsim.playlab.tw/docs/getting-started/env-setup) and verify dependencies
- **Simulation hangs?** → Enable [MT_DEBUG logging](https://acalsim.playlab.tw/docs/for-users/logging#multi-threaded-debug-logging)
- **Race conditions?** → Review [ThreadManager documentation](https://acalsim.playlab.tw/docs/for-developers/thread-manager)
- **Memory leaks?** → Ensure proper use of [RecycleContainer](https://acalsim.playlab.tw/docs/for-users/recycle-container)

---

## Reference Projects

Learn by example from test projects:

| Project | Location | Description |
|---------|----------|-------------|
| **testAccelerator** | `src/testAccelerator/` | Simple accelerator with NoC, cache, and memory |
| **testSimPort** | `src/testSimPort/` | Demonstrates SimPort usage and timing |
| **testChannel** | `src/testChannel/` | Inter-simulator communication patterns |
| **testSTSystemC** | `src/testSTSystemC/` | SystemC integration example |

---

## External Resources

- **GitHub Repository**: https://github.com/ACAL-Playlab/ACALSim
- **Issue Tracker**: https://github.com/ACAL-Playlab/ACALSim/issues

---

## Documentation Versioning

- **Current Version**: 0.1.0
- **Last Updated**: November 2025
- **Major Changes**:
  - November 2025: ThreadManager naming refactoring (V# → descriptive names)
  - November 2025: Added migration guide
  - November 2025: ExitEvent memory leak fix

---

## Contributing to Documentation

Found an error or want to improve the documentation? Contributions are welcome!

1. Documentation source: `docs/` directory
2. Follow the Apache 2.0 license header format
3. Use clear examples and diagrams
4. Test all code snippets

---

## Need Help?

- 📖 **Can't find what you're looking for?** Check the [Getting Started](./getting-started/for-users.md) guide
- 🐛 **Found a bug?** Report it on [GitHub Issues](https://github.com/ACAL-Playlab/ACALSim/issues)
- 💬 **Have a question?** Check existing documentation or open a discussion

---

**Happy Simulating with ACALSim! 🚀**
