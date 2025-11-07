# Migration Guide: ThreadManager Naming Refactoring

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

## Overview

As of November 2025, ACALSim has refactored the ThreadManager naming to use descriptive names instead of version numbers. This makes it easier to understand which ThreadManager is appropriate for your use case.

## What Changed

### Production ThreadManagers

The following production-ready ThreadManagers have been renamed:

| Old Name | New Name | Backward Compatible Alias |
|----------|----------|---------------------------|
| V1 | **PriorityQueue** | V1 |
| V2 | **Barrier** | V2 |
| V3 | **PrebuiltTaskList** | V3 |
| V6 | **LocalTaskQueue** | V6 |

### Experimental ThreadManagers

The following experimental ThreadManagers have been moved to `include/sim/experimental/`:

- **V4**: Dedicated thread per simulator (no thread pooling)
- **V5**: Simplified V1 with active bit mask removed
- **V7**: Hybrid C++20 barriers + pre-built task lists
- **V8**: Experimental refinement of V3

⚠️ **These are research prototypes and NOT recommended for production use.**

## Migration Path

### For Command-Line Arguments

**Option 1: Use new descriptive names (recommended)**
```bash
# Old
./my_simulation --threadmanager 1
./my_simulation --threadmanager 3

# New (recommended)
./my_simulation --threadmanager PriorityQueue
./my_simulation --threadmanager PrebuiltTaskList
```

**Option 2: Continue using numeric aliases (backward compatible)**
```bash
# Still works!
./my_simulation --threadmanager 1
./my_simulation --threadmanager 3
```

### For Configuration Files

**Old configuration:**
```json
{
  "threadmanager": "1"
}
```

**New configuration (recommended):**
```json
{
  "threadmanager": "PriorityQueue"
}
```

**Backward compatible (still works):**
```json
{
  "threadmanager": "V1"
}
```

### For Source Code

**If you're using C++ enum values:**

```cpp
// Old
ThreadManagerVersion version = ThreadManagerVersion::V1;

// New (recommended)
ThreadManagerVersion version = ThreadManagerVersion::PriorityQueue;

// Backward compatible (still works)
ThreadManagerVersion version = ThreadManagerVersion::V1;
```

## Choosing the Right ThreadManager

### Quick Reference

| Use Case | Recommended ThreadManager |
|----------|---------------------------|
| Default / Don't know | **PriorityQueue** (V1) |
| Sparse activation patterns (e.g., DGXSim) | **PriorityQueue** (V1) |
| Memory-intensive workloads (e.g., GPUSim) | **PrebuiltTaskList** (V3) |
| Lock contention with V1 | **LocalTaskQueue** (V6) |
| C++20 barrier synchronization | **Barrier** (V2) |

### Decision Tree

1. **Start with PriorityQueue** - It's the default and works well for most cases
2. **Profile your simulation** - Use ACALSim's profiling tools
3. **Check for bottlenecks**:
   - High memory usage? → Try **PrebuiltTaskList**
   - Lock contention in task queue? → Try **LocalTaskQueue**
4. **Run experiments** - Compare performance with different ThreadManagers

## What's Not Affected

✅ **No breaking changes if you:**
- Use numeric aliases (1, 2, 3, 6)
- Use V# aliases (V1, V2, V3, V6)
- Don't explicitly specify a ThreadManager (default is still PriorityQueue/V1)

## New Features

### Added to Regression Tests

The refactoring also added comprehensive regression testing for:
- **PrebuiltTaskList** (V3) - previously missing!
- **LocalTaskQueue** (V6) - previously missing!

All production ThreadManagers are now validated in the regression test suite.

### Bug Fixes

The following critical bug fix from HPCSim was backported:

- **ExitEvent memory leak fix** - Prevents 197+ ExitEvent objects from leaking during simulation cleanup

## Directory Structure Changes

```
include/sim/
├── ThreadManagerV1/         # PriorityQueue (production)
├── ThreadManagerV2/         # Barrier (production)
├── ThreadManagerV3/         # PrebuiltTaskList (production)
├── ThreadManagerV6/         # LocalTaskQueue (production)
├── experimental/            # NEW: Experimental versions
│   ├── README.md
│   ├── ThreadManagerV4/
│   ├── ThreadManagerV5/
│   ├── ThreadManagerV7/
│   └── ThreadManagerV8/
└── utils/                   # NEW: Shared utilities
    └── ConcurrentTaskQueue.hh
```

## References

- [ThreadManager Documentation](./for-developers/thread-manager.md)
- [PriorityQueue (V1) Documentation](./for-developers/thread-manager-v1.md)
- [Experimental ThreadManagers README](../include/sim/experimental/README.md)

## Support

If you encounter any issues during migration:

1. Check that your command-line arguments use valid ThreadManager names
2. Verify regression tests pass: `make regression`
3. Review the experimental ThreadManager README if using V4, V5, V7, or V8
4. Report issues at: https://github.com/anthropics/claude-code/issues

---

**Last Updated:** November 6, 2025
