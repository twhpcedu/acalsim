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

# PyTorch Python 3.13 Compatibility Fix

## The Problem

PyTorch 2.1.0 is not compatible with Python 3.13. The `_PyEval_EvalFrameDefault` function signature changed in Python 3.13.

Error:
```
/home/debian/pytorch/torch/csrc/dynamo/eval_frame.c:5:13: error: conflicting types for '_PyEval_EvalFrameDefault'
```

## Solutions

### Solution 1: Downgrade to Python 3.11 (Recommended)

Python 3.11 is fully supported by PyTorch 2.1.0.

```bash
# Inside Debian
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 2
sudo update-alternatives --config python3
# Choose python3.11

# Verify
python3 --version
# Should show: Python 3.11.x

# Reinstall pip for Python 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
python3.11 -m pip install numpy typing-extensions pyyaml

# Clean and rebuild PyTorch
cd /home/debian/pytorch
rm -rf build/
python3.11 setup.py install 2>&1 | tee ../pytorch-py311.log
```

### Solution 2: Use PyTorch 2.2+ (Supports Python 3.13)

Switch to a newer PyTorch version that supports Python 3.13:

```bash
cd /home/debian

# Remove old PyTorch
rm -rf pytorch

# Clone PyTorch 2.2 or later
git clone --depth 1 --branch v2.2.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Install dependencies
pip3 install typing-extensions pyyaml requests packaging

# Build with Python 3.13
export USE_SLEEF=0 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0
export BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1
python3 setup.py install 2>&1 | tee ../pytorch-2.2-py313.log
```

### Solution 3: Patch PyTorch 2.1 for Python 3.13

Apply compatibility patch to eval_frame.c:

```bash
cd /home/debian/pytorch

# Create patch
cat > /tmp/eval_frame_py313.patch << 'EOF'
--- a/torch/csrc/dynamo/eval_frame.c
+++ b/torch/csrc/dynamo/eval_frame.c
@@ -2,7 +2,11 @@
 #include <torch/csrc/utils/python_compat.h>

 #ifdef _WIN32
+#if PY_VERSION_HEX >= 0x030d0000
+  PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, struct _PyInterpreterFrame *frame, int exc) {
+#else
   PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, PyFrameObject *frame, int throwflag) {
+#endif
     Py_FatalError("Dynamo not supported on Windows");
   }
 #else
EOF

# Apply patch
patch -p1 < /tmp/eval_frame_py313.patch

# Rebuild
rm -rf build/
export USE_SLEEF=0 MAX_JOBS=4 USE_OPENBLAS=1 BUILD_TEST=0
python3 setup.py install 2>&1 | tee ../pytorch-patched-py313.log
```

### Solution 4: Disable TorchDynamo (Quick Fix)

Disable the problematic dynamo feature:

```bash
cd /home/debian/pytorch

# Disable dynamo compilation
export USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0
export BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1
export USE_SLEEF=0
export BUILD_CAFFE2_OPS=0  # Disables problematic eval_frame.c

# Clean and rebuild
rm -rf build/
python3 setup.py install 2>&1 | tee ../pytorch-no-dynamo.log
```

---

## Recommended Approach

**For RISC-V on Debian, use Solution 1 (Python 3.11):**

1. Debian sid includes both Python 3.11 and 3.13
2. PyTorch 2.1.0 is thoroughly tested with Python 3.11
3. Python 3.13 support is still experimental in PyTorch

```bash
# Quick commands
sudo apt install -y python3.11 python3.11-dev
sudo update-alternatives --config python3  # Choose 3.11
python3 --version  # Verify 3.11
cd /home/debian/pytorch
rm -rf build/
python3 setup.py install 2>&1 | tee ../pytorch-final.log
```

---

## Alternative: Use Pre-built Wheels (If Available)

Check if Debian has pre-built PyTorch for RISC-V:

```bash
apt search python3-torch
# If found:
sudo apt install python3-torch python3-torch-cpu
```

---

## Python Version Compatibility Matrix

| PyTorch Version | Python 3.11 | Python 3.12 | Python 3.13 |
|----------------|-------------|-------------|-------------|
| 2.0.x          | ✅          | ❌          | ❌          |
| 2.1.x          | ✅          | ✅          | ❌          |
| 2.2.x          | ✅          | ✅          | ✅          |
| 2.3.x+         | ✅          | ✅          | ✅          |

---

**Created**: 2025-11-20
**Issue**: Python 3.13 incompatibility with PyTorch 2.1.0
**Recommended**: Downgrade to Python 3.11
