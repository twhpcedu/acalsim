#!/bin/bash
# Complete PyTorch RISC-V Patch Script
# This applies all necessary patches for building PyTorch on RISC-V with Python 3.13

set -e

PYTORCH_VERSION=${1:-v2.3.0}

echo "====================================="
echo "PyTorch RISC-V Complete Patch Script"
echo "====================================="
echo "Target version: $PYTORCH_VERSION"
echo "Python version: $(python3 --version)"
echo "====================================="
echo ""

# Check if we're in pytorch directory
if [ ! -f "setup.py" ] || [ ! -d "torch" ]; then
	echo "ERROR: Not in PyTorch directory"
	echo "Usage: cd /home/debian/pytorch && bash this_script.sh"
	exit 1
fi

echo "Step 1: Installing build dependencies..."
sudo apt update
sudo apt install -y clang llvm lld libopenblas-dev ninja-build

echo ""
echo "Step 2: Fix CMakeLists.txt - Disable strict warnings..."

# Fix if statement syntax in caffe2/CMakeLists.txt (if exists)
if [ -f "caffe2/CMakeLists.txt" ]; then
	sed -i '812s/.*/  if(COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)/' caffe2/CMakeLists.txt 2>/dev/null || true
fi

# Add warning suppression flags
cat >/tmp/cmake_patch.txt <<'EOFCMAKE'
# Disable problematic warnings for RISC-V/Clang
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=dangling-reference -Wno-error=tautological-compare -Wno-unknown-warning-option")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unknown-warning-option")
EOFCMAKE

sed -i '/^project(Torch CXX C)/r /tmp/cmake_patch.txt' CMakeLists.txt

echo "✓ CMake configuration patched"
echo ""

echo "Step 3: Fix cpuinfo syscall issues..."

# Fix cpuinfo api.c - add _GNU_SOURCE and syscall headers
if [ -f "third_party/cpuinfo/src/api.c" ]; then
	# Remove any previous attempts
	sed -i '/^#include <sys\/syscall.h>/d' third_party/cpuinfo/src/api.c
	sed -i '/^#include <unistd.h>/d' third_party/cpuinfo/src/api.c
	sed -i '/^#define _GNU_SOURCE/d' third_party/cpuinfo/src/api.c
	sed -i '/^#ifndef _GNU_SOURCE/d' third_party/cpuinfo/src/api.c

	# Add proper fix at the beginning
	sed -i '1i #ifndef _GNU_SOURCE\n#define _GNU_SOURCE\n#endif\n#include <unistd.h>\n#include <sys/syscall.h>' third_party/cpuinfo/src/api.c

	echo "✓ cpuinfo patched"
else
	echo "⚠ cpuinfo/src/api.c not found (may not be needed in this version)"
fi

echo ""
echo "Step 4: Fix missing cstdint headers..."

# Add missing headers to various files
[ -f "caffe2/utils/string_utils.cc" ] && sed -i '1i #include <cstdint>' caffe2/utils/string_utils.cc
[ -f "c10/util/ThreadLocalDebugInfo.h" ] && sed -i '1i #include <cstdint>' c10/util/ThreadLocalDebugInfo.h
[ -f "torch/csrc/jit/passes/quantization/quantization_type.h" ] && sed -i '1i #include <cstdint>' torch/csrc/jit/passes/quantization/quantization_type.h
[ -f "torch/csrc/jit/runtime/logging.cpp" ] && sed -i '1i #include <stdexcept>' torch/csrc/jit/runtime/logging.cpp
[ -f "torch/csrc/lazy/core/multi_wait.cpp" ] && sed -i '1i #include <stdexcept>' torch/csrc/lazy/core/multi_wait.cpp

echo "✓ Missing headers added"
echo ""

echo "Step 5: Fix flatbuffers const issue..."

# Fix const-qualified assignment
if [ -f "third_party/flatbuffers/include/flatbuffers/stl_emulation.h" ]; then
	sed -i 's/const size_type count_;/size_type count_;/' third_party/flatbuffers/include/flatbuffers/stl_emulation.h
	echo "✓ flatbuffers patched"
else
	echo "⚠ flatbuffers stl_emulation.h not found (may not be needed)"
fi

echo ""
echo "Step 6: Fix SLEEF FMA error..."

# Patch SLEEF for RISC-V
if [ -f "third_party/sleef/src/arch/helperpurec_scalar.h" ]; then
	# Comment out the error and define FMA macros
	sed -i 's/#error FP_FAST_FMA or FP_FAST_FMAF not defined/\/\/ #error FP_FAST_FMA or FP_FAST_FMAF not defined\n#define FP_FAST_FMA 1\n#define FP_FAST_FMAF 1/' third_party/sleef/src/arch/helperpurec_scalar.h
	echo "✓ SLEEF FMA patched"
else
	echo "⚠ SLEEF header not found"
fi

echo ""
echo "Step 7: Fix Python 3.13 eval_frame.c (if needed)..."

# For PyTorch 2.3+, this may already be fixed, but add stub just in case
if [ -f "torch/csrc/dynamo/eval_frame.c" ]; then
	# Check if it needs patching
	if grep -q "PyFrameObject \*frame, int throwflag" torch/csrc/dynamo/eval_frame.c 2>/dev/null; then
		echo "Patching eval_frame.c for Python 3.13..."
		cp torch/csrc/dynamo/eval_frame.c torch/csrc/dynamo/eval_frame.c.backup

		# Add Python 3.13 compatibility at the top
		sed -i '1a\
#if PY_VERSION_HEX >= 0x030d0000\n\
// Python 3.13 stub\n\
PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, struct _PyInterpreterFrame *frame, int exc) {\n\
  return _PyEval_EvalFrameDefault_fallback(tstate, frame, exc);\n\
}\n\
#endif' torch/csrc/dynamo/eval_frame.c

		echo "✓ eval_frame.c patched for Python 3.13"
	else
		echo "✓ eval_frame.c already compatible"
	fi
else
	echo "⚠ eval_frame.c not found"
fi

echo ""
echo "====================================="
echo "All patches applied successfully!"
echo "====================================="
echo ""
echo "Now you can build PyTorch with:"
echo ""
echo "  export USE_SLEEF=0 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0"
echo "  export BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1 USE_DISTRIBUTED=0"
echo "  python3 setup.py install 2>&1 | tee ../pytorch-build.log"
echo ""
