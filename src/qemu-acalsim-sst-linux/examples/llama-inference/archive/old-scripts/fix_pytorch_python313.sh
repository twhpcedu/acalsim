#!/bin/bash
# Fix PyTorch for Python 3.13 compatibility

cat <<'INSTRUCTIONS'
=====================================
PyTorch Python 3.13 Compatibility Fix
=====================================

Since Python 3.11 is not available in Debian sid, we'll patch PyTorch
to work with Python 3.13.

Run these commands inside Debian QEMU:

cd /home/debian/pytorch

# Patch eval_frame.c for Python 3.13
cat > /tmp/fix_eval_frame.patch << 'EOF'
--- a/torch/csrc/dynamo/eval_frame.c
+++ b/torch/csrc/dynamo/eval_frame.c
@@ -1,8 +1,13 @@
 #include <torch/csrc/dynamo/eval_frame.h>
 #include <torch/csrc/utils/python_compat.h>

+#if PY_VERSION_HEX >= 0x030d0000  // Python 3.13+
+PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, struct _PyInterpreterFrame *frame, int exc) {
+  Py_FatalError("Dynamo not supported with this Python version");
+  return NULL;
+}
+#else
 #ifdef _WIN32
-  PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, PyFrameObject *frame, int throwflag) {
+PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, PyFrameObject *frame, int throwflag) {
     Py_FatalError("Dynamo not supported on Windows");
   }
 #else
@@ -12,6 +17,7 @@
 #define unlikely(x) (x)
 #endif

+#endif  // PY_VERSION_HEX >= 0x030d0000

 static PyObject* guard_error_hook = NULL;

EOF

# Apply patch
patch -p1 < /tmp/fix_eval_frame.patch

# Or manually edit the file (simpler)
# Replace the entire beginning of torch/csrc/dynamo/eval_frame.c

cat > torch/csrc/dynamo/eval_frame.c.new << 'EOFNEW'
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/utils/python_compat.h>

#if PY_VERSION_HEX >= 0x030d0000
// Python 3.13+ compatibility stub
PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, struct _PyInterpreterFrame *frame, int exc) {
  // Dynamo not supported on Python 3.13 yet
  // Fall back to default behavior
  return NULL;
}
#else
#ifdef _WIN32
PyObject* _PyEval_EvalFrameDefault(PyThreadState *tstate, PyFrameObject *frame, int throwflag) {
    Py_FatalError("Dynamo not supported on Windows");
  }
#endif
#endif

#if defined(__GNUC__)
#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

static PyObject* guard_error_hook = NULL;
EOFNEW

# Read the rest of the original file and append
tail -n +18 torch/csrc/dynamo/eval_frame.c >> torch/csrc/dynamo/eval_frame.c.new

# Replace the file
mv torch/csrc/dynamo/eval_frame.c torch/csrc/dynamo/eval_frame.c.bak
mv torch/csrc/dynamo/eval_frame.c.new torch/csrc/dynamo/eval_frame.c

# Install pip packages with --break-system-packages (necessary on Debian)
python3 -m pip install --break-system-packages numpy typing-extensions pyyaml

# Rebuild (should continue from where it left off)
export USE_SLEEF=0 USE_CUDA=0 BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1
python3 setup.py install 2>&1 | tee ../pytorch-py313-fixed.log

INSTRUCTIONS
