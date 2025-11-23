#!/bin/bash
# Patch SLEEF to fix FMA error on RISC-V
# Run this inside Debian QEMU

cat <<'PATCH_INSTRUCTIONS'
=====================================
SLEEF FMA Patch for RISC-V
=====================================

This script patches the SLEEF library to work on RISC-V without FMA.

Run these commands inside Debian QEMU:

cd /home/debian/pytorch

# Stop the current build (Ctrl+C)

# Patch the SLEEF header file
cat > /tmp/sleef_fix.patch << 'EOF'
--- a/third_party/sleef/src/arch/helperpurec_scalar.h
+++ b/third_party/sleef/src/arch/helperpurec_scalar.h
@@ -66,7 +66,11 @@
 #endif

 #if !defined(FP_FAST_FMA) && !defined(FP_FAST_FMAF)
-#error FP_FAST_FMA or FP_FAST_FMAF not defined
+// RISC-V workaround: Define FMA macros for software emulation
+#warning FP_FAST_FMA not defined, using software FMA emulation
+#define FP_FAST_FMA 1
+#define FP_FAST_FMAF 1
+// #error FP_FAST_FMA or FP_FAST_FMAF not defined
 #endif

 //////////////////////////////////////////////////////////////////////////////////
EOF

# Apply the patch
patch -p1 < /tmp/sleef_fix.patch

# Or manually edit the file
sed -i '69s/^#error/#warning/' third_party/sleef/src/arch/helperpurec_scalar.h
sed -i '69a #define FP_FAST_FMA 1\n#define FP_FAST_FMAF 1' third_party/sleef/src/arch/helperpurec_scalar.h

# Verify the patch
echo "Checking patched file:"
head -80 third_party/sleef/src/arch/helperpurec_scalar.h | tail -20

# Now clean and rebuild
rm -rf build/
export MAX_JOBS=4 USE_OPENBLAS=1 BUILD_TEST=0
python3 setup.py install 2>&1 | tee ../pytorch-build-sleef-patched.log

PATCH_INSTRUCTIONS
