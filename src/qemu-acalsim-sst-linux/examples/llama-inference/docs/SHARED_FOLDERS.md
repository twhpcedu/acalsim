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

# Shared Folder Setup - Complete Guide

## Overview

The shared folder allows seamless file access across three layers:
- **Mac Host** → **Docker Container** → **QEMU Debian Guest**

## Folder Mapping

```
┌─────────────────────────────────────────────────────────────┐
│ Mac Host (your laptop)                                       │
│ /Users/weifen/work/acal/acalsim-workspace/projects          │
│                                                              │
│                              ↕                               │
│                                                              │
│ Docker Container (acalsim-workspace)                        │
│ /home/user/projects                                         │
│                                                              │
│                              ↕                               │
│                                                              │
│ QEMU Debian Guest (RISC-V Linux)                            │
│ /mnt/shared (auto-mounted via /etc/fstab)                   │
└─────────────────────────────────────────────────────────────┘
```

## Current Configuration

### QEMU Script Updated
File: `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/run_qemu_debian_dqib.sh`

**Changes made:**
1. Added shared folder variables:
   ```bash
   SHARED_HOST_DIR="/home/user/projects"
   SHARED_MOUNT_TAG="hostshare"
   ```

2. Added virtfs parameter to QEMU command:
   ```bash
   -virtfs local,path=$SHARED_HOST_DIR,mount_tag=$SHARED_MOUNT_TAG,security_model=passthrough,id=hostshare
   ```

### Debian Guest Auto-mount
File: `/etc/fstab` (in Debian guest)

**Entry added:**
```
hostshare /mnt/shared 9p trans=virtio,version=9p2000.L,rw,_netdev 0 0
```

This ensures the shared folder is automatically mounted on boot.

## How to Use

### Next Boot
After you restart QEMU with the updated script:

1. **Boot QEMU** (the script now includes virtfs)
2. **Login to Debian** (debian/debian or root/root)
3. **Shared folder is auto-mounted** at `/mnt/shared`

### Verify It Works

**From Mac:**
```bash
# Create a test file on Mac
echo "Hello from Mac" > /Users/weifen/work/acal/acalsim-workspace/projects/test.txt
```

**From Docker:**
```bash
docker exec -it acalsim-workspace bash
cat /home/user/projects/test.txt
# Should show: Hello from Mac
```

**From QEMU Debian:**
```bash
cat /mnt/shared/test.txt
# Should show: Hello from Mac
```

## Common Operations

### Create Files from Each Layer

**Mac:**
```bash
cd /Users/weifen/work/acal/acalsim-workspace/projects
mkdir shared-data
echo "Mac file" > shared-data/from-mac.txt
```

**Docker:**
```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/shared-data
echo "Docker file" > from-docker.txt
```

**QEMU Debian:**
```bash
cd /mnt/shared/shared-data
echo "Debian file" > from-debian.txt
```

All files will be visible from all three layers!

### Share PyTorch Code

You can now edit PyTorch code on your Mac and build it in QEMU:

**Mac (using VSCode or any editor):**
```bash
# Edit files in:
/Users/weifen/work/acal/acalsim-workspace/projects/pytorch-dev/
```

**QEMU Debian:**
```bash
cd /mnt/shared/pytorch-dev
# Build and test
python3 setup.py develop
```

### Transfer Large Files

**From Mac to QEMU:**
```bash
# Just copy to projects directory on Mac
cp ~/Downloads/large-dataset.tar.gz /Users/weifen/work/acal/acalsim-workspace/projects/

# Immediately available in QEMU
cd /mnt/shared
tar xzf large-dataset.tar.gz
```

## Troubleshooting

### Shared folder not mounted in QEMU

**Check if virtfs is working:**
```bash
# In QEMU Debian
dmesg | grep 9p
# Should show 9p initialization messages
```

**Manual mount (if auto-mount fails):**
```bash
sudo mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/shared
```

### Permission issues

**Option 1: Fix ownership in Debian**
```bash
sudo chown -R debian:debian /mnt/shared
```

**Option 2: Use sudo to write files**
```bash
sudo touch /mnt/shared/testfile
sudo chown debian:debian /mnt/shared/testfile
```

### Changes not appearing

**Force sync:**
```bash
# In Debian
sync
```

**Or remount with sync option:**
```bash
sudo umount /mnt/shared
sudo mount -t 9p -o trans=virtio,version=9p2000.L,sync hostshare /mnt/shared
```

## Best Practices

1. **Keep source code in shared folder**: Edit on Mac, build in QEMU
2. **Use subdirectories**: Organize projects, data, and builds separately
3. **Regular sync**: Run `sync` after important file operations in Debian
4. **Permissions**: Files created in Debian may have different ownership - use `chown` as needed
5. **Large files**: The shared folder works well for datasets and build artifacts

## Example: PyTorch Development Workflow

```bash
# 1. On Mac: Clone PyTorch to shared location
cd /Users/weifen/work/acal/acalsim-workspace/projects
git clone https://github.com/pytorch/pytorch pytorch-riscv
cd pytorch-riscv
git checkout v2.4.0
git submodule update --init --recursive

# 2. On Mac: Edit code with your favorite IDE/editor
code .  # Or use any editor

# 3. In QEMU Debian: Build
cd /mnt/shared/pytorch-riscv
# Apply patches and build (as per the build guide)
export USE_CUDA=0 USE_OPENBLAS=1 MAX_JOBS=4
pip3 install --break-system-packages --no-build-isolation --no-deps -v -e .

# 4. Test in QEMU
python3 -c "import torch; print(torch.__version__)"

# 5. Results are in shared folder, accessible from Mac!
```

## Restart Instructions

Since QEMU is currently running with the old configuration, you need to restart it:

1. **In QEMU console**: Press `Ctrl-A` then `X` to exit

   OR

   **Via SSH**: `sudo poweroff`

2. **Restart QEMU** with the updated script:
   ```bash
   docker exec -it acalsim-workspace bash
   cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
   ./run_qemu_debian_dqib.sh
   ```

3. **After boot, verify**:
   ```bash
   # In Debian
   ls /mnt/shared
   # Should show your projects directory contents
   ```

## Summary

✅ **Mac**: `/Users/weifen/work/acal/acalsim-workspace/projects`
✅ **Docker**: `/home/user/projects`
✅ **QEMU**: `/mnt/shared` (auto-mounted)
✅ **Real-time sync**: Changes appear immediately
✅ **Persistent**: Configured to auto-mount on boot

You can now seamlessly work across all three environments!
