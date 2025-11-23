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

# ACALSIM QEMU Debian RISC-V Workspace

Pre-configured QEMU environment with Debian RISC-V and PyTorch 2.4.

## Quick Start

### Option 1: Use Pre-built Docker Image (Recommended)

```bash
# Pull from Docker Hub
docker pull mibojobo/debian-pytorch-workspace:latest

# Run the container
docker run -it --name pytorch-workspace \
    -p 2222:2222 \
    mibojobo/debian-pytorch-workspace:latest

# Inside container, start QEMU
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh

# From host, SSH into QEMU
ssh -p 2222 debian@localhost
# Password: debian

# Verify PyTorch
python3 -c "import torch; print(torch.__version__)"
```

### Option 2: Build from Scratch

See [docs/BUILD_FROM_SCRATCH.md](docs/BUILD_FROM_SCRATCH.md)

## What's Included

- ✅ **Debian RISC-V** (sid/unstable) with 100GB virtual disk
- ✅ **Python 3.13.9**
- ✅ **PyTorch 2.4.0** (compiled for RISC-V with all patches)
- ✅ **QEMU 7.0.0** RISC-V emulator
- ✅ **Development Tools**: GCC, Clang, CMake, Ninja, OpenBLAS
- ✅ **Shared Folder**: `/mnt/shared` auto-mounted
- ✅ **SSH Access**: Port 2222

## Directory Structure

```
.
├── README.md                       # This file
├── run_qemu_debian_dqib.sh        # Main script to boot Debian
├── run_qemu_initramfs.sh          # Boot with initramfs
├── run_qemu.sh                    # General QEMU launcher
├── scripts/                       # Setup and build scripts
│   ├── download_debian_image.sh
│   ├── install_pytorch_from_source.sh
│   ├── setup_debian_riscv.sh
│   ├── setup_buildroot_python.sh
│   └── setup_persistent_*.sh
├── llama/                         # LLaMA inference example
│   ├── llama_inference.py
│   ├── llama_sst_backend.py
│   ├── sst_config_llama.py
│   └── run_sst.sh
├── docs/                          # Documentation
└── archive/                       # Old/deprecated files
```

## Usage

### Boot Debian QEMU

```bash
./run_qemu_debian_dqib.sh
```

**Login credentials:**
- User: `debian` / `debian`
- Root: `root` / `root`

**SSH access:**
```bash
ssh -p 2222 debian@localhost
```

### Access Shared Folder

Files in `/home/user/projects` (Docker) are accessible at `/mnt/shared` (QEMU).

**Example:**
```bash
# On host Mac
echo "test" > /Users/weifen/work/acal/acalsim-workspace/projects/test.txt

# In QEMU Debian
cat /mnt/shared/test.txt
```

### Test PyTorch

```bash
# SSH into QEMU
ssh -p 2222 debian@localhost

# Run Python test
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y

print(f"Test: {z.tolist()}")
print("✓ PyTorch is working!")
EOF
```

### Shutdown QEMU

```bash
# From within QEMU
sudo poweroff

# Or from QEMU console
# Press: Ctrl-A, then X
```

## System Requirements

- **Docker**: Docker Desktop or Docker Engine
- **Disk Space**: 100GB free
- **RAM**: 32GB recommended (minimum 16GB)
- **CPU**: 4+ cores recommended

## Documentation

- **[Build from Scratch](docs/BUILD_FROM_SCRATCH.md)** - Complete build guide
- **[Shared Folders](docs/SHARED_FOLDERS.md)** - Setup and usage
- **[PyTorch Guide](docs/PYTORCH_RISCV.md)** - PyTorch on RISC-V
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues

## LLaMA Inference Example

This directory also contains a LLaMA inference example using SST backend:

```bash
cd llama/
./run_sst.sh
```

See `llama/README.md` for details.

## Advanced Usage

### Mount Additional Folders

Edit `run_qemu_debian_dqib.sh` and add more virtfs mounts:

```bash
-virtfs local,path=/host/path,mount_tag=mytag,security_model=passthrough,id=myid
```

Then in QEMU:
```bash
sudo mount -t 9p -o trans=virtio mytag /mnt/mydir
```

### Increase Memory/CPU

Edit `run_qemu_debian_dqib.sh`:
```bash
-smp 8       # 8 CPU cores
-m 64G       # 64GB RAM
```

### Network Port Forwarding

Add more port forwards in `run_qemu_debian_dqib.sh`:
```bash
-netdev user,id=net0,hostfwd=tcp:127.0.0.1:8080-:80,hostfwd=tcp:127.0.0.1:2222-:22
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Issues

Report issues at: https://github.com/yourusername/acalsim-debian-pytorch-workspace/issues

## License

[Your License Here]

## Citation

If you use this in your research:

```bibtex
@software{acalsim_pytorch_workspace,
  title = {ACALSIM Debian PyTorch RISC-V Workspace},
  author = {Your Name},
  year = {2025},
  url = {https://hub.docker.com/r/mibojobo/debian-pytorch-workspace}
}
```

## Credits

- **Debian RISC-V**: Debian Quick Image Builder (DQIB)
- **QEMU**: QEMU Project
- **PyTorch**: PyTorch Team
- **RISC-V**: RISC-V International

## Version History

### v1.0.0 (2025-11-23)
- Initial release
- Debian RISC-V (sid)
- Python 3.13.9
- PyTorch 2.4.0
- QEMU 7.0.0
- Shared folder support
- SSH access configured

---

**Status**: Production Ready ✅
**Last Updated**: 2025-11-23
