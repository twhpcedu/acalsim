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

# QEMU Boot Success Report

## Summary
Successfully booted RISC-V Linux in QEMU with buildroot rootfs and reached login prompt.

## Configuration
- **QEMU Version**: 7.0.0 with virtio-sst device
- **Kernel**: Linux 6.18.0-rc6 (built from v6.1 tag)
- **Rootfs**: Buildroot 2024.02 (89M compressed initramfs)
- **Architecture**: RISC-V 64-bit (rv64)
- **Memory**: 8GB
- **CPUs**: 4 cores

## Boot Command
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

## Boot Configuration Details
```bash
QEMU_BIN=/home/user/qemu-build/qemu/build/qemu-system-riscv64
KERNEL=/home/user/linux/arch/riscv/boot/Image
INITRAMFS=/home/user/initramfs-buildroot.cpio.gz
  -> Links to: /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz
```

## Verified Working Features
✅ OpenSBI firmware initialization
✅ Linux kernel boot (6.18.0-rc6)
✅ 4 CPUs brought online successfully
✅ VirtIO SST device initialization
✅ VirtIO network device (virtio-net)
✅ Buildroot initramfs extraction (91MB)
✅ System services (syslogd, klogd, udev)
✅ Network configuration via DHCP (10.0.2.15)
✅ IPv6 auto-configuration
✅ SSH server (dropbear) started
✅ Login prompt displayed

## Boot Log Highlights
```
[    2.468978] Freeing unused kernel image (initmem) memory: 2364K
[    2.469751] Run /init as init process
Saving 256 bits of non-creditable seed for next boot
Starting syslogd: OK
Starting klogd: OK
Running sysctl: OK
Populating /dev using udev: done
Starting network: OK
Starting dhcpcd: OK
Starting dropbear sshd: OK

Welcome to ACAL Simulator RISC-V Linux
acalsim-riscv login:
```

## Network Configuration
- **IP Address**: 10.0.2.15 (DHCP)
- **Gateway**: 10.0.2.2
- **DNS**: 10.0.2.3
- **IPv6**: fec0::801f:234a:a9db:22ef/64
- **SSH**: Running on default port

## Exit QEMU
Press: `Ctrl-A` then `X`

## Next Steps
1. Test login with credentials from buildroot config
2. Verify Python and ML tools in rootfs
3. Test networking and SSH connectivity
4. Run llama inference workload

## Build Date
Successfully booted on: 2025-11-20 15:18:10 UTC
