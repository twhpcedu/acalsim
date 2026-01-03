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

# Configure Sudo in Debian DQIB RISC-V

## Quick Setup (Run Inside Debian as root)

```bash
# Boot Debian DQIB
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh

# Login as root (password: root)
# Then run this one-liner:
apt update && \
apt install -y sudo && \
usermod -aG sudo debian && \
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && \
chmod 440 /etc/sudoers.d/debian && \
echo "✓ Sudo configured successfully"
```

---

## Step-by-Step Instructions

### 1. Boot Debian DQIB

```bash
docker exec -it acalsim-workspace bash

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh
```

### 2. Login as Root

```
Debian GNU/Linux bookworm/sid debian ttyS0

debian login: root
Password: root
```

### 3. Install Sudo Package

```bash
# Update package lists
apt update

# Install sudo
apt install -y sudo

# Verify installation
sudo --version
```

### 4. Add debian User to sudo Group

```bash
# Add debian to sudo group
usermod -aG sudo debian

# Verify group membership
groups debian
# Should show: debian : debian sudo
```

### 5. Configure Passwordless Sudo (Optional)

```bash
# Create sudoers file for debian user
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian

# Set correct permissions
chmod 440 /etc/sudoers.d/debian

# Verify syntax
visudo -c
```

### 6. Test Sudo Access

```bash
# Switch to debian user
su - debian

# Test sudo
sudo whoami
# Should print: root

# Test without password
sudo apt update
```

---

## Alternative: Configure Root Sudo Access

If you want to use root with sudo:

```bash
# Root already has full privileges, but to explicitly use sudo:
apt update && apt install -y sudo

# Root can use sudo without additional configuration
sudo whoami
```

---

## Verification Commands

```bash
# Check sudo is installed
dpkg -l | grep sudo

# Check debian user groups
id debian

# Check sudoers configuration
cat /etc/sudoers.d/debian

# Test sudo access as debian
su - debian -c "sudo whoami"
```

---

## Sudo Configuration Options

### Option 1: Sudo with Password Prompt
```bash
# Adds debian to sudo group (will prompt for password)
usermod -aG sudo debian
```

### Option 2: Passwordless Sudo (Recommended for Development)
```bash
# Allows debian to use sudo without password
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian
chmod 440 /etc/sudoers.d/debian
```

### Option 3: Specific Command Sudo
```bash
# Allow specific commands without password
echo "debian ALL=(ALL) NOPASSWD: /usr/bin/apt, /usr/bin/apt-get" > /etc/sudoers.d/debian
chmod 440 /etc/sudoers.d/debian
```

---

## Post-Configuration Usage

### Install Packages as debian User

```bash
# Login as debian (password: debian)
su - debian

# Install packages with sudo
sudo apt update
sudo apt install -y python3-pip build-essential cmake git

# Install Python packages
pip3 install numpy torch
```

### Run Commands as Root

```bash
# As debian user
sudo systemctl status ssh
sudo journalctl -f
sudo apt upgrade -y
```

---

## Troubleshooting

### Error: "debian is not in the sudoers file"

```bash
# Login as root
su - root

# Add debian to sudo group again
usermod -aG sudo debian

# Or create sudoers file manually
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian
chmod 440 /etc/sudoers.d/debian
```

### Error: "sudo: command not found"

```bash
# Install sudo as root
apt update
apt install -y sudo
```

### Error: "syntax error in sudoers file"

```bash
# Check syntax
visudo -c

# Fix permissions
chmod 440 /etc/sudoers.d/debian

# Verify content
cat /etc/sudoers.d/debian
```

---

## Complete Setup Script

Save this as a script to run inside Debian:

```bash
#!/bin/bash
# setup_sudo.sh - Run this inside Debian DQIB as root

set -e

echo "Installing sudo..."
apt update
apt install -y sudo

echo "Adding debian user to sudo group..."
usermod -aG sudo debian

echo "Configuring passwordless sudo..."
cat > /etc/sudoers.d/debian << 'EOF'
debian ALL=(ALL) NOPASSWD:ALL
EOF

chmod 440 /etc/sudoers.d/debian

echo "Verifying configuration..."
visudo -c

echo "Testing sudo access..."
su - debian -c "sudo whoami"

echo ""
echo "✓ Sudo configured successfully!"
echo ""
echo "You can now login as debian and use sudo:"
echo "  su - debian"
echo "  sudo apt install <package>"
```

---

## Next Steps After Sudo Setup

Once sudo is configured, you can proceed with PyTorch installation:

```bash
# Login as debian
su - debian

# Install build dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    ninja-build

# Build PyTorch (see BUILD_PYTORCH_IN_QEMU.md)
cd /home/debian
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=0
python3 setup.py install
```

---

## SSH Access with Sudo

The DQIB image has SSH forwarding configured (port 2222 → 22):

```bash
# From host machine
ssh -p 2222 debian@localhost

# Inside SSH session, use sudo
sudo apt update
sudo apt install <package>
```

---

**Created**: 2025-11-20
**For**: Debian DQIB RISC-V (dqib_riscv64-virt)
**Default Users**: root/root, debian/debian
**Related Docs**: run_qemu_debian_dqib.sh, BUILD_PYTORCH_IN_QEMU.md
