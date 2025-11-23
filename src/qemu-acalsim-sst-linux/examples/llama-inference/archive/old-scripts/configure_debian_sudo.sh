#!/bin/bash
# Configure sudo permissions in Debian DQIB RISC-V

# This script helps you configure sudo in the Debian DQIB image
# Run this INSIDE the Debian QEMU guest after booting

echo "Debian Sudo Configuration Script"
echo "================================"
echo ""
echo "Run these commands after logging in as root:"
echo ""
echo "1. Install sudo (if not already installed):"
echo "   apt update"
echo "   apt install -y sudo"
echo ""
echo "2. Add debian user to sudo group:"
echo "   usermod -aG sudo debian"
echo ""
echo "3. Verify sudo group membership:"
echo "   groups debian"
echo ""
echo "4. Test sudo access (login as debian user):"
echo "   su - debian"
echo "   sudo whoami"
echo ""
echo "5. (Optional) Allow sudo without password:"
echo "   echo 'debian ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/debian"
echo "   chmod 440 /etc/sudoers.d/debian"
echo ""
echo "================================"
echo ""
echo "Quick command to run as root:"
echo ""
cat << 'SUDOCMD'
apt update && \
apt install -y sudo && \
usermod -aG sudo debian && \
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && \
chmod 440 /etc/sudoers.d/debian && \
echo "✓ Sudo configured for debian user"
SUDOCMD
