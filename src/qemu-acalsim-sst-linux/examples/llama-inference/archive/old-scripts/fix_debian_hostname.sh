#!/bin/bash
# Fix hostname resolution warning in Debian DQIB

echo "Fix Debian Hostname Resolution"
echo "==============================="
echo ""
echo "Run these commands inside Debian as root to fix the"
echo "'unable to resolve host debian' warning:"
echo ""
cat << 'FIXCMD'
# Add hostname to /etc/hosts
echo "127.0.0.1 debian" >> /etc/hosts

# Verify
cat /etc/hosts

# Test sudo again (should work without warning)
su - debian -c "sudo whoami"
FIXCMD
echo ""
echo "Or run this one-liner as root:"
echo ""
echo 'echo "127.0.0.1 debian" >> /etc/hosts'
