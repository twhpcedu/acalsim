#!/bin/bash
# Launch QEMU Debian with SST connection for LLAMA inference
#
# This wraps run_qemu_debian_dqib.sh with SST integration
#
# Copyright 2023-2025 Playlab/ACAL
# Licensed under the Apache License, Version 2.0

# Simply call the Debian QEMU script (which already has SST socket support)
exec ./run_qemu_debian_dqib.sh
