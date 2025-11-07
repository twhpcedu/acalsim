#!/bin/sh

# ==============================================================================
# PyTorch Model Generator Script
# ==============================================================================
#
# Description:
#   This script generates PyTorch TorchScript models required for AI accelerator
#   simulations in the ACALSim framework. Currently generates ResNet-18 model
#   files used by testBlackBear and other neural network accelerator tests.
#
# Usage:
#   ./gen-models.sh
#
# Requirements:
#   - Python 3.x with PyTorch installed
#   - models/reset18.py script must be executable
#
# Generated Files:
#   - Creates .pt (TorchScript) model files in appropriate directories
#   - Models are used by testBlackBear for neural network inference simulation
#
# Workflow:
#   1. Navigate to project root directory
#   2. Execute models/reset18.py to generate ResNet-18 TorchScript model
#   3. Model files are saved for use in accelerator simulations
#
# Exit Codes:
#   0: Success - All models generated successfully
#   1: Failure - Directory navigation or model generation failed
#
# Author: ACAL Playlab
# Copyright: 2023-2025 Playlab/ACAL
# ==============================================================================

set -e # Exit immediately if any command fails

# Navigate to project root directory
cd "$(dirname "$0")"/../ || exit

# Generate ResNet-18 TorchScript model
# This model is used by testBlackBear AI accelerator simulation
DIR=$(dirname "$0")
"${DIR}"/models/reset18.py
