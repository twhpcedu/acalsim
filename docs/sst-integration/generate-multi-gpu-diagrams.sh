#!/bin/bash
# Generate Hybrid Multi-GPU Architecture Diagrams
#
# Copyright 2023-2026 Playlab/ACAL
# Licensed under the Apache License, Version 2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================================================="
echo "Generating Hybrid Multi-GPU Architecture Diagrams"
echo "==================================================================="
echo ""

cd "$SCRIPT_DIR"

# Check if plantuml is available
if ! command -v plantuml &>/dev/null; then
	echo "Error: plantuml not found"
	echo ""
	echo "Please install PlantUML:"
	echo "  Ubuntu: sudo apt install plantuml"
	echo "  macOS:  brew install plantuml"
	echo ""
	exit 1
fi

# Generate diagrams
echo "→ Generating architecture diagram..."
plantuml -tpng hybrid-multi-gpu-architecture.puml
echo "  ✓ hybrid-multi-gpu-architecture.png"

echo "→ Generating sequence diagram..."
plantuml -tpng hybrid-multi-gpu-sequence.puml
echo "  ✓ hybrid-multi-gpu-sequence.png"

echo "→ Generating topology diagram..."
plantuml -tpng hybrid-multi-gpu-topology.puml
echo "  ✓ hybrid-multi-gpu-topology.png"

echo ""
echo "==================================================================="
echo "✓ All diagrams generated successfully!"
echo "==================================================================="
echo ""
echo "Generated files:"
echo "  • hybrid-multi-gpu-architecture.png"
echo "  • hybrid-multi-gpu-sequence.png"
echo "  • hybrid-multi-gpu-topology.png"
echo ""
echo "View documentation: HYBRID_MULTI_GPU_DIAGRAMS.md"
echo "==================================================================="
