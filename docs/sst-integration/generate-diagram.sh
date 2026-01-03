#!/bin/bash
# Generate PNG diagram from PlantUML source
#
# Copyright 2023-2026 Playlab/ACAL
# Licensed under the Apache License, Version 2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUML_FILE="$SCRIPT_DIR/architecture-diagram.puml"
OUTPUT_FILE="$SCRIPT_DIR/architecture-diagram.png"

echo "==================================================================="
echo "ACALSim SST Integration - Diagram Generator"
echo "==================================================================="
echo ""

# Method 1: Check if plantuml is installed locally
if command -v plantuml &>/dev/null; then
	echo "✓ Found PlantUML installed locally"
	echo "→ Generating PNG diagram..."
	plantuml -tpng "$PUML_FILE"
	echo "✓ Generated: $OUTPUT_FILE"
	echo ""
	echo "Done! Open the PNG with:"
	echo "  open $OUTPUT_FILE"
	exit 0
fi

# Method 2: Check if Java is available and download PlantUML jar
if command -v java &>/dev/null; then
	echo "✓ Found Java installed"

	PLANTUML_JAR="$SCRIPT_DIR/plantuml.jar"

	if [ ! -f "$PLANTUML_JAR" ]; then
		echo "→ Downloading PlantUML jar..."
		curl -L -o "$PLANTUML_JAR" "https://github.com/plantuml/plantuml/releases/download/v1.2024.3/plantuml-1.2024.3.jar"
		echo "✓ Downloaded PlantUML jar"
	fi

	echo "→ Generating PNG diagram..."
	java -jar "$PLANTUML_JAR" -tpng "$PUML_FILE"
	echo "✓ Generated: $OUTPUT_FILE"
	echo ""
	echo "Done! Open the PNG with:"
	echo "  open $OUTPUT_FILE"
	exit 0
fi

# Method 3: Use PlantUML web service
echo "⚠ Neither PlantUML nor Java found locally"
echo "→ Using PlantUML web service..."
echo ""

# Encode the PlantUML file for web service
ENCODED=$(python3 -c "
import zlib
import base64
import sys

def encode_plantuml(plantuml_text):
    zlibbed_str = zlib.compress(plantuml_text.encode('utf-8'))
    compressed_string = zlibbed_str[2:-4]
    return base64.urlsafe_b64encode(compressed_string).decode('utf-8').replace('=', '')

with open('$PUML_FILE', 'r') as f:
    content = f.read()
    print(encode_plantuml(content))
")

URL="http://www.plantuml.com/plantuml/png/$ENCODED"

echo "→ Downloading PNG from PlantUML web service..."
curl -o "$OUTPUT_FILE" "$URL"
echo "✓ Generated: $OUTPUT_FILE"
echo ""
echo "Done! Open the PNG with:"
echo "  open $OUTPUT_FILE"
echo ""
echo "==================================================================="
echo "Tip: Install PlantUML locally for faster generation:"
echo "  macOS:  brew install plantuml"
echo "  Ubuntu: sudo apt install plantuml"
echo "  Or download from: https://plantuml.com/download"
echo "==================================================================="
