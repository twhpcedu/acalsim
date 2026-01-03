#!/bin/bash

# Generate PNG diagrams for multi-rank architecture documentation
# Usage: ./generate-multi-rank-diagrams.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Generating multi-rank architecture diagrams..."

# Check if plantuml is available
if ! command -v plantuml &>/dev/null; then
	echo "Error: plantuml not found. Please install it:"
	echo "  brew install plantuml      # macOS"
	echo "  sudo apt install plantuml  # Ubuntu"
	exit 1
fi

# Generate PNG diagrams
echo "Generating multi-rank-simple.png..."
plantuml -tpng multi-rank-simple.puml

echo "Generating multi-rank-minimal.png..."
plantuml -tpng multi-rank-minimal.puml

echo "Generating multi-rank-layers.png..."
plantuml -tpng multi-rank-layers.puml

echo "Generating multi-rank-scalable-architecture.png..."
plantuml -tpng multi-rank-scalable-architecture.puml

echo "Generating multi-rank-deployment.png..."
plantuml -tpng multi-rank-deployment.puml

# Also generate PDF for paper
echo "Generating PDF versions for paper..."
plantuml -tpdf multi-rank-simple.puml
plantuml -tpdf multi-rank-minimal.puml
plantuml -tpdf multi-rank-layers.puml
plantuml -tpdf multi-rank-scalable-architecture.puml
plantuml -tpdf multi-rank-deployment.puml

echo ""
echo "✅ All diagrams generated successfully!"
echo ""
echo "Generated files:"
echo "  📄 For IEEE 2-column paper (RECOMMENDED):"
echo "     - multi-rank-simple.png/pdf      (2 ranks, 4 GPUs)"
echo "     - multi-rank-minimal.png/pdf     (2 ranks, most compact)"
echo ""
echo "  📄 For presentation/detailed documentation:"
echo "     - multi-rank-layers.png/pdf      (layered architecture)"
echo "     - multi-rank-scalable-architecture.png/pdf  (3 ranks, 12 GPUs)"
echo "     - multi-rank-deployment.png/pdf  (deployment example)"
echo ""
echo "For paper (LaTeX):"
echo "  Use .pdf files with \includegraphics[width=\\textwidth]{...}"
echo ""
echo "For web/GitHub:"
echo "  Use .png files in Markdown"
