#!/bin/bash

# Generate scalable multi-rank architecture diagram
# Usage: ./generate-scalable-diagram.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DIAGRAM="scalable-multi-rank-architecture"

echo "Generating ${DIAGRAM} diagram..."

# Generate PNG (for papers and presentations)
plantuml -tpng "${DIAGRAM}.puml"

# Generate PDF (for high-quality papers)
plantuml -tpdf "${DIAGRAM}.puml"

# Generate SVG (for scalable graphics)
plantuml -tsvg "${DIAGRAM}.puml"

if [ $? -eq 0 ]; then
	echo "✅ Generated ${DIAGRAM}.png, ${DIAGRAM}.pdf, ${DIAGRAM}.svg"
	ls -lh "${DIAGRAM}.png" "${DIAGRAM}.pdf" "${DIAGRAM}.svg"
else
	echo "❌ Error generating diagram"
	exit 1
fi
