#!/bin/bash
# Build main.tex -> main.pdf via latexmk
# Usage:
#   ./build.sh           one-shot build
#   ./build.sh --watch   rebuild on save
#   ./build.sh --clean   remove all build artifacts
set -e
cd "$(dirname "$0")"

case "$1" in
    --clean)
        latexmk -C main.tex
        ;;
    --watch)
        latexmk -pdf -pvc -interaction=nonstopmode main.tex
        ;;
    *)
        latexmk -pdf -interaction=nonstopmode main.tex
        ;;
esac
