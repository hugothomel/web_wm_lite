#!/bin/bash
# Copy ORT WASM files to public/wasm/ for serving
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_DIR="$PROJECT_DIR/public/wasm"
ORT_DIR="$PROJECT_DIR/node_modules/onnxruntime-web/dist"

mkdir -p "$WASM_DIR"

echo "Copying ORT WASM files from $ORT_DIR to $WASM_DIR..."
cp "$ORT_DIR"/*.wasm "$WASM_DIR/" 2>/dev/null || true
cp "$ORT_DIR"/*.mjs "$WASM_DIR/" 2>/dev/null || true

echo "WASM files copied:"
ls -lh "$WASM_DIR/"
