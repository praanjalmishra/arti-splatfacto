#!/bin/bash
set -e

echo "Building ArtiSplatfacto Docker image..."
docker compose build

echo ""
echo "✅ Build complete!"
echo ""
