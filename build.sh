#!/bin/bash
set -e

echo "Installing Git LFS..."
apt-get update
apt-get install -y git-lfs

echo "Pulling Git LFS files..."
git lfs pull

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Build complete!"
