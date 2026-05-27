#!/bin/bash
set -e

echo "======================================="
echo "Building Tomato Leaf Disease API"
echo "======================================="

echo "Python version:"
python --version

echo ""
echo "Installing system dependencies..."
apt-get update
apt-get install -y git-lfs

echo ""
echo "Pulling Git LFS files..."
git lfs pull

echo ""
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================="
echo "Build complete!"
echo "======================================="

