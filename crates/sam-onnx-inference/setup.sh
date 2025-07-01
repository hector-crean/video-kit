#!/bin/bash
# Setup script for SAM ONNX inference environment using uv

set -e

echo "Setting up SAM ONNX inference environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create a virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv

# Activate the virtual environment and install dependencies
echo "Installing Python dependencies..."
uv pip install -e .

# Also install segment-anything from GitHub
echo "Installing segment-anything package..."
uv pip install git+https://github.com/facebookresearch/segment-anything.git

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate  # On Unix/macOS"
echo "  .venv\\Scripts\\activate     # On Windows"
echo ""
echo "To convert a SAM model to ONNX:"
echo "  uv run python scripts/convert_sam_to_onnx.py --model-type vit_b"
echo ""
echo "Or use the installed script:"
echo "  uv run convert-sam --model-type vit_b" 