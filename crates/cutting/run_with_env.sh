#!/bin/bash

# Set up GStreamer environment for Homebrew installation
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export GST_PLUGIN_PATH="/opt/homebrew/lib/gstreamer-1.0"

# Disable Python GI which is causing conflicts
export GST_PLUGIN_SCANNER_1_0=""

echo "Setting up GStreamer environment..."
echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
echo "GST_PLUGIN_PATH: $GST_PLUGIN_PATH"

# Check if GStreamer tools work
echo "Testing GStreamer installation..."
gst-launch-1.0 --version

echo "Running the example..."
cargo run --example run 