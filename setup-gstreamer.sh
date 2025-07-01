#!/bin/bash

# GStreamer environment setup for video-kit project
# Run with: source ./setup-gstreamer.sh

echo "Setting up GStreamer environment..."

# GStreamer Framework paths
export DYLD_LIBRARY_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/bin:$PATH"

# Verify setup
echo "✅ GStreamer version: $(pkg-config --modversion gstreamer-1.0 2>/dev/null || echo 'Not found')"
echo "✅ Library path: $DYLD_LIBRARY_PATH"
echo "✅ PKG_CONFIG_PATH: $PKG_CONFIG_PATH"

echo "Environment ready! You can now run: cargo run --example splice_video" 