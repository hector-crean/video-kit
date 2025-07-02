# Video Kit - Tools for preprocessing videos




## 🏗️ Architecture

Video Kit is organized as a Rust workspace with specialized crates for different aspects of video processing:

```
video-kit/
├── cli/                     # Command-line interface
├── crates/
│   ├── cutting/             # Core video cutting and manipulation
│   ├── subtitles/           # speech analysis and subtitle extraction using Deepgram
│   ├── mask/                # Image masking and outline extraction
│   ├── gpu_processor_burn/  # GPU processing with Burn framework
│   ├── gpu_processor_bevy/  # GPU processing with Bevy engine
│   ├── sam-onnx-inference/  # Segment Anything Model (ONNX)
│   ├── sam2/                # Segment Anything Model 2 (Python integration)
│   ├── video_kit_common/    # Shared types and utilities
│   └── deepgram-rust-sdk/   # Official Deepgram SDK for speech processing
```

## 📦 Quick Start

### Prerequisites

```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu/Debian

# Set up Deepgram API key (for AI features)
export DEEPGRAM_API_KEY="your_deepgram_api_key"
```

### CLI Usage

```bash
# Clone and build
git clone <repository-url>
cd video-kit
cargo build --release
```



## 🎯 Crate Overview

### 🔧 [CLI](cli/README.md)
**Command-line video processing tool**
- Smart video processing with natural break detection
- Configuration-driven workflows (JSON/TOML)
- Integration with all video-kit crates
- Batch processing capabilities

### ✂️ [Cutting](crates/cutting/readme.md)
**Core video processing library**
- Multi-backend support (FFmpeg, GStreamer)
- High-level operations (splice, reverse, loop, sequentise)
- Streaming API for complex pipelines

### 🧠 [Subtitles](crates/subtitles/README.md)
**AI-powered speech processing**
- Deepgram integration for high-accuracy transcription
- Smart video cutting that respects speech boundaries
- Multiple output formats (SRT, VTT, JSON)
- Speaker identification and language detection

### 🎨 [Mask](crates/mask/README.md)
**Image masking and outline extraction**
- Hole detection and complex shape handling
- Pipeline architecture for composable processing

### ⚡ [GPU Processor (Burn)](crates/gpu_processor_burn/README.md)
**GPU image processing**
- Custom compute shaders and kernels

### 🎮 [GPU Processor (Bevy)](crates/gpu_processor_bevy/README.md)
**Game engine-powered image processing**
- Real-time processing capabilities
- Interactive applications and live processing

### 🔍 [SAM ONNX Inference](crates/sam-onnx-inference/README.md)
**Segment Anything Model implementation (WIP)**
- Native Rust implementation with Burn
- ONNX compatibility for PyTorch models
- Multiple prompt types (points, boxes, masks)

### 📹 [SAM2](crates/sam2/README.md)
**Segment Anything Model 2 integration**
- Python interop for SAM2 video processing
- Video object segmentation and tracking

### 🎤 [Deepgram Rust SDK](crates/deepgram-rust-sdk/README.md)
**Official Deepgram speech recognition SDK**


## 🎬 Use Cases

## 🔧 Configuration Examples

### Basic Configuration (JSON)
```json
{
  "path": "input.mp4",
  "output_dir": "output",
  "clips": [
    {
      "name": "intro",
      "description": "Introduction segment",
      "operation": {
        "type": "splice",
        "params": {
          "segments": [[0.0, 30.0]]
        }
      }
    }
  ]
}
```



## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com) for advanced speech recognition
- [Burn](https://github.com/tracel-ai/burn) for GPU ML framework
- [Bevy](https://bevyengine.org) for game engine integration
- [Meta AI](https://ai.meta.com) for Segment Anything Models
- FFmpeg and GStreamer communities for multimedia processing

---

