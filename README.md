# Video Kit - Tools for preprocessing videos

<video width="600" controls>
  <source src="assets/clickable_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<small>
Set of tools developed for turning video animations into interactive presentations, but the tools have wider applicability.

The initial ambition was to use segment anything 2 to generate masks for objects within a video. We can then setup a 
<a href="https://github.com/hector-crean/ran/blob/ab68788e8c4b30dd1ecbcc4dd8289601d0fc3f4a/src/components/webgpu-canvas.tsx#L211">compute shader pipeline</a>, 
where we readback the colour of the mask we are clicking on, and use this to infer the object we are interacting with. 

This can in turn be used to drive UI effects within a fragment shader. 
</small>

Where possible the tools have to be 'backend agnostic'. The `cutting` crate for instance derives a driver trait, which has been implemented for ffmpeg and gstreamer,
but could be implemented for any 'backend'. 

```rs
pub trait Driver {
    /// The type of source this driver works with
    type Source: Source;
    /// The type of sink this driver works with  
    type Sink: Sink;
    /// The internal representation of a video/audio stream with operations applied
    type Stream: Clone;

    /// Load a source into the driver's internal stream representation
    fn load(&self, source: &Self::Source) -> Result<Self::Stream, DriverError>;
    
    /// Apply a splice operation to the stream, returning a new stream
    fn splice(&self, stream: Self::Stream, segments: &[Range<f64>]) -> Result<Self::Stream, DriverError>;
    
    /// Apply frame extraction to the stream (changes the stream to image sequence)
    fn extract_frames(&self, stream: Self::Stream, sequentise: &Sequentise) -> Result<Self::Stream, DriverError>;
    
    /// Apply reverse operation to the stream
    fn reverse(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError>;
    
    /// Apply loop creation to the stream
    fn create_loop(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError>;
    
    /// Materialize the stream to a sink (execute the pipeline)
    fn save(&self, stream: Self::Stream, sink: &Self::Sink) -> Result<(), DriverError>;
}

```
I've used this pattern elsewhere when making a (rich text editor)[https://github.com/hector-crean/bluebook/blob/c78fbcfee24d173a331fa6f94e35786fa6560ccf/bluebook_core/src/text_buffer.rs#L65], or
when creating APIs, and (not wanting to commit to a particular database)[https://github.com/hector-crean/crayon/blob/b7ec8989d650756f3b15af05c99504d9acbeb3ad/server/src/lib.rs#L104]. 

This used of traits can sometimes overcomplicate things, but does provide useful agnosticism.

Some thought has also been put to make the crates easily convertible into an MCP, which we have in fact done for the `mask` crate (crates/mask/src/mcp/mod.rs). Many of use now code using editors which can act as MCP
clients, and do tool calling. It's very useful to be able to do tool calling with our custom tools right from our editor. I've also included a `cli` crate, which aims to expose all the tools via the command line





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

