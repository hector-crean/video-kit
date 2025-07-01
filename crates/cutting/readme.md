# Cutting - Intelligent Video Processing Library

A powerful, backend-agnostic video processing library for Rust that provides high-level operations for video cutting, splicing, and manipulation. Features a trait-based architecture supporting multiple backends (FFmpeg, GStreamer) and intelligent subtitle-based cutting capabilities.

## 🚀 Features

- **Multi-Backend Support**: Pluggable backends (FFmpeg, GStreamer) via trait system
- **Intelligent Cutting**: AI-powered cutting that respects speech boundaries using subtitle analysis
- **High-Level Operations**: Simple commands for common video processing tasks
- **Streaming API**: Low-level streaming interface for complex pipeline building
- **Timeline Operations**: Splice, sequence extraction, reverse, and loop creation
- **Configuration-Driven**: JSON/TOML configuration support for complex workflows
- **Error Handling**: Comprehensive error types with detailed context

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs           // Main API and Runner orchestrator
├── driver/          // Backend implementations
│   ├── mod.rs       // Driver trait definitions
│   ├── ffmpeg/      // FFmpeg backend implementation
│   └── gstreamer/   // GStreamer backend implementation
└── sources.rs       // Source and Sink abstractions
```

### Design Patterns

- **Strategy Pattern**: Interchangeable video processing backends
- **Builder Pattern**: Fluent API for pipeline construction
- **Command Pattern**: High-level operation encapsulation
- **Pipeline Pattern**: Streaming operations for complex processing

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
cutting = { version = "0.1", features = ["ffmpeg"] }
```

### Basic Usage

```rust
use cutting::{Runner, CutVideoOperation};

// Create a simple FFmpeg-based runner
let runner = Runner::ffmpeg_default("input.mp4", "output.mp4")?;

// Execute a splice operation
runner.execute(CutVideoOperation::Splice {
    segments: vec![0.0..30.0, 60.0..90.0]
})?;
```

### Advanced Pipeline Usage

```rust
use cutting::{Runner, driver::ffmpeg::FFmpegDriver};

let runner = Runner::ffmpeg_default("input.mp4", "output.mp4")?;

// Use the streaming API for complex operations
runner.execute_stream(|stream, driver| {
    stream
        .splice(driver, &[10.0..50.0])?
        .reverse(driver)?
        .create_loop(driver)
})?;
```

## 🎬 Supported Operations

### High-Level Commands

```rust
pub enum CutVideoOperation {
    /// Splice video into segments based on timeline
    Splice { segments: Vec<Range<f64>> },
    
    /// Extract clip as sequence of images
    Sequentise(Sequentise),
    
    /// Reverse video playback
    Reverse,
    
    /// Create seamless loop (forward + reverse)
    CreateLoop,
}
```

### Streaming Operations

The low-level streaming API provides:

- **Frame Extraction**: Extract individual frames or sequences
- **Timeline Manipulation**: Precise cutting and splicing
- **Effects Processing**: Reverse, loop, and transition effects
- **Format Conversion**: Cross-format processing pipelines

## 🔧 Backend Support

### FFmpeg Backend (Recommended)

```rust
use cutting::Runner;

// Default FFmpeg runner
let runner = Runner::ffmpeg_default("input.mp4", "output.mp4")?;

// Custom FFmpeg path
let runner = Runner::ffmpeg_with_path("input.mp4", "output.mp4", "/usr/local/bin/ffmpeg")?;
```

**Features:**
- Universal format support
- Hardware acceleration
- Professional-grade processing
- Extensive codec support

### GStreamer Backend

```rust
use cutting::Runner;

// GStreamer runner (requires GStreamer installation)
let runner = Runner::gstreamer_default("input.mp4", "output.mp4")?;

// Check GStreamer availability
if Runner::gstreamer_available() {
    // Proceed with GStreamer
}
```

**Features:**
- Plugin-based architecture
- Real-time streaming
- Linux multimedia integration
- Professional pipeline construction

## 🧠 Intelligent Cutting Integration

When combined with the `subtitles` crate, cutting supports intelligent video processing:

```rust
use cutting::{Runner, CutVideoOperation};
use subtitles::{SmartClipper, SubtitleConfig};

// Find optimal cut points that respect speech boundaries
let clipper = SmartClipper::new(api_key)?;
let optimal_cuts = clipper.find_optimal_cuts(
    video_path,
    &[(10.0, 30.0), (60.0, 90.0)], // Desired segments
    &SubtitleConfig::default()
).await?;

// Apply intelligent cuts
for cut in optimal_cuts {
    let runner = Runner::ffmpeg_default("input.mp4", &format!("output_{}.mp4", cut.name))?;
    runner.execute(CutVideoOperation::Splice {
        segments: vec![cut.optimal_start..cut.optimal_end]
    })?;
}
```

## 📋 Configuration-Driven Processing

Support for JSON and TOML configuration files:

```toml
[video]
input_path = "input.mp4"
output_format = "mp4"

[[clips]]
name = "intro"
start = 0.0
end = 30.0
operation = { type = "splice", params = { segments = [[0.0, 30.0]] } }

[[clips]]
name = "outro"
start = 270.0
end = 300.0
operation = { type = "reverse" }
```

## 🔍 Error Handling

Comprehensive error types for robust applications:

```rust
use cutting::{CutError, Runner};

match runner.execute(operation) {
    Ok(()) => println!("Processing completed successfully"),
    Err(CutError::Backend(e)) => eprintln!("Backend error: {}", e),
    Err(CutError::InvalidPath(path)) => eprintln!("Invalid path: {}", path),
    Err(CutError::InvalidCommand) => eprintln!("Invalid operation"),
}
```

### Error Types

```rust
#[derive(Error, Debug)]
pub enum CutError {
    #[error("Backend error: {0}")]
    Backend(#[from] DriverError),
    #[error("Invalid command")]
    InvalidCommand,
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("GStreamer initialization failed")]
    GStreamerInit(#[from] gstreamer::glib::Error),
}
```

## 🚀 Performance

- **Multi-Backend**: Choose optimal backend for your use case
- **Streaming**: Memory-efficient processing for large videos
- **Hardware Acceleration**: GPU support via FFmpeg and GStreamer
- **Parallel Processing**: Concurrent segment processing

## 🔗 Integration

### With Subtitles Crate

```rust
use cutting::{Runner, CutVideoOperation};
use subtitles::{SmartClipper, SubtitleConfig};

// Get AI-recommended cut points
let clipper = SmartClipper::new(api_key)?;
let optimal_cuts = clipper.find_optimal_cuts(
    "video.mp4".as_ref(),
    &[(10.0, 30.0), (60.0, 90.0)], // Desired segments
    &SubtitleConfig::default()
).await?;

// Apply intelligent cuts
for cut in optimal_cuts {
    let runner = Runner::ffmpeg_default("input.mp4", &format!("clip_{}.mp4", cut.name))?;
    runner.execute(CutVideoOperation::Splice {
        segments: vec![cut.optimal_start..cut.optimal_end]
    })?;
}
```

### With Video Kit Common

```rust
use cutting::{Runner, CutVideoOperation};
use video_kit_common::{TimestampRange, VideoMetadata};

// Use shared types for consistency
let segments: Vec<TimestampRange> = vec![
    TimestampRange::new(0.0, 30.0)?,
    TimestampRange::new(60.0, 90.0)?,
];

let ranges: Vec<std::ops::Range<f64>> = segments
    .into_iter()
    .map(|seg| seg.start..seg.end)
    .collect();

let runner = Runner::ffmpeg_default("input.mp4", "output.mp4")?;
runner.execute(CutVideoOperation::Splice { segments: ranges })?;
```

## 📚 Examples

### Basic Video Splitting

```rust
use cutting::{Runner, CutVideoOperation};

let runner = Runner::ffmpeg_default("lecture.mp4", "intro.mp4")?;
runner.execute(CutVideoOperation::Splice {
    segments: vec![0.0..30.0] // First 30 seconds
})?;
```

### Creating Video Loops

```rust
// Create a seamless loop
let runner = Runner::ffmpeg_default("action.mp4", "action_loop.mp4")?;
runner.execute(CutVideoOperation::CreateLoop)?;
```

### Extracting Frame Sequences

```rust
use cutting::{Runner, CutVideoOperation, Sequentise};

let runner = Runner::ffmpeg_default("video.mp4", "frames/")?;
runner.execute(CutVideoOperation::Sequentise(Sequentise {
    period: 10.0..20.0,  // Extract from 10s to 20s
    fps: Some(1.0),      // 1 frame per second
    format: Some("png".to_string()),
    quality: Some(95),
}))?;
```

## 🔧 Backend Configuration

### FFmpeg Advanced Usage

```rust
use cutting::driver::ffmpeg::{FFmpegDriver, FFmpegOptions};

let options = FFmpegOptions {
    hardware_acceleration: Some("nvenc".to_string()),
    preset: Some("fast".to_string()),
    crf: Some(23),
    threads: Some(8),
};

let driver = FFmpegDriver::with_options(options)?;
```

### GStreamer Pipeline Customization

```rust
use cutting::driver::gstreamer::{GStreamerDriver, PipelineConfig};

let config = PipelineConfig {
    video_encoder: "x264enc".to_string(),
    audio_encoder: "aacenc".to_string(),
    muxer: "mp4mux".to_string(),
    quality: Some("high".to_string()),
};

let driver = GStreamerDriver::with_config(config)?;
```

## 📖 Documentation

- [API Documentation](https://docs.rs/cutting)
- [Backend Guide](docs/backends.md)
- [Configuration Reference](docs/configuration.md)
- [Examples Repository](examples/)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Test specific backend
cargo test --features ffmpeg
cargo test --features gstreamer

# Run examples
cargo run --example splice_video --features ffmpeg
cargo run --example smart_cutting_with_subtitles --features ffmpeg,subtitles
```
    Err(CutError::InvalidPath(path)) => eprintln!("Invalid path: {}", path),
    Err(CutError::GStreamerInit(e)) => eprintln!("GStreamer init failed: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 🚀 Performance Tips

1. **Choose the Right Backend**: FFmpeg for flexibility, GStreamer for performance
2. **Batch Operations**: Group multiple operations into streaming pipelines
3. **Memory Management**: Use streaming APIs for large files
4. **Format Optimization**: Match input/output formats when possible

## 🔧 Extension Points

### Custom Backends

Implement the `Driver` trait for custom backends:

```rust
use cutting::driver::{Driver, StreamOps};

struct CustomDriver {
    // Your implementation
}

impl Driver for CustomDriver {
    type Source = CustomSource;
    type Sink = CustomSink;
    type Stream = CustomStream;
    
    fn load(&self, source: &Self::Source) -> Result<Self::Stream, DriverError> {
        // Implementation
    }
}
```

### Custom Operations

Extend with custom streaming operations:

```rust
use cutting::driver::StreamOps;

trait CustomStreamOps: StreamOps {
    fn apply_custom_effect(&self, driver: &Self::Driver) -> Result<Self, DriverError>;
}
```

## 🔗 Integration

Works seamlessly with other video-kit crates:

- **subtitles**: Intelligent cutting based on speech analysis
- **mask**: Video masking and region-based processing
- **gpu_processor_***: GPU-accelerated effects and processing

## 📚 Examples

See the `examples/` directory for complete examples:

- `splice_video.rs`: Basic video splicing
- `ffmpeg_example.rs`: FFmpeg backend usage
- `poster_generation.rs`: Frame extraction and poster creation
- `smart_cutting_with_subtitles.rs`: Intelligent subtitle-based cutting

## 🧪 Testing

```bash
# Run tests with FFmpeg backend
cargo test --features ffmpeg

# Run tests with GStreamer backend  
cargo test --features gstreamer

# Run all tests
cargo test --features ffmpeg,gstreamer
```

## 📄 License

MIT OR Apache-2.0

