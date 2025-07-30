# Cutting - GStreamer Video Processing

A simple, fluent Rust API for video processing using GStreamer. Designed for elegant operation chaining and pipeline-based video manipulation.

## Why GStreamer?

- **🔗 Natural Pipeline Architecture**: Perfect for fluent API chaining
- **📊 Streaming Processing**: Memory-efficient, no intermediate files
- **🛠️ Composable Elements**: Each operation is a separate, linkable element
- **⚡ High Performance**: Direct memory-to-memory processing
- **🎯 Simple**: No complex filter strings or monolithic commands

## Quick Start

```rust
use cutting::{quick_cut, quick_loop, Runner};
use std::ops::Range;

// Quick operations
quick_cut("input.mp4", "cut.mp4", &Range { start: 10.0, end: 30.0 })?;
quick_loop("input.mp4", "loop.mp4", Some(&Range { start: 5.0, end: 15.0 }))?;

// Fluent API
let runner = Runner::new("input.mp4", "output.mp4")?;
runner.execute_stream(|stream| {
    stream
        .cut(&Range { start: 2.0, end: 25.0 })?
        .reverse()?
        .ping_pong()
})?;
```

## API Levels

### 1. Quick Functions (Simplest)
```rust
use cutting::{quick_cut, quick_loop, quick_reverse};

quick_cut("input.mp4", "output.mp4", &Range { start: 10.0, end: 30.0 })?;
quick_loop("input.mp4", "loop.mp4", None)?; // Loop entire video
quick_reverse("input.mp4", "reverse.mp4", Some(&Range { start: 5.0, end: 15.0 }))?;
```

### 2. Runner API (Balanced)
```rust
use cutting::{Runner, Cut, CutVideoOperation};

let runner = Runner::new("input.mp4", "output.mp4")?;

// Command-based
let cut_op = CutVideoOperation::Cut(Cut { 
    period: Range { start: 10.0, end: 30.0 } 
});
runner.execute(cut_op)?;

// Fluent chaining
runner.execute_stream(|stream| {
    stream.cut(&Range { start: 10.0, end: 30.0 })?.reverse()
})?;
```

### 3. Direct Driver (Maximum Control)
```rust
use cutting::{GStreamerDriver, GStreamerFileSource, GStreamerFileSink};

let driver = GStreamerDriver::new()?;
let source = GStreamerFileSource::new("input.mp4");
let sink = GStreamerFileSink::new("output.mp4");

driver.load(&source)?
    .cut(&Range { start: 10.0, end: 30.0 })?
    .reverse()?
    .ping_pong()?
    .save(&driver, &sink)?;
```

## Operations

- **Cut**: Extract time ranges
- **Reverse**: Reverse video playback
- **PingPong**: Create seamless back-and-forth loops
- **ExtractFrames**: Extract frame sequences

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cutting = { path = "path/to/cutting" }
```

Requires GStreamer to be installed on your system.

## Examples

Check the `examples/` directory for complete usage examples:

- `simple_gstreamer.rs` - Basic API usage
- `elegant_loop_generation.rs` - Various loop creation methods
- `poster_generation.rs` - Frame extraction techniques

## Architecture

This crate uses GStreamer's pipeline-based architecture, where each operation becomes a pipeline element that can be naturally chained together. This is much cleaner than FFmpeg's monolithic command approach for fluent APIs.

```
filesrc -> cut_element -> reverse_element -> pingpong_element -> filesink
```

Each operation is a separate GStreamer element that links to the next, creating a natural processing pipeline.

