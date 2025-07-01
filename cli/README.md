# Video Kit CLI - Command-Line Video Processing Tool

A powerful command-line interface for the video-kit ecosystem, providing intelligent video processing workflows with AI-powered features. Combines video cutting, subtitle extraction, and smart editing capabilities in a single, easy-to-use tool.

## 🚀 Features

- **Smart Video Processing**: AI-powered cutting that respects speech boundaries
- **Subtitle Integration**: Extract subtitles and use them for intelligent editing decisions
- **Configuration-Driven**: JSON/TOML configuration files for complex workflows
- **Batch Processing**: Process multiple videos or segments efficiently
- **Natural Break Detection**: Find optimal cutting points automatically
- **Multi-Format Support**: Handle various video and audio formats
- **Real-Time Preview**: Preview processing results before committing changes

## 🏗️ Architecture

### Core Components

```
src/
├── bin/
│   └── video_cli.rs        // Main CLI application
├── lib.rs                  // CLI library and data structures
├── commands/               // Individual command implementations
│   ├── process.rs          // Video processing command
│   ├── generate.rs         // Configuration generation
│   ├── analyze.rs          // Video analysis
│   └── validate.rs         // Configuration validation
├── config/                 // Configuration handling
│   ├── input_video.rs      // Input video configuration
│   ├── clip.rs             // Clip definition structures
│   └── validation.rs       // Configuration validation
└── utils/                  // CLI utilities and helpers
    ├── output.rs           // Output formatting
    ├── progress.rs         // Progress reporting
    └── errors.rs           // Error handling
```

## 📦 Installation

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd video-kit

# Build the CLI
cargo build --release --bin video_cli

# Install globally (optional)
cargo install --path cli
```

### Prerequisites

- **FFmpeg**: Required for video processing
- **Deepgram API Key**: Required for subtitle extraction and smart cutting

```bash
# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html

# Set Deepgram API key
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"
```

## 🎬 Quick Start

### 1. Process Video from Configuration

```bash
# Create a basic configuration
video_cli generate-skeleton \
    --input example.mp4 \
    --output-dir ./clips \
    --json-output config.json

# Process the video
video_cli process --config config.json
```

### 2. Smart Natural Break Detection

```bash
# Analyze video for natural speech breaks
video_cli generate-skeleton \
    --input lecture.mp4 \
    --output-dir ./segments \
    --json-output lecture_config.json \
    --min-break-duration 2.0 \
    --max-segment-duration 300.0 \
    --language en
```

## 📋 Commands

### `process` - Process Video

Process video according to JSON configuration with intelligent cutting.

```bash
video_cli process [OPTIONS]
```

**Options:**
- `--config <PATH>`: Path to JSON configuration file

**Example:**
```bash
# Process with configuration file
video_cli process --config my_video_config.json

# Configuration file example (config.json):
{
  "path": "input_video.mp4",
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
    },
    {
      "name": "main_content",
      "description": "Main content",
      "operation": {
        "type": "splice", 
        "params": {
          "segments": [[35.0, 120.0]]
        }
      }
    }
  ]
}
```

### `generate-skeleton` - Generate Configuration

Generate a configuration skeleton from natural speech breaks.

```bash
video_cli generate-skeleton [OPTIONS]
```

**Options:**
- `--input <PATH>`: Input video file
- `--output-dir <PATH>`: Output directory for clips
- `--json-output <PATH>`: Output JSON configuration file
- `--min-break-duration <SECONDS>`: Minimum silence duration for breaks (default: 2.0)
- `--max-segment-duration <SECONDS>`: Maximum segment duration (default: 300.0)
- `--api-key <KEY>`: Deepgram API key (or use DEEPGRAM_API_KEY env var)
- `--language <LANG>`: Language for transcription (auto-detect if not specified)

**Example:**
```bash
# Generate configuration with natural breaks
video_cli generate-skeleton \
    --input presentation.mp4 \
    --output-dir ./presentation_clips \
    --json-output presentation_config.json \
    --min-break-duration 1.5 \
    --max-segment-duration 600.0 \
    --language en

# Output: presentation_config.json with intelligently detected segments
```

## 🧠 Smart Features

### Natural Break Detection

The CLI automatically analyzes speech patterns to find optimal cutting points:

```bash
# Analyze video for natural breaks
video_cli generate-skeleton \
    --input interview.mp4 \
    --output-dir ./interview_segments \
    --json-output interview_config.json \
    --min-break-duration 2.0

# Results in configuration like:
{
  "path": "interview.mp4",
  "output_dir": "./interview_segments",
  "clips": [
    {
      "name": "segment_001",
      "description": "Natural break at 2.3s - 45.7s",
      "operation": {
        "type": "splice",
        "params": {
          "segments": [[2.3, 45.7]]
        }
      }
    }
  ]
}
```

### Intelligent Segment Creation

Automatically creates logical segments based on:

- **Speech Pauses**: Natural breaks in conversation
- **Speaker Changes**: Transitions between different speakers
- **Content Boundaries**: Topic or scene changes
- **Duration Limits**: Maximum segment length constraints

### Language Support

Supports multiple languages for accurate speech analysis:

```bash
# Spanish content
video_cli generate-skeleton \
    --input contenido.mp4 \
    --language es \
    --json-output config_es.json

# French content  
video_cli generate-skeleton \
    --input contenu.mp4 \
    --language fr \
    --json-output config_fr.json

# Auto-detection
video_cli generate-skeleton \
    --input multilingual.mp4 \
    --json-output config_auto.json
```

## ⚙️ Configuration Format

### Complete Configuration Example

```json
{
  "path": "input_video.mp4",
  "output_dir": "output_clips",
  "clips": [
    {
      "name": "introduction",
      "description": "Opening introduction segment",
      "operation": {
        "type": "splice",
        "params": {
          "segments": [[0.0, 30.5]]
        }
      }
    },
    {
      "name": "main_demo",
      "description": "Main demonstration with reversal effect",
      "operation": {
        "type": "reverse"
      }
    },
    {
      "name": "tutorial_sequence",
      "description": "Step-by-step tutorial",
      "operation": {
        "type": "sequentise",
        "params": {
          "period": {
            "start": 120.0,
            "end": 180.0
          },
          "fps": 1.0,
          "format": "png",
          "quality": 95
        }
      }
    },
    {
      "name": "loop_demo",
      "description": "Seamless loop demonstration",
      "operation": {
        "type": "create_loop"
      }
    }
  ]
}
```

### Supported Operations

#### Splice Operation
Extract specific time ranges:
```json
{
  "type": "splice",
  "params": {
    "segments": [
      [0.0, 30.0],
      [60.0, 90.0],
      [120.0, 150.0]
    ]
  }
}
```

#### Sequentise Operation
Extract frames as image sequence:
```json
{
  "type": "sequentise", 
  "params": {
    "period": {
      "start": 10.0,
      "end": 20.0  
    },
    "fps": 2.0,
    "format": "png",
    "quality": 90
  }
}
```

#### Reverse Operation
Reverse video playback:
```json
{
  "type": "reverse"
}
```

#### Create Loop Operation
Create seamless loop:
```json
{
  "type": "create_loop"
}
```

## 🎯 Use Cases

### Educational Content

```bash
# Process lecture video with natural breaks
video_cli generate-skeleton \
    --input lecture_series_01.mp4 \
    --output-dir ./lecture_segments \
    --json-output lecture_config.json \
    --min-break-duration 3.0 \
    --max-segment-duration 600.0 \
    --language en

# Results in segments at natural topic boundaries
video_cli process --config lecture_config.json
```

### Interview Processing

```bash
# Multi-speaker interview with speaker detection
video_cli generate-skeleton \
    --input interview.mp4 \
    --output-dir ./interview_clips \
    --json-output interview_config.json \
    --min-break-duration 1.0 \
    --language en

# Edit configuration to add speaker labels
# Then process
video_cli process --config interview_config.json
```

### Podcast Segmentation

```bash
# Podcast episode with automatic chapter detection
video_cli generate-skeleton \
    --input podcast_episode_05.mp4 \
    --output-dir ./podcast_chapters \
    --json-output podcast_config.json \
    --min-break-duration 2.5 \
    --max-segment-duration 900.0

# Process into chapters
video_cli process --config podcast_config.json
```

### Content Creation

```bash
# YouTube video with optimal cut points
video_cli generate-skeleton \
    --input raw_content.mp4 \
    --output-dir ./youtube_segments \
    --json-output content_config.json \
    --min-break-duration 1.5 \
    --max-segment-duration 300.0

# Manual editing of config for specific requirements
# Then process
video_cli process --config content_config.json
```

## 🔧 Advanced Usage

### Batch Processing

```bash
# Process multiple videos
for video in videos/*.mp4; do
    base_name=$(basename "$video" .mp4)
    
    # Generate configuration
    video_cli generate-skeleton \
        --input "$video" \
        --output-dir "./processed/$base_name" \
        --json-output "./configs/${base_name}_config.json" \
        --min-break-duration 2.0
    
    # Process video
    video_cli process --config "./configs/${base_name}_config.json"
done
```

### Custom Workflows

```bash
# 1. Generate initial configuration
video_cli generate-skeleton \
    --input source.mp4 \
    --output-dir ./clips \
    --json-output config.json \
    --min-break-duration 2.0

# 2. Manual editing of config.json for custom requirements

# 3. Validate configuration
jq '.' config.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"

# 4. Process with custom configuration
video_cli process --config config.json
```

### Integration with Other Tools

```bash
# Combine with other video processing tools
video_cli generate-skeleton \
    --input input.mp4 \
    --output-dir ./segments \
    --json-output segments_config.json

# Process segments
video_cli process --config segments_config.json

# Post-process with external tools
for clip in segments/*.mp4; do
    # Apply additional processing
    ffmpeg -i "$clip" -vf "scale=1280:720" "processed_$(basename "$clip")"
done
```

## 🚦 Error Handling

### Common Issues and Solutions

#### Missing API Key
```bash
# Error: Deepgram API key not provided
# Solution:
export DEEPGRAM_API_KEY="your_api_key_here"
# OR
video_cli generate-skeleton --api-key "your_api_key_here" ...
```

#### Invalid Video Format
```bash
# Error: Unsupported video format
# Solution: Convert to supported format first
ffmpeg -i input.avi -c copy input.mp4
video_cli generate-skeleton --input input.mp4 ...
```

#### Configuration Errors
```bash
# Validate JSON configuration
jq '.' config.json

# Check for required fields
jq '.path, .output_dir, .clips' config.json
```

## 🔗 Integration

### With Video Kit Ecosystem

The CLI integrates seamlessly with other video-kit crates:

```rust
// In your Rust code
use cli::{InputVideo, Clip};
use cutting::{Runner, CutVideoOperation};
use subtitles::{SmartClipper, SubtitleConfig};

// Load CLI configuration
let input_video = InputVideo::from_json_file("config.json")?;

// Process with cutting crate
for clip in input_video.clips {
    if let Some(operation) = clip.operation {
        let runner = Runner::ffmpeg_default(&input_video.path, &output_path)?;
        runner.execute(operation)?;
    }
}
```

### Environment Variables

```bash
# Deepgram API key
export DEEPGRAM_API_KEY="your_api_key"

# FFmpeg path (if not in PATH)
export FFMPEG_PATH="/usr/local/bin/ffmpeg"

# Default output directory
export VIDEO_KIT_OUTPUT_DIR="./output"

# Log level
export RUST_LOG="info"
```

## 📚 Examples

### Example Configurations

See the `assets/` directory:
- `video-simple.json`: Basic video processing
- `backup.json`: Complex multi-operation workflow

### Sample Commands

```bash
# Basic processing
video_cli process --config assets/video-simple.json

# Generate configuration for long-form content
video_cli generate-skeleton \
    --input documentary.mp4 \
    --output-dir ./doc_segments \
    --json-output doc_config.json \
    --min-break-duration 5.0 \
    --max-segment-duration 1200.0

# Educational content with precise breaks
video_cli generate-skeleton \
    --input course_module.mp4 \
    --output-dir ./course_segments \
    --json-output course_config.json \
    --min-break-duration 2.0 \
    --max-segment-duration 600.0 \
    --language en
```

## 🧪 Testing

```bash
# Run CLI tests
cargo test --bin video_cli

# Test with sample video
video_cli generate-skeleton \
    --input assets/NOV271_Zigakibart_MoA_1080p_HD_240923a.mp4 \
    --output-dir ./test_output \
    --json-output test_config.json \
    --min-break-duration 2.0

# Validate generated configuration
video_cli process --config test_config.json
```

## 📄 License

MIT OR Apache-2.0 