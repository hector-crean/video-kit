# Video Kit Common - Shared Types and Utilities

A foundational library providing shared data structures, utilities, and cross-crate compatibility helpers for the Video Kit ecosystem. Ensures consistent types, serialization, and configuration across all video-kit crates.

## 🚀 Features

- **Shared Data Types**: Common structures for video metadata, timing, and configuration
- **Serialization Support**: Full serde and JSON Schema integration for configuration files
- **Configuration System**: Validation and parsing utilities for complex workflows
- **Cross-Crate Compatibility**: Ensures type consistency across the video-kit ecosystem
- **Utility Functions**: Common operations for time, geometry, and file handling
- **Error Handling**: Standardized error types with comprehensive context

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs              // Main exports and module organization
├── types/              // Core data structures
│   ├── timing.rs       // Time-related types and utilities
│   ├── geometry.rs     // Geometric primitives and operations
│   ├── metadata.rs     // Video and audio metadata structures
│   └── config.rs       // Configuration types and validation
├── utils/              // Utility functions
│   ├── time.rs         // Time formatting and conversion
│   ├── file.rs         // File handling and path utilities
│   └── validation.rs   // Input validation helpers
├── error.rs            // Error types and handling
└── serde_helpers.rs    // Custom serialization helpers
```

### Design Principles

- **Type Safety**: Strongly typed interfaces prevent common errors
- **Serialization First**: All types support serde with JSON Schema generation
- **Extensibility**: Designed for easy extension and customization
- **Performance**: Zero-cost abstractions where possible

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
video_kit_common = { version = "0.1" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Basic Usage

```rust
use video_kit_common::{
    TimestampRange, VideoMetadata, ConfigurationValidator
};

// Work with time ranges
let range = TimestampRange::new(10.5, 30.2)?;
println!("Duration: {:.1}s", range.duration());

// Video metadata
let metadata = VideoMetadata {
    duration: 120.5,
    width: 1920,
    height: 1080,
    framerate: 30.0,
    codec: "h264".to_string(),
};

// Serialize to JSON
let json = serde_json::to_string_pretty(&metadata)?;
println!("{}", json);
```

### Configuration Validation

```rust
use video_kit_common::{ConfigurationValidator, ValidationRule};

let validator = ConfigurationValidator::new()
    .add_rule(ValidationRule::PositiveDuration)
    .add_rule(ValidationRule::ValidPath)
    .add_rule(ValidationRule::SupportedFormat);

// Validate configuration
let config = load_config("config.json")?;
validator.validate(&config)?;
```

## 🕐 Time and Duration Types

### TimestampRange

```rust
use video_kit_common::TimestampRange;

// Create time ranges
let range = TimestampRange::new(10.0, 30.0)?;
let range_from_duration = TimestampRange::from_start_duration(10.0, 20.0)?;

// Operations
assert_eq!(range.duration(), 20.0);
assert!(range.contains(15.0));
assert!(range.overlaps(&TimestampRange::new(25.0, 35.0)?));

// Serialization support
let json = serde_json::to_string(&range)?;
// Output: {"start": 10.0, "end": 30.0}
```

### Timestamp Utilities

```rust
use video_kit_common::utils::{format_timestamp, parse_timestamp};

// Format timestamps
assert_eq!(format_timestamp(125.5), "00:02:05.500");
assert_eq!(format_timestamp(3665.0), "01:01:05.000");

// Parse timestamps
assert_eq!(parse_timestamp("00:02:05.500")?, 125.5);
assert_eq!(parse_timestamp("1:01:05")?, 3665.0);
```

## 📐 Geometry Types

### Point and Rectangle

```rust
use video_kit_common::{Point2D, Rectangle, Size};

// 2D points
let point = Point2D::new(100.0, 200.0);
let translated = point.translate(50.0, -25.0);

// Rectangles
let rect = Rectangle::new(Point2D::new(10.0, 20.0), Size::new(100.0, 50.0));
assert!(rect.contains(Point2D::new(50.0, 30.0)));
assert_eq!(rect.area(), 5000.0);

// Serialization
let json = serde_json::to_string(&rect)?;
```

### Geometric Operations

```rust
use video_kit_common::geometry::{BoundingBox, Transform2D};

// Bounding box calculations
let points = vec![
    Point2D::new(10.0, 20.0),
    Point2D::new(100.0, 80.0),
    Point2D::new(50.0, 150.0),
];
let bbox = BoundingBox::from_points(&points);

// 2D transformations
let transform = Transform2D::translation(50.0, 100.0)
    .scale(2.0, 2.0)
    .rotate(45.0);
let transformed_point = transform.apply(Point2D::new(10.0, 10.0));
```

## 📊 Metadata Types

### Video Metadata

```rust
use video_kit_common::{VideoMetadata, AudioMetadata, StreamInfo};

let video_meta = VideoMetadata {
    duration: 120.5,
    width: 1920,
    height: 1080,
    framerate: 29.97,
    codec: "h264".to_string(),
    bitrate: Some(5000000),
    pixel_format: Some("yuv420p".to_string()),
    color_space: Some("bt709".to_string()),
};

let audio_meta = AudioMetadata {
    duration: 120.5,
    sample_rate: 48000,
    channels: 2,
    codec: "aac".to_string(),
    bitrate: Some(192000),
    channel_layout: Some("stereo".to_string()),
};

// Complete stream information
let stream_info = StreamInfo {
    video: Some(video_meta),
    audio: Some(audio_meta),
    format: "mp4".to_string(),
    file_size: Some(50 * 1024 * 1024), // 50MB
};
```

### Metadata Utilities

```rust
use video_kit_common::metadata::{aspect_ratio, is_standard_resolution};

// Calculate aspect ratio
let ratio = aspect_ratio(1920, 1080); // Returns (16, 9)

// Check standard resolutions
assert!(is_standard_resolution(1920, 1080)); // 1080p
assert!(is_standard_resolution(3840, 2160)); // 4K
```

## ⚙️ Configuration Support

### Configuration Types

```rust
use video_kit_common::{
    ProcessingConfig, OutputConfig, QualitySettings, FormatSettings
};

#[derive(Serialize, Deserialize, JsonSchema)]
struct AppConfig {
    processing: ProcessingConfig,
    output: OutputConfig,
    quality: QualitySettings,
}

let config = AppConfig {
    processing: ProcessingConfig {
        backend: "ffmpeg".to_string(),
        threads: Some(8),
        memory_limit: Some(2048), // MB
        temp_dir: None,
    },
    output: OutputConfig {
        format: FormatSettings::MP4 {
            codec: "h264".to_string(),
            preset: "medium".to_string(),
        },
        quality: QualitySettings::Bitrate(5000000),
        audio_settings: Some(AudioSettings::default()),
    },
    quality: QualitySettings::CRF(23),
};

// Generate JSON Schema
let schema = schemars::schema_for!(AppConfig);
```

### Configuration Validation

```rust
use video_kit_common::{ConfigValidator, ValidationError};

let validator = ConfigValidator::new()
    .require_field("processing.backend")
    .validate_range("quality.crf", 0..=51)
    .validate_positive("processing.threads")
    .custom_validation(|config| {
        // Custom validation logic
        if config.processing.backend == "ffmpeg" {
            if config.processing.threads.unwrap_or(1) > 16 {
                return Err(ValidationError::InvalidValue(
                    "Too many threads for FFmpeg".to_string()
                ));
            }
        }
        Ok(())
    });

validator.validate(&config)?;
```

## 🔧 Utility Functions

### File Operations

```rust
use video_kit_common::utils::{
    ensure_output_dir, get_file_extension, is_video_file, generate_output_path
};

// Directory management
ensure_output_dir("./output/clips")?;

// File type detection
assert!(is_video_file("video.mp4"));
assert!(!is_video_file("audio.mp3"));

// Path utilities
let ext = get_file_extension("video.mp4"); // Some("mp4")
let output = generate_output_path("input.mp4", "clips", "processed"); 
// Returns: "clips/processed.mp4"
```

### Format Conversion

```rust
use video_kit_common::utils::{
    seconds_to_timecode, timecode_to_seconds, format_file_size
};

// Time format conversion
let timecode = seconds_to_timecode(3725.5); // "01:02:05.500"
let seconds = timecode_to_seconds("01:02:05.500")?; // 3725.5

// File size formatting
assert_eq!(format_file_size(1536), "1.5 KB");
assert_eq!(format_file_size(2048576), "2.0 MB");
```

## 🚨 Error Handling

### Standardized Error Types

```rust
use video_kit_common::{VideoKitError, Result};

#[derive(thiserror::Error, Debug)]
pub enum VideoKitError {
    #[error("Invalid time range: start {start} >= end {end}")]
    InvalidTimeRange { start: f64, end: f64 },
    
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    
    #[error("Unsupported format: {format}")]
    UnsupportedFormat { format: String },
    
    #[error("Configuration validation failed: {details}")]
    ValidationFailed { details: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// Usage in functions
fn process_video(path: &str) -> Result<()> {
    if !std::path::Path::new(path).exists() {
        return Err(VideoKitError::FileNotFound {
            path: path.to_string(),
        });
    }
    Ok(())
}
```

## 📝 JSON Schema Support

### Automatic Schema Generation

```rust
use video_kit_common::generate_config_schema;
use schemars::JsonSchema;

#[derive(Serialize, Deserialize, JsonSchema)]
struct MyConfig {
    video_path: String,
    output_dir: String,
    segments: Vec<TimestampRange>,
}

// Generate JSON Schema
let schema = generate_config_schema::<MyConfig>();
std::fs::write("config-schema.json", &schema)?;
```

## 🔗 Integration Examples

### With Cutting Crate

```rust
use video_kit_common::{TimestampRange, ProcessingConfig};
use cutting::{Runner, CutVideoOperation};

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

### With Configuration Files

```rust
use video_kit_common::{ConfigurationLoader, FormatDetector};

// Auto-detect and load configuration
let config = ConfigurationLoader::from_file("config.toml")?;
// or
let config = ConfigurationLoader::from_file("config.json")?;

// Validate before use
config.validate()?;

// Convert between formats
config.save_as_json("config.json")?;
config.save_as_toml("config.toml")?;
```

## 📚 Documentation

- [API Documentation](https://docs.rs/video_kit_common)
- [Type Reference](docs/types.md)
- [Configuration Guide](docs/configuration.md)
- [Integration Examples](examples/)

## 🔄 Version Compatibility

This crate maintains compatibility across the video-kit ecosystem:

- All video-kit crates use `video_kit_common` types
- JSON Schema ensures configuration compatibility
- Semantic versioning for breaking changes
- Migration guides for major updates 