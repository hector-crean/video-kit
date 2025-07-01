# Subtitles - AI-Powered Speech Processing Library

A comprehensive Rust library for speech-to-text transcription, subtitle extraction, and intelligent video analysis using Deepgram's industry-leading speech recognition API. Features smart cutting capabilities that respect speech boundaries for professional video editing.

## 🚀 Features

- **High-Accuracy Transcription**: Powered by Deepgram's Nova-3 model
- **Multiple Output Formats**: SRT, VTT, and JSON subtitle formats
- **Smart Video Cutting**: Find optimal cut points that avoid interrupting speech
- **Speaker Identification**: Diarization support for multi-speaker content
- **Language Detection**: Automatic language detection or manual specification
- **Audio Extraction**: Automatic audio extraction from video files
- **Word-Level Timing**: Precise word-level timestamps for fine-grained control
- **Natural Break Detection**: Identify natural pauses for seamless editing

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs              // Main API and data structures
├── extractor/          // Speech-to-text processing
├── clipper/            // Intelligent cutting algorithms
├── exporter/           // Multi-format subtitle export
└── config/             // Configuration and options
```

### Key Types

- **`SubtitleExtractor`**: Main client for speech processing
- **`SmartClipper`**: Intelligent cutting with speech awareness
- **`Subtitle`**: Standard subtitle with timing and metadata
- **`SpeechSegment`**: Detailed speech analysis with word-level data
- **`OptimalCut`**: Intelligent cut recommendations with confidence scores

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
subtitles = { version = "0.1" }
tokio = { version = "1.0", features = ["full"] }
```

### Basic Subtitle Extraction

```rust
use subtitles::{SubtitleExtractor, SubtitleConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extractor = SubtitleExtractor::new("your_deepgram_api_key".to_string())?;
    
    let subtitles = extractor.extract_subtitles(
        "video.mp4".as_ref(),
        &SubtitleConfig::default()
    ).await?;
    
    // Export to SRT format
    let srt_content = SubtitleExporter::to_srt(&subtitles);
    std::fs::write("subtitles.srt", srt_content)?;
    
    Ok(())
}
```

### Smart Video Cutting

```rust
use subtitles::{SmartClipper, SubtitleConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let clipper = SmartClipper::new("your_deepgram_api_key".to_string())?;
    
    // Find optimal cut points for desired segments
    let optimal_cuts = clipper.find_optimal_cuts(
        "video.mp4".as_ref(),
        &[(10.0, 30.0), (60.0, 90.0)], // Desired segments
        &SubtitleConfig::default()
    ).await?;
    
    for cut in optimal_cuts {
        println!("Segment: {:.1}s - {:.1}s", cut.optimal_start, cut.optimal_end);
        println!("Adjustment: {}", cut.adjustment_reason);
        println!("Confidence: {:.2}%", cut.start_confidence * 100.0);
    }
    
    Ok(())
}
```

## 🎛️ Configuration Options

### Comprehensive Configuration

```rust
use subtitles::SubtitleConfig;

let config = SubtitleConfig {
    language: Some("en".to_string()),          // Language (auto-detect if None)
    identify_speakers: true,                   // Enable speaker diarization
    word_timestamps: true,                     // Word-level timestamps
    filter_profanity: false,                   // Profanity filtering
    custom_vocabulary: vec![                   // Custom vocabulary
        "VideoKit".to_string(),
        "Deepgram".to_string()
    ],
    model: Some("nova-3".to_string()),         // Model selection
    punctuation: true,                         // Auto punctuation
    utterances: true,                          // Sentence segmentation
    audio_bitrate: 128,                        // Audio quality (kbps)
    auto_extract_audio: true,                  // Auto video→audio extraction
};
```

### Supported Languages

Full support for Deepgram's language models:

- English (en), Spanish (es), French (fr), German (de)
- Italian (it), Portuguese (pt), Russian (ru), Japanese (ja)
- Korean (ko), Chinese (zh), Dutch (nl), Turkish (tr)
- Polish (pl), Swedish (sv), Bulgarian (bg), Czech (cs)
- Danish (da), Greek (el), and more...

## 🧠 Intelligent Cutting Features

### Natural Break Detection

```rust
use subtitles::{SmartClipper, SubtitleConfig};

let clipper = SmartClipper::new(api_key)?;

// Find natural pauses longer than 2 seconds
let natural_breaks = clipper.find_natural_breaks(
    "video.mp4".as_ref(),
    2.0, // Minimum break duration
    &SubtitleConfig::default()
).await?;

println!("Found {} natural break points", natural_breaks.len());
for &timestamp in &natural_breaks {
    println!("Natural break at {:.1}s", timestamp);
}
```

### Optimal Cut Analysis

The `OptimalCut` struct provides detailed cut recommendations:

```rust
pub struct OptimalCut {
    pub original_start: f64,      // Your requested start time
    pub original_end: f64,        // Your requested end time  
    pub optimal_start: f64,       // AI-recommended start time
    pub optimal_end: f64,         // AI-recommended end time
    pub start_confidence: f64,    // Confidence in start adjustment
    pub end_confidence: f64,      // Confidence in end adjustment
    pub adjustment_reason: String, // Human-readable explanation
}
```

**Example Output:**
```
Original: 15.0s - 45.0s
Optimal:  14.2s - 46.8s
Reason: Adjusted start to avoid cutting mid-word, extended end to complete sentence
Confidence: Start 95%, End 87%
```

## 📝 Export Formats

### SRT (SubRip) Format

```rust
use subtitles::SubtitleExporter;

let srt_content = SubtitleExporter::to_srt(&subtitles);
// Output:
// 1
// 00:00:00,000 --> 00:00:05,200
// Welcome to VideoKit, the comprehensive video processing toolkit.
```

### VTT (WebVTT) Format

```rust
let vtt_content = SubtitleExporter::to_vtt(&subtitles);
// Output:
// WEBVTT
//
// 00:00:00.000 --> 00:00:05.200
// Welcome to VideoKit, the comprehensive video processing toolkit.
```

### JSON Format

```rust
let json_content = SubtitleExporter::to_json(&subtitles)?;
// Structured JSON with full metadata, timestamps, and confidence scores
```

## 🎥 Video Format Support

### Supported Input Formats

**Video:** MP4, AVI, MOV, MKV, WebM, FLV, 3GP, M4V  
**Audio:** MP3, WAV, FLAC, AAC, OGG, M4A, WMA

### Automatic Audio Extraction

When processing video files, the library automatically:

1. Extracts audio using FFmpeg
2. Optimizes for speech recognition (configurable bitrate)
3. Processes with Deepgram
4. Cleans up temporary files

```rust
let config = SubtitleConfig {
    auto_extract_audio: true,    // Enable automatic extraction
    audio_bitrate: 128,          // Quality vs. speed balance
    ..Default::default()
};
```

## 🔍 Advanced Speech Analysis

### Detailed Speech Segments

```rust
use subtitles::{SubtitleExtractor, SpeechSegment};

let segments = extractor.extract_speech_segments(
    "video.mp4".as_ref(),
    &config
).await?;

for segment in segments {
    println!("Speaker {}: \"{}\"", 
             segment.speaker.unwrap_or(0), 
             segment.transcript);
    println!("Timing: {:.1}s - {:.1}s", segment.start, segment.end);
    println!("Confidence: {:.2}%", segment.confidence * 100.0);
    
    // Access individual words with timestamps
    for word in segment.words {
        println!("  Word: \"{}\" at {:.1}s", word.word, word.start);
    }
}
```

### Speaker Diarization

```rust
let config = SubtitleConfig {
    identify_speakers: true,     // Enable speaker identification
    ..Default::default()
};

let segments = extractor.extract_speech_segments("video.mp4".as_ref(), &config).await?;

// Group by speaker
let mut speakers: std::collections::HashMap<i32, Vec<_>> = std::collections::HashMap::new();
for segment in segments {
    if let Some(speaker_id) = segment.speaker {
        speakers.entry(speaker_id).or_default().push(segment);
    }
}

println!("Found {} distinct speakers", speakers.len());
```

## ⚡ Performance Optimization

### Best Practices

1. **Batch Processing**: Process multiple files with the same client
2. **Audio Quality**: Balance between quality (higher bitrate) and processing speed
3. **Model Selection**: Choose appropriate model for your use case
4. **Language Specification**: Specify language when known for faster processing

### Recommended Settings

```rust
// High accuracy (slower)
let high_accuracy_config = SubtitleConfig {
    model: Some("nova-3".to_string()),
    audio_bitrate: 192,
    word_timestamps: true,
    utterances: true,
    ..Default::default()
};

// Fast processing (lower accuracy)
let fast_config = SubtitleConfig {
    model: Some("base".to_string()),
    audio_bitrate: 96,
    word_timestamps: false,
    utterances: false,
    ..Default::default()
};
```

## 🔐 API Key Management

### Environment Variables

```bash
export DEEPGRAM_API_KEY="your_api_key_here"
```

### Programmatic Setup

```rust
// From environment variable
let api_key = std::env::var("DEEPGRAM_API_KEY")
    .expect("DEEPGRAM_API_KEY environment variable not set");

let extractor = SubtitleExtractor::new(api_key)?;
```

### API Key Acquisition

1. Sign up at [Deepgram Console](https://console.deepgram.com/)
2. Create a new project
3. Generate an API key
4. $200 in free credits for new accounts

## 🧪 Error Handling

Comprehensive error types for robust applications:

```rust
use subtitles::SubtitleError;

match extractor.extract_subtitles(path, &config).await {
    Ok(subtitles) => println!("Extracted {} subtitles", subtitles.len()),
    Err(SubtitleError::Deepgram(e)) => eprintln!("Deepgram API error: {}", e),
    Err(SubtitleError::InvalidFormat(fmt)) => eprintln!("Unsupported format: {}", fmt),
    Err(SubtitleError::AudioExtractionFailed(e)) => eprintln!("Audio extraction failed: {}", e),
    Err(SubtitleError::NoResults) => eprintln!("No speech detected in audio"),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 🔗 Integration

Works seamlessly with other video-kit crates:

- **cutting**: Intelligent video cutting based on speech analysis
- **mask**: Region-based subtitle analysis
- **CLI**: Command-line subtitle extraction and processing

### Integration Example

```rust
use cutting::{Runner, CutVideoOperation};
use subtitles::{SmartClipper, SubtitleConfig};

// Extract optimal cuts
let clipper = SmartClipper::new(api_key)?;
let cuts = clipper.find_optimal_cuts(
    "input.mp4".as_ref(),
    &[(0.0, 30.0), (60.0, 90.0)],
    &SubtitleConfig::default()
).await?;

// Apply cuts using the cutting crate
for (i, cut) in cuts.iter().enumerate() {
    let runner = Runner::ffmpeg_default("input.mp4", &format!("segment_{}.mp4", i))?;
    runner.execute(CutVideoOperation::Splice {
        segments: vec![cut.optimal_start..cut.optimal_end]
    })?;
}
```

## 📚 Examples

See the `examples/` directory:

- `extract_subtitles.rs`: Basic subtitle extraction
- `smart_cutting.rs`: Intelligent video cutting
- `speaker_analysis.rs`: Multi-speaker content analysis
- `format_conversion.rs`: Export to multiple formats

## 🧪 Testing

```bash
# Run tests (requires DEEPGRAM_API_KEY environment variable)
cargo test

# Run with logging
RUST_LOG=debug cargo test
```

## 📄 License

MIT OR Apache-2.0 