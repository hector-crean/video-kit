# SAM2 - Segment Anything Model 2 Python Integration

A Rust wrapper for Meta's Segment Anything Model 2 (SAM 2), providing seamless integration with Python-based SAM 2 for advanced video object segmentation and tracking. Features efficient video processing, object tracking across frames, and high-quality mask propagation.

## 🚀 Features

- **SAM 2 Integration**: Full access to Meta's SAM 2 video segmentation capabilities
- **Video Object Segmentation**: Track and segment objects across video sequences
- **Real-time Processing**: Efficient frame-by-frame processing with memory optimization
- **Object Tracking**: Persistent object tracking with automatic re-identification
- **Mask Propagation**: Intelligent mask propagation using temporal consistency
- **Python Interop**: Seamless integration with Python SAM 2 implementation
- **Memory Efficient**: Optimized for long video sequences with memory management

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs                 // Main Rust API and Python bridge
├── python/               // Python integration modules
│   ├── sam2_predictor.py  // SAM 2 predictor wrapper
│   ├── video_processor.py // Video processing pipeline
│   ├── object_tracker.py  // Object tracking and memory
│   ├── mask_propagator.py // Mask propagation algorithms
│   ├── memory_manager.py  // Memory bank management
│   ├── utils.py          // Utility functions
│   └── models/           // Model configuration and loading
│       ├── sam2_hiera_l.py
│       ├── sam2_hiera_b.py
│       └── sam2_hiera_s.py
└── integration.rs        // Python-Rust communication
```

### Processing Pipeline

```
Video Input
     │
     ▼
┌──────────────────┐
│  Frame Extractor │
│                  │
│ • Decode frames  │
│ • Preprocessing  │
│ • Batch loading  │
└──────────────────┘
     │
     ▼
┌──────────────────┐    ┌─────────────────┐
│   SAM 2 Model    │    │  Memory Bank    │
│                  │◄──►│                 │
│ • Video Encoder  │    │ • Object Store  │
│ • Object Decoder │    │ • Feature Cache │
│ • Mask Predictor │    │ • Temporal Info │
└──────────────────┘    └─────────────────┘
     │
     ▼
┌──────────────────┐
│ Mask Propagation │
│                  │
│ • Temporal Cons. │
│ • Quality Check  │
│ • Refinement     │
└──────────────────┘
     │
     ▼
Video Masks Output
```

## 📦 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download SAM 2 checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

Add to your `Cargo.toml`:

```toml
[dependencies]
sam2 = { version = "0.1" }
tokio = { version = "1.0", features = ["full"] }
serde_json = "1.0"
```

### Basic Video Segmentation

```rust
use sam2::{Sam2Predictor, VideoProcessor, PromptPoint};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize SAM 2 predictor
    let predictor = Sam2Predictor::new(
        "sam2_hiera_large.pt",
        Some("uv") // Use uv for Python environment
    ).await?;
    
    // Load video
    let video_path = Path::new("input_video.mp4");
    let mut processor = VideoProcessor::new(predictor, video_path).await?;
    
    // Initialize with first frame prompt
    let initial_prompt = PromptPoint::new(300.0, 200.0, true);
    let object_id = processor.add_object(0, vec![initial_prompt]).await?;
    
    // Process all frames
    while let Some(frame_result) = processor.next_frame().await? {
        println!("Frame {}: {} objects detected", 
                 frame_result.frame_idx, 
                 frame_result.objects.len());
        
        // Save masks
        for (obj_id, mask) in frame_result.objects {
            mask.save(&format!("masks/frame_{:04}_obj_{}.png", 
                              frame_result.frame_idx, obj_id))?;
        }
    }
    
    Ok(())
}
```

### Interactive Video Segmentation

```rust
use sam2::{InteractiveProcessor, UserInteraction, CorrectionType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let mut interactive = InteractiveProcessor::new(predictor).await?;
    
    // Load video
    interactive.load_video("input_video.mp4").await?;
    
    // Initial segmentation
    let initial_prompt = PromptPoint::new(250.0, 180.0, true);
    let object_id = interactive.segment_object(0, vec![initial_prompt]).await?;
    
    // Process frames with user corrections
    for frame_idx in 1..100 {
        let result = interactive.process_frame(frame_idx).await?;
        
        // Simulate user interaction (in real app, this comes from UI)
        if result.needs_correction(object_id) {
            let correction = UserInteraction {
                frame_idx,
                object_id,
                correction_type: CorrectionType::AddPositivePoint,
                point: (320.0, 190.0),
            };
            
            interactive.apply_correction(correction).await?;
        }
        
        // Save corrected masks
        let corrected_result = interactive.get_frame_result(frame_idx).await?;
        corrected_result.save_masks(&format!("corrected_frame_{:04}", frame_idx))?;
    }
    
    Ok(())
}
```

## 🎬 Video Processing Features

### Multi-Object Tracking

```rust
use sam2::{MultiObjectTracker, ObjectPrompt, TrackingConfig};

async fn track_multiple_objects() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let config = TrackingConfig {
        max_objects: 10,
        memory_frames: 30,
        similarity_threshold: 0.8,
        re_identification: true,
    };
    
    let mut tracker = MultiObjectTracker::new(predictor, config).await?;
    tracker.load_video("multi_object_video.mp4").await?;
    
    // Add multiple objects
    let objects = vec![
        ObjectPrompt {
            frame_idx: 0,
            points: vec![PromptPoint::new(100.0, 150.0, true)],
            label: "person_1".to_string(),
        },
        ObjectPrompt {
            frame_idx: 0, 
            points: vec![PromptPoint::new(300.0, 200.0, true)],
            label: "person_2".to_string(),
        },
        ObjectPrompt {
            frame_idx: 5,
            points: vec![PromptPoint::new(450.0, 180.0, true)],
            label: "car_1".to_string(),
        },
    ];
    
    for prompt in objects {
        tracker.add_object(prompt).await?;
    }
    
    // Process entire video
    let results = tracker.process_video().await?;
    
    // Generate tracking report
    let report = tracker.generate_tracking_report(&results)?;
    report.save("tracking_report.json")?;
    
    Ok(())
}
```

### Temporal Consistency

```rust
use sam2::{TemporalProcessor, ConsistencyConfig, TemporalSmoothing};

async fn ensure_temporal_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_base_plus.pt", None).await?;
    
    let consistency_config = ConsistencyConfig {
        temporal_window: 5,
        smoothing_factor: 0.7,
        occlusion_handling: true,
        re_emergence_detection: true,
        mask_interpolation: TemporalSmoothing::Gaussian { sigma: 1.5 },
    };
    
    let mut processor = TemporalProcessor::new(predictor, consistency_config).await?;
    processor.load_video("input_video.mp4").await?;
    
    // Process with temporal consistency
    let initial_prompt = PromptPoint::new(200.0, 150.0, true);
    let object_id = processor.add_object(0, vec![initial_prompt]).await?;
    
    let results = processor.process_with_consistency().await?;
    
    // Save temporally consistent masks
    for (frame_idx, frame_result) in results.iter().enumerate() {
        frame_result.save_masks(&format!("consistent_frame_{:04}", frame_idx))?;
    }
    
    Ok(())
}
```

## 🎯 Advanced Features

### Memory Management

```rust
use sam2::{MemoryBank, MemoryConfig, MemoryStrategy};

// Configure memory management for long videos
let memory_config = MemoryConfig {
    max_memory_frames: 50,
    strategy: MemoryStrategy::AdaptiveLRU,
    compression_enabled: true,
    feature_cache_size: 1000,
    temporal_decay: 0.95,
};

let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?
    .with_memory_config(memory_config);
```

### Quality Assessment

```rust
use sam2::{QualityAssessment, QualityMetrics, QualityThreshold};

async fn assess_segmentation_quality() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let mut processor = VideoProcessor::new(predictor, "input_video.mp4").await?;
    
    let quality_thresholds = QualityThreshold {
        min_confidence: 0.7,
        min_iou_consistency: 0.8,
        max_drift_pixels: 50.0,
        min_object_area: 100,
    };
    
    processor.set_quality_thresholds(quality_thresholds);
    
    // Process with quality monitoring
    while let Some(result) = processor.next_frame().await? {
        for (object_id, mask) in &result.objects {
            let quality = processor.assess_mask_quality(*object_id, mask)?;
            
            if quality.confidence < 0.7 {
                println!("Low confidence for object {} in frame {}", 
                         object_id, result.frame_idx);
                
                // Trigger re-segmentation or user intervention
                processor.request_user_correction(*object_id, result.frame_idx).await?;
            }
        }
    }
    
    Ok(())
}
```

### Real-Time Processing

```rust
use sam2::{RealTimeProcessor, StreamConfig, LatencyOptimization};

async fn real_time_segmentation() -> Result<(), Box<dyn std::error::Error>> {
    let stream_config = StreamConfig {
        target_fps: 30.0,
        max_latency_ms: 33, // ~30 FPS
        quality_preset: QualityPreset::Balanced,
        optimization: LatencyOptimization::Aggressive,
    };
    
    let predictor = Sam2Predictor::new("sam2_hiera_base_plus.pt", None).await?
        .with_optimization(stream_config.optimization);
    
    let mut real_time = RealTimeProcessor::new(predictor, stream_config).await?;
    
    // Initialize with camera or stream
    real_time.initialize_camera(0).await?; // Camera index 0
    
    // Add initial object
    let initial_frame = real_time.capture_frame().await?;
    let prompt = PromptPoint::new(320.0, 240.0, true);
    let object_id = real_time.add_object(&initial_frame, vec![prompt]).await?;
    
    // Real-time processing loop
    loop {
        let frame = real_time.capture_frame().await?;
        let result = real_time.process_frame(&frame).await?;
        
        // Display results
        real_time.display_results(&result).await?;
        
        // Check for exit condition
        if real_time.should_exit().await? {
            break;
        }
    }
    
    Ok(())
}
```

## 🔧 Python Integration

### Custom Python Scripts

```python
# python/custom_processor.py
import torch
from sam2.build_sam import build_sam2_video_predictor

class CustomSAM2Processor:
    def __init__(self, checkpoint_path, model_cfg="sam2_hiera_l.yaml"):
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
        self.inference_state = None
    
    def initialize_video(self, video_path):
        """Initialize video processing state"""
        self.inference_state = self.predictor.init_state(video_path=video_path)
        return {"status": "initialized", "frame_count": len(self.inference_state["video_segments"])}
    
    def add_object_prompt(self, frame_idx, points, labels):
        """Add object with point prompts"""
        obj_id, out_mask_logits = self.predictor.add_new_points(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=None,  # Let SAM 2 assign ID
            points=points,
            labels=labels,
        )
        return {"object_id": obj_id, "mask_logits": out_mask_logits.cpu().numpy()}
    
    def propagate_video(self):
        """Propagate masks across entire video"""
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                "object_ids": out_obj_ids,
                "mask_logits": {obj_id: mask.cpu().numpy() for obj_id, mask in out_mask_logits.items()}
            }
        return video_segments
```

### Rust-Python Bridge

```rust
use sam2::PythonMLProcessor;
use serde_json::{json, Value};

async fn custom_python_integration() -> Result<(), Box<dyn std::error::Error>> {
    let processor = PythonMLProcessor::new(
        "python/custom_processor.py".to_string(),
        Some("uv".to_string())
    );
    
    // Initialize video
    let init_params = json!({
        "video_path": "input_video.mp4",
        "model_checkpoint": "sam2_hiera_large.pt"
    });
    
    let init_result = processor.process_image("", init_params).await?;
    println!("Initialized: {:?}", init_result);
    
    // Add object prompt
    let prompt_params = json!({
        "frame_idx": 0,
        "points": [[300.0, 200.0]],
        "labels": [1]  // Positive point
    });
    
    let object_result = processor.process_image("add_object", prompt_params).await?;
    let object_id = object_result["object_id"].as_i64().unwrap();
    
    // Propagate through video
    let propagate_result = processor.process_image("propagate", json!({})).await?;
    
    // Process results
    if let Value::Object(video_segments) = &propagate_result["video_segments"] {
        for (frame_idx, frame_data) in video_segments {
            println!("Frame {}: {} objects", frame_idx, 
                    frame_data["object_ids"].as_array().unwrap().len());
        }
    }
    
    Ok(())
}
```

## 🎨 Use Cases

### Video Editing Workflows

```rust
use sam2::{EditingProcessor, EditOperation, MaskComposition};

async fn video_editing_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let mut editor = EditingProcessor::new(predictor).await?;
    
    // Load source video
    editor.load_video("source_video.mp4").await?;
    
    // Segment main subject
    let subject_prompt = PromptPoint::new(320.0, 240.0, true);
    let subject_id = editor.segment_object(0, vec![subject_prompt]).await?;
    
    // Apply editing operations
    let operations = vec![
        EditOperation::BackgroundReplacement {
            background_video: "new_background.mp4".to_string(),
            blend_mode: BlendMode::AlphaBlend,
        },
        EditOperation::ColorGrading {
            object_id: subject_id,
            saturation: 1.2,
            brightness: 0.1,
            contrast: 1.1,
        },
        EditOperation::MotionBlur {
            object_id: subject_id,
            blur_strength: 2.0,
            direction: MotionDirection::Automatic,
        },
    ];
    
    for operation in operations {
        editor.apply_operation(operation).await?;
    }
    
    // Export final video
    editor.export_video("edited_output.mp4").await?;
    
    Ok(())
}
```

### Sports Analysis

```rust
use sam2::{SportsAnalyzer, PlayerTracking, GameEvent};

async fn sports_video_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let mut analyzer = SportsAnalyzer::new(predictor).await?;
    
    // Load game footage
    analyzer.load_video("basketball_game.mp4").await?;
    
    // Initialize player tracking
    let players = vec![
        ("Player 1", PromptPoint::new(200.0, 300.0, true)),
        ("Player 2", PromptPoint::new(400.0, 280.0, true)),
        ("Player 3", PromptPoint::new(600.0, 320.0, true)),
    ];
    
    for (name, prompt) in players {
        analyzer.add_player(name, 0, vec![prompt]).await?;
    }
    
    // Analyze game
    let analysis_result = analyzer.analyze_video().await?;
    
    // Generate statistics
    for player_stats in analysis_result.player_statistics {
        println!("Player {}: {} movements, avg speed {:.1} px/frame",
                 player_stats.name,
                 player_stats.movement_events.len(),
                 player_stats.average_speed);
    }
    
    // Export player tracks
    analyzer.export_tracking_data("game_analysis.json").await?;
    
    Ok(())
}
```

## 🔗 Integration

### With Video Kit Ecosystem

```rust
use sam2::Sam2Predictor;
use cutting::{Runner, CutVideoOperation};
use mask::{Pipeline as MaskPipeline};

// Intelligent video cutting based on object segmentation
async fn sam2_video_cutting() -> Result<(), Box<dyn std::error::Error>> {
    let predictor = Sam2Predictor::new("sam2_hiera_large.pt", None).await?;
    let mut processor = VideoProcessor::new(predictor, "input_video.mp4").await?;
    
    // Segment main subject
    let subject_prompt = PromptPoint::new(320.0, 240.0, true);
    let subject_id = processor.add_object(0, vec![subject_prompt]).await?;
    
    // Find frames where subject is present
    let results = processor.process_video().await?;
    let subject_frames: Vec<usize> = results.iter()
        .enumerate()
        .filter(|(_, result)| result.objects.contains_key(&subject_id))
        .map(|(idx, _)| idx)
        .collect();
    
    // Create video segments based on subject presence
    let mut segments = Vec::new();
    let mut start = None;
    
    for (i, &frame_idx) in subject_frames.iter().enumerate() {
        if start.is_none() {
            start = Some(frame_idx);
        }
        
        // Check for gap in presence
        if i + 1 < subject_frames.len() && subject_frames[i + 1] - frame_idx > 30 {
            if let Some(start_frame) = start {
                segments.push((start_frame as f64 / 30.0)..(frame_idx as f64 / 30.0));
            }
            start = None;
        }
    }
    
    // Apply cuts using cutting crate
    let runner = Runner::ffmpeg_default("input_video.mp4", "output_video.mp4")?;
    runner.execute(CutVideoOperation::Splice { segments })?;
    
    Ok(())
}
```

## 📚 Examples

See the `examples/` directory:

- `basic_video_segmentation.rs`: Simple video object segmentation
- `multi_object_tracking.rs`: Track multiple objects in video
- `interactive_correction.rs`: Interactive segmentation with corrections
- `real_time_processing.rs`: Real-time video processing
- `sports_analysis.rs`: Sports video analysis example

## 🧪 Testing

```bash
# Run tests (requires Python SAM 2 environment)
cargo test

# Test with specific model
cargo test -- --test-threads=1

# Run examples
cargo run --example basic_video_segmentation
cargo run --example multi_object_tracking
```

## 📄 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

- Meta AI for SAM 2 and the original implementation
- The Python SAM 2 team for the excellent foundation 