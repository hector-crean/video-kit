# SAM ONNX Inference - Segment Anything Model with Burn

A high-performance implementation of Meta's Segment Anything Model (SAM) using the Burn deep learning framework with ONNX interoperability. Provides efficient, GPU-accelerated image segmentation with support for points, boxes, and mask prompts.

## 🚀 Features

- **SAM Model Support**: Full implementation of Segment Anything Model
- **ONNX Compatibility**: Import PyTorch SAM models via ONNX
- **Burn Integration**: Built on Burn framework for efficient GPU inference
- **Multiple Prompts**: Support for point, box, and mask prompting
- **Batch Processing**: Efficient batch inference for multiple images
- **Cross-Platform**: GPU acceleration on Windows, macOS, Linux
- **Memory Efficient**: Optimized memory usage for large images
- **Easy Integration**: Simple API for embedding in applications

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs                  // Main API and model wrapper
├── model/                  // SAM model implementation
│   ├── image_encoder.rs    // Vision Transformer image encoder
│   ├── prompt_encoder.rs   // Prompt embedding (points, boxes, masks)
│   ├── mask_decoder.rs     // Mask prediction decoder
│   ├── sam.rs              // Complete SAM model
│   └── config.rs           // Model configuration
├── preprocessing.rs        // Image preprocessing and normalization
├── postprocessing.rs       // Mask postprocessing and refinement
└── bin/
    └── sam.rs              // CLI tool for inference
```

### Model Architecture

```
Input Image (1024x1024)
        │
        ▼
┌─────────────────┐      ┌──────────────────┐
│  Image Encoder  │      │  Prompt Encoder  │
│ (Vision ViT-H)  │      │ (Points/Boxes)   │
│                 │      │                  │
│ Outputs:        │      │ Outputs:         │
│ • Image         │      │ • Sparse         │
│   Embeddings    │      │   Embeddings     │
│ • Positional    │      │ • Dense          │
│   Encodings     │      │   Embeddings     │
└─────────────────┘      └──────────────────┘
        │                         │
        │                         │
        └──────────┬──────────────┘
                   │
                   ▼
        ┌─────────────────┐
        │  Mask Decoder   │
        │                 │
        │ • Transformer   │
        │ • IoU Prediction│
        │ • Mask Output   │
        └─────────────────┘
                   │
                   ▼
        Output Masks + Scores
```

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
sam-inference = { version = "0.1" }
burn = { version = "0.17", features = ["wgpu"] }
image = "0.25"
```

### Basic Usage

```rust
use sam_inference::{Sam, SamConfig, PromptPoint, Device};
use image::open;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize device and load model
    let device = Device::default();
    let config = SamConfig::vit_h(); // SAM ViT-H model
    let mut sam = Sam::load(&config, &device).await?;
    
    // Load and preprocess image
    let image = open("input.jpg")?.to_rgb8();
    let processed_image = sam.preprocess_image(&image)?;
    
    // Create point prompt (x, y, is_positive)
    let point = PromptPoint::new(512.0, 384.0, true);
    
    // Generate mask
    let result = sam.segment(&processed_image, &[point]).await?;
    
    // Save results
    for (i, mask) in result.masks.iter().enumerate() {
        mask.save(&format!("mask_{}.png", i))?;
    }
    
    Ok(())
}
```

### Advanced Prompting

```rust
use sam_inference::{PromptPoint, PromptBox, PromptMask, MultiPrompt};

// Multiple points
let points = vec![
    PromptPoint::new(100.0, 100.0, true),   // Positive point
    PromptPoint::new(200.0, 200.0, false),  // Negative point
    PromptPoint::new(300.0, 150.0, true),   // Another positive
];

// Bounding box prompt
let bbox = PromptBox::new(50.0, 50.0, 400.0, 300.0);

// Previous mask as prompt
let previous_mask = load_mask("previous_result.png")?;
let mask_prompt = PromptMask::from_image(&previous_mask)?;

// Combine multiple prompt types
let multi_prompt = MultiPrompt::new()
    .with_points(points)
    .with_box(bbox)
    .with_mask(mask_prompt);

let result = sam.segment(&processed_image, &multi_prompt).await?;
```

## 🎯 Prompt Types

### Point Prompts

```rust
use sam_inference::PromptPoint;

// Single positive point
let positive_point = PromptPoint::new(x, y, true);

// Single negative point (exclude this area)
let negative_point = PromptPoint::new(x, y, false);

// Multiple points for complex shapes
let complex_points = vec![
    PromptPoint::new(100.0, 100.0, true),
    PromptPoint::new(150.0, 120.0, true),
    PromptPoint::new(80.0, 140.0, false),   // Exclude this region
];
```

### Box Prompts

```rust
use sam_inference::PromptBox;

// Bounding box (x1, y1, x2, y2)
let bbox = PromptBox::new(50.0, 50.0, 300.0, 200.0);

// Convert from different formats
let bbox_from_center = PromptBox::from_center_size(175.0, 125.0, 250.0, 150.0);
let bbox_from_corners = PromptBox::from_corners((50.0, 50.0), (300.0, 200.0));
```

### Mask Prompts

```rust
use sam_inference::PromptMask;

// Use previous segmentation as prompt
let previous_result = sam.segment(&image, &initial_prompt).await?;
let refined_prompt = PromptMask::from_tensor(&previous_result.masks[0]);

// Load external mask
let external_mask = load_mask_image("external_mask.png")?;
let mask_prompt = PromptMask::from_image(&external_mask)?;
```

## 🔧 Model Configuration

### Available Models

```rust
use sam_inference::SamConfig;

// SAM ViT-H (best quality, largest model)
let vit_h_config = SamConfig::vit_h();

// SAM ViT-L (good balance of quality and speed)
let vit_l_config = SamConfig::vit_l();

// SAM ViT-B (fastest, smallest model)
let vit_b_config = SamConfig::vit_b();

// Custom configuration
let custom_config = SamConfig::builder()
    .encoder_embed_dim(1280)
    .encoder_depth(32)
    .encoder_num_heads(16)
    .prompt_embed_dim(256)
    .image_size(1024)
    .patch_size(16)
    .build();
```

### Model Loading

```rust
use sam_inference::{Sam, ModelSource};

// Load from local ONNX file
let sam = Sam::from_onnx_file("sam_vit_h_4b8939.onnx", &device).await?;

// Load from URL
let sam = Sam::from_url(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    &device
).await?;

// Load from PyTorch checkpoint
let sam = Sam::from_pytorch_checkpoint("sam_checkpoint.pth", &device).await?;
```

## 🎨 Batch Processing

### Multiple Images

```rust
use sam_inference::{BatchProcessor, ImageBatch};

async fn process_batch() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::default();
    let sam = Sam::load(&SamConfig::vit_h(), &device).await?;
    let mut batch_processor = BatchProcessor::new(sam, 4); // Batch size of 4
    
    // Create batch
    let mut batch = ImageBatch::new();
    for i in 0..20 {
        let image = load_image(&format!("input_{}.jpg", i))?;
        let point = PromptPoint::new(256.0, 256.0, true);
        batch.add_item(image, vec![point]);
    }
    
    // Process batch
    let results = batch_processor.process_batch(&batch).await?;
    
    // Save results
    for (i, result) in results.iter().enumerate() {
        result.save_masks(&format!("output_{}", i))?;
    }
    
    Ok(())
}
```

### Video Segmentation

```rust
use sam_inference::{VideoProcessor, TrackingState};

async fn process_video() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::default();
    let sam = Sam::load(&SamConfig::vit_h(), &device).await?;
    let mut video_processor = VideoProcessor::new(sam);
    
    // Initialize with first frame
    let first_frame = load_frame("frame_000.jpg")?;
    let initial_prompt = PromptPoint::new(300.0, 200.0, true);
    let tracking_state = video_processor.initialize(&first_frame, &[initial_prompt]).await?;
    
    // Process subsequent frames
    for frame_idx in 1..1000 {
        let frame = load_frame(&format!("frame_{:03}.jpg", frame_idx))?;
        let result = video_processor.track_frame(&frame, &tracking_state).await?;
        
        // Update tracking state
        tracking_state.update(&result);
        
        // Save mask
        result.primary_mask().save(&format!("mask_{:03}.png", frame_idx))?;
    }
    
    Ok(())
}
```

## 🔍 Advanced Features

### Mask Refinement

```rust
use sam_inference::{MaskRefinement, RefinementOptions};

// Enable iterative refinement
let refinement_options = RefinementOptions {
    max_iterations: 3,
    iou_threshold: 0.8,
    score_threshold: 0.9,
    use_stability_score: true,
};

let mut sam = Sam::load(&config, &device).await?
    .with_refinement(refinement_options);

// Automatic refinement during inference
let result = sam.segment_with_refinement(&image, &prompts).await?;
```

### Multi-Scale Processing

```rust
use sam_inference::{MultiScaleProcessor, ScaleConfig};

// Process at multiple scales for better accuracy
let scale_config = ScaleConfig {
    scales: vec![0.5, 1.0, 1.5],
    fusion_method: FusionMethod::WeightedAverage,
    weight_by_confidence: true,
};

let mut multi_scale_sam = MultiScaleProcessor::new(sam, scale_config);
let result = multi_scale_sam.segment(&image, &prompts).await?;
```

### Custom Post-Processing

```rust
use sam_inference::{PostProcessor, MorphologyOp};

// Custom post-processing pipeline
let post_processor = PostProcessor::new()
    .add_operation(MorphologyOp::Opening { kernel_size: 3 })
    .add_operation(MorphologyOp::Closing { kernel_size: 5 })
    .add_smoothing(sigma: 1.0)
    .add_hole_filling(min_hole_size: 100);

let raw_result = sam.segment(&image, &prompts).await?;
let refined_result = post_processor.process(&raw_result)?;
```

## 🧪 Evaluation and Metrics

### Quality Assessment

```rust
use sam_inference::{MetricsCalculator, GroundTruth};

// Calculate segmentation metrics
let metrics_calc = MetricsCalculator::new();
let ground_truth = GroundTruth::from_mask_file("ground_truth.png")?;

let result = sam.segment(&image, &prompts).await?;
let metrics = metrics_calc.calculate(&result.primary_mask(), &ground_truth)?;

println!("IoU: {:.3}", metrics.iou);
println!("Dice: {:.3}", metrics.dice);
println!("Precision: {:.3}", metrics.precision);
println!("Recall: {:.3}", metrics.recall);
```

### Benchmark Suite

```rust
use sam_inference::{BenchmarkSuite, BenchmarkConfig};

// Run comprehensive benchmarks
let benchmark_config = BenchmarkConfig {
    dataset_path: "benchmark_dataset/",
    models: vec![
        SamConfig::vit_b(),
        SamConfig::vit_l(),
        SamConfig::vit_h(),
    ],
    prompt_types: vec![
        PromptType::SinglePoint,
        PromptType::MultiplePoints,
        PromptType::BoundingBox,
    ],
};

let benchmark = BenchmarkSuite::new(benchmark_config);
let results = benchmark.run().await?;

// Generate report
results.save_report("benchmark_results.json")?;
results.generate_html_report("benchmark_report.html")?;
```

## 🚀 Performance Optimization

### Memory Management

```rust
use sam_inference::{MemoryConfig, CacheStrategy};

// Configure memory usage
let memory_config = MemoryConfig {
    max_gpu_memory: 8 * 1024 * 1024 * 1024, // 8GB
    cache_strategy: CacheStrategy::LRU,
    preload_model: true,
    enable_gradient_checkpointing: true,
};

let sam = Sam::load(&config, &device).await?
    .with_memory_config(memory_config);
```

### Inference Optimization

```rust
use sam_inference::{InferenceConfig, PrecisionMode};

// Optimize inference speed
let inference_config = InferenceConfig {
    precision_mode: PrecisionMode::FP16,
    enable_tensor_cores: true,
    batch_size: 8,
    enable_jit_compilation: true,
};

let sam = Sam::load(&config, &device).await?
    .with_inference_config(inference_config);
```

## 🔗 Integration

### With Video Kit Ecosystem

```rust
use sam_inference::Sam;
use mask::{Pipeline as MaskPipeline, ComplexShape};
use cutting::{Runner, CutVideoOperation};

// Integrate SAM with mask processing
async fn sam_mask_integration() -> Result<(), Box<dyn std::error::Error>> {
    let sam = Sam::load(&SamConfig::vit_h(), &device).await?;
    
    // Generate mask with SAM
    let image = load_image("input.jpg")?;
    let point = PromptPoint::new(300.0, 200.0, true);
    let sam_result = sam.segment(&image, &[point]).await?;
    
    // Convert to mask crate format
    let mask_image = sam_result.primary_mask().to_grayscale_image()?;
    
    // Process with mask crate
    let mask_pipeline = MaskPipeline::builder()
        .with_simplification(2.0)
        .build();
    let outline_result = mask_pipeline.process(&mask_image)?;
    
    // Use with cutting crate
    // ... video processing based on detected masks
    
    Ok(())
}
```

## 📚 Examples

### CLI Tool

```bash
# Segment with point prompt
sam --model vit_h --image input.jpg --point 300,200 --output masks/

# Segment with bounding box
sam --model vit_l --image input.jpg --box 100,100,400,300 --output masks/

# Batch processing
sam --model vit_b --input-dir images/ --prompt-file prompts.json --output-dir results/

# Video processing
sam --model vit_h --video input.mp4 --initial-prompt 300,200 --output-dir video_masks/
```

### Python Script Integration

```bash
# Convert PyTorch model to ONNX
python scripts/convert_sam_to_onnx.py --checkpoint sam_vit_h.pth --output sam_vit_h.onnx

# Evaluate on dataset
python scripts/evaluate_sam.py --model sam_vit_h.onnx --dataset coco_val2017
```

## 🧪 Testing

```bash
# Run tests
cargo test

# Run with specific model
cargo test --features download -- --test-threads=1

# Benchmark performance
cargo run --example benchmark --release

# CLI tool
cargo run --bin sam -- --help
```

## 🔧 Model Conversion

### From PyTorch

```python
# scripts/convert_sam_to_onnx.py
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load PyTorch model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")

# Convert to ONNX
torch.onnx.export(
    sam,
    (dummy_input,),
    "sam_vit_h.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
)
```

### Model Download

```bash
# Download official SAM models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 📄 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

- Meta AI for the original Segment Anything Model
- Burn team for the excellent deep learning framework
- ONNX community for interoperability standards 