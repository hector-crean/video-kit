# GPU Processor (Burn) - High-Performance GPU Image Processing

A high-performance GPU-accelerated image processing library built on the [Burn](https://github.com/tracel-ai/burn) deep learning framework. Provides custom kernels for advanced image processing operations including edge detection, morphological operations, and machine learning-powered image analysis.

## 🚀 Features

- **GPU Acceleration**: Leverages WebGPU/WGPU for cross-platform GPU computing
- **Custom Kernels**: Hand-optimized compute shaders for image processing
- **Burn Integration**: Built on Burn's tensor operations for ML compatibility
- **Batch Processing**: Efficient batch processing of multiple images
- **Advanced Algorithms**: Edge detection, morphological operations, and custom filters
- **Memory Efficient**: Optimized memory management for large image processing
- **Cross-Platform**: Works on Windows, macOS, Linux with GPU support

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs              // Main API and tensor operations
├── backend.rs          // Burn backend configuration and initialization
├── dataset.rs          // Image dataset handling and batching
├── kernel.rs           // Custom kernel implementations
├── kernels/            // Individual kernel modules
│   ├── blur.rs         // Gaussian and box blur
│   ├── edge.rs         // Edge detection (Sobel, Canny, etc.)
│   ├── morphology.rs   // Erosion, dilation, opening, closing
│   ├── threshold.rs    // Adaptive and global thresholding
│   └── outline.rs      // Contour detection and outline extraction
└── model.rs            // Machine learning model integration
```

### Design Principles

- **Tensor-First**: All operations work with Burn tensors for ML compatibility
- **Kernel-Based**: Custom compute shaders for maximum performance
- **Composable**: Operations can be chained for complex processing pipelines
- **Memory Efficient**: Minimize GPU-CPU data transfers

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
burn_processor = { version = "0.1" }
burn = { version = "0.17", features = ["wgpu"] }
burn-wgpu = "0.17"
image = "0.25"
```

### Basic Usage

```rust
use burn_processor::{BurnBackend, ImageProcessor, ProcessingPipeline};
use burn::tensor::Tensor;
use image::open;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU backend
    let device = BurnBackend::default_device();
    let processor = ImageProcessor::new(&device);
    
    // Load image
    let img = open("input.png")?.to_rgb8();
    let tensor = processor.image_to_tensor(&img);
    
    // Apply Gaussian blur
    let blurred = processor.gaussian_blur(&tensor, 2.0)?;
    
    // Convert back to image
    let result_img = processor.tensor_to_image(&blurred);
    result_img.save("output.png")?;
    
    Ok(())
}
```

### Advanced Pipeline Processing

```rust
use burn_processor::{ProcessingPipeline, operations::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    
    // Create processing pipeline
    let pipeline = ProcessingPipeline::new(&device)
        .add_operation(GaussianBlur::new(1.5))
        .add_operation(SobelEdgeDetection::new())
        .add_operation(AdaptiveThreshold::new(11, 2.0))
        .add_operation(MorphologyClose::new(3))
        .build();
    
    // Process single image
    let input = load_image("input.png")?;
    let result = pipeline.process(&input)?;
    save_image(&result, "output.png")?;
    
    Ok(())
}
```

## 🧮 Supported Operations

### Basic Filters

```rust
use burn_processor::operations::*;

// Gaussian blur with configurable sigma
let blur = GaussianBlur::new(2.0);

// Box filter for fast blurring
let box_filter = BoxFilter::new(5);

// Median filter for noise reduction
let median = MedianFilter::new(3);
```

### Edge Detection

```rust
// Sobel edge detection
let sobel = SobelEdgeDetection::new();

// Laplacian edge detection
let laplacian = LaplacianEdgeDetection::new();

// Canny edge detection with thresholds
let canny = CannyEdgeDetection::new(50.0, 150.0);
```

### Morphological Operations

```rust
// Basic morphological operations
let erosion = MorphologyErode::new(3);      // 3x3 kernel
let dilation = MorphologyDilate::new(5);    // 5x5 kernel

// Compound operations
let opening = MorphologyOpen::new(3);       // Erosion followed by dilation
let closing = MorphologyClose::new(3);      // Dilation followed by erosion
```

### Thresholding

```rust
// Global threshold
let global_thresh = GlobalThreshold::new(128);

// Adaptive threshold with block size and C parameter
let adaptive_thresh = AdaptiveThreshold::new(11, 2.0);

// Otsu's automatic threshold
let otsu_thresh = OtsuThreshold::new();
```

## 🎯 Batch Processing

Process multiple images efficiently:

```rust
use burn_processor::{BatchProcessor, ImageBatch};

fn process_batch() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    let processor = BatchProcessor::new(&device, 32); // Batch size of 32
    
    // Load images into batch
    let mut batch = ImageBatch::new();
    for i in 0..100 {
        let img = load_image(&format!("input_{}.png", i))?;
        batch.add_image(img);
    }
    
    // Process entire batch
    let pipeline = ProcessingPipeline::new(&device)
        .add_operation(GaussianBlur::new(1.0))
        .add_operation(SobelEdgeDetection::new())
        .build();
    
    let results = processor.process_batch(&batch, &pipeline)?;
    
    // Save results
    for (i, result) in results.iter().enumerate() {
        save_image(result, &format!("output_{}.png", i))?;
    }
    
    Ok(())
}
```

## 🔧 Custom Kernels

Define your own GPU kernels:

```rust
use burn_processor::{CustomKernel, KernelBuilder};
use cubecl::prelude::*;

#[cube(launch)]
fn custom_sharpen_kernel(
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    #[comptime] strength: f32,
) {
    let pos = ABSOLUTE_POS;
    
    if pos.x < input.shape(2) && pos.y < input.shape(1) {
        let center = input[pos.y][pos.x];
        let neighbors = 
            input[pos.y - 1][pos.x] +
            input[pos.y + 1][pos.x] +
            input[pos.y][pos.x - 1] +
            input[pos.y][pos.x + 1];
        
        let sharpened = center + strength * (center * 4.0 - neighbors);
        output[pos.y][pos.x] = clamp(sharpened, 0.0, 1.0);
    }
}

struct CustomSharpen {
    strength: f32,
}

impl CustomKernel for CustomSharpen {
    fn apply(&self, input: &Tensor<BurnBackend>, device: &Device) -> Tensor<BurnBackend> {
        let output = Tensor::zeros(input.shape(), device);
        
        custom_sharpen_kernel::launch::<BurnBackend>(
            &input.client(),
            CubeCount::Static(input.shape(2), input.shape(1), 1),
            input,
            &output,
            self.strength,
        );
        
        output
    }
}
```

## 🧠 Machine Learning Integration

Leverage Burn's ML capabilities:

```rust
use burn_processor::{MLProcessor, FeatureExtractor};
use burn::nn::{conv::Conv2d, Linear, ReLU};

// Extract features for ML models
fn extract_features() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    let processor = MLProcessor::new(&device);
    
    // Preprocess images for ML
    let pipeline = ProcessingPipeline::new(&device)
        .add_operation(Resize::new(224, 224))
        .add_operation(Normalize::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        .build();
    
    let preprocessed = pipeline.process(&input_image)?;
    
    // Extract features using CNN
    let feature_extractor = FeatureExtractor::load_pretrained("resnet18", &device)?;
    let features = feature_extractor.extract_features(&preprocessed)?;
    
    Ok(())
}
```

## 🎨 Advanced Examples

### Real-time Edge Detection

```rust
use burn_processor::{RealTimeProcessor, StreamProcessor};

fn real_time_edge_detection() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    let mut stream_processor = StreamProcessor::new(&device);
    
    // Configure real-time pipeline
    let pipeline = ProcessingPipeline::new(&device)
        .add_operation(GaussianBlur::new(1.0))
        .add_operation(SobelEdgeDetection::new())
        .add_operation(AdaptiveThreshold::new(9, 1.5))
        .build();
    
    stream_processor.set_pipeline(pipeline);
    
    // Process video frames
    while let Some(frame) = capture_frame()? {
        let processed = stream_processor.process_frame(&frame)?;
        display_frame(&processed)?;
    }
    
    Ok(())
}
```

### Multi-Scale Processing

```rust
use burn_processor::{MultiScaleProcessor, ScalePyramid};

fn multi_scale_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    let processor = MultiScaleProcessor::new(&device);
    
    // Create scale pyramid
    let scales = vec![1.0, 0.75, 0.5, 0.25];
    let pyramid = ScalePyramid::new(&input_image, &scales)?;
    
    // Process at each scale
    let mut results = Vec::new();
    for scale_image in pyramid.iter() {
        let processed = processor.detect_features(scale_image)?;
        results.push(processed);
    }
    
    // Combine multi-scale results
    let combined = processor.combine_scales(&results)?;
    
    Ok(())
}
```

## ⚡ Performance Optimization

### Memory Management

```rust
use burn_processor::{MemoryPool, TensorCache};

// Use memory pools for frequent allocations
let device = BurnBackend::default_device();
let memory_pool = MemoryPool::new(&device, 1024 * 1024 * 100); // 100MB pool

// Cache tensors for reuse
let tensor_cache = TensorCache::new(100); // Cache up to 100 tensors

let processor = ImageProcessor::new(&device)
    .with_memory_pool(memory_pool)
    .with_tensor_cache(tensor_cache);
```

### Asynchronous Processing

```rust
use burn_processor::{AsyncProcessor, ProcessingQueue};

async fn async_processing() -> Result<(), Box<dyn std::error::Error>> {
    let device = BurnBackend::default_device();
    let async_processor = AsyncProcessor::new(&device);
    
    // Create processing queue
    let mut queue = ProcessingQueue::new(10); // Queue capacity
    
    // Submit processing tasks
    for i in 0..100 {
        let img = load_image(&format!("input_{}.png", i))?;
        let future = async_processor.process_async(&img);
        queue.submit(future).await?;
    }
    
    // Collect results
    let results = queue.collect_all().await?;
    
    Ok(())
}
```

## 🧪 Testing and Benchmarking

Built-in performance testing:

```bash
# Run performance benchmarks
cargo run --example benchmark --release

# Test with different image sizes
cargo run --example scaling_test --release

# Memory usage analysis
cargo run --example memory_test --features profiling
```

Example benchmark output:
```
Gaussian Blur (512x512): 2.3ms
Sobel Edge Detection (512x512): 1.8ms
Morphology Close (512x512): 4.1ms
Batch Processing (32x 256x256): 15.2ms
```

## 🔗 Integration

Works seamlessly with other video-kit crates:

### With Mask Crate

```rust
use burn_processor::{ImageProcessor, operations::*};
use mask::{Pipeline as MaskPipeline, algorithms::*};

// GPU preprocessing for mask extraction
let gpu_processor = ImageProcessor::new(&device);
let preprocessed = gpu_processor
    .gaussian_blur(&input_tensor, 1.0)?
    .adaptive_threshold(11, 2.0)?;

// Convert to CPU for mask processing
let cpu_image = gpu_processor.tensor_to_image(&preprocessed);

// Extract mask outlines
let mask_pipeline = MaskPipeline::builder()
    .with_simplification(2.0)
    .build();
let result = mask_pipeline.process(&cpu_image)?;
```

### With Cutting Crate

```rust
use burn_processor::{FrameProcessor, VideoAnalyzer};
use cutting::{Runner, CutVideoOperation};

// Analyze video frames for optimal cut points
let frame_processor = FrameProcessor::new(&device);
let analyzer = VideoAnalyzer::new(frame_processor);

let cut_points = analyzer.find_scene_changes("video.mp4")?;

// Use detected cuts with cutting crate
let runner = Runner::ffmpeg_default("input.mp4", "output.mp4")?;
runner.execute(CutVideoOperation::Splice { segments: cut_points })?;
```

## 📚 Examples

See the `examples/` directory:

- `outline.rs`: GPU-accelerated outline detection
- `batch_processing.rs`: Efficient batch image processing
- `custom_kernel.rs`: Creating custom GPU kernels
- `real_time.rs`: Real-time image processing
- `ml_features.rs`: Feature extraction for ML models

## 🧪 Testing

```bash
# Run tests (requires GPU)
cargo test

# Run with WebGPU backend
cargo test --features webgpu

# Run performance tests
cargo test --release --features benchmarks
```

## 🔧 Dependencies

- **Burn**: Deep learning framework and tensor operations
- **WGPU**: Cross-platform GPU compute
- **CubeCL**: GPU kernel compilation
- **Image**: Image loading and saving

## 📄 License

MIT OR Apache-2.0 