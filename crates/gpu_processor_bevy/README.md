# GPU Processor (Bevy) - Game Engine-Powered Image Processing

A GPU-accelerated image processing library built on the [Bevy](https://bevyengine.org/) game engine's powerful rendering pipeline. Leverages Bevy's asset system, compute shaders, and ECS architecture for high-performance, real-time image processing applications.

## 🚀 Features

- **Bevy Integration**: Built on Bevy's proven rendering infrastructure
- **Compute Shaders**: WGSL-based compute shaders for GPU processing
- **ECS Architecture**: Entity-Component-System for scalable processing pipelines
- **Asset System**: Seamless image loading and management via Bevy's asset system
- **Real-Time Processing**: Optimized for interactive and real-time applications
- **Plugin Architecture**: Modular design with customizable processing plugins
- **Cross-Platform**: Supports all platforms Bevy runs on (Windows, macOS, Linux, Web)

## 🏗️ Architecture

### Core Components

```
src/
├── lib.rs              // Main plugin and API
├── processor.rs        // Core image processing systems
├── image_format.rs     // Image format handling and conversion
├── shaders/            // WGSL compute shaders
│   ├── blur.wgsl       // Gaussian and box blur shaders
│   ├── edge.wgsl       // Edge detection shaders
│   ├── threshold.wgsl  // Thresholding operations
│   └── morphology.wgsl // Morphological operations
├── components.rs       // ECS components for processing
├── resources.rs        // Global resources and configuration
└── systems.rs          // Bevy systems for processing pipeline
```

### Design Philosophy

- **ECS-First**: Everything is modeled as entities, components, and systems
- **Shader-Based**: Core operations implemented as compute shaders
- **Asset-Driven**: Images processed through Bevy's asset pipeline
- **Plugin System**: Extensible through Bevy's plugin architecture

## 📦 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
gpu_processor = { version = "0.1" }
bevy = { version = "0.14", features = ["default"] }
```

### Basic Setup

```rust
use bevy::prelude::*;
use gpu_processor::{GpuProcessorPlugin, ProcessImageEvent, ImageProcessor};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GpuProcessorPlugin)
        .add_systems(Startup, setup_processing)
        .add_systems(Update, process_images)
        .run();
}

fn setup_processing(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    // Load image as asset
    let image_handle = asset_server.load("input.png");
    
    // Create processing entity
    commands.spawn((
        ImageProcessor::new(image_handle),
        ProcessingPipeline::default()
            .with_gaussian_blur(2.0)
            .with_edge_detection()
            .with_threshold(128),
    ));
}

fn process_images(
    mut processor_query: Query<&mut ImageProcessor>,
    mut events: EventWriter<ProcessImageEvent>,
) {
    for processor in processor_query.iter_mut() {
        if processor.is_ready() {
            events.send(ProcessImageEvent {
                entity: processor.entity(),
                operation: ProcessingOperation::ExecutePipeline,
            });
        }
    }
}
```

### Advanced Pipeline Configuration

```rust
use gpu_processor::{
    ProcessingPipeline, BlurComponent, EdgeDetectionComponent,
    ThresholdComponent, MorphologyComponent
};

fn setup_advanced_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let image_handle = asset_server.load("complex_image.png");
    
    commands.spawn((
        ImageProcessor::new(image_handle),
        ProcessingPipeline::new()
            .add_stage(BlurComponent::gaussian(1.5))
            .add_stage(EdgeDetectionComponent::sobel())
            .add_stage(ThresholdComponent::adaptive(11, 2.0))
            .add_stage(MorphologyComponent::closing(3))
            .with_output_path("processed_image.png"),
    ));
}
```

## 🎮 ECS Components

### Processing Components

```rust
use gpu_processor::components::*;

// Blur operations
#[derive(Component)]
pub struct BlurComponent {
    pub blur_type: BlurType,
    pub strength: f32,
}

impl BlurComponent {
    pub fn gaussian(sigma: f32) -> Self { /* ... */ }
    pub fn box_filter(radius: u32) -> Self { /* ... */ }
    pub fn motion(angle: f32, distance: f32) -> Self { /* ... */ }
}

// Edge detection
#[derive(Component)]
pub struct EdgeDetectionComponent {
    pub algorithm: EdgeAlgorithm,
    pub threshold_low: f32,
    pub threshold_high: f32,
}

// Morphological operations
#[derive(Component)]
pub struct MorphologyComponent {
    pub operation: MorphologyOp,
    pub kernel_size: u32,
    pub iterations: u32,
}
```

### Processing Pipeline

```rust
#[derive(Component)]
pub struct ProcessingPipeline {
    stages: Vec<ProcessingStage>,
    current_stage: usize,
    output_path: Option<String>,
}

impl ProcessingPipeline {
    pub fn new() -> Self { /* ... */ }
    
    pub fn add_stage<T: Into<ProcessingStage>>(mut self, stage: T) -> Self {
        self.stages.push(stage.into());
        self
    }
    
    pub fn with_output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = Some(path.into());
        self
    }
}
```

## 🎨 Compute Shaders

### Custom Shader Integration

```rust
use gpu_processor::{ComputeShaderComponent, ShaderRegistry};

// Register custom shader
fn setup_custom_shaders(mut shader_registry: ResMut<ShaderRegistry>) {
    shader_registry.register(
        "custom_filter",
        include_str!("shaders/custom_filter.wgsl")
    );
}

// Use custom shader in component
#[derive(Component)]
pub struct CustomFilterComponent {
    pub shader_handle: Handle<Shader>,
    pub parameters: CustomFilterParams,
}

fn apply_custom_filter(
    mut query: Query<(&mut ImageProcessor, &CustomFilterComponent)>,
    mut compute_pass: ResMut<ComputePass>,
) {
    for (mut processor, filter) in query.iter_mut() {
        compute_pass.dispatch_custom_shader(
            &filter.shader_handle,
            processor.image_dimensions(),
            &filter.parameters,
        );
        processor.mark_dirty();
    }
}
```

### WGSL Shader Example

```wgsl
// shaders/custom_filter.wgsl
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: CustomFilterParams;

struct CustomFilterParams {
    strength: f32,
    threshold: f32,
    _padding: vec2<f32>,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coord = vec2<i32>(global_id.xy);
    
    if (coord.x >= i32(dimensions.x) || coord.y >= i32(dimensions.y)) {
        return;
    }
    
    let center = textureLoad(input_texture, coord, 0);
    let processed = apply_custom_effect(center, params);
    
    textureStore(output_texture, coord, processed);
}

fn apply_custom_effect(color: vec4<f32>, params: CustomFilterParams) -> vec4<f32> {
    // Custom processing logic here
    let intensity = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    if (intensity > params.threshold) {
        return mix(color, vec4<f32>(1.0, 1.0, 1.0, color.a), params.strength);
    } else {
        return mix(color, vec4<f32>(0.0, 0.0, 0.0, color.a), params.strength);
    }
}
```

## 🔄 Real-Time Processing

### Interactive Applications

```rust
use gpu_processor::{RealTimeProcessor, ProcessingState};

#[derive(Component)]
pub struct InteractiveProcessor {
    pub auto_update: bool,
    pub last_update: f64,
    pub update_interval: f64,
}

fn real_time_processing_system(
    time: Res<Time>,
    mut query: Query<(&mut ImageProcessor, &mut InteractiveProcessor)>,
    keyboard: Res<Input<KeyCode>>,
) {
    for (mut processor, mut interactive) in query.iter_mut() {
        // Update processing on key press or timer
        let should_update = keyboard.just_pressed(KeyCode::Space) ||
            (interactive.auto_update && 
             time.elapsed_seconds_f64() - interactive.last_update > interactive.update_interval);
        
        if should_update {
            processor.queue_processing();
            interactive.last_update = time.elapsed_seconds_f64();
        }
    }
}
```

### Live Camera Feed Processing

```rust
use gpu_processor::{CameraFeedComponent, LiveProcessor};

fn setup_camera_processing(
    mut commands: Commands,
    mut camera_feed: ResMut<CameraFeedComponent>,
) {
    // Setup camera capture
    camera_feed.initialize_capture(0)?; // Camera index 0
    
    commands.spawn((
        LiveProcessor::new(),
        ProcessingPipeline::new()
            .add_stage(BlurComponent::gaussian(1.0))
            .add_stage(EdgeDetectionComponent::canny(50.0, 150.0))
            .with_real_time_output(),
    ));
}

fn process_camera_feed(
    mut live_query: Query<&mut LiveProcessor>,
    camera_feed: Res<CameraFeedComponent>,
    mut render_target: ResMut<RenderTarget>,
) {
    if let Some(frame) = camera_feed.get_latest_frame() {
        for mut processor in live_query.iter_mut() {
            let processed_frame = processor.process_frame(frame)?;
            render_target.update_texture(processed_frame);
        }
    }
}
```

## 🎯 Batch Processing

### Multi-Image Processing

```rust
use gpu_processor::{BatchProcessor, ImageBatch};

#[derive(Resource)]
pub struct BatchProcessingQueue {
    pub pending_batches: Vec<ImageBatch>,
    pub completed_batches: Vec<ProcessedBatch>,
}

fn setup_batch_processing(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let mut batch = ImageBatch::new("landscape_processing");
    
    // Add multiple images to batch
    for i in 0..50 {
        let handle = asset_server.load(&format!("landscapes/image_{:03}.png", i));
        batch.add_image(handle, format!("processed_{:03}.png", i));
    }
    
    commands.insert_resource(BatchProcessingQueue {
        pending_batches: vec![batch],
        completed_batches: vec![],
    });
}

fn process_batches(
    mut batch_queue: ResMut<BatchProcessingQueue>,
    mut batch_processor: ResMut<BatchProcessor>,
) {
    if let Some(batch) = batch_queue.pending_batches.pop() {
        let processing_future = batch_processor.process_batch_async(batch);
        // Handle async processing...
    }
}
```

## 🎪 Event System

### Processing Events

```rust
use gpu_processor::events::*;

#[derive(Event)]
pub struct ProcessImageEvent {
    pub entity: Entity,
    pub operation: ProcessingOperation,
}

#[derive(Event)]
pub struct ProcessingCompleteEvent {
    pub entity: Entity,
    pub result: ProcessingResult,
    pub timing: ProcessingTiming,
}

fn handle_processing_events(
    mut complete_events: EventReader<ProcessingCompleteEvent>,
    mut commands: Commands,
) {
    for event in complete_events.read() {
        info!("Processing completed for entity {:?} in {:.2}ms", 
              event.entity, event.timing.total_duration_ms);
        
        // Save result if specified
        if let Some(output_path) = &event.result.output_path {
            event.result.save_to_file(output_path)?;
        }
        
        // Clean up processing components
        commands.entity(event.entity)
            .remove::<ProcessingPipeline>()
            .insert(ProcessingComplete);
    }
}
```

## 🔧 Performance Optimization

### GPU Memory Management

```rust
use gpu_processor::{GpuMemoryPool, TextureCache};

fn setup_optimized_processing(mut commands: Commands) {
    commands.insert_resource(GpuMemoryPool::new(256 * 1024 * 1024)); // 256MB pool
    commands.insert_resource(TextureCache::new(100)); // Cache 100 textures
}

#[derive(Resource)]
pub struct ProcessingConfig {
    pub max_concurrent_operations: usize,
    pub texture_compression: TextureCompression,
    pub gpu_memory_limit: usize,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 4,
            texture_compression: TextureCompression::BC7,
            gpu_memory_limit: 512 * 1024 * 1024, // 512MB
        }
    }
}
```

### Async Processing

```rust
use gpu_processor::{AsyncProcessor, ProcessingTask};
use bevy::tasks::{AsyncComputeTaskPool, Task};

#[derive(Component)]
pub struct AsyncProcessingTask {
    pub task: Task<ProcessingResult>,
}

fn spawn_async_processing(
    mut commands: Commands,
    query: Query<(Entity, &ImageProcessor), With<QueuedForProcessing>>,
) {
    let task_pool = AsyncComputeTaskPool::get();
    
    for (entity, processor) in query.iter() {
        let image_data = processor.get_image_data().clone();
        let pipeline = processor.get_pipeline().clone();
        
        let task = task_pool.spawn(async move {
            AsyncProcessor::process_image_async(image_data, pipeline).await
        });
        
        commands.entity(entity)
            .insert(AsyncProcessingTask { task })
            .remove::<QueuedForProcessing>();
    }
}

fn handle_async_results(
    mut commands: Commands,
    mut query: Query<(Entity, &mut AsyncProcessingTask)>,
) {
    for (entity, mut task) in query.iter_mut() {
        if let Some(result) = future::block_on(future::poll_once(&mut task.task)) {
            commands.entity(entity)
                .insert(ProcessingResult(result))
                .remove::<AsyncProcessingTask>();
        }
    }
}
```

## 🔗 Integration

### With Video Kit Ecosystem

```rust
use gpu_processor::BevyImageProcessor;
use mask::{Pipeline as MaskPipeline};
use cutting::{Runner, CutVideoOperation};

// GPU preprocessing for mask extraction
fn gpu_mask_preprocessing(
    mut query: Query<(&mut ImageProcessor, &MaskPreprocessing)>,
) {
    for (mut processor, _) in query.iter_mut() {
        processor.queue_pipeline(
            ProcessingPipeline::new()
                .add_stage(BlurComponent::gaussian(1.0))
                .add_stage(ThresholdComponent::adaptive(11, 2.0))
                .add_stage(MorphologyComponent::closing(3))
        );
    }
}

// Integration with cutting crate for video analysis
fn analyze_video_frames(
    frame_query: Query<&ProcessedFrame>,
    mut scene_analyzer: ResMut<SceneAnalyzer>,
) {
    for frame in frame_query.iter() {
        if frame.is_scene_boundary() {
            scene_analyzer.add_cut_point(frame.timestamp);
        }
    }
}
```

## 📚 Examples

See the `examples/` directory:

- `basic_processing.rs`: Simple image processing pipeline
- `real_time_camera.rs`: Live camera feed processing
- `batch_processing.rs`: Batch processing multiple images
- `custom_shaders.rs`: Creating custom compute shaders
- `interactive_app.rs`: Interactive processing application

## 🧪 Testing

```bash
# Run tests
cargo test

# Run examples
cargo run --example basic_processing
cargo run --example real_time_camera

# Performance benchmarks
cargo run --example benchmark --release
```

## 🎮 Bevy Compatibility

Compatible with Bevy 0.14.x. The plugin integrates seamlessly with:

- **Bevy's Asset System**: For image loading and management
- **Bevy's Render Pipeline**: For GPU compute integration
- **Bevy's ECS**: For component-based processing
- **Bevy's Plugin System**: For modular functionality

## 📄 License

MIT OR Apache-2.0 