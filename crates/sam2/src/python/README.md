# SAM2 Video Prediction with uv

This project uses uv to run video predictions with Meta's SAM2 (Segment Anything Model 2) for video segmentation.

## Setup

The project is already configured with all necessary dependencies. To get started:

```bash
# Install dependencies (already done)
uv sync

# Test the installation
uv run python simple_example.py
```

## Usage

### Quick Start with Any Video

```bash
# Run inference on any video with default settings (center point, middle frame)
uv run python run_inference.py path/to/your/video.mp4

# Run with custom point coordinates
uv run python run_inference.py path/to/your/video.mp4 --points "100,200"

# Run with custom frame and multiple points
uv run python run_inference.py path/to/your/video.mp4 --frame-idx 25 --points "100,200 300,400"

# Run with bounding box instead of points
uv run python run_inference.py path/to/your/video.mp4 --box "100,100,400,300"

# 🎨 NEW: Colored Masks - generate videos with uniquely colored masks using prime numbers!
uv run python run_inference.py path/to/your/video.mp4 --colored-masks

# 🌐 NEW: Web-Optimized Videos - H.264 encoding for perfect web compatibility!
uv run python run_inference.py path/to/your/video.mp4 --web-optimized --quality high

# 🔢 NEW: Direct RGB-to-ID Encoding - revolutionary lossless encoding with perfect decoding!
uv run python run_inference.py path/to/your/video.mp4 --direct-encoded

# 🎭 NEW: Unique IDs - assign unique alpha values to each object for post-processing!
uv run python unique_ids_example.py

# 🌈 NEW: Colored masks example with multiple objects
uv run python colored_masks_example.py

# 🎯 NEW: Direct encoding example with mathematical demonstration
uv run python direct_encoded_example.py

# Limit number of frames processed
uv run python run_inference.py path/to/your/video.mp4 --max-frames 50

# Control output frame types
uv run python run_inference.py path/to/your/video.mp4 --no-originals --no-binary-masks
```

### Output Structure

The script generates several types of output:

1. **Visualization**: `{output}_results.png` - Side-by-side comparison of key frames
2. **Masked Video**: `{output}_masked_video.mp4` - Video with mask overlays
3. **🎨 Colored Masks Video**: `{output}_colored_masks.mp4` - Video with uniquely colored masks using prime number combinations
4. **🌐 Web-Optimized Video**: `{output}_web_optimized.mp4` - H.264 encoded MP4 with optimal web settings
5. **🔢 Direct-Encoded Video**: `{output}_direct_encoded.webm` - VP9 lossless WebM with perfect RGB-to-ID encoding
6. **Alpha Channel Video**: `{output}_alpha.webm` - WebM with VP9 codec, masks encoded in alpha channel
7. **🎭 Unique IDs Alpha Video**: `{output}_unique_ids_alpha.webm` - Each object has a unique alpha value based on its ID
8. **Individual Frames Directory**: `{output}_frames/` containing:
   - `originals/` - Original video frames as PNG files
   - `overlays/` - Frames with colored mask overlays
   - `binary_masks/` - Pure black/white mask images

### 🎨 Colored Masks with Prime Number Strategy

The new colored masks feature generates videos with uniquely colored masks that can handle overlapping regions intelligently:

**Key Features:**
- **Unique Colors**: Each mask gets a distinct RGB color generated using prime number combinations
- **Overlap Handling**: When masks overlap, prime factorization ensures unique identification
- **Transparent Background**: Areas without masks remain rgba(0,0,0,0) (transparent black)
- **Blended Overlaps**: Overlapping regions show blended colors of contributing masks

**Prime Number Strategy:**
- Each object is assigned a unique prime number (2, 3, 5, 7, 11, ...)
- Pixel values are multiplied by their object's prime number
- Overlapping regions have composite prime values (e.g., 2×3=6 for objects 1&2 overlapping)
- Prime factorization reveals exactly which objects contribute to each pixel
- Colors are blended proportionally for overlapping areas

**Benefits:**
- **Perfect Overlap Detection**: Mathematical guarantee that each overlap combination is unique
- **Visual Clarity**: Each object has a distinct, mathematically-derived color
- **Post-Processing Friendly**: Easy to identify and manipulate specific objects or overlaps
- **Scalable**: Works with any number of objects (limited only by prime number availability)

```bash
# Generate colored masks video
uv run python run_inference.py video.mp4 --colored-masks

# Run the multi-object colored masks example
uv run python colored_masks_example.py
```

### 🌐 Web-Optimized Video Encoding

The new web-optimized encoding generates MP4 videos with H.264 codec that are perfect for web deployment:

**Key Features:**
- **H.264 Codec**: Universal browser support, excellent compression
- **Fast Start**: Optimized for web streaming with `movflags +faststart`
- **Quality Presets**: High, medium, low quality options
- **Automatic Fallback**: Falls back to OpenCV if ffmpeg unavailable
- **Mobile Friendly**: Compatible with all mobile devices

**Codec Hierarchy:**
1. **H.264/AVC**: Best web compatibility (requires ffmpeg)
2. **VP9**: Modern WebM codec, excellent compression
3. **VP8**: Older WebM codec, good compatibility  
4. **mp4v**: Fallback codec (less efficient)

**Quality Presets:**
- **High**: CRF 18, slow preset (best quality, larger files)
- **Medium**: CRF 23, medium preset (balanced)
- **Low**: CRF 28, fast preset (smaller files, faster encoding)

```bash
# Generate web-optimized video with different quality settings
uv run python run_inference.py video.mp4 --web-optimized --quality high
uv run python run_inference.py video.mp4 --web-optimized --quality medium  
uv run python run_inference.py video.mp4 --web-optimized --quality low

# Combine with colored masks for best web experience
uv run python run_inference.py video.mp4 --web-optimized --colored-masks --quality medium
```

**Web Deployment Benefits:**
- ✅ **Universal Compatibility**: Works on all modern browsers
- ✅ **Progressive Download**: Starts playing before fully downloaded
- ✅ **Mobile Optimized**: Perfect for responsive web design
- ✅ **CDN Friendly**: Efficient bandwidth usage
- ✅ **SEO Benefits**: Fast loading improves page performance

### 🔢 Direct RGB-to-ID Encoding (Revolutionary!)

The **groundbreaking** direct RGB-to-ID encoding provides perfect mathematical encoding where each RGB pixel directly encodes which objects are present with **100% accuracy guarantee**:

**Revolutionary Features:**
- **Perfect Encoding**: RGB values directly map to object combinations via prime numbers
- **Lossless Decoding**: Mathematical guarantee of perfect RGB-to-ID recovery
- **Prime Mathematics**: Each object gets a unique prime (2,3,5,7,11...)
- **Overlap Perfection**: Overlaps multiply primes (Objects 1+2 = 2×3 = 6)
- **Web Compatible**: VP9 lossless WebM format for universal browser support

**Mathematical Foundation:**
```
Object 1 → Prime 2 → RGB(2,0,0)
Object 2 → Prime 3 → RGB(3,0,0)  
Object 3 → Prime 5 → RGB(5,0,0)

Overlaps (multiply primes):
Objects 1+2 → 2×3=6 → RGB(6,0,0) → Decode via factorization: [1,2]
Objects 1+3 → 2×5=10 → RGB(10,0,0) → Decode: [1,3]
Objects 1+2+3 → 2×3×5=30 → RGB(30,0,0) → Decode: [1,2,3]
```

**Perfect Decoding Process:**
1. Read RGB pixel: `RGB(6,0,0)`
2. Convert to integer: `6`
3. Prime factorization: `6 = 2 × 3`
4. Map primes to objects: `{2: Object1, 3: Object2}`
5. Result: `[Object1, Object2]` ✅

**Technical Specifications:**
- **24-bit RGB Space**: 0 to 16,777,215 possible values
- **Object Limit**: ~10-12 objects (due to prime multiplication limits)
- **Codec**: VP9 Lossless in WebM format (critical for data integrity!)
- **Background**: RGB(0,0,0) = no objects present
- **Mathematical Guarantee**: Every RGB value uniquely decodes to object set

```bash
# Generate direct-encoded video (lossless, perfect decoding)
uv run python run_inference.py video.mp4 --direct-encoded

# Show mathematical demonstration only
uv run python direct_encoded_example.py --math-only

# Full demo with video processing
uv run python direct_encoded_example.py video.mp4
```

**Use Cases:**
- 🎯 **Data Analysis**: Perfect object identification from RGB values
- 🔬 **Scientific Applications**: Lossless data integrity requirements
- 🎬 **Post-Production**: Frame-by-frame object extraction and manipulation
- 📊 **Computer Vision**: Training data with perfect ground truth
- 🌐 **Web Applications**: Lossless mask data embedded in standard video

**Advantages over Other Methods:**
- ❌ **Alpha Encoding**: Limited to ~255 objects, lossy compression issues
- ❌ **Colored Masks**: Visually appealing but no mathematical decoding guarantee
- ❌ **Separate Files**: Multiple files, sync issues, more complex deployment
- ✅ **Direct RGB**: Single file, perfect decoding, mathematical guarantee, web-ready

**Example Decoding Code:**
```python
from video_prediction import SAM2VideoProcessor

processor = SAM2VideoProcessor("facebook/sam2-hiera-large")

# Load your direct-encoded video and prime mapping
prime_mapping = {1: 2, 2: 3, 3: 5}  # Saved during encoding

# Read a pixel from the video
r, g, b = 6, 0, 0  # Example pixel

# Decode to object IDs
object_ids = processor.decode_rgb_to_object_ids(r, g, b, prime_mapping)
print(f"RGB({r},{g},{b}) contains objects: {object_ids}")  # [1, 2]
```

### 🎭 Unique IDs & Alpha Values

The unique IDs feature allows you to segment multiple objects with each getting a unique identifier and alpha value:

- **Multiple Prompts**: Support for multiple point and box prompts in a single video
- **Sequential IDs**: Each prompt gets a unique sequential ID (1, 2, 3, ...)
- **Unique Alpha Values**: Each object gets a unique alpha value distributed from 0.1 to 1.0
- **Video Propagation**: All objects are tracked and propagated through the entire video

**Benefits of Unique Alpha Values:**
- Each object can be individually controlled in post-processing
- Alpha values can be used to identify specific objects (Object 1 = α 0.1, Object 2 = α 0.55, etc.)
- Perfect for compositing and visual effects workflows
- Enables object-specific editing and analysis
- Import into video editors that support alpha channels for advanced compositing

### Alpha Channel Videos

Generate WebM videos with VP9 codec where masks are encoded in the alpha channel:

```bash
# Basic alpha channel video (using OpenCV)
uv run python run_inference.py video.mp4 --alpha-video

# High-quality alpha channel video (using ffmpeg - recommended)
uv run python run_inference.py video.mp4 --alpha-video --use-ffmpeg

# Both regular and alpha videos
uv run python run_inference.py video.mp4 --alpha-video --use-ffmpeg
```

**Alpha Channel Benefits:**
- Masks are embedded in the video file itself
- Perfect transparency support
- Smaller file sizes than separate mask files
- Easy to composite over other backgrounds
- Professional video editing software compatible

### Basic Usage (Your Pattern)

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device="mps")

with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    state = predictor.init_state(your_video_path)  # Can be JPEG folder or MP4
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state, 
        frame_idx=0,
        obj_id=1,
        points=[[100, 150]],
        labels=[1]
    )
    
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # Process your masks here - they work perfectly!
        pass
```

### Advanced Usage with SAM2VideoProcessor

```python
from video_prediction import SAM2VideoProcessor

# Initialize processor
processor = SAM2VideoProcessor("facebook/sam2-hiera-large")

# Run inference with point prompts
frames, masks, temp_dir = processor.predict_video_with_points(
    video_path="your_video.mp4",
    frame_idx=50,  # Middle frame
    points=[[960, 540]],  # Center point
    max_frames=100
)

# Save results in multiple formats
processor.visualize_results(frames, masks, 50, [[960, 540]])
processor.save_masks_as_video(frames, masks, "output_video.mp4")
processor.save_mask_frames_to_directory(
    frames, 
    masks, 
    output_dir="output_frames",
    save_originals=True,
    save_overlays=True,
    save_binary_masks=True
)

# Clean up
processor.cleanup_temp_dir(temp_dir)
```

### Available Scripts

1. **`run_inference.py`** - Command-line script for any video file
2. **`simple_example.py`** - Minimal example following your exact pattern
3. **`video_prediction.py`** - Full-featured class with utilities for:
   - Loading videos from files or image directories
   - Visualizing results
   - Saving masked videos
   - **NEW**: Saving individual mask frames to organized directories
   - Device auto-detection

### Running the Demo

```bash
# Run with your own video
uv run python run_inference.py videos/your_video.mp4

# Or run the built-in example (processes scene_03_04.mp4)
uv run python run_inference.py
```

### Using with Your Own Videos

#### Option 1: Command Line (Recommended)
```bash
uv run python run_inference.py path/to/video.mp4 --points "x,y" --max-frames 100
```

#### Option 2: Python Code
```python
from video_prediction import SAM2VideoProcessor

processor = SAM2VideoProcessor()
frames, masks, temp_dir = processor.predict_video_with_points(
    video_path="your_video.mp4",
    frame_idx=25,
    points=[[400, 300]],
    max_frames=50
)
# Save individual frames for analysis
processor.save_mask_frames_to_directory(frames, masks, "analysis_frames")
processor.cleanup_temp_dir(temp_dir)
```

### Prompt Formats

#### Unique IDs for Multiple Objects
```bash
# Run the unique IDs example with multiple prompts
uv run python unique_ids_example.py

# Programmatic usage with custom prompts
python -c "
from video_prediction import SAM2VideoProcessor
prompts = [
    {'frame_idx': 25, 'points': [[400, 300]]},
    {'frame_idx': 25, 'points': [[800, 300]]},
    {'frame_idx': 25, 'box': [100, 100, 300, 250]}
]
processor = SAM2VideoProcessor()
frames, masks, alpha_map, temp_dir = processor.predict_video_with_unique_ids('video.mp4', prompts)
"
```

#### Point Prompts
```bash
# Single point
--points "400,300"

# Multiple points  
--points "400,300 600,500"
```

#### Box Prompts
```bash
# Bounding box [x1,y1,x2,y2]
--box "100,100,500,400"
```

### Output Options

```bash
# Control what frame types to save
--no-originals      # Skip saving original frames
--no-overlays       # Skip saving overlay frames  
--no-binary-masks   # Skip saving binary mask images

# Alpha channel video options
--alpha-video       # Generate WebM with masks in alpha channel
--use-ffmpeg        # Use ffmpeg for better alpha support (requires ffmpeg)

# Example: Only save overlay frames + alpha video
uv run python run_inference.py video.mp4 --no-originals --no-binary-masks --alpha-video --use-ffmpeg
```

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for best performance
- **Memory**: At least 8GB RAM, 4GB+ VRAM for GPU
- **CPU**: Works on CPU but will be slower

The code automatically detects your hardware:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon GPU)  
- CPU (fallback)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce video resolution or number of frames
2. **Model download issues**: Ensure internet connection and Hugging Face access
3. **MPS issues on Apple Silicon**: Code automatically adjusts tensor types

### Test Installation

```bash
# Quick test
uv run python -c "from sam2.sam2_video_predictor import SAM2VideoPredictor; print('SAM2 installed successfully!')"
```

## Dependencies

- `torch>=2.0.0` - PyTorch for deep learning
- `sam-2` - Meta's SAM2 model (installed from GitHub)
- `opencv-python` - Video processing
- `matplotlib` - Visualization
- `pillow` - Image processing
- `transformers` - Model loading
- `numpy` - Array operations

## Examples

See the example scripts for different use cases:
- Basic usage: `simple_example.py`
- Command line: `run_inference.py path/to/video.mp4`
- Full features: Use `SAM2VideoProcessor` class directly
