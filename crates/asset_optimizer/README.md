# Asset Optimizer

A high-performance Rust tool that uses ffmpeg to optimize video and image assets for web delivery. Converts MP4 videos to WebM and PNG images to WebP format with configurable quality settings and parallel processing.

## Features

- 🎬 **Video Optimization**: Convert MP4 to WebM using VP9 codec
- 🖼️ **Image Optimization**: Convert PNG to WebP format
- ⚡ **Parallel Processing**: Process multiple files concurrently
- 📊 **Progress Tracking**: Real-time progress bars for all conversions
- 🔧 **Configurable Quality**: Adjust video CRF and WebP quality settings
- 📈 **Savings Report**: Shows file size reductions after optimization
- 🔍 **Dry Run Mode**: Preview what will be converted without processing

## Prerequisites

1. **Rust** (latest stable version)

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **FFmpeg** with WebP and VP9 support

   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Installation

1. Clone or create the project:

   ```bash
   cd asset_optimizer
   ```

2. Build the project:

   ```bash
   cargo build --release
   ```

3. The binary will be available at `target/release/asset_optimizer`

## Usage

### Basic Usage

```bash
# Convert all assets in public/assets/
cargo run

# Or use the compiled binary
./target/release/asset_optimizer
```

### Advanced Usage

```bash
# Custom input/output directories
cargo run -- -i /path/to/input -o /path/to/output

# Adjust quality settings
cargo run -- --video-crf 25 --webp-quality 90

# Increase concurrent processing
cargo run -- --concurrent 8

# Preview what will be converted (dry run)
cargo run -- --dry-run
```

### Command Line Options

| Option           | Short | Description                           | Default                   |
| ---------------- | ----- | ------------------------------------- | ------------------------- |
| `--input`        | `-i`  | Input directory containing assets     | `public/assets`           |
| `--output`       | `-o`  | Output directory for optimized assets | `public/assets/optimized` |
| `--video-crf`    |       | Video quality (0-63, lower = better)  | `30`                      |
| `--webp-quality` |       | WebP quality (0-100, higher = better) | `85`                      |
| `--concurrent`   | `-c`  | Maximum concurrent conversions        | `4`                       |
| `--dry-run`      |       | Show what would be converted          | `false`                   |

### Quality Settings Guide

**Video CRF (Constant Rate Factor):**

- `18-23`: High quality, larger files
- `28-32`: Good quality, balanced size (recommended)
- `35-40`: Lower quality, smaller files

**WebP Quality:**

- `80-100`: High quality, larger files
- `70-85`: Good quality, balanced size (recommended)
- `50-70`: Lower quality, smaller files

## Example Output

```
🔍 Scanning for assets in: public/assets
📊 Found 26 videos and 371 images to optimize

🎬 Converting videos to WebM...
✅ [########################################] 26/26 Scene_3.1.mp4

🖼️  Converting images to WebP...
✅ [########################################] 371/371 Scene_2.2.1_00001.png

✅ Asset optimization complete!

📊 Size Comparison:
  Original: 1.0 GB
  Optimized: 650.0 MB
  Savings: 350.0 MB (35.0%)
```

## Integration with Your Project

After optimization, update your code to use the new formats with fallbacks:

### Video Elements

```html
<video controls>
  <source src="/assets/optimized/Scene_3.1.webm" type="video/webm" />
  <source src="/assets/Scene_3.1.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
```

### Image Elements

```html
<picture>
  <source srcset="/assets/optimized/image.webp" type="image/webp" />
  <img src="/assets/image.png" alt="Description" />
</picture>
```

### Next.js Integration

```jsx
// components/OptimizedVideo.tsx
export function OptimizedVideo({ src, ...props }) {
  const baseName = src.replace(/\.[^/.]+$/, "");

  return (
    <video {...props}>
      <source src={`${baseName}.webm`} type="video/webm" />
      <source src={`${baseName}.mp4`} type="video/mp4" />
    </video>
  );
}

// components/OptimizedImage.tsx
export function OptimizedImage({ src, alt, ...props }) {
  const baseName = src.replace(/\.[^/.]+$/, "");

  return (
    <picture>
      <source srcSet={`${baseName}.webp`} type="image/webp" />
      <img src={src} alt={alt} {...props} />
    </picture>
  );
}
```

## Performance Benefits

Based on typical results:

- **Video files**: 30-40% size reduction
- **PNG images**: 25-35% size reduction
- **Overall**: 300-400MB savings on a 1GB asset folder
- **Load times**: Significantly faster, especially on mobile

## Browser Support

- **WebM**: Chrome 6+, Firefox 4+, Opera 10.6+, Edge 14+
- **WebP**: Chrome 23+, Firefox 65+, Safari 14+, Edge 18+

The script generates optimized versions while keeping originals as fallbacks, ensuring universal compatibility.

## Troubleshooting

### FFmpeg not found

```bash
# Verify ffmpeg installation
ffmpeg -version

# Check if VP9 and WebP codecs are available
ffmpeg -codecs | grep -E "(vp9|webp)"
```

### Permission errors

```bash
# Make sure output directory is writable
chmod -R 755 public/assets/optimized
```

### Out of memory errors

```bash
# Reduce concurrent processing
cargo run -- --concurrent 2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - feel free to use in your projects!
