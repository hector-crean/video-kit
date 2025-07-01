# Mask Outline Extraction with Enhanced geo Algorithms

A powerful, trait-based Rust library for extracting polyline outlines from mask images, now enhanced with sophisticated algorithms from the [geo](https://docs.rs/geo/0.30.0/geo/algorithm/index.html) crate. Supports complex shapes with holes, multiple processing algorithms, and a composable pipeline architecture.

## 🚀 Enhanced Features

- **Advanced Simplification**: Douglas-Peucker and Visvalingam-Whyatt algorithms from geo crate
- **Professional Smoothing**: Chaikin's algorithm for high-quality curve smoothing
- **Geometric Operations**: Convex hull, validation, and advanced geometric computations
- **Pipeline Architecture**: Composable processing steps with multiple preprocessing and post-processing options
- **Topology-Aware**: Proper handling of holes and complex topological relationships
- **Multiple Export Formats**: Native Rust structures and standards-compliant GeoJSON

## 🔬 Algorithm Showcase

### Simplification Algorithms

Our library now leverages the well-tested, optimized algorithms from the geo crate:

1. **Douglas-Peucker** (`geo::Simplify`): Industry-standard line simplification
2. **Visvalingam-Whyatt** (`geo::SimplifyVw`): Area-based simplification for smoother results
3. **Chaikin Smoothing** (`geo::ChaikinSmoothing`): Professional curve smoothing

### Geometric Operations

- **Area & Centroid**: Using `geo::Area` and `geo::Centroid` for accurate calculations
- **Convex Hull**: `geo::ConvexHull` for shape analysis
- **Validation**: Built-in geometry validation with OGC compliance checking
- **Bounding Rectangles**: `geo::BoundingRect` for spatial indexing

## 📊 Performance Results

From our enhanced demo, leveraging geo crate algorithms achieves:

- **Douglas-Peucker**: 96.3% point reduction (540 → 20 points)
- **Visvalingam-Whyatt**: 93.5% point reduction (540 → 35 points)  
- **Comprehensive Pipeline**: 95.8% reduction with multi-step processing
- **Chaikin Smoothing**: High-quality curve interpolation

## 🎯 Quick Start

### Basic Usage with Enhanced Algorithms

```rust
use mask::Pipeline;

// Douglas-Peucker simplification (geo crate)
let pipeline = Pipeline::builder()
    .with_simplification(2.0)
    .build();

// Visvalingam-Whyatt simplification (geo crate)  
let pipeline = Pipeline::builder()
    .with_vw_simplification(2.0)
    .build();

// Chaikin smoothing (geo crate)
let pipeline = Pipeline::builder()
    .with_chaikin_smoothing(2)  // 2 iterations
    .build();
```

### Comprehensive Processing Pipeline

```rust
use mask::{Pipeline, algorithms::*};

let pipeline = Pipeline::builder()
    .add_preprocessor(GaussianBlurPreprocessor { sigma: 1.0 })
    .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
    .set_hole_detector(AreaBasedHoleDetector::default())
    .with_chaikin_smoothing(1)
    .with_simplification(1.5)
    .with_validation()
    .build();

let result = pipeline.process(&image)?;
result.save_geojson("output.geojson")?;
```

### Geometric Analysis

```rust
for shape in result.shapes {
    let area = shape.area();           // Using geo::Area
    let centroid = shape.centroid();   // Using geo::Centroid
    let bbox = shape.bounding_box();   // Built-in implementation
    let perimeter = shape.perimeter(); // Calculated perimeter
    
    println!("Shape: area={}, centroid=({}, {})", 
             area, centroid[0], centroid[1]);
}
```

## 🏗️ Architecture

### Modular Design

```
src/
├── lib.rs              // Main exports and documentation  
├── error.rs            // Enhanced error handling with thiserror
├── types.rs            // Core data structures with geo integration
├── traits/             // Trait definitions for extensibility
├── algorithms/         // Algorithm implementations
│   ├── preprocessing.rs    // Image preprocessing (blur, threshold)
│   ├── extraction.rs       // Contour extraction (imageproc)
│   ├── detection.rs        // Hole detection algorithms
│   └── simplification.rs   // geo crate algorithm integration
├── pipeline/           // Composable pipeline system
├── io/                 // GeoJSON I/O with official geojson crate
└── manager.rs          // Legacy compatibility layer
```

### Key Enhancements

1. **geo Crate Integration**: Professional-grade geometric algorithms
2. **imageproc Leveraging**: Optimized contour detection and image processing
3. **Type Safety**: Comprehensive error handling with `thiserror`
4. **Standards Compliance**: Official `geojson` crate for proper typing
5. **Performance**: Optimized algorithms with significant point reduction

## 🔧 Available Algorithms

### From geo Crate
- **Simplification**: Douglas-Peucker, Visvalingam-Whyatt
- **Smoothing**: Chaikin's algorithm with configurable iterations
- **Geometric**: Area, centroid, bounding rectangles, convex hull
- **Validation**: OGC-compliant geometry validation

### From imageproc Crate  
- **Contour Detection**: Suzuki-Abe border following algorithm
- **Preprocessing**: Connected components, morphological operations
- **Region Analysis**: Advanced shape analysis capabilities

### Custom Implementations
- **Hole Detection**: Containment-based and area-ratio algorithms
- **Multi-step Preprocessing**: Gaussian blur, adaptive thresholding
- **Pipeline Composition**: Flexible algorithm chaining

## 📈 Benchmarks

Our enhanced algorithms achieve excellent performance:

| Algorithm | Original Points | Final Points | Reduction |
|-----------|----------------|--------------|-----------|
| Douglas-Peucker (ε=2.0) | 540 | 20 | 96.3% |
| Visvalingam-Whyatt (ε=2.0) | 540 | 35 | 93.5% |
| Comprehensive Pipeline | 850 | 36 | 95.8% |

## 🎮 Demo Results

Run `cargo run` to see the enhanced algorithms in action:

```bash
🎯 Mask Outline Extraction Demo with Enhanced geo Algorithms
============================================================

🔬 Demo 1: Enhanced Simplification Algorithms
---------------------------------------------
   📊 Original total points: 540
   🔹 Douglas-Peucker (ε=2.0): 20 points (96.3% reduction)
   🔸 Visvalingam-Whyatt (ε=2.0): 35 points (93.5% reduction)

🎨 Demo 2: geo Crate Algorithm Showcase  
---------------------------------------
   🌊 Chaikin smoothing: 1 shapes smoothed with 2 iterations
   🔺 Convex hull: 2 shapes converted to convex hulls
      • Hull 0: area=20010.0, centroid=(150.0, 125.0)

🚀 Demo 3: Comprehensive Processing Pipeline
--------------------------------------------
   📈 Pipeline: 2 preprocessors, 1 contour extractor, 1 hole detector, 3 postprocessors
   📊 Result: 3 valid shapes
   📏 Total area: 13988.3, Average perimeter: 375.9
   🎯 Point reduction: 95.8%
```

## 📦 Dependencies

```toml
[dependencies]
image = "0.25"
imageproc = "0.25"  
geo = "0.30"         # Enhanced geometric algorithms
geo-types = "0.7"    # Geometric primitives
geojson = "0.24"     # Official GeoJSON support
thiserror = "1.0"    # Enhanced error handling
serde = { version = "1.0", features = ["derive"] }
```

## 🎯 Use Cases

- **GIS Applications**: High-quality geographic data processing
- **Computer Vision**: Shape analysis and feature extraction  
- **Cartography**: Map simplification and generalization
- **CAD Systems**: Technical drawing outline extraction
- **Game Development**: Collision mesh generation
- **Scientific Computing**: Geometric analysis and visualization

## 🔬 Technical Excellence

- **Algorithm Quality**: Professional algorithms from the geo crate ecosystem
- **Performance**: Significant point reduction with quality preservation
- **Standards Compliance**: OGC-compatible geometry validation
- **Type Safety**: Comprehensive error handling and type checking
- **Extensibility**: Trait-based architecture for custom implementations
- **Backward Compatibility**: Legacy API support for smooth migration

## 🚀 Future Enhancements

- **Advanced Morphology**: Leverage more imageproc morphological operations
- **Spatial Indexing**: R-tree integration for large-scale processing
- **Parallel Processing**: Multi-threaded pipeline execution
- **Additional Algorithms**: More geometric operations from geo ecosystem
- **Performance Optimization**: SIMD acceleration for critical paths

This enhanced version demonstrates the power of leveraging well-established, optimized algorithms from the Rust geo ecosystem while maintaining a clean, extensible architecture. 