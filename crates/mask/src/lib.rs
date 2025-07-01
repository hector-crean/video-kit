//! # Mask Outline Extraction Library
//! 
//! A flexible, trait-based library for extracting outlines from mask images.
//! Supports complex shapes with holes, multiple processing algorithms, and
//! a composable pipeline architecture.
//! 
//! ## Core Features
//! 
//! - **Trait-based Architecture**: Implement custom algorithms by implementing traits
//! - **Pipeline System**: Compose multiple processing steps in a flexible pipeline
//! - **Hole Detection**: Detect and handle shapes with holes (topology-aware)
//! - **GeoJSON Support**: Export/import to standard GeoJSON format
//! - **Multiple Algorithms**: Various preprocessing, extraction, and simplification algorithms
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use mask::Pipeline;
//! use image::open;
//! 
//! // Create a pipeline with default settings
//! let pipeline = Pipeline::builder()
//!     .build();
//! 
//! // Process an image
//! let image = open("mask.png")?.to_luma8();
//! let result = pipeline.process(&image)?;
//! 
//! // Export to GeoJSON
//! result.save_geojson("output.geojson")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//! 
//! ## Custom Pipeline
//! 
//! ```rust,no_run
//! use mask::{Pipeline, algorithms::*};
//! 
//! let pipeline = Pipeline::builder()
//!     .add_preprocessor(GaussianBlurPreprocessor { sigma: 1.0 })
//!     .add_preprocessor(ThresholdPreprocessor { threshold: 150 })
//!     .set_hole_detector(AreaBasedHoleDetector::default())
//!     .with_simplification(2.0)
//!     .build();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Core modules
pub mod error;
pub mod types;
pub mod traits;
pub mod algorithms;
pub mod pipeline;
pub mod io;
pub mod manager; // Legacy compatibility
pub mod mcp;
pub mod typed_geojson;

// Re-exports for convenience
pub use error::{MaskError, Result};
pub use types::{ComplexShape, ComputedOutline};
pub use traits::*;
pub use algorithms::*;
pub use pipeline::{Pipeline, builder::PipelineBuilder};
pub use io::*;
pub use manager::{MaskManager, MaskManagerCommand}; // Legacy

/// Type aliases for common extractor configurations
pub type SimpleExtractor = StandardOutlineExtractor<
    ThresholdPreprocessor,
    ImageprocContourExtractor,
    NoHoleDetector,
>;

pub type HoleAwareExtractor = StandardOutlineExtractor<
    ThresholdPreprocessor,
    ImageprocContourExtractor,
    ContainmentHoleDetector,
>;

/// Standard outline extractor implementation
#[derive(Debug)]
pub struct StandardOutlineExtractor<P, C, H> 
where
    P: ImagePreprocessor,
    C: ContourExtractor,
    H: HoleDetector,
{
    pub preprocessor: P,
    pub contour_extractor: C,
    pub hole_detector: H,
}

impl<P, C, H> StandardOutlineExtractor<P, C, H>
where
    P: ImagePreprocessor,
    C: ContourExtractor,
    H: HoleDetector,
{
    pub fn new(preprocessor: P, contour_extractor: C, hole_detector: H) -> Self {
        Self {
            preprocessor,
            contour_extractor,
            hole_detector,
        }
    }
}

impl<P, C, H> OutlineExtractor for StandardOutlineExtractor<P, C, H>
where
    P: ImagePreprocessor,
    C: ContourExtractor,
    H: HoleDetector,
{
    fn extract_outlines(&self, image: &image::GrayImage) -> Result<Vec<ComplexShape>> {
        let binary_image = self.preprocessor.preprocess(image)?;
        let contours = self.contour_extractor.extract_contours(&binary_image)?;
        self.hole_detector.detect_holes(contours)
    }
}

impl Default for SimpleExtractor {
    fn default() -> Self {
        Self::new(
            ThresholdPreprocessor::default(),
            ImageprocContourExtractor,
            NoHoleDetector,
        )
    }
}

impl Default for HoleAwareExtractor {
    fn default() -> Self {
        Self::new(
            ThresholdPreprocessor::default(),
            ImageprocContourExtractor,
            ContainmentHoleDetector,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_image() -> GrayImage {
        let mut img = GrayImage::new(100, 100);
        for y in 20..80 {
            for x in 20..80 {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }
        img
    }

    #[test]
    fn test_pipeline_basic() {
        let pipeline = Pipeline::builder().build();
        let image = create_test_image();
        
        let result = pipeline.process(&image).expect("Should process successfully");
        assert!(!result.shapes.is_empty(), "Should find at least one shape");
        assert_eq!(result.image_width, 100);
        assert_eq!(result.image_height, 100);
    }

    #[test]
    fn test_pipeline_with_simplification() {
        let pipeline = Pipeline::builder()
            .with_simplification(2.0)
            .build();
        let image = create_test_image();
        
        let result = pipeline.process(&image).expect("Should process successfully");
        assert!(!result.shapes.is_empty(), "Should find at least one shape");
    }

    #[test]
    fn test_legacy_manager_compatibility() {
        let mut manager = MaskManager::new();
        let image = create_test_image();
        manager.set_image(image);
        
        let result = manager.execute(MaskManagerCommand::ExtractOutline)
            .expect("Should extract outline successfully");
        
        assert!(!result.shapes.is_empty(), "Should find at least one shape");
    }

    #[test]
    fn test_geojson_export() {
        let pipeline = Pipeline::builder().build();
        let image = create_test_image();
        
        let result = pipeline.process(&image).expect("Should process successfully");
        let geojson = result.to_geojson().expect("Should create GeoJSON");
        assert!(!geojson.features.is_empty());
    }

    #[test]
    fn test_custom_extractor() {
        let extractor = StandardOutlineExtractor::new(
            ThresholdPreprocessor { threshold: 100 },
            ImageprocContourExtractor,
            NoHoleDetector,
        );
        
        let image = create_test_image();
        let shapes = extractor.extract_outlines(&image).expect("Should extract outlines");
        assert!(!shapes.is_empty(), "Should find at least one shape");
    }
}