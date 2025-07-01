pub mod builder;

use image::GrayImage;
use crate::{
    error::Result,
    types::ComputedOutline,
    traits::{ImagePreprocessor, ContourExtractor, HoleDetector, ShapePostProcessor},
};

/// A flexible pipeline for outline extraction with multiple processing stages
pub struct Pipeline {
    preprocessors: Vec<Box<dyn ImagePreprocessor>>,
    contour_extractor: Box<dyn ContourExtractor>,
    hole_detector: Box<dyn HoleDetector>,
    postprocessors: Vec<Box<dyn ShapePostProcessor>>,
}

impl Pipeline {
    /// Create a new pipeline builder
    pub fn builder() -> builder::PipelineBuilder {
        builder::PipelineBuilder::new()
    }

    /// Create a new pipeline with the given components
    pub fn new(
        preprocessors: Vec<Box<dyn ImagePreprocessor>>,
        contour_extractor: Box<dyn ContourExtractor>,
        hole_detector: Box<dyn HoleDetector>,
        postprocessors: Vec<Box<dyn ShapePostProcessor>>,
    ) -> Self {
        Self {
            preprocessors,
            contour_extractor,
            hole_detector,
            postprocessors,
        }
    }

    /// Process an image through the entire pipeline
    pub fn process(&self, image: &GrayImage) -> Result<ComputedOutline> {
        // Step 1: Apply all preprocessors in sequence
        let mut processed_image = image.clone();
        for preprocessor in &self.preprocessors {
            processed_image = preprocessor.preprocess(&processed_image)?;
        }

        // Step 2: Extract contours
        let contours = self.contour_extractor.extract_contours(&processed_image)?;

        // Step 3: Detect holes and create shapes
        let mut shapes = self.hole_detector.detect_holes(contours)?;

        // Step 4: Apply all post-processors in sequence
        for postprocessor in &self.postprocessors {
            postprocessor.process(&mut shapes)?;
        }

        // Step 5: Filter out invalid shapes (those with no points)
        shapes.retain(|shape| !shape.exterior.is_empty());

        Ok(ComputedOutline {
            shapes,
            image_width: image.width(),
            image_height: image.height(),
        })
    }

    /// Get information about the pipeline configuration
    pub fn info(&self) -> String {
        format!(
            "Pipeline: {} preprocessors, 1 contour extractor, 1 hole detector, {} postprocessors",
            self.preprocessors.len(),
            self.postprocessors.len()
        )
    }
} 