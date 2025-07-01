pub mod preprocessing;
pub mod extraction;
pub mod detection;
pub mod simplification;

pub use preprocessing::*;
pub use extraction::*;
pub use detection::*;
pub use simplification::*;

use crate::{
    error::Result,
    types::ComplexShape,
    traits::{ImagePreprocessor, ContourExtractor, HoleDetector, OutlineExtractor},
};

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