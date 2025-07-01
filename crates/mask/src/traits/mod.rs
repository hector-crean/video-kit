use image::GrayImage;
use crate::{error::Result, types::ComplexShape};

/// Trait for image preprocessing algorithms
pub trait ImagePreprocessor: Send + Sync  {
    /// Preprocess the input image (e.g., blur, threshold)
    fn preprocess(&self, image: &GrayImage) -> Result<GrayImage>;
}

/// Trait for contour extraction algorithms
pub trait ContourExtractor: Send + Sync  {
    /// Extract contours from a binary image
    fn extract_contours(&self, image: &GrayImage) -> Result<Vec<Vec<[f32; 2]>>>;
}

/// Trait for hole detection algorithms
pub trait HoleDetector: Send + Sync {
    /// Detect holes in the given contours and return complex shapes
    fn detect_holes(&self, contours: Vec<Vec<[f32; 2]>>) -> Result<Vec<ComplexShape>>;
}

/// Trait for shape simplification algorithms
pub trait ShapeSimplifier: Send + Sync {
    /// Simplify the shapes by reducing point count
    fn simplify(&self, shapes: &mut [ComplexShape], tolerance: f32) -> Result<()>;
}

/// Trait for shape post-processing algorithms
pub trait ShapePostProcessor: Send + Sync {
    /// Post-process the extracted shapes
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()>;
}

/// Main trait for outline extraction
pub trait OutlineExtractor: Send + Sync {
    /// Extract outlines from a grayscale image
    fn extract_outlines(&self, image: &GrayImage) -> Result<Vec<ComplexShape>>;
} 