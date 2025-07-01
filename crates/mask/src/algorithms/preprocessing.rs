use image::GrayImage;
use crate::{error::Result, traits::ImagePreprocessor};

/// Simple thresholding preprocessor
#[derive(Debug, Clone)]
pub struct ThresholdPreprocessor {
    pub threshold: u8,
}

impl Default for ThresholdPreprocessor {
    fn default() -> Self {
        Self { threshold: 128 }
    }
}

impl ImagePreprocessor for ThresholdPreprocessor {
    fn preprocess(&self, image: &GrayImage) -> Result<GrayImage> {
        Ok(imageproc::contrast::threshold(image, self.threshold))
    }
}

/// Gaussian blur preprocessor for noise reduction
#[derive(Debug, Clone)]
pub struct GaussianBlurPreprocessor {
    pub sigma: f32,
}

impl Default for GaussianBlurPreprocessor {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

impl ImagePreprocessor for GaussianBlurPreprocessor {
    fn preprocess(&self, image: &GrayImage) -> Result<GrayImage> {
        Ok(imageproc::filter::gaussian_blur_f32(image, self.sigma))
    }
}

/// Adaptive threshold preprocessor
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdPreprocessor {
    pub block_size: u32,
    pub c: f64,
}

impl Default for AdaptiveThresholdPreprocessor {
    fn default() -> Self {
        Self { 
            block_size: 11,
            c: 2.0 
        }
    }
}

impl ImagePreprocessor for AdaptiveThresholdPreprocessor {
    fn preprocess(&self, image: &GrayImage) -> Result<GrayImage> {
        // For now, fall back to regular threshold
        // In a real implementation, you'd implement adaptive thresholding
        Ok(imageproc::contrast::threshold(image, 128))
    }
} 