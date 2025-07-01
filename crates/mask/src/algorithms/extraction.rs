use image::GrayImage;
use crate::{error::Result, traits::ContourExtractor};

/// Imageproc-based contour extractor
#[derive(Debug, Clone, Default)]
pub struct ImageprocContourExtractor;

impl ContourExtractor for ImageprocContourExtractor {
    fn extract_contours(&self, binary_image: &GrayImage) -> Result<Vec<Vec<[f32; 2]>>> {
        let contours = imageproc::contours::find_contours::<i32>(binary_image);
        
        let result = contours
            .into_iter()
            .map(|contour| {
                contour.points
                    .iter()
                    .map(|p| [p.x as f32, p.y as f32])
                    .collect()
            })
            .collect();
        
        Ok(result)
    }
} 