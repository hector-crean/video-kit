use burn::prelude::*;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use image::{DynamicImage, ImageBuffer, Rgb, imageops::FilterType};
use anyhow::Result;

/// SAM preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target image size (SAM uses 1024x1024)
    pub target_size: (u32, u32),
    /// Pixel mean for normalization (ImageNet values)
    pub pixel_mean: [f32; 3],
    /// Pixel std for normalization (ImageNet values)
    pub pixel_std: [f32; 3],
    /// Whether to maintain aspect ratio during resize
    pub keep_aspect_ratio: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (1024, 1024),
            pixel_mean: [123.675, 116.28, 103.53],   // ImageNet mean
            pixel_std: [58.395, 57.12, 57.375],      // ImageNet std
            keep_aspect_ratio: true,
        }
    }
}

/// Preprocess an image for SAM inference
/// 
/// This function:
/// 1. Resizes the image to 1024x1024 (with optional aspect ratio preservation)
/// 2. Converts to RGB format
/// 3. Normalizes using ImageNet statistics
/// 4. Converts to tensor format [1, 3, 1024, 1024]
pub fn preprocess_image<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    preprocess_image_with_config(image, &PreprocessConfig::default(), device)
}

/// Preprocess an image with custom configuration
pub fn preprocess_image_with_config<B: Backend>(
    image: &DynamicImage,
    config: &PreprocessConfig,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let (target_width, target_height) = config.target_size;
    
    // Resize image
    let resized_image = if config.keep_aspect_ratio {
        resize_with_aspect_ratio(image, target_width, target_height)
    } else {
        image.resize_exact(target_width, target_height, FilterType::Lanczos3)
    };
    
    // Convert to RGB
    let rgb_image = resized_image.to_rgb8();
    
    // Convert to tensor format
    let tensor = rgb_to_tensor(&rgb_image, config, device)?;
    
    Ok(tensor)
}

/// Resize image while maintaining aspect ratio and padding if necessary
fn resize_with_aspect_ratio(
    image: &DynamicImage, 
    target_width: u32, 
    target_height: u32
) -> DynamicImage {
    let (original_width, original_height) = (image.width(), image.height());
    
    // Calculate scaling factor to fit within target size
    let scale_x = target_width as f32 / original_width as f32;
    let scale_y = target_height as f32 / original_height as f32;
    let scale = scale_x.min(scale_y);
    
    // Calculate new dimensions
    let new_width = (original_width as f32 * scale) as u32;
    let new_height = (original_height as f32 * scale) as u32;
    
    // Resize to new dimensions
    let resized = image.resize(new_width, new_height, FilterType::Lanczos3);
    
    // Create a new image with target dimensions and paste the resized image
    let mut result = DynamicImage::new_rgb8(target_width, target_height);
    
    // Calculate padding offsets to center the image
    let offset_x = (target_width - new_width) / 2;
    let offset_y = (target_height - new_height) / 2;
    
    // Copy the resized image to the center of the target canvas
    for y in 0..new_height {
        for x in 0..new_width {
            if let Some(pixel) = resized.as_rgb8() {
                if let Some(pixel_val) = pixel.get_pixel_checked(x, y) {
                    result.as_mut_rgb8().unwrap().put_pixel(
                        x + offset_x,
                        y + offset_y,
                        *pixel_val,
                    );
                }
            }
        }
    }
    
    result
}

/// Convert RGB image to tensor format
fn rgb_to_tensor<B: Backend>(
    rgb_image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    config: &PreprocessConfig,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let (width, height) = rgb_image.dimensions();
    let mut pixel_data = Vec::with_capacity((width * height * 3) as usize);
    
    // Convert RGB pixels to normalized float values
    // Format: [R, G, B, R, G, B, ...] for the entire image
    for pixel in rgb_image.pixels() {
        let r = (pixel[0] as f32 - config.pixel_mean[0]) / config.pixel_std[0];
        let g = (pixel[1] as f32 - config.pixel_mean[1]) / config.pixel_std[1];
        let b = (pixel[2] as f32 - config.pixel_mean[2]) / config.pixel_std[2];
        
        pixel_data.extend_from_slice(&[r, g, b]);
    }
    
    // Reshape to [1, 3, height, width] format
    let data = TensorData::new(pixel_data, [1, 3, height as usize, width as usize]);
    Ok(Tensor::from_data(data, device))
}

/// Get original image dimensions before preprocessing
pub fn get_original_dimensions(image: &DynamicImage) -> (u32, u32) {
    (image.width(), image.height())
}

/// Calculate scaling factors for converting coordinates back to original image space
pub fn get_scale_factors(
    original_size: (u32, u32),
    processed_size: (u32, u32),
    keep_aspect_ratio: bool,
) -> (f32, f32) {
    let (orig_w, orig_h) = original_size;
    let (proc_w, proc_h) = processed_size;
    
    if keep_aspect_ratio {
        // When aspect ratio is preserved, both dimensions use the same scale
        let scale = (orig_w as f32 / proc_w as f32).max(orig_h as f32 / proc_h as f32);
        (scale, scale)
    } else {
        // When aspect ratio is not preserved, scale each dimension independently
        (orig_w as f32 / proc_w as f32, orig_h as f32 / proc_h as f32)
    }
}

/// Transform coordinates from processed image space back to original image space
pub fn transform_coordinates_to_original(
    coords: &[(f32, f32)],
    original_size: (u32, u32),
    processed_size: (u32, u32),
    keep_aspect_ratio: bool,
) -> Vec<(f32, f32)> {
    let (scale_x, scale_y) = get_scale_factors(original_size, processed_size, keep_aspect_ratio);
    
    coords.iter().map(|&(x, y)| {
        if keep_aspect_ratio {
            // Account for centering when aspect ratio is preserved
            let (orig_w, orig_h) = original_size;
            let (proc_w, proc_h) = processed_size;
            
            let effective_w = orig_w as f32 / scale_x;
            let effective_h = orig_h as f32 / scale_y;
            
            let offset_x = (proc_w as f32 - effective_w) / 2.0;
            let offset_y = (proc_h as f32 - effective_h) / 2.0;
            
            let adjusted_x = (x - offset_x) * scale_x;
            let adjusted_y = (y - offset_y) * scale_y;
            
            (adjusted_x, adjusted_y)
        } else {
            (x * scale_x, y * scale_y)
        }
    }).collect()
}

/// Create a batch of preprocessed images
pub fn preprocess_image_batch<B: Backend>(
    images: &[DynamicImage],
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let config = PreprocessConfig::default();
    let mut batch_tensors = Vec::new();
    
    for image in images {
        let tensor = preprocess_image_with_config(image, &config, device)?;
        batch_tensors.push(tensor);
    }
    
    // Stack tensors along batch dimension
    Ok(Tensor::cat(batch_tensors, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use image::{RgbImage, Rgb};
    
    type TestBackend = NdArray;

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        
        // Fill with a simple pattern
        for y in 0..height {
            for x in 0..width {
                let intensity = ((x + y) * 255 / (width + height)) as u8;
                image.put_pixel(x, y, Rgb([intensity, intensity, intensity]));
            }
        }
        
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn test_preprocess_image() {
        let device = Default::default();
        let image = create_test_image(512, 512);
        
        let tensor = preprocess_image::<TestBackend>(&image, &device).unwrap();
        
        // Check tensor dimensions
        assert_eq!(tensor.dims(), [1, 3, 1024, 1024]);
    }

    #[test]
    fn test_aspect_ratio_preservation() {
        let image = create_test_image(800, 600); // 4:3 aspect ratio
        let resized = resize_with_aspect_ratio(&image, 1024, 1024);
        
        assert_eq!(resized.width(), 1024);
        assert_eq!(resized.height(), 1024);
    }

    #[test]
    fn test_coordinate_transformation() {
        let original_size = (800, 600);
        let processed_size = (1024, 1024);
        let coords = vec![(100.0, 150.0), (400.0, 300.0)];
        
        let transformed = transform_coordinates_to_original(
            &coords,
            original_size,
            processed_size,
            true,
        );
        
        assert_eq!(transformed.len(), coords.len());
        // Coordinates should be scaled appropriately
        assert!(transformed[0].0 > 0.0 && transformed[0].1 > 0.0);
    }

    #[test]
    fn test_scale_factors() {
        let (scale_x, scale_y) = get_scale_factors((800, 600), (1024, 1024), true);
        
        // With aspect ratio preservation, scales should be equal
        assert_eq!(scale_x, scale_y);
        
        let (scale_x2, scale_y2) = get_scale_factors((800, 600), (1024, 1024), false);
        
        // Without aspect ratio preservation, scales can differ
        assert_ne!(scale_x2, scale_y2);
    }

    #[test]
    fn test_preprocessing_config() {
        let config = PreprocessConfig::default();
        assert_eq!(config.target_size, (1024, 1024));
        assert_eq!(config.pixel_mean, [123.675, 116.28, 103.53]);
        assert!(config.keep_aspect_ratio);
    }

    #[test]
    fn test_batch_preprocessing() {
        let device = Default::default();
        let images = vec![
            create_test_image(512, 512),
            create_test_image(256, 256),
            create_test_image(1024, 768),
        ];
        
        let batch_tensor = preprocess_image_batch::<TestBackend>(&images, &device).unwrap();
        
        // Check batch dimensions
        assert_eq!(batch_tensor.dims(), [3, 3, 1024, 1024]);
    }
} 