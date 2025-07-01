use burn::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use image::{ImageBuffer, Luma, Rgb, RgbImage, GrayImage};
use anyhow::Result;

/// Configuration for postprocessing SAM outputs
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    /// Threshold for binarizing masks
    pub mask_threshold: f32,
    /// Whether to apply smoothing to masks
    pub apply_smoothing: bool,
    /// Color for visualizing masks
    pub mask_color: [u8; 3],
    /// Alpha value for mask overlay
    pub mask_alpha: f32,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            mask_threshold: 0.0,
            apply_smoothing: false,
            mask_color: [255, 0, 0], // Red
            mask_alpha: 0.6,
        }
    }
}

/// Convert SAM mask tensors to binary mask images
pub fn masks_to_images<B: Backend>(
    masks: Tensor<B, 4>,
    output_width: u32,
    output_height: u32,
) -> Result<Vec<ImageBuffer<Luma<u8>, Vec<u8>>>> {
    masks_to_images_with_config(masks, output_width, output_height, &PostprocessConfig::default())
}

/// Convert SAM mask tensors to binary mask images with custom configuration
pub fn masks_to_images_with_config<B: Backend>(
    masks: Tensor<B, 4>,
    output_width: u32,
    output_height: u32,
    config: &PostprocessConfig,
) -> Result<Vec<ImageBuffer<Luma<u8>, Vec<u8>>>> {
    let [batch_size, num_masks, height, width] = masks.dims();
    let mut result_masks = Vec::new();
    
    // Extract mask data
    let mask_data = masks.to_data();
    let values = mask_data.as_slice::<f32>().unwrap();
    
    for batch_idx in 0..batch_size {
        for mask_idx in 0..num_masks {
            let mut mask_pixels = Vec::with_capacity((output_width * output_height) as usize);
            
            for y in 0..height {
                for x in 0..width {
                    let idx = batch_idx * num_masks * height * width 
                            + mask_idx * height * width 
                            + y * width + x;
                    
                    let mask_value = values[idx];
                    
                    // Apply threshold and convert to binary
                    let binary_value = if mask_value > config.mask_threshold { 255u8 } else { 0u8 };
                    mask_pixels.push(binary_value);
                }
            }
            
            // Resize to output dimensions if necessary
            let mask_image = if width == output_width as usize && height == output_height as usize {
                ImageBuffer::from_raw(output_width, output_height, mask_pixels)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create mask image"))?
            } else {
                let temp_image = ImageBuffer::from_raw(width as u32, height as u32, mask_pixels)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create temporary mask image"))?;
                resize_mask_image(&temp_image, output_width, output_height)
            };
            
            // Apply smoothing if requested
            let final_mask = if config.apply_smoothing {
                smooth_mask_image(&mask_image)
            } else {
                mask_image
            };
            
            result_masks.push(final_mask);
        }
    }
    
    Ok(result_masks)
}

/// Resize a mask image to the specified dimensions
fn resize_mask_image(
    mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    target_width: u32,
    target_height: u32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (current_width, current_height) = mask.dimensions();
    
    if current_width == target_width && current_height == target_height {
        return mask.clone();
    }
    
    let mut resized_pixels = Vec::with_capacity((target_width * target_height) as usize);
    
    // Simple nearest neighbor resizing
    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = (x as f32 * current_width as f32 / target_width as f32) as u32;
            let src_y = (y as f32 * current_height as f32 / target_height as f32) as u32;
            
            let src_x = src_x.min(current_width - 1);
            let src_y = src_y.min(current_height - 1);
            
            let pixel_value = mask.get_pixel(src_x, src_y)[0];
            resized_pixels.push(pixel_value);
        }
    }
    
    ImageBuffer::from_raw(target_width, target_height, resized_pixels)
        .expect("Failed to create resized mask image")
}

/// Apply smoothing to a mask image
fn smooth_mask_image(mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = mask.dimensions();
    let mut smoothed = mask.clone();
    
    // Simple 3x3 box filter
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut sum = 0u32;
            let mut count = 0u32;
            
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    
                    sum += mask.get_pixel(nx, ny)[0] as u32;
                    count += 1;
                }
            }
            
            let average = (sum / count) as u8;
            smoothed.put_pixel(x, y, Luma([average]));
        }
    }
    
    smoothed
}

/// Create a visualization of masks overlaid on the original image
pub fn visualize_masks_on_image(
    original_image: &image::DynamicImage,
    masks: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    config: &PostprocessConfig,
) -> Result<RgbImage> {
    let rgb_image = original_image.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    let mut result = rgb_image.clone();
    
    for (mask_idx, mask) in masks.iter().enumerate() {
        // Resize mask to match image if necessary
        let resized_mask = if mask.dimensions() != (width, height) {
            resize_mask_image(mask, width, height)
        } else {
            mask.clone()
        };
        
        // Use different colors for different masks
        let mask_color = get_mask_color(mask_idx, config);
        
        // Overlay mask on image
        for y in 0..height {
            for x in 0..width {
                let mask_value = resized_mask.get_pixel(x, y)[0];
                
                if mask_value > 128 { // If mask is active at this pixel
                    let original_pixel = result.get_pixel(x, y);
                    let blended_pixel = blend_colors(
                        [original_pixel[0], original_pixel[1], original_pixel[2]],
                        mask_color,
                        config.mask_alpha,
                    );
                    
                    result.put_pixel(x, y, Rgb(blended_pixel));
                }
            }
        }
    }
    
    Ok(result)
}

/// Get a color for a specific mask index
fn get_mask_color(mask_idx: usize, config: &PostprocessConfig) -> [u8; 3] {
    // Use different colors for different masks
    let colors = [
        [255, 0, 0],   // Red
        [0, 255, 0],   // Green
        [0, 0, 255],   // Blue
        [255, 255, 0], // Yellow
        [255, 0, 255], // Magenta
        [0, 255, 255], // Cyan
    ];
    
    if mask_idx < colors.len() {
        colors[mask_idx]
    } else {
        config.mask_color
    }
}

/// Blend two colors with alpha
fn blend_colors(base: [u8; 3], overlay: [u8; 3], alpha: f32) -> [u8; 3] {
    let alpha = alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    
    [
        (base[0] as f32 * inv_alpha + overlay[0] as f32 * alpha) as u8,
        (base[1] as f32 * inv_alpha + overlay[1] as f32 * alpha) as u8,
        (base[2] as f32 * inv_alpha + overlay[2] as f32 * alpha) as u8,
    ]
}

/// Extract the best mask from multiple mask predictions based on IoU scores
pub fn select_best_mask<B: Backend>(
    masks: Tensor<B, 4>,
    iou_scores: Option<Tensor<B, 2>>,
) -> Result<Tensor<B, 3>> {
    let [batch_size, num_masks, height, width] = masks.dims();
    
    if let Some(scores) = iou_scores {
        // Select mask with highest IoU score
        let best_indices = scores.argmax(1);
        let best_indices_data = best_indices.to_data();
        let indices = best_indices_data.as_slice::<i64>().unwrap();
        
        let mut best_masks: Vec<Tensor<B, 3>> = Vec::new();
        
        for batch_idx in 0..batch_size {
            let best_mask_idx = indices[batch_idx] as usize;
            let mask = masks.clone().slice([batch_idx..(batch_idx + 1), best_mask_idx..(best_mask_idx + 1)]);
            let squeezed_mask: Tensor<B, 3> = mask.squeeze_dims(&[0, 1]);
            best_masks.push(squeezed_mask);
        }
        
        Ok(Tensor::stack(best_masks, 0))
    } else {
        // Default to first mask
        Ok(masks.slice([0..batch_size, 0..1]).squeeze_dims(&[1]))
    }
}

/// Post-process coordinates to match original image dimensions
pub fn postprocess_coordinates(
    coords: &[(f32, f32)],
    original_size: (u32, u32),
    processed_size: (u32, u32),
) -> Vec<(f32, f32)> {
    crate::preprocessing::transform_coordinates_to_original(
        coords,
        original_size,
        processed_size,
        true, // Assuming aspect ratio was preserved
    )
}

/// Create contours from binary masks
pub fn masks_to_contours(
    masks: &[ImageBuffer<Luma<u8>, Vec<u8>>],
) -> Result<Vec<Vec<(u32, u32)>>> {
    let mut all_contours = Vec::new();
    
    for mask in masks {
        let contours = extract_contours_from_mask(mask);
        all_contours.push(contours);
    }
    
    Ok(all_contours)
}

/// Extract contours from a single binary mask
fn extract_contours_from_mask(mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<(u32, u32)> {
    let (width, height) = mask.dimensions();
    let mut contour_points = Vec::new();
    
    // Simple edge detection - look for transitions from 0 to 255
    for y in 0..height {
        for x in 0..width {
            let current_pixel = mask.get_pixel(x, y)[0];
            
            if current_pixel > 128 { // If current pixel is part of mask
                // Check if it's on the edge (has a non-mask neighbor)
                let mut is_edge = false;
                
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let neighbor_pixel = mask.get_pixel(nx as u32, ny as u32)[0];
                            if neighbor_pixel <= 128 {
                                is_edge = true;
                                break;
                            }
                        } else {
                            // Edge of image
                            is_edge = true;
                        }
                    }
                    if is_edge { break; }
                }
                
                if is_edge {
                    contour_points.push((x, y));
                }
            }
        }
    }
    
    contour_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    
    type TestBackend = NdArray;

    fn create_test_masks() -> Tensor<TestBackend, 4> {
        // Create a simple test mask: 2x2 image with a diagonal pattern
        let mask_data = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        
        let data = TensorData::new(mask_data, [1, 1, 2, 2]);
        Tensor::from_data(data, &Default::default())
    }

    #[test]
    fn test_masks_to_images() {
        let masks = create_test_masks();
        let images = masks_to_images(masks, 4, 4).unwrap();
        
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].dimensions(), (4, 4));
    }

    #[test]
    fn test_postprocess_config() {
        let config = PostprocessConfig::default();
        assert_eq!(config.mask_threshold, 0.0);
        assert_eq!(config.mask_color, [255, 0, 0]);
        assert!(!config.apply_smoothing);
    }

    #[test]
    fn test_mask_color_selection() {
        let config = PostprocessConfig::default();
        
        let color0 = get_mask_color(0, &config);
        let color1 = get_mask_color(1, &config);
        
        assert_ne!(color0, color1); // Different masks should have different colors
    }

    #[test]
    fn test_color_blending() {
        let base = [100, 100, 100];
        let overlay = [255, 0, 0];
        let alpha = 0.5;
        
        let blended = blend_colors(base, overlay, alpha);
        
        // Should be somewhere between base and overlay
        assert!(blended[0] > base[0] && blended[0] < overlay[0]);
    }

    #[test]
    fn test_mask_resizing() {
        // Create a 2x2 mask
        let mask_data = vec![255u8, 0, 0, 255];
        let small_mask = ImageBuffer::from_raw(2, 2, mask_data).unwrap();
        
        // Resize to 4x4
        let large_mask = resize_mask_image(&small_mask, 4, 4);
        
        assert_eq!(large_mask.dimensions(), (4, 4));
    }

    #[test]
    fn test_best_mask_selection() {
        let masks = create_test_masks();
        
        // Test without IoU scores (should select first mask)
        let best_mask = select_best_mask(masks.clone(), None).unwrap();
        assert_eq!(best_mask.dims(), [1, 2, 2]);
        
        // Test with IoU scores
        let iou_scores = Tensor::from_floats([[0.8]], &Default::default());
        let best_mask_with_scores = select_best_mask(masks, Some(iou_scores)).unwrap();
        assert_eq!(best_mask_with_scores.dims(), [1, 2, 2]);
    }
} 