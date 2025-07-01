use crate::backend::ProcessingBackend;
use burn::tensor::{Float, Tensor};

/// Applies the outline extraction operation on a tensor.
pub fn extract_outline<B: ProcessingBackend, const D: usize>(tensor: Tensor<B, D, Float>) -> Tensor<B, D, Float> {
    B::extract_outline(tensor)
}

/// Applies Gaussian blur using GPU kernel
pub fn gaussian_blur<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    radius: u32,
) -> Tensor<B, D, Float> {
    B::gaussian_blur(tensor, radius)
}

/// Applies threshold operation using GPU kernel
pub fn threshold<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    threshold_value: f32,
    max_value: f32,
) -> Tensor<B, D, Float> {
    B::threshold(tensor, threshold_value, max_value)
}

/// Applies morphological erosion using GPU kernel
pub fn erode<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    kernel_size: u32,
) -> Tensor<B, D, Float> {
    B::morphology(tensor, kernel_size, true)
}

/// Applies morphological dilation using GPU kernel
pub fn dilate<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    kernel_size: u32,
) -> Tensor<B, D, Float> {
    B::morphology(tensor, kernel_size, false)
}

/// Applies fused pipeline (blur + edge detection + threshold) in single GPU pass
pub fn fused_pipeline<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    blur_radius: u32,
    threshold_value: f32,
    edge_threshold: f32,
) -> Tensor<B, D, Float> {
    B::fused_pipeline(tensor, blur_radius, threshold_value, edge_threshold)
}

// Chain operations together efficiently
/// Demonstrates chaining operations: blur -> outline -> threshold
pub fn blur_outline_threshold<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    blur_radius: u32,
    threshold_value: f32,
) -> Tensor<B, D, Float> {
    let blurred = gaussian_blur(tensor, blur_radius);
    let outlined = extract_outline(blurred);
    threshold(outlined, threshold_value, 1.0)
}

/// Advanced pipeline: blur -> outline -> morphology -> threshold
pub fn advanced_pipeline<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    blur_radius: u32,
    morph_kernel_size: u32,
    threshold_value: f32,
) -> Tensor<B, D, Float> {
    let blurred = gaussian_blur(tensor, blur_radius);
    let outlined = extract_outline(blurred);
    let eroded = erode(outlined, morph_kernel_size);  // Clean up noise
    let dilated = dilate(eroded, morph_kernel_size); // Restore shape
    threshold(dilated, threshold_value, 1.0)
}

/// Simple pipeline using the fused kernel for maximum efficiency
pub fn efficient_pipeline<B: ProcessingBackend, const D: usize>(
    tensor: Tensor<B, D, Float>,
    blur_radius: u32,
    threshold_value: f32,
    edge_threshold: f32,
) -> Tensor<B, D, Float> {
    // Single GPU kernel dispatch for blur + edge detection + threshold
    fused_pipeline(tensor, blur_radius, threshold_value, edge_threshold)
} 