use crate::kernel::{
    extract_outline_kernel, gaussian_blur_kernel, threshold_kernel, 
    morphology_kernel, fused_pipeline_kernel
};
use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor, TensorPrimitive},
};
use burn_wgpu::{
    BoolElement, CubeBackend, FloatElement, IntElement, WgpuRuntime,
};

/// The backend trait for our custom image processing operations.
pub trait ProcessingBackend: Backend {
    /// Extracts the outline of an image represented by a tensor.
    fn extract_outline<const D: usize>(tensor: Tensor<Self, D, Float>) -> Tensor<Self, D, Float>;
    
    /// Applies Gaussian blur to an image tensor.
    fn gaussian_blur<const D: usize>(tensor: Tensor<Self, D, Float>, radius: u32) -> Tensor<Self, D, Float>;
    
    /// Applies threshold operation to create binary image.
    fn threshold<const D: usize>(tensor: Tensor<Self, D, Float>, threshold_value: f32, max_value: f32) -> Tensor<Self, D, Float>;
    
    /// Applies morphological operations (erosion or dilation).
    fn morphology<const D: usize>(tensor: Tensor<Self, D, Float>, kernel_size: u32, is_erosion: bool) -> Tensor<Self, D, Float>;
    
    /// Applies fused pipeline (blur + edge detection + threshold) in single pass.
    fn fused_pipeline<const D: usize>(
        tensor: Tensor<Self, D, Float>, 
        blur_radius: u32, 
        threshold_value: f32, 
        edge_threshold: f32
    ) -> Tensor<Self, D, Float>;
}

impl<F: FloatElement, I: IntElement, B: BoolElement> ProcessingBackend
    for CubeBackend<WgpuRuntime, F, I, B>
{
    fn extract_outline<const D: usize>(tensor: Tensor<Self, D, Float>) -> Tensor<Self, D, Float> {
        let result_tensor = extract_outline_kernel::<F>(tensor.into_primitive().tensor());
        Tensor::from_primitive(TensorPrimitive::Float(result_tensor))
    }
    
    fn gaussian_blur<const D: usize>(tensor: Tensor<Self, D, Float>, radius: u32) -> Tensor<Self, D, Float> {
        let result_tensor = gaussian_blur_kernel::<F>(tensor.into_primitive().tensor(), radius);
        Tensor::from_primitive(TensorPrimitive::Float(result_tensor))
    }
    
    fn threshold<const D: usize>(tensor: Tensor<Self, D, Float>, threshold_value: f32, max_value: f32) -> Tensor<Self, D, Float> {
        let result_tensor = threshold_kernel::<F>(tensor.into_primitive().tensor(), threshold_value, max_value);
        Tensor::from_primitive(TensorPrimitive::Float(result_tensor))
    }
    
    fn morphology<const D: usize>(tensor: Tensor<Self, D, Float>, kernel_size: u32, is_erosion: bool) -> Tensor<Self, D, Float> {
        let result_tensor = morphology_kernel::<F>(tensor.into_primitive().tensor(), kernel_size, is_erosion);
        Tensor::from_primitive(TensorPrimitive::Float(result_tensor))
    }
    
    fn fused_pipeline<const D: usize>(
        tensor: Tensor<Self, D, Float>, 
        blur_radius: u32, 
        threshold_value: f32, 
        edge_threshold: f32
    ) -> Tensor<Self, D, Float> {
        let result_tensor = fused_pipeline_kernel::<F>(
            tensor.into_primitive().tensor(), 
            blur_radius, 
            threshold_value, 
            edge_threshold
        );
        Tensor::from_primitive(TensorPrimitive::Float(result_tensor))
    }
} 