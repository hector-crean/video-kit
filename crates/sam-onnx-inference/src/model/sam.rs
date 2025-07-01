use burn::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use super::{ViT, ViTConfig, PromptEncoder, MaskDecoder};

/// The main SAM (Segment Anything Model) implementation
#[derive(Module, Debug)]
pub struct Sam<B: Backend> {
    /// Image encoder (Vision Transformer)
    pub image_encoder: ViT<B>,
    /// Prompt encoder for points, boxes, and mask prompts
    pub prompt_encoder: PromptEncoder<B>,
    /// Mask decoder that generates segmentation masks
    pub mask_decoder: MaskDecoder<B>,
    /// Pixel mean for normalization
    pixel_mean: Tensor<B, 4>,
    /// Pixel std for normalization
    pixel_std: Tensor<B, 4>,
}

impl<B: Backend> Sam<B> {
    /// Create a new SAM model with the given configuration
    pub fn new(config: &ViTConfig, device: &B::Device) -> Self {
        let image_encoder = ViT::new(config, device);
        let prompt_encoder = PromptEncoder::new(config.embed_dim, device);
        let mask_decoder = MaskDecoder::new(config.embed_dim, device);
        
        // ImageNet normalization values used by SAM
        let pixel_mean = Tensor::from_floats([[[[123.675]], [[116.28]], [[103.53]]]], device);
        let pixel_std = Tensor::from_floats([[[[58.395]], [[57.12]], [[57.375]]]], device);

        Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_mean,
            pixel_std,
        }
    }

    /// Encode an image into feature embeddings
    pub fn encode_image(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        // Normalize the image using ImageNet statistics
        let normalized_image = (image - self.pixel_mean.clone()) / self.pixel_std.clone();
        
        // Extract features using the image encoder
        self.image_encoder.forward(normalized_image)
    }

    /// Encode prompts (points, boxes, masks) into embeddings
    pub fn encode_prompts(
        &self,
        points: Option<(&[(f32, f32)], &[i32])>, // (coordinates, labels)
        boxes: Option<&[(f32, f32, f32, f32)]>,   // (x1, y1, x2, y2)
        masks: Option<Tensor<B, 4>>,               // Previous mask predictions
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        self.prompt_encoder.forward(points, boxes, masks)
    }

    /// Decode masks from image and prompt embeddings
    pub fn decode_masks(
        &self,
        image_embeddings: Tensor<B, 4>,
        point_embeddings: Tensor<B, 3>,
        mask_embeddings: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.mask_decoder.predict_masks(image_embeddings, point_embeddings, mask_embeddings)
    }

    /// Full forward pass with point prompts
    pub fn forward_points(
        &self,
        image: Tensor<B, 4>,
        points: &[(f32, f32)],
        labels: &[i32],
    ) -> Tensor<B, 4> {
        // Encode image
        let image_embeddings = self.encode_image(image);
        
        // Encode prompts
        let (point_embeddings, mask_embeddings) = self.encode_prompts(
            Some((points, labels)),
            None,
            None,
        );
        
        // Decode masks
        self.decode_masks(image_embeddings, point_embeddings, mask_embeddings)
    }

    /// Full forward pass with box prompts
    pub fn forward_boxes(
        &self,
        image: Tensor<B, 4>,
        boxes: &[(f32, f32, f32, f32)],
    ) -> Tensor<B, 4> {
        // Encode image
        let image_embeddings = self.encode_image(image);
        
        // Encode prompts
        let (point_embeddings, mask_embeddings) = self.encode_prompts(
            None,
            Some(boxes),
            None,
        );
        
        // Decode masks
        self.decode_masks(image_embeddings, point_embeddings, mask_embeddings)
    }

    /// Get the device this model is on
    pub fn device(&self) -> B::Device {
        self.pixel_mean.device()
    }

    /// Set the model to evaluation mode (mainly affects batch norm, dropout, etc.)
    pub fn eval(self) -> Self {
        // In Burn, modules don't have explicit train/eval modes like PyTorch
        // But we can add this for API compatibility
        self
    }
}

/// Configuration for SAM models
#[derive(Debug, Clone)]
pub struct SamConfig {
    pub vit_config: ViTConfig,
    pub prompt_embed_dim: usize,
    pub mask_threshold: f32,
}

impl SamConfig {
    /// Create configuration for SAM-B (Base) model
    pub fn sam_vit_b() -> Self {
        Self {
            vit_config: ViTConfig {
                image_size: 1024,
                patch_size: 16,
                num_classes: 0,
                embed_dim: 768,
                depth: 12,
                num_heads: 12,
                mlp_ratio: 4.0,
                qkv_bias: true,
                window_size: 14,
                global_attn_indices: vec![2, 5, 8, 11],
            },
            prompt_embed_dim: 256,
            mask_threshold: 0.0,
        }
    }

    /// Create configuration for SAM-L (Large) model
    pub fn sam_vit_l() -> Self {
        Self {
            vit_config: ViTConfig {
                image_size: 1024,
                patch_size: 16,
                num_classes: 0,
                embed_dim: 1024,
                depth: 24,
                num_heads: 16,
                mlp_ratio: 4.0,
                qkv_bias: true,
                window_size: 14,
                global_attn_indices: vec![5, 11, 17, 23],
            },
            prompt_embed_dim: 256,
            mask_threshold: 0.0,
        }
    }

    /// Create configuration for SAM-H (Huge) model
    pub fn sam_vit_h() -> Self {
        Self {
            vit_config: ViTConfig {
                image_size: 1024,
                patch_size: 16,
                num_classes: 0,
                embed_dim: 1280,
                depth: 32,
                num_heads: 16,
                mlp_ratio: 4.0,
                qkv_bias: true,
                window_size: 14,
                global_attn_indices: vec![7, 15, 23, 31],
            },
            prompt_embed_dim: 256,
            mask_threshold: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;

    #[test]
    fn test_sam_creation() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = SamConfig::sam_vit_b().vit_config;
        let sam = Sam::<TestBackend>::new(&config, &device);
        
        // Test that the model was created successfully
        assert_eq!(sam.device(), device);
    }

    #[test]
    fn test_sam_configs() {
        let sam_b = SamConfig::sam_vit_b();
        assert_eq!(sam_b.vit_config.embed_dim, 768);
        
        let sam_l = SamConfig::sam_vit_l();
        assert_eq!(sam_l.vit_config.embed_dim, 1024);
        
        let sam_h = SamConfig::sam_vit_h();
        assert_eq!(sam_h.vit_config.embed_dim, 1280);
    }

    #[test]
    fn test_image_encoding() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = SamConfig::sam_vit_b().vit_config;
        let sam = Sam::<TestBackend>::new(&config, &device);
        
        // Test image encoding with a dummy input
        let image = Tensor::random([1, 3, 1024, 1024], Distribution::Normal(0.0, 1.0), &device);
        let embeddings = sam.encode_image(image);
        
        // Check that embeddings have expected shape
        let [batch, channels, height, width] = embeddings.dims();
        assert_eq!(batch, 1);
        assert!(channels > 0);
        assert!(height > 0);
        assert!(width > 0);
    }
} 