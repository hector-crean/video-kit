use burn::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use image::{DynamicImage, ImageBuffer, Luma};
use anyhow::Result;

pub mod model;
pub mod weights;
pub mod preprocessing;
pub mod postprocessing;

use model::sam::Sam;

/// SAM model variants with their corresponding configurations
#[derive(Debug, Clone)]
pub enum SamVariant {
    Base,
    Large, 
    Huge,
}

impl SamVariant {
    /// Get the Vision Transformer configuration for this SAM variant
    pub fn vit_config(&self) -> model::vit::ViTConfig {
        match self {
            SamVariant::Base => model::vit::ViTConfig {
                image_size: 1024,
                patch_size: 16,
                num_classes: 0, // No classification head
                embed_dim: 768,
                depth: 12,
                num_heads: 12,
                mlp_ratio: 4.0,
                qkv_bias: true,
                window_size: 14,
                global_attn_indices: vec![2, 5, 8, 11],
            },
            SamVariant::Large => model::vit::ViTConfig {
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
            SamVariant::Huge => model::vit::ViTConfig {
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
        }
    }

    /// Get the filename for the checkpoint
    pub fn checkpoint_filename(&self) -> &'static str {
        match self {
            SamVariant::Base => "sam_vit_b_01ec64.pth",
            SamVariant::Large => "sam_vit_l_0b3195.pth",
            SamVariant::Huge => "sam_vit_h_4b8939.pth",
        }
    }

    /// Get the download URL for the checkpoint
    pub fn checkpoint_url(&self) -> &'static str {
        match self {
            SamVariant::Base => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            SamVariant::Large => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            SamVariant::Huge => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
    }
}

/// Create a SAM model with the specified variant
pub fn create_sam_model<B: Backend>(
    variant: SamVariant,
    device: &B::Device,
) -> Sam<B> {
    let config = variant.vit_config();
    Sam::new(&config, device)
}

/// Load a SAM model with pretrained weights
pub fn load_sam_model<B: Backend>(
    variant: SamVariant,
    checkpoint_path: &str,
    device: &B::Device,
) -> anyhow::Result<Sam<B>> {
    weights::load_sam_weights(&variant, checkpoint_path, device)
}

/// SAM model wrapper that handles initialization, weight loading, and inference
pub struct SamModel<B: Backend> {
    model: Sam<B>,
    variant: SamVariant,
}

impl<B: Backend> SamModel<B> {
    /// Create a new SAM model with the specified variant
    pub fn new(variant: SamVariant, device: &B::Device) -> Self {
        let config = variant.vit_config();
        let model = Sam::new(&config, device);
        
        Self { model, variant }
    }

    /// Load pretrained weights from a PyTorch checkpoint
    pub fn load_weights(&mut self, checkpoint_path: &str, device: &B::Device) -> Result<()> {
        self.model = weights::load_sam_weights(&self.variant, checkpoint_path, device)?;
        Ok(())
    }

    /// Run inference on an image with point prompts
    pub fn predict_with_points(
        &self,
        image: &DynamicImage,
        points: &[(f32, f32)],
        labels: &[i32],
    ) -> Result<Vec<ImageBuffer<Luma<u8>, Vec<u8>>>> {
        // Preprocess image
        let image_tensor = preprocessing::preprocess_image::<B>(image, &self.model.device())?;
        
        // Encode image
        let image_embeddings = self.model.encode_image(image_tensor);
        
        // Encode prompts  
        let (point_embeddings, mask_embeddings) = self.model.encode_prompts(
            Some((points, labels)),
            None, // No box prompts
            None, // No mask prompts
        );
        
        // Decode masks
        let masks = self.model.decode_masks(
            image_embeddings,
            point_embeddings,
            mask_embeddings,
        );
        
        // Postprocess masks
        let mask_images = postprocessing::masks_to_images(masks, image.width(), image.height())?;
        
        Ok(mask_images)
    }

    /// Run inference on an image with box prompts
    pub fn predict_with_boxes(
        &self,
        image: &DynamicImage,
        boxes: &[(f32, f32, f32, f32)],
    ) -> Result<Vec<ImageBuffer<Luma<u8>, Vec<u8>>>> {
        // Preprocess image
        let image_tensor = preprocessing::preprocess_image::<B>(image, &self.model.device())?;
        
        // Encode image
        let image_embeddings = self.model.encode_image(image_tensor);
        
        // Encode prompts
        let (point_embeddings, mask_embeddings) = self.model.encode_prompts(
            None, // No point prompts
            Some(boxes),
            None, // No mask prompts
        );
        
        // Decode masks
        let masks = self.model.decode_masks(
            image_embeddings,
            point_embeddings,
            mask_embeddings,
        );
        
        // Postprocess masks
        let mask_images = postprocessing::masks_to_images(masks, image.width(), image.height())?;
        
        Ok(mask_images)
    }

    /// Get the SAM variant
    pub fn variant(&self) -> &SamVariant {
        &self.variant
    }
}

/// Utility function to check if checkpoint exists (and download if download feature is enabled)
pub fn ensure_checkpoint_exists(variant: &SamVariant, checkpoints_dir: &str) -> Result<String> {
    let checkpoint_path = format!("{}/{}", checkpoints_dir, variant.checkpoint_filename());
    
    if std::path::Path::new(&checkpoint_path).exists() {
        return Ok(checkpoint_path);
    }
    
    #[cfg(feature = "download")]
    {
        std::fs::create_dir_all(checkpoints_dir)?;
        
        log::info!("Downloading {} checkpoint...", variant.checkpoint_filename());
        let response = ureq::get(variant.checkpoint_url()).call()?;
        
        let mut file = std::fs::File::create(&checkpoint_path)?;
        std::io::copy(&mut response.into_reader(), &mut file)?;
        
        log::info!("Downloaded checkpoint to {}", checkpoint_path);
        Ok(checkpoint_path)
    }
    
    #[cfg(not(feature = "download"))]
    {
        Err(anyhow::anyhow!(
            "Checkpoint file not found: {}. Please download it manually from {} or compile with --features download",
            checkpoint_path,
            variant.checkpoint_url()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;

    #[test]
    fn test_sam_variant_configs() {
        let base_config = SamVariant::Base.vit_config();
        assert_eq!(base_config.embed_dim, 768);
        assert_eq!(base_config.depth, 12);
        
        let large_config = SamVariant::Large.vit_config();
        assert_eq!(large_config.embed_dim, 1024);
        assert_eq!(large_config.depth, 24);
        
        let huge_config = SamVariant::Huge.vit_config();
        assert_eq!(huge_config.embed_dim, 1280);
        assert_eq!(huge_config.depth, 32);
    }

    #[test]
    fn test_create_sam_model() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let model = create_sam_model::<TestBackend>(SamVariant::Base, &device);
        
        // Test that the model was created successfully
        assert_eq!(model.device(), device);
    }

    #[test]
    fn test_checkpoint_filenames() {
        assert_eq!(SamVariant::Base.checkpoint_filename(), "sam_vit_b_01ec64.pth");
        assert_eq!(SamVariant::Large.checkpoint_filename(), "sam_vit_l_0b3195.pth");
        assert_eq!(SamVariant::Huge.checkpoint_filename(), "sam_vit_h_4b8939.pth");
    }

    #[test]
    fn test_checkpoint_urls() {
        let base_url = SamVariant::Base.checkpoint_url();
        assert!(base_url.starts_with("https://"));
        assert!(base_url.contains("sam_vit_b"));
        
        let large_url = SamVariant::Large.checkpoint_url();
        assert!(large_url.starts_with("https://"));
        assert!(large_url.contains("sam_vit_l"));
        
        let huge_url = SamVariant::Huge.checkpoint_url();
        assert!(huge_url.starts_with("https://"));
        assert!(huge_url.contains("sam_vit_h"));
    }
} 