use burn::prelude::*;
use burn::nn::{
    Embedding, EmbeddingConfig, 
    conv::{Conv2d, Conv2dConfig}, 
    LayerNorm, LayerNormConfig, 
    PaddingConfig2d
};
use burn::tensor::{Tensor, activation, backend::Backend, Distribution};
use burn::module::{Module, Param};

/// Prompt encoder that converts various prompt types into embeddings
#[derive(Module, Debug)]
pub struct PromptEncoder<B: Backend> {
    /// Embedding dimension
    embed_dim: usize,
    /// Image embedding size (H, W)
    image_embedding_size: (usize, usize),
    /// Input image size (H, W)
    input_image_size: (usize, usize),
    /// Embeddings for point prompts
    point_embeddings: Vec<Param<Tensor<B, 2>>>,
    /// Embedding for "not a point" 
    not_a_point_embed: Param<Tensor<B, 2>>,
    /// Convolutional layers for mask prompts
    mask_input_conv: Vec<Conv2d<B>>,
    /// Layer norm for mask features
    mask_input_norm: LayerNorm<B>,
    /// Number of mask tokens
    num_mask_tokens: usize,
    /// Mask tokens
    mask_tokens: Param<Tensor<B, 2>>,
    /// IoU token
    iou_token: Param<Tensor<B, 2>>,
}

impl<B: Backend> PromptEncoder<B> {
    pub fn new(embed_dim: usize, device: &B::Device) -> Self {
        let image_embedding_size = (64, 64); // 1024 / 16 = 64
        let input_image_size = (1024, 1024);
        
        // Point embeddings for positive/negative points
        let point_embeddings = vec![
            // Positive point embedding
            Param::from_tensor(Tensor::random([1, embed_dim], Distribution::Normal(0.0, 1.0), device)),
            // Negative point embedding
            Param::from_tensor(Tensor::random([1, embed_dim], Distribution::Normal(0.0, 1.0), device)),
        ];
        
        // Not-a-point embedding
        let not_a_point_embed = Param::from_tensor(
            Tensor::random([1, embed_dim], Distribution::Normal(0.0, 1.0), device)
        );
        
        // Mask input processing
        let mask_input_conv = vec![
            Conv2dConfig::new([1, embed_dim / 4], [2, 2])
                .with_stride([2, 2])
                .init(device),
            Conv2dConfig::new([embed_dim / 4, embed_dim / 4], [2, 2])
                .with_stride([2, 2])
                .init(device),
            Conv2dConfig::new([embed_dim / 4, embed_dim], [1, 1])
                .init(device),
        ];
        
        let mask_input_norm = LayerNormConfig::new(embed_dim).init(device);
        
        // Mask tokens
        let num_mask_tokens = 4;
        let mask_tokens = Param::from_tensor(
            Tensor::random([num_mask_tokens, embed_dim], Distribution::Normal(0.0, 1.0), device)
        );
        
        // IoU prediction token
        let iou_token = Param::from_tensor(
            Tensor::random([1, embed_dim], Distribution::Normal(0.0, 1.0), device)
        );

        Self {
            embed_dim,
            image_embedding_size,
            input_image_size,
            point_embeddings,
            not_a_point_embed,
            mask_input_conv,
            mask_input_norm,
            num_mask_tokens,
            mask_tokens,
            iou_token,
        }
    }

    pub fn forward(
        &self,
        points: Option<(&[(f32, f32)], &[i32])>, // (coordinates, labels)
        boxes: Option<&[(f32, f32, f32, f32)]>,   // (x1, y1, x2, y2)
        masks: Option<Tensor<B, 4>>,               // Previous mask predictions
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let device = self.point_embeddings[0].device();
        let batch_size = 1; // For simplicity, assume batch size 1
        
        // Process point prompts
        let point_embeddings = if let Some((coords, labels)) = points {
            self.encode_points(coords, labels, &device)
        } else if let Some(boxes) = boxes {
            self.encode_boxes(boxes, &device)
        } else {
            // No point prompts, use not-a-point embedding
            self.not_a_point_embed.val().unsqueeze_dim(0)
        };
        
        // Process mask prompts
        let mask_embeddings = if let Some(masks) = masks {
            self.encode_masks(masks)
        } else {
            // No mask prompts, return empty tensor
            Tensor::zeros([batch_size, 0, self.embed_dim], &device)
        };
        
        // Combine embeddings
        let sparse_embeddings = if mask_embeddings.dims()[1] > 0 {
            Tensor::cat(vec![point_embeddings, mask_embeddings], 1)
        } else {
            point_embeddings
        };
        
        // Dense embeddings (for masks)
        let dense_embeddings = Tensor::zeros(
            [batch_size, self.embed_dim, self.image_embedding_size.0, self.image_embedding_size.1],
            &device
        );
        
        (sparse_embeddings, dense_embeddings)
    }

    fn encode_points(&self, coords: &[(f32, f32)], labels: &[i32], device: &B::Device) -> Tensor<B, 3> {
        let mut embeddings = Vec::new();
        
        for (i, &(x, y)) in coords.iter().enumerate() {
            let label = labels.get(i).copied().unwrap_or(1); // Default to positive
            
            // Get base embedding based on label
            let base_embed = if label == 1 {
                self.point_embeddings[0].val() // Positive point
            } else {
                self.point_embeddings[1].val() // Negative point
            };
            
            // Add positional encoding
            let pos_embed = self.get_positional_encoding(x, y, device);
            let point_embed = base_embed + pos_embed;
            
            embeddings.push(point_embed);
        }
        
        if embeddings.is_empty() {
            self.not_a_point_embed.val().unsqueeze_dim(0)
        } else {
            Tensor::stack(embeddings, 1)
        }
    }

    fn encode_boxes(&self, boxes: &[(f32, f32, f32, f32)], device: &B::Device) -> Tensor<B, 3> {
        let mut embeddings = Vec::new();
        
        for &(x1, y1, x2, y2) in boxes {
            // Encode each box as two corner points
            let corner1_embed = self.point_embeddings[0].val() + self.get_positional_encoding(x1, y1, device);
            let corner2_embed = self.point_embeddings[0].val() + self.get_positional_encoding(x2, y2, device);
            
            embeddings.push(corner1_embed);
            embeddings.push(corner2_embed);
        }
        
        if embeddings.is_empty() {
            self.not_a_point_embed.val().unsqueeze_dim(0)
        } else {
            Tensor::stack(embeddings, 1)
        }
    }

    fn encode_masks(&self, masks: Tensor<B, 4>) -> Tensor<B, 3> {
        // Process mask through convolutions
        let mut x = masks;
        
        for conv in &self.mask_input_conv {
            x = conv.forward(x);
            x = activation::gelu(x);
        }
        
        // Global average pooling
        let [batch_size, channels, height, width] = x.dims();
        x = x.mean_dim(2).mean_dim(2); // [batch_size, channels]
        
        // Reshape to [batch_size, 1, channels] for consistency
        x.reshape([batch_size, 1, channels])
    }

    fn get_positional_encoding(&self, x: f32, y: f32, device: &B::Device) -> Tensor<B, 2> {
        // Scale coordinates to [0, 1] range
        let x_scaled = x / self.input_image_size.1 as f32;
        let y_scaled = y / self.input_image_size.0 as f32;
        
        // Simple positional encoding using sine/cosine
        let mut pos_embed = Vec::new();
        
        // Generate frequency components
        for i in 0..(self.embed_dim / 4) {
            let freq = 2.0_f32.powi(i as i32);
            pos_embed.push((x_scaled * freq).sin());
            pos_embed.push((x_scaled * freq).cos());
            pos_embed.push((y_scaled * freq).sin());
            pos_embed.push((y_scaled * freq).cos());
        }
        
        // Pad to exact embed_dim if needed
        while pos_embed.len() < self.embed_dim {
            pos_embed.push(0.0);
        }
        pos_embed.truncate(self.embed_dim);
        
        Tensor::<B, 1>::from_floats(pos_embed.as_slice(), device).reshape([1, self.embed_dim])
    }

    /// Get mask tokens for the decoder
    pub fn get_mask_tokens(&self) -> Tensor<B, 2> {
        self.mask_tokens.val()
    }

    /// Get IoU token for quality prediction
    pub fn get_iou_token(&self) -> Tensor<B, 2> {
        self.iou_token.val()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;

    #[test]
    fn test_prompt_encoder_creation() {
        let device = Default::default();
        let encoder = PromptEncoder::<TestBackend>::new(256, &device);
        
        assert_eq!(encoder.embed_dim, 256);
        assert_eq!(encoder.num_mask_tokens, 4);
    }

    #[test]
    fn test_point_encoding() {
        let device = Default::default();
        let encoder = PromptEncoder::<TestBackend>::new(256, &device);
        
        let points = vec![(100.0, 150.0), (200.0, 250.0)];
        let labels = vec![1, 0]; // positive, negative
        
        let (sparse_embeddings, dense_embeddings) = encoder.forward(
            Some((&points, &labels)),
            None,
            None,
        );
        
        assert_eq!(sparse_embeddings.dims()[1], 2); // Two points
        assert_eq!(sparse_embeddings.dims()[2], 256); // Embed dim
        assert_eq!(dense_embeddings.dims()[1], 256); // Embed dim
    }

    #[test]
    fn test_box_encoding() {
        let device = Default::default();
        let encoder = PromptEncoder::<TestBackend>::new(256, &device);
        
        let boxes = vec![(50.0, 60.0, 150.0, 160.0)]; // x1, y1, x2, y2
        
        let (sparse_embeddings, dense_embeddings) = encoder.forward(
            None,
            Some(&boxes),
            None,
        );
        
        assert_eq!(sparse_embeddings.dims()[1], 2); // Two corners per box
        assert_eq!(sparse_embeddings.dims()[2], 256); // Embed dim
        assert_eq!(dense_embeddings.dims()[1], 256); // Embed dim
    }

    #[test]
    fn test_no_prompts() {
        let device = Default::default();
        let encoder = PromptEncoder::<TestBackend>::new(256, &device);
        
        let (sparse_embeddings, dense_embeddings) = encoder.forward(None, None, None);
        
        assert_eq!(sparse_embeddings.dims()[1], 1); // Not-a-point embedding
        assert_eq!(sparse_embeddings.dims()[2], 256); // Embed dim
        assert_eq!(dense_embeddings.dims()[1], 256); // Embed dim
    }
} 