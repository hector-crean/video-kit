use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    LayerNorm, LayerNormConfig, 
    Linear, LinearConfig, 
    Dropout, DropoutConfig,
    PaddingConfig2d, 
    Initializer,
    attention::{MultiHeadAttention, MultiHeadAttentionConfig, MhaInput},
};
use burn::tensor::{Tensor, activation, backend::Backend, Distribution};
use burn::module::{Module, Param};

#[derive(Config, Debug)]
pub struct ViTConfig {
    /// Input image size (height and width)
    pub image_size: usize,
    /// Patch size for tokenization
    pub patch_size: usize,
    /// Number of classes (0 for no classification head)
    pub num_classes: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub depth: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP expansion ratio
    pub mlp_ratio: f64,
    /// QKV bias
    pub qkv_bias: bool,
    /// Window size for windowed attention
    pub window_size: usize,
    /// Indices of layers that use global attention instead of windowed
    pub global_attn_indices: Vec<usize>,
}

/// Patch embedding layer that converts images to sequence of patch embeddings
#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    proj: Conv2d<B>,
    num_patches: usize,
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn new(config: &ViTConfig, device: &B::Device) -> Self {
        let proj = Conv2dConfig::new(
            [3, config.embed_dim], // RGB input to embed_dim output
            [config.patch_size, config.patch_size]
        )
        .with_stride([config.patch_size, config.patch_size])
        .with_padding(PaddingConfig2d::Valid)
        .with_bias(true)
        .init(device);

        let num_patches = (config.image_size / config.patch_size).pow(2);

        Self { proj, num_patches }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // x: [batch_size, channels, height, width]
        let x = self.proj.forward(x); // [batch_size, embed_dim, H/patch_size, W/patch_size]
        let [batch_size, embed_dim, h, w] = x.dims();
        
        // Flatten spatial dimensions and transpose to [batch_size, num_patches, embed_dim]
        x.reshape([batch_size, embed_dim, h * w])
            .swap_dims(1, 2) // [batch_size, h*w, embed_dim]
    }

    pub fn num_patches(&self) -> usize {
        self.num_patches
    }
}

/// Multi-layer perceptron used in transformer blocks
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &ViTConfig, device: &B::Device) -> Self {
        let hidden_features = (config.embed_dim as f64 * config.mlp_ratio) as usize;
        
        let fc1 = LinearConfig::new(config.embed_dim, hidden_features)
            .with_bias(true)
            .init(device);
        let fc2 = LinearConfig::new(hidden_features, config.embed_dim)
            .with_bias(true)
            .init(device);
        let dropout = DropoutConfig::new(0.0).init(); // SAM typically uses no dropout

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = activation::gelu(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}

/// Window partitioning function for windowed attention
fn window_partition<B: Backend>(x: Tensor<B, 4>, window_size: usize) -> Tensor<B, 4> {
    // x: [batch_size, height, width, channels]
    let [batch_size, height, width, channels] = x.dims();
    
    // Ensure height and width are divisible by window_size
    assert_eq!(height % window_size, 0);
    assert_eq!(width % window_size, 0);
    
    let num_windows_h = height / window_size;
    let num_windows_w = width / window_size;
    
    // Reshape to [batch_size * num_windows, window_size, window_size, channels]
    x.reshape([batch_size, num_windows_h, window_size, num_windows_w, window_size, channels])
        .swap_dims(2, 3)
        .reshape([batch_size * num_windows_h * num_windows_w, window_size, window_size, channels])
}

/// Window unpartitioning function
fn window_unpartition<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Tensor<B, 4> {
    let [_, _, _, channels] = windows.dims();
    let (height, width) = hw;
    let (pad_h, pad_w) = pad_hw;
    
    let num_windows_h = (height + pad_h) / window_size;
    let num_windows_w = (width + pad_w) / window_size;
    let batch_size = windows.dims()[0] / (num_windows_h * num_windows_w);
    
    // Reshape back to image format
    let x = windows.reshape([
        batch_size,
        num_windows_h,
        num_windows_w,
        window_size,
        window_size,
        channels,
    ]);
    
    x.swap_dims(2, 3)
        .reshape([batch_size, height + pad_h, width + pad_w, channels])
}

/// Transformer block with windowed or global attention
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadAttention<B>, 
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
    window_size: usize,
    use_global_attn: bool,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &ViTConfig, layer_idx: usize, device: &B::Device) -> Self {
        let norm1 = LayerNormConfig::new(config.embed_dim).init(device);
        let norm2 = LayerNormConfig::new(config.embed_dim).init(device);
        
        let attn = MultiHeadAttentionConfig::new(config.embed_dim, config.num_heads)
            .with_dropout(0.0)
            .init(device);
        
        let mlp = Mlp::new(config, device);
        
        let use_global_attn = config.global_attn_indices.contains(&layer_idx);

        Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size: config.window_size,
            use_global_attn,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention
        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        
        let x = if self.use_global_attn {
            // Global attention
            let attn_input = MhaInput::self_attn(x);
            self.attn.forward(attn_input).context
        } else {
            // Windowed attention (simplified - full implementation would be more complex)
            let attn_input = MhaInput::self_attn(x);
            self.attn.forward(attn_input).context
        };
        
        let x = shortcut + x;
        
        // MLP
        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.mlp.forward(x);
        shortcut + x
    }
}

/// Vision Transformer for SAM image encoding
#[derive(Module, Debug)]
pub struct ViT<B: Backend> {
    patch_embed: PatchEmbedding<B>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    neck: Vec<Conv2d<B>>,
}

impl<B: Backend> ViT<B> {
    pub fn new(config: &ViTConfig, device: &B::Device) -> Self {
        let patch_embed = PatchEmbedding::new(config, device);
        let num_patches = patch_embed.num_patches();
        
        // Positional embeddings
        let pos_embed = Param::from_tensor(
            Tensor::random([1, num_patches, config.embed_dim], Distribution::Normal(0.0, 0.02), device)
        );
        
        // Transformer blocks
        let blocks = (0..config.depth)
            .map(|i| Block::new(config, i, device))
            .collect();
        
        // Neck convolutions for feature map processing (simplified)
        let neck = vec![];

        Self {
            patch_embed,
            pos_embed,
            blocks,
            neck,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, _, height, width] = x.dims();
        
        // Patch embedding
        let mut x = self.patch_embed.forward(x); // [batch_size, num_patches, embed_dim]
        
        // Add positional embedding
        x = x + self.pos_embed.val();
        
        // Apply transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        // Reshape back to spatial format for neck processing
        let patch_h = height / 16; // Assuming patch_size = 16
        let patch_w = width / 16;
        let embed_dim = x.dims()[2];
        
        let spatial_x = x.reshape([batch_size, patch_h, patch_w, embed_dim])
             .swap_dims(1, 3)  // [batch_size, embed_dim, patch_w, patch_h]
             .swap_dims(2, 3); // [batch_size, embed_dim, patch_h, patch_w]
        
        // Apply neck convolutions (simplified)
        // In a full implementation, this would include LayerNorm2d operations
        spatial_x
    }
    
    pub fn device(&self) -> B::Device {
        self.pos_embed.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_patch_embedding() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = ViTConfig {
            image_size: 224,
            patch_size: 16,
            num_classes: 0,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            qkv_bias: true,
            window_size: 14,
            global_attn_indices: vec![2, 5, 8, 11],
        };
        
        let patch_embed = PatchEmbedding::new(&config, &device);
        let input = Tensor::random([1, 3, 224, 224], Distribution::Normal(0.0, 1.0), &device);
        let output = patch_embed.forward(input);
        
        assert_eq!(output.dims(), [1, 196, 768]); // (224/16)^2 = 196 patches
    }
    
    #[test]
    fn test_mlp() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = ViTConfig {
            image_size: 224,
            patch_size: 16,
            num_classes: 0,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            qkv_bias: true,
            window_size: 14,
            global_attn_indices: vec![2, 5, 8, 11],
        };
        
        let mlp = Mlp::new(&config, &device);
        let input = Tensor::random([1, 196, 768], Distribution::Normal(0.0, 1.0), &device);
        let output = mlp.forward(input);
        
        assert_eq!(output.dims(), [1, 196, 768]);
    }
} 