//! crates/gpu_processor/src/image_format.rs

use bevy::render::render_resource::TextureFormat;
use serde::{Deserialize, Serialize};

/// A serializable and cross-platform representation of GPU texture formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuImageFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
    // Add other formats as needed
}

impl From<TextureFormat> for GpuImageFormat {
    fn from(format: TextureFormat) -> Self {
        match format {
            TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => Self::Rgba8Unorm,
            TextureFormat::Rgba16Float => Self::Rgba16Float,
            TextureFormat::Rgba32Float => Self::Rgba32Float,
            // Add other format mappings here
            _ => panic!("Unsupported texture format: {:?}", format),
        }
    }
}

impl From<GpuImageFormat> for TextureFormat {
    fn from(format: GpuImageFormat) -> Self {
        match format {
            GpuImageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            GpuImageFormat::Rgba16Float => TextureFormat::Rgba16Float,
            GpuImageFormat::Rgba32Float => TextureFormat::Rgba32Float,
        }
    }
} 