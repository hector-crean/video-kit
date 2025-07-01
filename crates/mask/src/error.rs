use thiserror::Error;

#[derive(Error, Debug)]
pub enum MaskError {
    #[error("Failed to load image: {0}")]
    ImageLoad(#[from] image::ImageError),
    
    #[error("No image loaded")]
    NoImageLoaded,
    
    #[error("Image processing error: {0}")]
    ImageProcessing(String),
    
    #[error("Geometric computation error: {0}")]
    GeometricComputation(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("GeoJSON error: {0}")]
    GeoJson(#[from] geojson::Error),
}

pub type Result<T> = std::result::Result<T, MaskError>; 