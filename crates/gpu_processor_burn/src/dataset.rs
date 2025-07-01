//! Dataset utilities for GPU image processing with batch processing capabilities.
//! 
//! This module provides convenient batch processing utilities that can work with
//! image collections and process them efficiently using our custom GPU kernels.

use burn::{
    prelude::*,
    tensor::{TensorData, Shape},
};
use crate::backend::ProcessingBackend;
use crate::ops::{gaussian_blur, extract_outline, threshold, advanced_pipeline, efficient_pipeline};
use image::{ImageBuffer, DynamicImage, Luma};
use std::path::{Path, PathBuf};
use std::fs;

/// A processed image batch ready for GPU operations.
#[derive(Debug, Clone)]
pub struct ProcessedBatch<B: Backend> {
    /// Batch of processed images as tensors.
    pub images: Tensor<B, 3, Float>, // [batch_size, height, width]
    
    /// Original image paths for reference.
    pub paths: Vec<String>,
    
    /// Image dimensions (width, height) for each image in batch.
    pub dimensions: Vec<(u32, u32)>,
}

/// A simple image dataset item.
#[derive(Debug, Clone)]
pub struct ImageItem {
    /// Path to the image file.
    pub path: PathBuf,
    
    /// Optional label/category.
    pub label: Option<String>,
    
    /// Image data as normalized f32 values.
    pub pixels: Vec<f32>,
    
    /// Image dimensions (width, height).
    pub dimensions: (u32, u32),
}

/// Configuration for batch processing operations.
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Gaussian blur radius (0 to disable).
    pub blur_radius: u32,
    
    /// Threshold value for binarization.
    pub threshold_value: f32,
    
    /// Maximum pixel value after thresholding.
    pub max_value: f32,
    
    /// Morphological kernel size.
    pub morph_kernel_size: u32,
    
    /// Edge detection threshold for fused pipeline.
    pub edge_threshold: f32,
    
    /// Use fused pipeline instead of individual operations.
    pub use_fused_pipeline: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            blur_radius: 2,
            threshold_value: 0.15,
            max_value: 1.0,
            morph_kernel_size: 2,
            edge_threshold: 0.1,
            use_fused_pipeline: false,
        }
    }
}

/// Simple image dataset for batch processing.
pub struct ImageDataset {
    items: Vec<ImageItem>,
}

impl ImageDataset {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }
    
    /// Load images from a directory (non-recursive).
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut items = Vec::new();
        let dir_path = path.as_ref();
        
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let file_path = entry.path();
            
            // Check if it's an image file
            if let Some(extension) = file_path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    match ext_str.to_lowercase().as_str() {
                        "jpg" | "jpeg" | "png" | "bmp" => {
                            if let Ok(item) = Self::load_image_item(&file_path, None) {
                                items.push(item);
                            }
                        }
                        _ => continue,
                    }
                }
            }
        }
        
        Ok(Self { items })
    }
    
    /// Load images from a directory with subdirectories as labels.
    pub fn from_classification_directory<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut items = Vec::new();
        let root_path = path.as_ref();
        
        for entry in fs::read_dir(root_path)? {
            let entry = entry?;
            let class_path = entry.path();
            
            if class_path.is_dir() {
                let class_name = class_path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|s| s.to_string());
                
                for image_entry in fs::read_dir(&class_path)? {
                    let image_entry = image_entry?;
                    let image_path = image_entry.path();
                    
                    if let Some(extension) = image_path.extension() {
                        if let Some(ext_str) = extension.to_str() {
                            match ext_str.to_lowercase().as_str() {
                                "jpg" | "jpeg" | "png" | "bmp" => {
                                    if let Ok(item) = Self::load_image_item(&image_path, class_name.clone()) {
                                        items.push(item);
                                    }
                                }
                                _ => continue,
                            }
                        }
                    }
                }
            }
        }
        
        Ok(Self { items })
    }
    
    /// Add a single image to the dataset.
    pub fn add_image<P: AsRef<Path>>(&mut self, path: P, label: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        let item = Self::load_image_item(path.as_ref(), label)?;
        self.items.push(item);
        Ok(())
    }
    
    /// Get the number of images in the dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    /// Get an image by index.
    pub fn get(&self, index: usize) -> Option<&ImageItem> {
        self.items.get(index)
    }
    
    /// Get all items.
    pub fn items(&self) -> &[ImageItem] {
        &self.items
    }
    
    /// Load a single image item from disk.
    fn load_image_item(path: &Path, label: Option<String>) -> Result<ImageItem, Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let img = img.to_luma8(); // Convert to grayscale
        let (width, height) = img.dimensions();
        
        // Convert to normalized f32 values
        let pixels: Vec<f32> = img.into_raw()
            .into_iter()
            .map(|pixel| pixel as f32 / 255.0)
            .collect();
        
        Ok(ImageItem {
            path: path.to_path_buf(),
            label,
            pixels,
            dimensions: (width, height),
        })
    }
    
    /// Create batch indices for processing.
    pub fn create_batch_indices(&self, batch_size: usize) -> Vec<Vec<usize>> {
        (0..self.len())
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

/// Batch processor for efficient GPU image processing.
pub struct ImageBatchProcessor<B: ProcessingBackend> {
    device: B::Device,
    config: ProcessingConfig,
}

impl<B: ProcessingBackend> ImageBatchProcessor<B> {
    /// Create a new batch processor with the given device and configuration.
    pub fn new(device: B::Device, config: ProcessingConfig) -> Self {
        Self { device, config }
    }
    
    /// Load images from a dataset and convert to tensor format.
    pub fn load_batch_from_dataset(
        &self,
        dataset: &ImageDataset,
        batch_indices: &[usize],
    ) -> Result<ProcessedBatch<B>, Box<dyn std::error::Error>> {
        let mut all_pixels: Vec<f32> = Vec::new();
        let mut paths = Vec::new();
        let mut dimensions = Vec::new();
        
        // For batch processing, we need consistent dimensions
        // Here we'll find the most common dimensions or use the first image's dimensions
        let first_item = dataset.get(batch_indices[0])
            .ok_or("Invalid batch index")?;
        let (target_width, target_height) = first_item.dimensions;
        
        for &idx in batch_indices {
            let item = dataset.get(idx).ok_or(format!("Invalid index: {}", idx))?;
            
            // For simplicity, we assume all images have the same dimensions
            // In practice, you'd want to resize images to a common size
            if item.dimensions != (target_width, target_height) {
                return Err(format!(
                    "Inconsistent image dimensions: expected {}x{}, got {}x{}",
                    target_width, target_height, item.dimensions.0, item.dimensions.1
                ).into());
            }
            
            all_pixels.extend(item.pixels.iter());
            paths.push(item.path.display().to_string());
            dimensions.push(item.dimensions);
        }
        
        // Create batch tensor [batch_size, height, width]
        let batch_size = batch_indices.len();
        let tensor_data = TensorData::new(
            all_pixels, 
            Shape::new([batch_size, target_height as usize, target_width as usize])
        );
        let images = Tensor::from_data(tensor_data, &self.device);
        
        Ok(ProcessedBatch {
            images,
            paths,
            dimensions,
        })
    }
    
    /// Process a batch of images using individual GPU operations.
    pub fn process_batch_individual(&self, batch: ProcessedBatch<B>) -> ProcessedBatch<B> {
        let processed_images = if self.config.blur_radius > 0 {
            let blurred = self.process_batch_operation(batch.images.clone(), |img| {
                gaussian_blur(img, self.config.blur_radius)
            });
            
            let outlined = self.process_batch_operation(blurred, |img| {
                extract_outline(img)
            });
            
            self.process_batch_operation(outlined, |img| {
                threshold(img, self.config.threshold_value, self.config.max_value)
            })
        } else {
            let outlined = self.process_batch_operation(batch.images.clone(), |img| {
                extract_outline(img)
            });
            
            self.process_batch_operation(outlined, |img| {
                threshold(img, self.config.threshold_value, self.config.max_value)
            })
        };
        
        ProcessedBatch {
            images: processed_images,
            paths: batch.paths,
            dimensions: batch.dimensions,
        }
    }
    
    /// Process a batch of images using the fused GPU pipeline.
    pub fn process_batch_fused(&self, batch: ProcessedBatch<B>) -> ProcessedBatch<B> {
        let processed_images = self.process_batch_operation(batch.images.clone(), |img| {
            efficient_pipeline(
                img, 
                self.config.blur_radius, 
                self.config.threshold_value, 
                self.config.edge_threshold
            )
        });
        
        ProcessedBatch {
            images: processed_images,
            paths: batch.paths,
            dimensions: batch.dimensions,
        }
    }
    
    /// Process a batch using the configured processing method.
    pub fn process_batch(&self, batch: ProcessedBatch<B>) -> ProcessedBatch<B> {
        if self.config.use_fused_pipeline {
            self.process_batch_fused(batch)
        } else {
            self.process_batch_individual(batch)
        }
    }
    
    /// Helper to apply an operation to each image in a batch.
    fn process_batch_operation<F>(&self, batch_tensor: Tensor<B, 3, Float>, operation: F) -> Tensor<B, 3, Float>
    where
        F: Fn(Tensor<B, 2, Float>) -> Tensor<B, 2, Float>,
    {
        let batch_size = batch_tensor.dims()[0];
        let mut processed_images: Vec<Tensor<B, 3, Float>> = Vec::new();
        
        for i in 0..batch_size {
            // Extract single image from batch [height, width]
            let image = batch_tensor.clone().slice([i..i+1, 0..batch_tensor.dims()[1], 0..batch_tensor.dims()[2]])
                .squeeze(0); // Remove batch dimension
                
            // Apply operation
            let processed = operation(image);
            
            // Add batch dimension back
            let processed_with_batch: Tensor<B, 3, Float> = processed.unsqueeze_dim(0);
            processed_images.push(processed_with_batch);
        }
        
        // Concatenate all processed images back into batch
        Tensor::cat(processed_images, 0)
    }
    
    /// Save processed batch results to individual image files.
    pub fn save_batch_results(
        &self, 
        batch: &ProcessedBatch<B>, 
        output_dir: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;
        
        let batch_size = batch.images.dims()[0];
        
        for i in 0..batch_size {
            // Extract single image
            let image_tensor: Tensor<B, 2, Float> = batch.images.clone()
                .slice([i..i+1, 0..batch.images.dims()[1], 0..batch.images.dims()[2]])
                .squeeze(0);
                
            // Convert to image data
            let image_data = image_tensor.into_data();
            let pixels: Vec<f32> = image_data.convert::<f32>().to_vec().unwrap();
            
            // Normalize to u8 range
            let max_val = pixels.iter().cloned().fold(0.0f32, f32::max);
            let min_val = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = max_val - min_val;
            
            let pixel_data: Vec<u8> = if range > 0.0 {
                pixels.iter().map(|&p| (((p - min_val) / range) * 255.0) as u8).collect()
            } else {
                vec![0; pixels.len()]
            };
            
            // Create output filename
            let input_path = Path::new(&batch.paths[i]);
            let filename = input_path.file_stem().unwrap().to_str().unwrap();
            let output_path = Path::new(output_dir).join(format!("{}_processed.png", filename));
            
            // Save image
            let (width, height) = batch.dimensions[i];
            let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixel_data)
                .ok_or("Failed to create image buffer")?;
            
            img.save(output_path)?;
        }
        
        Ok(())
    }
}

/// Print dataset statistics.
pub fn print_dataset_info(dataset: &ImageDataset) {
    println!("Dataset Information:");
    println!("  Total images: {}", dataset.len());
    
    if let Some(first_item) = dataset.get(0) {
        println!("  First image path: {}", first_item.path.display());
        println!("  Image dimensions: {}x{}", first_item.dimensions.0, first_item.dimensions.1);
        println!("  Pixels count: {}", first_item.pixels.len());
        
        if let Some(label) = &first_item.label {
            println!("  First image label: {}", label);
        }
    }
    
    // Count labels if available
    let mut label_counts = std::collections::HashMap::new();
    for item in dataset.items() {
        if let Some(label) = &item.label {
            *label_counts.entry(label.clone()).or_insert(0) += 1;
        }
    }
    
    if !label_counts.is_empty() {
        println!("  Label distribution:");
        for (label, count) in label_counts {
            println!("    {}: {} images", label, count);
        }
    }
} 