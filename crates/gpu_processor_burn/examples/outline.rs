use burn::{
    backend::wgpu::WgpuDevice,
    prelude::*,
    tensor::{TensorData, Shape},
};
use burn_processor::ops::extract_outline;
use image::{ImageBuffer, Luma};

// Use the raw CubeBackend instead of the fusion backend
type MyBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();

    // Load the brain image
    let img_path = "crates/burn_processor/examples/brain_image.jpg";
    println!("Loading image from: {}", img_path);
    
    let img = image::open(img_path)?.to_luma8();
    let (width, height) = img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    // Convert image to normalized float data
    let input_data: Vec<f32> = img
        .into_raw()
        .into_iter()
        .map(|pixel| pixel as f32 / 255.0) // Normalize to [0, 1]
        .collect();

    // Convert to a Burn tensor
    let data = TensorData::new(input_data, Shape::new([height as usize, width as usize]));
    let input_tensor: Tensor<MyBackend, 2> = Tensor::from_data(data, &device);

    println!("Processing image with outline extraction...");
    
    // Apply the outline extraction
    let output_tensor = extract_outline(input_tensor);

    // Convert the output tensor back to data
    let output_data_tensor = output_tensor.into_data();
    let output_data: Vec<f32> = output_data_tensor.convert::<f32>().to_vec().unwrap();
    
    // Normalize output data to [0, 255] range for saving as image
    let max_val = output_data.iter().cloned().fold(0.0f32, f32::max);
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = max_val - min_val;
    
    let output_img_data: Vec<u8> = if range > 0.0 {
        output_data.iter().map(|&p| (((p - min_val) / range) * 255.0) as u8).collect()
    } else {
        vec![0; output_data.len()]
    };

    let output_img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, output_img_data)
        .ok_or("Failed to create output image buffer")?;

    // Save the output image
    let output_path = "crates/burn_processor/examples/brain_outline_output.png";
    output_img.save(output_path)?;

    println!("Outline image saved to {}", output_path);
    println!("Output value range: {:.3} to {:.3}", min_val, max_val);

    Ok(())
} 