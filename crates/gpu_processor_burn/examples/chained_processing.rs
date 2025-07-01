use burn::{
    backend::wgpu::WgpuDevice,
    prelude::*,
    tensor::{TensorData, Shape},
};
use burn_processor::ops::{
    gaussian_blur, extract_outline, threshold, erode, dilate,
    blur_outline_threshold, advanced_pipeline, efficient_pipeline, fused_pipeline
};
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

    println!("\n=== Testing Individual Operations ===");
    
    // Test individual operations
    println!("1. Applying Gaussian blur...");
    let blurred = gaussian_blur(input_tensor.clone(), 3);
    save_tensor_as_image(&blurred, width, height, "crates/burn_processor/examples/01_blurred.png")?;

    println!("2. Applying outline extraction...");
    let outlined = extract_outline(input_tensor.clone());
    save_tensor_as_image(&outlined, width, height, "crates/burn_processor/examples/02_outlined.png")?;

    println!("3. Applying threshold...");
    let thresholded = threshold(outlined.clone(), 0.1, 1.0);
    save_tensor_as_image(&thresholded, width, height, "crates/burn_processor/examples/03_thresholded.png")?;

    println!("4. Applying erosion...");
    let eroded = erode(thresholded.clone(), 3);
    save_tensor_as_image(&eroded, width, height, "crates/burn_processor/examples/04_eroded.png")?;

    println!("5. Applying dilation...");
    let dilated = dilate(eroded.clone(), 3);
    save_tensor_as_image(&dilated, width, height, "crates/burn_processor/examples/05_dilated.png")?;

    println!("\n=== Testing Chained Operations ===");

    // Test chained operations
    println!("6. Chain: Blur -> Outline -> Threshold");
    let chain1 = blur_outline_threshold(input_tensor.clone(), 2, 0.15);
    save_tensor_as_image(&chain1, width, height, "crates/burn_processor/examples/06_chain_blur_outline_threshold.png")?;

    println!("7. Advanced chain: Blur -> Outline -> Morphology -> Threshold");
    let chain2 = advanced_pipeline(input_tensor.clone(), 2, 2, 0.15);
    save_tensor_as_image(&chain2, width, height, "crates/burn_processor/examples/07_advanced_pipeline.png")?;

    println!("8. Fused pipeline (single GPU kernel)");
    let fused = efficient_pipeline(input_tensor.clone(), 2, 0.5, 0.15);
    save_tensor_as_image(&fused, width, height, "crates/burn_processor/examples/08_fused_pipeline.png")?;

    println!("\n=== Demonstrating Different Processing Styles ===");

    // Demonstrate different approaches to the same result
    println!("9. Multi-step processing (multiple kernel dispatches)");
    let step1 = gaussian_blur(input_tensor.clone(), 3);
    let step2 = extract_outline(step1);
    let step3 = threshold(step2, 0.1, 1.0);
    let step4 = erode(step3, 2);
    let multi_step = dilate(step4, 2);
    save_tensor_as_image(&multi_step, width, height, "crates/burn_processor/examples/09_multi_step.png")?;

    println!("10. Single-step fused processing (one kernel dispatch)");
    let single_step = fused_pipeline(input_tensor.clone(), 3, 0.5, 0.1);
    save_tensor_as_image(&single_step, width, height, "crates/burn_processor/examples/10_single_step_fused.png")?;

    println!("\n=== Performance Comparison ===");
    
    // Demonstrate the performance difference between approaches
    use std::time::Instant;
    
    println!("Timing multi-step approach...");
    let start = Instant::now();
    for _ in 0..10 {
        let _result = advanced_pipeline(input_tensor.clone(), 2, 2, 0.15);
    }
    let multi_step_time = start.elapsed();
    
    println!("Timing fused approach...");
    let start = Instant::now();
    for _ in 0..10 {
        let _result = efficient_pipeline(input_tensor.clone(), 2, 0.5, 0.15);
    }
    let fused_time = start.elapsed();
    
    println!("Multi-step approach: {:?}", multi_step_time);
    println!("Fused approach: {:?}", fused_time);
    println!("Speedup: {:.2}x", multi_step_time.as_secs_f64() / fused_time.as_secs_f64());

    println!("\n=== Computation Graph Benefits ===");
    println!("All operations are automatically:");
    println!("- Lazily evaluated until data is needed");
    println!("- Optimized by Burn's graph optimizer");
    println!("- Executed efficiently on GPU");
    println!("- Memory managed automatically");
    println!("- Parallelized across available compute units");

    println!("\nAll processed images saved to crates/burn_processor/examples/ directory!");
    Ok(())
}

fn save_tensor_as_image<B: burn::prelude::Backend>(
    tensor: &Tensor<B, 2>,
    width: u32,
    height: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert the output tensor back to data
    let output_data_tensor = tensor.clone().into_data();
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
    


    output_img.save(path)?;
    println!("  Saved: {}", path);
    
    Ok(())
} 