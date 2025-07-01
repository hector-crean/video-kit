use burn::{
    backend::wgpu::WgpuDevice,
    prelude::*,
};
use burn_processor::dataset::{ImageDataset, ImageBatchProcessor, ProcessingConfig, print_dataset_info};
use std::time::Instant;

// Use the raw CubeBackend instead of the fusion backend
type MyBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();

    println!("=== Burn GPU Batch Processing Demo ===\n");

    // Method 1: Create dataset from single directory (all images in one folder)
    println!("1. Creating dataset from examples directory...");
    let mut dataset = ImageDataset::new();
    
    // Add our test image manually since we only have one
    dataset.add_image("crates/burn_processor/examples/brain_image.jpg", Some("brain".to_string()))?;
    
    // Print dataset info
    print_dataset_info(&dataset);

    // Create processing configurations
    let individual_config = ProcessingConfig {
        blur_radius: 2,
        threshold_value: 0.15,
        max_value: 1.0,
        morph_kernel_size: 2,
        edge_threshold: 0.1,
        use_fused_pipeline: false,
    };

    let fused_config = ProcessingConfig {
        blur_radius: 2,
        threshold_value: 0.15,
        max_value: 1.0,
        morph_kernel_size: 2,
        edge_threshold: 0.1,
        use_fused_pipeline: true,
    };

    // Create batch processors
    let individual_processor = ImageBatchProcessor::<MyBackend>::new(device.clone(), individual_config);
    let fused_processor = ImageBatchProcessor::<MyBackend>::new(device.clone(), fused_config);

    println!("\n2. Processing single image batch...");
    
    // Process a single batch (just our one image)
    let batch_indices = vec![0]; // Single image batch
    
    // Load batch from dataset
    println!("   Loading batch from dataset...");
    let batch = individual_processor.load_batch_from_dataset(&dataset, &batch_indices)?;
    println!("   Batch loaded: {} images", batch.images.dims()[0]);

    // Process with individual operations
    println!("   Processing with individual GPU kernels...");
    let start = Instant::now();
    let processed_individual = individual_processor.process_batch(batch.clone());
    let individual_time = start.elapsed();
    println!("   Individual processing time: {:?}", individual_time);

    // Process with fused pipeline
    println!("   Processing with fused GPU kernel...");
    let start = Instant::now();
    let processed_fused = fused_processor.process_batch(batch.clone());
    let fused_time = start.elapsed();
    println!("   Fused processing time: {:?}", fused_time);

    // Save results
    println!("\n3. Saving processed results...");
    individual_processor.save_batch_results(
        &processed_individual, 
        "crates/burn_processor/examples/batch_output/individual"
    )?;
    
    fused_processor.save_batch_results(
        &processed_fused,
        "crates/burn_processor/examples/batch_output/fused"
    )?;
    
    println!("   Results saved to crates/burn_processor/examples/batch_output/");

    // Demonstrate batch processing performance with repeated operations
    println!("\n4. Performance comparison (10 iterations)...");
    
    let iterations = 10;
    
    // Individual processing benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let batch = individual_processor.load_batch_from_dataset(&dataset, &batch_indices)?;
        let _processed = individual_processor.process_batch(batch);
    }
    let individual_total = start.elapsed();
    
    // Fused processing benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let batch = fused_processor.load_batch_from_dataset(&dataset, &batch_indices)?;
        let _processed = fused_processor.process_batch(batch);
    }
    let fused_total = start.elapsed();
    
    println!("   Individual pipeline total: {:?} (avg: {:?})", 
             individual_total, individual_total / iterations);
    println!("   Fused pipeline total: {:?} (avg: {:?})", 
             fused_total, fused_total / iterations);
    
    if fused_total < individual_total {
        let speedup = individual_total.as_secs_f64() / fused_total.as_secs_f64();
        println!("   Fused pipeline is {:.2}x faster!", speedup);
    }

    println!("\n=== Batch Processing Benefits ===");
    println!("✅ Structured dataset management");
    println!("✅ Efficient batch tensor operations"); 
    println!("✅ GPU kernel fusion for optimal performance");
    println!("✅ Automatic memory management");
    println!("✅ Support for different image formats and sizes");
    println!("✅ Classification dataset support with labels");

    println!("\n=== Usage Examples ===");
    println!("// Load from classification directory:");
    println!("let dataset = ImageDataset::from_classification_directory(\"path/to/classes\")?;");
    println!("");
    println!("// Load from flat directory:");
    println!("let dataset = ImageDataset::from_directory(\"path/to/images\")?;");
    println!("");
    println!("// Process in batches:");
    println!("let batch_indices = dataset.create_batch_indices(4); // Batch size 4");
    println!("for indices in batch_indices {{");
    println!("    let batch = processor.load_batch_from_dataset(&dataset, &indices)?;");
    println!("    let processed = processor.process_batch(batch);");
    println!("    processor.save_batch_results(&processed, \"output_dir\")?;");
    println!("}}");

    Ok(())
}

// Helper function to demonstrate creating a larger dataset
#[allow(dead_code)]
fn create_demo_dataset() -> Result<ImageDataset, Box<dyn std::error::Error>> {
    println!("Creating demonstration dataset...");
    
    // This would be used with a real dataset structure like:
    // dataset/
    //   class1/
    //     image1.jpg
    //     image2.png
    //   class2/
    //     image3.jpg
    //     image4.png
    
    let dataset = ImageDataset::from_classification_directory("path/to/your/dataset")?;
    
    println!("Dataset created with {} images", dataset.len());
    print_dataset_info(&dataset);
    
    // Create batches
    let batch_size = 4;
    let batches = dataset.create_batch_indices(batch_size);
    println!("Created {} batches of size {}", batches.len(), batch_size);
    
    Ok(dataset)
} 