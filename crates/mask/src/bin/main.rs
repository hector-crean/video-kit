use mask::{Pipeline, algorithms::*, manager::{MaskManager, MaskManagerCommand}};
use image::{GrayImage, Luma};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Mask Outline Extraction Demo with Enhanced geo Algorithms");
    println!("============================================================");
    
    // Create test images for demo
    let simple_image = create_simple_test_image();
    let donut_image = create_donut_test_image();
    let complex_image = create_complex_test_image();
    
    demo_enhanced_commands();
    demo_enhanced_algorithms(&simple_image)?;
    demo_geo_algorithms(&donut_image)?;
    demo_comprehensive_pipeline(&complex_image)?;
    
    println!("✅ All demos completed successfully!");
    println!("📁 Generated files: enhanced_*.geojson");
    
    Ok(())
}

fn demo_enhanced_commands() {
    println!("\n🔧 Demo: Enhanced MaskManagerCommand with schemars & strum");
    println!("----------------------------------------------------------");
    
    // Show all available command names
    println!("📋 Available commands:");
    for (i, name) in MaskManagerCommand::command_names().iter().enumerate() {
        println!("   {}. {}", i + 1, name);
    }
    
    // Show command descriptions and parameters
    let commands = vec![
        MaskManagerCommand::ExtractOutline,
        MaskManagerCommand::ExtractOutlineWithSimplification { tolerance: 2.0 },
        MaskManagerCommand::ExtractOutlineWithHoleDetection,
        MaskManagerCommand::ExtractOutlineWithCustomExtractor { 
            extractor_name: "custom_algorithm".to_string() 
        },
    ];
    
    println!("\n📝 Command details:");
    for cmd in commands {
        println!("   🔹 {}", cmd); // Uses strum Display
        println!("     Description: {}", cmd.description());
        
        let params = cmd.parameters_info();
        if !params.is_empty() {
            println!("     Parameters:");
            for (name, desc, required) in params {
                let req_marker = if required { "*" } else { " " };
                println!("       {}{}: {}", req_marker, name, desc);
            }
        } else {
            println!("     Parameters: None");
        }
        println!();
    }
    
    // Show JSON schema
    println!("📋 JSON Schema for MaskManagerCommand:");
    let schema = MaskManagerCommand::schema();
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
    
    // Demonstrate serialization/deserialization
    println!("\n📦 JSON Serialization Examples:");
    let test_commands = vec![
        MaskManagerCommand::ExtractOutline,
        MaskManagerCommand::ExtractOutlineWithSimplification { tolerance: 1.5 },
    ];
    
    for cmd in test_commands {
        let json = serde_json::to_string_pretty(&cmd).unwrap();
        println!("   Command: {}", cmd);
        println!("   JSON: {}", json);
        
        // Deserialize back
        let parsed: MaskManagerCommand = serde_json::from_str(&json).unwrap();
        println!("   Parsed back: {}", parsed);
        println!();
    }
}

fn demo_enhanced_algorithms(image: &GrayImage) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Demo 1: Enhanced Simplification Algorithms");
    println!("---------------------------------------------");
    
    // Douglas-Peucker vs Visvalingam-Whyatt comparison
    let base_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(ContainmentHoleDetector)
        .build();
    
    let result = base_pipeline.process(image)?;
    let original_points: usize = result.shapes.iter().map(|s| s.exterior.len()).sum();
    println!("   📊 Original total points: {}", original_points);
    
    // Douglas-Peucker simplification
    let dp_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(ContainmentHoleDetector)
        .with_simplification(2.0)
        .build();
    
    let dp_result = dp_pipeline.process(image)?;
    let dp_points: usize = dp_result.shapes.iter().map(|s| s.exterior.len()).sum();
    println!("   🔹 Douglas-Peucker (ε=2.0): {} points ({:.1}% reduction)", 
             dp_points, 100.0 * (original_points - dp_points) as f32 / original_points as f32);
    dp_result.save_geojson("enhanced_douglas_peucker.geojson")?;
    
    // Visvalingam-Whyatt simplification  
    let vw_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(ContainmentHoleDetector)
        .with_vw_simplification(2.0)
        .build();
    
    let vw_result = vw_pipeline.process(image)?;
    let vw_points: usize = vw_result.shapes.iter().map(|s| s.exterior.len()).sum();
    println!("   🔸 Visvalingam-Whyatt (ε=2.0): {} points ({:.1}% reduction)", 
             vw_points, 100.0 * (original_points - vw_points) as f32 / original_points as f32);
    vw_result.save_geojson("enhanced_visvalingam_whyatt.geojson")?;
    
    Ok(())
}

fn demo_geo_algorithms(image: &GrayImage) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎨 Demo 2: geo Crate Algorithm Showcase");
    println!("---------------------------------------");
    
    // Chaikin smoothing demonstration
    let chaikin_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(ContainmentHoleDetector)
        .with_chaikin_smoothing(2)  // 2 iterations
        .build();
    
    let chaikin_result = chaikin_pipeline.process(image)?;
    println!("   🌊 Chaikin smoothing: {} shapes smoothed with 2 iterations", 
             chaikin_result.shapes.len());
    chaikin_result.save_geojson("enhanced_chaikin_smoothed.geojson")?;
    
    // Convex hull transformation
    let convex_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(NoHoleDetector)  // Convex hulls don't have holes
        .with_convex_hull()
        .build();
    
    let convex_result = convex_pipeline.process(image)?;
    println!("   🔺 Convex hull: {} shapes converted to convex hulls", 
             convex_result.shapes.len());
    
    for (i, shape) in convex_result.shapes.iter().enumerate() {
        let area = shape.area();
        let centroid = shape.centroid();
        println!("      • Hull {}: area={:.1}, centroid=({:.1}, {:.1})", 
                i, area, centroid[0], centroid[1]);
    }
    convex_result.save_geojson("enhanced_convex_hulls.geojson")?;
    
    Ok(())
}

fn demo_comprehensive_pipeline(image: &GrayImage) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demo 3: Comprehensive Processing Pipeline");
    println!("--------------------------------------------");
    
    // Multi-step processing pipeline
    let comprehensive_pipeline = Pipeline::builder()
        .add_preprocessor(GaussianBlurPreprocessor { sigma: 1.0 })
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .set_hole_detector(AreaBasedHoleDetector::default())
        .with_chaikin_smoothing(1)
        .with_simplification(1.5)
        .with_validation()
        .build();
    
    let comp_result = comprehensive_pipeline.process(image)?;
    println!("   📈 Pipeline: {} → Gaussian Blur → Threshold → Hole Detection → Chaikin Smoothing → Douglas-Peucker → Validation", 
             comprehensive_pipeline.info());
    println!("   📊 Result: {} valid shapes", comp_result.shapes.len());
    
    // Analysis of results
    let total_area: f32 = comp_result.shapes.iter().map(|s| s.area()).sum();
    let avg_perimeter: f32 = comp_result.shapes.iter()
        .map(|s| s.perimeter())
        .sum::<f32>() / comp_result.shapes.len() as f32;
    
    println!("   📏 Total area: {:.1}, Average perimeter: {:.1}", total_area, avg_perimeter);
    
    for (i, shape) in comp_result.shapes.iter().enumerate() {
        let centroid = shape.centroid();
        println!("   • Shape {}: {} exterior points, {} holes, centroid=({:.1}, {:.1})", 
                i, shape.exterior.len(), shape.holes.len(), centroid[0], centroid[1]);
    }
    
    comp_result.save_geojson("enhanced_comprehensive.geojson")?;
    
    // Compare against simpler pipeline
    let simple_pipeline = Pipeline::builder()
        .add_preprocessor(ThresholdPreprocessor { threshold: 128 })
        .build();
    
    let simple_result = simple_pipeline.process(image)?;
    let simple_points: usize = simple_result.shapes.iter().map(|s| s.exterior.len()).sum();
    let comp_points: usize = comp_result.shapes.iter().map(|s| s.exterior.len()).sum();
    
    println!("   📈 Comparison: Simple ({} points) vs Comprehensive ({} points)", 
             simple_points, comp_points);
    println!("   🎯 Point reduction: {:.1}%", 
             100.0 * (simple_points - comp_points) as f32 / simple_points as f32);
    
    Ok(())
}

fn create_simple_test_image() -> GrayImage {
    let mut img = GrayImage::new(200, 200);
    
    // Create a rectangle
    for y in 20..80 {
        for x in 20..120 {
            img.put_pixel(x, y, Luma([255u8]));
        }
    }
    
    // Create a circle
    let center_x = 150.0;
    let center_y = 150.0;
    let radius = 40.0;
    
    for y in 110..190 {
        for x in 110..190 {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            if dx*dx + dy*dy <= radius*radius {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }
    }
    
    img
}

fn create_donut_test_image() -> GrayImage {
    let mut img = GrayImage::new(300, 250);
    let center_x = 150.0;
    let center_y = 125.0;
    let outer_radius = 80.0;
    let inner_radius = 40.0;
    
    for y in 0..250 {
        for x in 0..300 {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist_sq = dx*dx + dy*dy;
            
            if dist_sq <= outer_radius*outer_radius && dist_sq >= inner_radius*inner_radius {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }
    }
    
    img
}

fn create_complex_test_image() -> GrayImage {
    let mut img = GrayImage::new(300, 250);
    
    // Shape 1: Rectangle with hole
    for y in 20..100 {
        for x in 20..120 {
            img.put_pixel(x, y, Luma([255u8]));
        }
    }
    // Hole in rectangle
    for y in 40..80 {
        for x in 50..90 {
            img.put_pixel(x, y, Luma([0u8]));
        }
    }
    
    // Shape 2: Circle with hole
    let center_x = 200.0;
    let center_y = 60.0;
    let outer_radius = 35.0;
    let inner_radius = 15.0;
    
    for y in 25..95 {
        for x in 165..235 {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist_sq = dx*dx + dy*dy;
            
            if dist_sq <= outer_radius*outer_radius && dist_sq >= inner_radius*inner_radius {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }
    }
    
    // Shape 3: Irregular shape
    for y in 150..220 {
        for x in 50..150 {
            let wave = (10.0 * (x as f32 / 20.0).sin()).abs() as u32;
            if y >= 150 + wave && y <= 200 + wave {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }
    }
    
    img
}