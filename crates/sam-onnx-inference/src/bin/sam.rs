use burn::prelude::*;
use burn::backend::Wgpu;
use clap::{Parser, Subcommand};
use image::DynamicImage;
use anyhow::Result;
use log::info;

use sam_inference::{
    SamVariant,
    preprocessing::{preprocess_image, get_original_dimensions},
    postprocessing::{masks_to_images, visualize_masks_on_image, PostprocessConfig},
    weights::load_sam_weights,
    model::sam::{Sam, SamConfig},
};

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "sam")]
#[command(about = "SAM (Segment Anything Model) inference with native Burn implementation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Segment objects using point prompts
    Points {
        /// Path to input image
        #[arg(short, long)]
        image: String,
        
        /// SAM model variant (base, large, huge)
        #[arg(short, long, default_value = "base")]
        model: String,
        
        /// Path to model checkpoint
        #[arg(short, long)]
        checkpoint: Option<String>,
        
        /// Point coordinates as "x1,y1 x2,y2 ..." 
        #[arg(short, long)]
        points: String,
        
        /// Point labels as "1 0 1 ..." (1=positive, 0=negative)
        #[arg(short, long, default_value = "1")]
        labels: String,
        
        /// Output directory for masks
        #[arg(short, long, default_value = "output")]
        output: String,
        
        /// Whether to save visualization
        #[arg(long)]
        visualize: bool,
    },
    
    /// Segment objects using box prompts
    Boxes {
        /// Path to input image
        #[arg(short, long)]
        image: String,
        
        /// SAM model variant (base, large, huge)
        #[arg(short, long, default_value = "base")]
        model: String,
        
        /// Path to model checkpoint
        #[arg(short, long)]
        checkpoint: Option<String>,
        
        /// Box coordinates as "x1,y1,x2,y2 x3,y3,x4,y4 ..."
        #[arg(short, long)]
        boxes: String,
        
        /// Output directory for masks
        #[arg(short, long, default_value = "output")]
        output: String,
        
        /// Whether to save visualization
        #[arg(long)]
        visualize: bool,
    },
    
    /// Generate example masks with random initialization (for testing)
    Demo {
        /// Path to input image
        #[arg(short, long)]
        image: String,
        
        /// Output directory for demo masks
        #[arg(short, long, default_value = "demo_output")]
        output: String,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Points { image, model, checkpoint, points, labels, output, visualize } => {
            run_point_segmentation(image, model, checkpoint, points, labels, output, visualize)
        }
        Commands::Boxes { image, model, checkpoint, boxes, output, visualize } => {
            run_box_segmentation(image, model, checkpoint, boxes, output, visualize)
        }
        Commands::Demo { image, output } => {
            run_demo(image, output)
        }
    }
}

fn run_point_segmentation(
    image_path: String,
    model_variant: String,
    checkpoint_path: Option<String>,
    points_str: String,
    labels_str: String,
    output_dir: String,
    visualize: bool,
) -> Result<()> {
    info!("Starting point-based segmentation");
    
    // Parse SAM variant
    let variant = match model_variant.as_str() {
        "base" => SamVariant::Base,
        "large" => SamVariant::Large,
        "huge" => SamVariant::Huge,
        _ => return Err(anyhow::anyhow!("Invalid model variant: {}", model_variant)),
    };
    
    // Parse points and labels
    let points = parse_points(&points_str)?;
    let labels = parse_labels(&labels_str, points.len())?;
    
    info!("Points: {:?}", points);
    info!("Labels: {:?}", labels);
    
    // Load image
    let image = image::open(&image_path)?;
    let original_size = get_original_dimensions(&image);
    info!("Image size: {:?}", original_size);
    
    // Create WGPU device and model
    let device = burn::backend::wgpu::WgpuDevice::default();
    let model = if let Some(checkpoint) = checkpoint_path {
        info!("Loading weights from: {}", checkpoint);
        load_sam_weights(&variant, &checkpoint, &device)?
    } else {
        info!("Using random initialization (no checkpoint provided)");
        let config = variant.vit_config();
        Sam::new(&config, &device)
    };
    
    // Preprocess image
    info!("Preprocessing image...");
    let processed_image = preprocess_image::<Backend>(&image, &device)?;
    
    // Run inference
    info!("Running SAM inference...");
    let masks = model.forward_points(processed_image, &points, &labels);
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Convert masks to images
    info!("Converting masks to images...");
    let mask_images = masks_to_images(masks, original_size.0, original_size.1)?;
    
    // Save masks
    for (i, mask) in mask_images.iter().enumerate() {
        let mask_path = format!("{}/mask_{}.png", output_dir, i);
        mask.save(&mask_path)?;
        info!("Saved mask: {}", mask_path);
    }
    
    // Create visualization if requested
    if visualize {
        info!("Creating visualization...");
        let config = PostprocessConfig::default();
        let viz = visualize_masks_on_image(&image, &mask_images, &config)?;
        let viz_path = format!("{}/visualization.png", output_dir);
        viz.save(&viz_path)?;
        info!("Saved visualization: {}", viz_path);
    }
    
    info!("Point segmentation complete!");
    Ok(())
}

fn run_box_segmentation(
    image_path: String,
    model_variant: String,
    checkpoint_path: Option<String>,
    boxes_str: String,
    output_dir: String,
    visualize: bool,
) -> Result<()> {
    info!("Starting box-based segmentation");
    
    // Parse SAM variant
    let variant = match model_variant.as_str() {
        "base" => SamVariant::Base,
        "large" => SamVariant::Large,
        "huge" => SamVariant::Huge,
        _ => return Err(anyhow::anyhow!("Invalid model variant: {}", model_variant)),
    };
    
    // Parse boxes
    let boxes = parse_boxes(&boxes_str)?;
    info!("Boxes: {:?}", boxes);
    
    // Load image
    let image = image::open(&image_path)?;
    let original_size = get_original_dimensions(&image);
    info!("Image size: {:?}", original_size);
    
    // Create WGPU device and model
    let device = burn::backend::wgpu::WgpuDevice::default();
    let model = if let Some(checkpoint) = checkpoint_path {
        info!("Loading weights from: {}", checkpoint);
        load_sam_weights(&variant, &checkpoint, &device)?
    } else {
        info!("Using random initialization (no checkpoint provided)");
        let config = variant.vit_config();
        Sam::new(&config, &device)
    };
    
    // Preprocess image
    info!("Preprocessing image...");
    let processed_image = preprocess_image::<Backend>(&image, &device)?;
    
    // Run inference
    info!("Running SAM inference...");
    let masks = model.forward_boxes(processed_image, &boxes);
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Convert masks to images
    info!("Converting masks to images...");
    let mask_images = masks_to_images(masks, original_size.0, original_size.1)?;
    
    // Save masks
    for (i, mask) in mask_images.iter().enumerate() {
        let mask_path = format!("{}/mask_{}.png", output_dir, i);
        mask.save(&mask_path)?;
        info!("Saved mask: {}", mask_path);
    }
    
    // Create visualization if requested
    if visualize {
        info!("Creating visualization...");
        let config = PostprocessConfig::default();
        let viz = visualize_masks_on_image(&image, &mask_images, &config)?;
        let viz_path = format!("{}/visualization.png", output_dir);
        viz.save(&viz_path)?;
        info!("Saved visualization: {}", viz_path);
    }
    
    info!("Box segmentation complete!");
    Ok(())
}

fn run_demo(image_path: String, output_dir: String) -> Result<()> {
    info!("Running SAM demo with random initialization");
    
    // Load image
    let image = image::open(&image_path)?;
    let original_size = get_original_dimensions(&image);
    info!("Image size: {:?}", original_size);
    
    // Create WGPU device and model (random initialization)
    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = SamVariant::Base.vit_config();
    let model = Sam::new(&config, &device);
    
    // Preprocess image
    info!("Preprocessing image...");
    let processed_image = preprocess_image::<Backend>(&image, &device)?;
    
    // Use center point as demo prompt
    let center_x = original_size.0 as f32 / 2.0;
    let center_y = original_size.1 as f32 / 2.0;
    let demo_points = vec![(center_x, center_y)];
    let demo_labels = vec![1]; // positive point
    
    info!("Using demo point: ({:.1}, {:.1})", center_x, center_y);
    
    // Run inference
    info!("Running SAM demo inference...");
    let masks = model.forward_points(processed_image, &demo_points, &demo_labels);
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Convert masks to images
    info!("Converting demo masks to images...");
    let mask_images = masks_to_images(masks, original_size.0, original_size.1)?;
    
    // Save masks
    for (i, mask) in mask_images.iter().enumerate() {
        let mask_path = format!("{}/demo_mask_{}.png", output_dir, i);
        mask.save(&mask_path)?;
        info!("Saved demo mask: {}", mask_path);
    }
    
    // Create visualization
    info!("Creating demo visualization...");
    let config = PostprocessConfig::default();
    let viz = visualize_masks_on_image(&image, &mask_images, &config)?;
    let viz_path = format!("{}/demo_visualization.png", output_dir);
    viz.save(&viz_path)?;
    info!("Saved demo visualization: {}", viz_path);
    
    info!("Demo complete! Note: This uses random weights, so masks won't be meaningful.");
    info!("To get real segmentation results, provide a trained checkpoint with --checkpoint");
    
    Ok(())
}

fn parse_points(points_str: &str) -> Result<Vec<(f32, f32)>> {
    let mut points = Vec::new();
    
    for point_str in points_str.split_whitespace() {
        let coords: Vec<&str> = point_str.split(',').collect();
        if coords.len() != 2 {
            return Err(anyhow::anyhow!("Invalid point format: {}", point_str));
        }
        
        let x: f32 = coords[0].parse()?;
        let y: f32 = coords[1].parse()?;
        points.push((x, y));
    }
    
    Ok(points)
}

fn parse_labels(labels_str: &str, num_points: usize) -> Result<Vec<i32>> {
    let labels: Result<Vec<i32>, _> = labels_str
        .split_whitespace()
        .map(|s| s.parse())
        .collect();
    
    let mut labels = labels?;
    
    // Pad with positive labels if not enough provided
    while labels.len() < num_points {
        labels.push(1);
    }
    
    // Truncate if too many provided
    labels.truncate(num_points);
    
    Ok(labels)
}

fn parse_boxes(boxes_str: &str) -> Result<Vec<(f32, f32, f32, f32)>> {
    let mut boxes = Vec::new();
    
    for box_str in boxes_str.split_whitespace() {
        let coords: Vec<&str> = box_str.split(',').collect();
        if coords.len() != 4 {
            return Err(anyhow::anyhow!("Invalid box format: {}", box_str));
        }
        
        let x1: f32 = coords[0].parse()?;
        let y1: f32 = coords[1].parse()?;
        let x2: f32 = coords[2].parse()?;
        let y2: f32 = coords[3].parse()?;
        boxes.push((x1, y1, x2, y2));
    }
    
    Ok(boxes)
} 