
use anyhow::{Context, Result};
use clap::{Arg, Command};
use futures::future::join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command as TokioCommand;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
struct ConversionSettings {
    video_crf: u8,
    webp_quality: u8,
    output_dir: PathBuf,
    max_concurrent: usize,
    dry_run: bool,
}

#[derive(Debug, Clone)]
struct AssetFile {
    input_path: PathBuf,
    output_path: PathBuf,
    file_type: AssetType,
}

#[derive(Debug, Clone)]
enum AssetType {
    Video,
    Image,
}

#[tokio::main]
async fn main() -> Result<()> {

    let settings = ConversionSettings {
        video_crf: 30,  // More conservative quality setting
        webp_quality: 80,
        output_dir: PathBuf::from("/Users/hectorcrean/rust/video-kit/crates/cutting/processed"),
        max_concurrent: 4,  // Reduced to avoid overwhelming the system
        dry_run: false
    };

    let input_dir = "/Users/hectorcrean/rust/video-kit/crates/cutting/output";
    
    println!("🔍 Scanning for assets in: {}", input_dir);
    let assets = discover_assets(input_dir, &settings.output_dir)?;
    
    if assets.is_empty() {
        println!("❌ No assets found to optimize");
        return Ok(());
    }

    let videos: Vec<_> = assets.iter().filter(|a| matches!(a.file_type, AssetType::Video)).collect();
    let images: Vec<_> = assets.iter().filter(|a| matches!(a.file_type, AssetType::Image)).collect();

    println!("📊 Found {} videos and {} images to optimize", videos.len(), images.len());
    
    if settings.dry_run {
        println!("\n🔮 Dry run mode - showing what would be converted:");
        for asset in &assets {
            println!("  {} -> {}", 
                asset.input_path.display(), 
                asset.output_path.display()
            );
        }
        return Ok(());
    }

    // Create output directory
    tokio::fs::create_dir_all(&settings.output_dir).await
        .context("Failed to create output directory")?;

    // Check if ffmpeg is available
    check_ffmpeg_availability().await?;

    // Process assets with progress tracking
    let multi_progress = MultiProgress::new();
    
    if !videos.is_empty() {
        println!("\n🎬 Converting videos to WebM...");
        process_videos(&videos, &settings, &multi_progress).await?;
    }
    
    if !images.is_empty() {
        println!("\n🖼️  Converting images to WebP...");
        process_images(&images, &settings, &multi_progress).await?;
    }

    println!("\n✅ Asset optimization complete!");
    calculate_and_display_savings(input_dir, &settings.output_dir).await?;

    // Uncomment the next line to test a single video file
    // test_single_video_conversion("/path/to/your/video.mp4", "/path/to/output/video.webm").await?;

    Ok(())
}

fn discover_assets(input_dir: &str, output_dir: &Path) -> Result<Vec<AssetFile>> {
    let mut assets = Vec::new();
    
    for entry in WalkDir::new(input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let (file_type, new_extension) = match extension.as_deref() {
            Some("mp4") => (AssetType::Video, "webm"),
            Some("png") => (AssetType::Image, "webp"),
            _ => continue,
        };

        let relative_path = path.strip_prefix(input_dir)
            .context("Failed to strip input directory prefix")?;
        
        let output_path = output_dir.join(relative_path).with_extension(new_extension);

        assets.push(AssetFile {
            input_path: path.to_path_buf(),
            output_path,
            file_type,
        });
    }

    Ok(assets)
}

async fn check_ffmpeg_availability() -> Result<()> {
    let output = TokioCommand::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;

    match output {
        Ok(status) if status.success() => Ok(()),
        _ => anyhow::bail!("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH"),
    }
}

async fn process_videos(
    videos: &[&AssetFile],
    settings: &ConversionSettings,
    multi_progress: &MultiProgress,
) -> Result<()> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(settings.max_concurrent));
    
    let futures = videos.iter().map(|asset| {
        let asset = (*asset).clone();
        let settings = settings.clone();
        let semaphore = semaphore.clone();
        let multi_progress = multi_progress.clone();
        
        async move {
            let _permit = semaphore.acquire().await?;
            
            let pb = multi_progress.add(ProgressBar::new(100));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}/{len:3} {msg}")
                    .unwrap()
                    .progress_chars("#>-")
            );
            
            let filename = asset.input_path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            pb.set_message(format!("Converting {}", filename));
            
            let result = convert_video(&asset, &settings, pb.clone()).await;
            
            if result.is_ok() {
                pb.set_message(format!("✅ {}", filename));
                pb.finish();
            } else {
                pb.set_message(format!("❌ {}", filename));
                pb.abandon();
            }
            
            result
        }
    });

    let results = join_all(futures).await;
    
    // Check for errors
    for result in results {
        result?;
    }

    Ok(())
}

async fn process_images(
    images: &[&AssetFile],
    settings: &ConversionSettings,
    multi_progress: &MultiProgress,
) -> Result<()> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(settings.max_concurrent));
    
    let futures = images.iter().map(|asset| {
        let asset = (*asset).clone();
        let settings = settings.clone();
        let semaphore = semaphore.clone();
        let multi_progress = multi_progress.clone();
        
        async move {
            let _permit = semaphore.acquire().await?;
            
            let pb = multi_progress.add(ProgressBar::new(100));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}/{len:3} {msg}")
                    .unwrap()
                    .progress_chars("#>-")
            );
            
            let filename = asset.input_path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            pb.set_message(format!("Converting {}", filename));
            
            let result = convert_image(&asset, &settings, pb.clone()).await;
            
            if result.is_ok() {
                pb.set_message(format!("✅ {}", filename));
                pb.finish();
            } else {
                pb.set_message(format!("❌ {}", filename));
                pb.abandon();
            }
            
            result
        }
    });

    let results = join_all(futures).await;
    
    // Check for errors
    for result in results {
        result?;
    }

    Ok(())
}

async fn convert_video(
    asset: &AssetFile,
    settings: &ConversionSettings,
    progress: ProgressBar,
) -> Result<()> {
    // Create output directory if it doesn't exist
    if let Some(parent) = asset.output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    progress.set_position(10);

    // First, let's check if the input file exists and get its info
    let input_path = &asset.input_path;
    if !input_path.exists() {
        anyhow::bail!("Input file does not exist: {}", input_path.display());
    }

    // Build ffmpeg command with more robust settings
    let mut cmd = TokioCommand::new("ffmpeg");
    cmd.arg("-i").arg(input_path)
        .arg("-c:v").arg("libvpx-vp9")
        .arg("-crf").arg(settings.video_crf.to_string())
        .arg("-b:v").arg("0")
        .arg("-deadline").arg("good")  // Faster encoding
        .arg("-cpu-used").arg("4")     // Faster encoding
        .arg("-row-mt").arg("1")       // Multi-threading
        .arg("-tile-columns").arg("2") // Better for web
        .arg("-frame-parallel").arg("1") // Parallel frame processing
        .arg("-y"); // Overwrite output file

    // Handle audio - use copy if available, otherwise skip audio
    cmd.arg("-c:a").arg("libopus")
        .arg("-b:a").arg("128k");

    cmd.arg(&asset.output_path);

    progress.set_position(20);

    // Capture stderr for debugging
    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("Failed to execute ffmpeg")?;

    progress.set_position(90);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // If the error is about audio, try without audio
        if stderr.contains("No audio stream") || stderr.contains("Invalid audio stream") {
            println!("🔄 Retrying without audio for: {}", asset.input_path.file_name().unwrap_or_default().to_string_lossy());
            
            let mut cmd_no_audio = TokioCommand::new("ffmpeg");
            cmd_no_audio.arg("-i").arg(input_path)
                .arg("-c:v").arg("libvpx-vp9")
                .arg("-crf").arg(settings.video_crf.to_string())
                .arg("-b:v").arg("0")
                .arg("-deadline").arg("good")
                .arg("-cpu-used").arg("4")
                .arg("-row-mt").arg("1")
                .arg("-tile-columns").arg("2")
                .arg("-frame-parallel").arg("1")
                .arg("-an") // No audio
                .arg("-y")
                .arg(&asset.output_path);

            let output_no_audio = cmd_no_audio
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
                .context("Failed to execute ffmpeg (no audio)")?;

            if !output_no_audio.status.success() {
                let stderr_no_audio = String::from_utf8_lossy(&output_no_audio.stderr);
                anyhow::bail!(
                    "ffmpeg failed to convert video (with and without audio): {}\nStderr: {}",
                    asset.input_path.display(),
                    stderr_no_audio
                );
            }
        } else {
            anyhow::bail!(
                "ffmpeg failed to convert video: {}\nStderr: {}",
                asset.input_path.display(),
                stderr
            );
        }
    }

    progress.set_position(100);
    Ok(())
}

async fn convert_image(
    asset: &AssetFile,
    settings: &ConversionSettings,
    progress: ProgressBar,
) -> Result<()> {
    // Create output directory if it doesn't exist
    if let Some(parent) = asset.output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    progress.set_position(10);

    let mut cmd = TokioCommand::new("ffmpeg");
    cmd.arg("-i").arg(&asset.input_path)
        .arg("-c:v").arg("libwebp")
        .arg("-quality").arg(settings.webp_quality.to_string())
        .arg("-y") // Overwrite output file
        .arg(&asset.output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    progress.set_position(20);

    let status = cmd.status().await
        .context("Failed to execute ffmpeg")?;

    progress.set_position(90);

    if !status.success() {
        anyhow::bail!("ffmpeg failed to convert image: {}", asset.input_path.display());
    }

    progress.set_position(100);
    Ok(())
}

async fn calculate_and_display_savings(input_dir: &str, output_dir: &Path) -> Result<()> {
    let input_size = get_directory_size(input_dir).await?;
    let output_size = get_directory_size(output_dir).await?;
    
    if input_size > 0 {
        println!("\n📊 Size Comparison:");
        println!("  Original: {}", format_bytes(input_size));
        println!("  Optimized: {}", format_bytes(output_size));
        
        if output_size <= input_size {
            let savings = input_size - output_size;
            let savings_percent = (savings as f64 / input_size as f64) * 100.0;
            println!("  Savings: {} ({:.1}%)", format_bytes(savings), savings_percent);
        } else {
            let increase = output_size - input_size;
            let increase_percent = (increase as f64 / input_size as f64) * 100.0;
            println!("  ⚠️  Size increased: {} ({:.1}%)", format_bytes(increase), increase_percent);
        }
    }
    
    Ok(())
}

async fn get_directory_size(dir: impl AsRef<Path>) -> Result<u64> {
    let mut total_size = 0;
    let mut stack = vec![dir.as_ref().to_path_buf()];
    
    while let Some(current_dir) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&current_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                if let Ok(metadata) = tokio::fs::metadata(&path).await {
                    total_size += metadata.len();
                }
            } else if path.is_dir() {
                stack.push(path);
            }
        }
    }
    
    Ok(total_size)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
} 

async fn test_single_video_conversion(input_path: &str, output_path: &str) -> Result<()> {
    println!("🧪 Testing video conversion:");
    println!("  Input: {}", input_path);
    println!("  Output: {}", output_path);
    
    let mut cmd = TokioCommand::new("ffmpeg");
    cmd.arg("-i").arg(input_path)
        .arg("-c:v").arg("libvpx-vp9")
        .arg("-crf").arg("30")
        .arg("-b:v").arg("0")
        .arg("-deadline").arg("good")
        .arg("-cpu-used").arg("4")
        .arg("-row-mt").arg("1")
        .arg("-tile-columns").arg("2")
        .arg("-frame-parallel").arg("1")
        .arg("-c:a").arg("libopus")
        .arg("-b:a").arg("128k")
        .arg("-y")
        .arg(output_path);

    println!("🔧 FFmpeg command:");
    println!("  {}", cmd.as_std().get_program().to_string_lossy());
    for arg in cmd.as_std().get_args() {
        println!("  {}", arg.to_string_lossy());
    }
    println!();

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("Failed to execute ffmpeg")?;

    if output.status.success() {
        println!("✅ Conversion successful!");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("❌ Conversion failed:");
        println!("{}", stderr);
        anyhow::bail!("Video conversion failed");
    }

    Ok(())
} 