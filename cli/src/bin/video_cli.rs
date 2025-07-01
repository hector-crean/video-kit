use clap::{Parser, Subcommand};
use cli::{InputVideo, Clip};
use color_eyre::eyre::Result;
use cutting::{CutError, Runner, CutVideoOperation};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};
use tracing_subscriber::{self, EnvFilter};

use subtitles::{SmartClipper, SubtitleConfig, SubtitleExporter, SubtitleExtractor};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process video using existing configuration file
    Process {
        /// Path to the JSON configuration file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Generate skeleton configuration from natural speech breaks
    GenerateSkeleton {
        /// Path to the input video file
        #[arg(short, long)]
        input: PathBuf,
        /// Output directory for clips
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Path to save the generated JSON configuration
        #[arg(short, long)]
        json_output: PathBuf,
        /// Minimum silence duration to consider a natural break (in seconds)
        #[arg(long, default_value = "2.0")]
        min_break_duration: f64,
        /// Deepgram API key (or set DEEPGRAM_API_KEY environment variable)
        #[arg(long)]
        api_key: Option<String>,
        /// Maximum segment duration in seconds (clips longer than this will be split)
        #[arg(long, default_value = "300.0")]
        max_segment_duration: f64,
        /// Language for transcription (auto-detect if not specified)
        #[arg(long)]
        language: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info"))
        )
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Process { config } => {
            process_video(config).await?;
        }
        Commands::GenerateSkeleton {
            input,
            output_dir,
            json_output,
            min_break_duration,
            api_key,
            max_segment_duration,
            language,
        } => {
            generate_skeleton(
                input,
                output_dir,
                json_output,
                *min_break_duration,
                api_key.as_deref(),
                *max_segment_duration,
                language.as_deref(),
            ).await?;
        }
    }

    Ok(())
}

async fn process_video(config_path: &Path) -> Result<()> {
    let input_video = InputVideo::from_json_file(config_path)?;
    info!("Input video: {:?}", input_video);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&input_video.output_dir)?;

    for clip in input_video.clips {
        let output_filename = format!("{}/{}.mp4", input_video.output_dir, clip.name);
        info!("Processing clip '{}' -> {}", clip.name, output_filename);
        
        let operation = match clip.operation.clone() {
            Some(op) => op,
            None => {
                warn!(
                    "No operation found for clip '{}': {}",
                    clip.name,
                    clip.description.clone().unwrap_or_default()
                );
                continue;
            }
        };

        let ffmpeg_runner = Runner::ffmpeg_default(&input_video.path, &output_filename)?;
        ffmpeg_runner.execute(operation)?;
    }

    info!("✅ Video processing completed!");
    Ok(())
}

async fn generate_skeleton(
    input_path: &Path,
    output_dir: &Path,
    json_output: &Path,
    min_break_duration: f64,
    api_key: Option<&str>,
    max_segment_duration: f64,
    language: Option<&str>,
) -> Result<()> {
    // Get API key from parameter or environment variable
    let api_key = api_key
        .map(|s| s.to_string())
        .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
        .ok_or_else(|| color_eyre::eyre::eyre!("Deepgram API key not provided. Use --api-key or set DEEPGRAM_API_KEY environment variable"))?;

    info!("🎵 Analyzing video for natural speech breaks: {:?}", input_path);
    info!("Minimum break duration: {:.1}s", min_break_duration);
    info!("Maximum segment duration: {:.1}s", max_segment_duration);

    // Create subtitle configuration
    let mut config = SubtitleConfig::default();
    config.language = language.map(|s| s.to_string());
    config.auto_extract_audio = true;
    config.utterances = true; // Enable utterances for better segmentation

    // Clone API key for multiple uses
    let api_key_cloned = api_key.clone();
    
    // Create smart clipper and find natural breaks
    let clipper = SmartClipper::new(api_key)?;
    let natural_breaks = clipper.find_natural_breaks(input_path, min_break_duration, &config).await?;

    info!("Found {} natural speech breaks", natural_breaks.len());

    // Create segments from natural breaks
    let mut segments = Vec::new();
    let mut current_start = 0.0;

    for &break_point in &natural_breaks {
        let segment_duration = break_point - current_start;
        
        // If segment is too long, split it further
        if segment_duration > max_segment_duration {
            // Split long segments into smaller chunks
            let num_chunks = (segment_duration / max_segment_duration).ceil() as usize;
            let chunk_duration = segment_duration / num_chunks as f64;
            
            for i in 0..num_chunks {
                let chunk_start = current_start + (i as f64 * chunk_duration);
                let chunk_end = if i == num_chunks - 1 {
                    break_point // Last chunk goes to the natural break
                } else {
                    chunk_start + chunk_duration
                };
                
                segments.push((chunk_start, chunk_end));
            }
        } else {
            segments.push((current_start, break_point));
        }
        
        current_start = break_point;
    }

    // Add final segment if there's remaining content
    // Get video duration by checking the last utterance or estimating
    let extractor = SubtitleExtractor::new(api_key_cloned)?;
    let utterances = extractor.extract_speech_segments(input_path, &config).await?;
    let video_end = utterances.last().map(|u| u.end + 5.0).unwrap_or(current_start + 60.0); // Add 5s buffer
    
    if current_start < video_end {
        segments.push((current_start, video_end));
    }

    info!("Created {} segments from natural breaks", segments.len());

    // Create clips from segments
    let clips: Vec<Clip> = segments
        .into_iter()
        .enumerate()
        .map(|(i, (start, end))| {
            let duration = end - start;
            Clip {
                name: format!("segment_{:03}", i + 1),
                description: Some(format!("Natural segment {} ({:.1}s - {:.1}s, duration: {:.1}s)", 
                    i + 1, start, end, duration)),
                operation: Some(CutVideoOperation::Splice {
                    segments: vec![start..end],
                }),
            }
        })
        .collect();

    // Create the InputVideo configuration
    let input_video = InputVideo {
        path: input_path.to_string_lossy().to_string(),
        output_dir: output_dir.to_string_lossy().to_string(),
        clips,
    };

    // Save to JSON file
    input_video.to_json_file(json_output)?;

    info!("✅ Generated skeleton configuration with {} clips", input_video.clips.len());
    info!("📄 Configuration saved to: {:?}", json_output);
    info!("🎬 Total segments: {}", input_video.clips.len());
    
 
    
    if input_video.clips.len() > 5 {
        info!("  ... and {} more clips", input_video.clips.len() - 5);
    }

    Ok(())
}
