use clap::{Parser, Subcommand};
use cli::{InputVideo, Clip};
use color_eyre::eyre::Result;
use cutting::{CutError, Runner, CutVideoOperation, Cut};
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
        
      

        let ffmpeg_runner = Runner::ffmpeg_default(&input_video.path, &output_filename)?;
        ffmpeg_runner.execute(clip.operation)?;
    }

    info!("✅ Video processing completed!");
    Ok(())
}
