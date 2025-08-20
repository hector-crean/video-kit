use subtitles::{
    SubtitleExtractor, SmartClipper, SubtitleConfig, SubtitleExporter,
    extract_subtitles, extract_speech_segments, find_optimal_cuts
};
use std::path::{Path, PathBuf};
use std::env;
use tokio;
use walkdir::WalkDir;
use glob::{Pattern, PatternError};
use thiserror::Error;
use std::process::Command;

async fn has_audio_stream(video_path: &Path) -> bool {
    let output = Command::new("ffprobe")
        .args(&[
            "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path.to_str().unwrap()
        ])
        .output();
    
    match output {
        Ok(result) => !result.stdout.is_empty(),
        Err(_) => false
    }
}


async fn file_extraction_handler(api_key: String, video_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️ Subtitle Extraction Example");
    println!("==============================");

 
    
    if !video_path.exists() {
        println!("❌ Video file not found: {:?}", video_path);
        println!("Please update the video_path in the example to point to your video file");
        return Ok(());
    }

    println!("📹 Processing video: {:?}", video_path);

    // Configure subtitle extraction with automatic audio extraction
    let config = SubtitleConfig {
        language: Some("en".to_string()), // English
        identify_speakers: true,          // Enable speaker identification
        word_timestamps: true,            // Get word-level timing
        filter_profanity: false,          // Keep original content
        custom_vocabulary: vec![          // Add domain-specific terms
          
        ],
        model: Some("nova-3".to_string()), // Use Deepgram's latest model
        punctuation: true,
        utterances: true,
        audio_bitrate: 128,               // 128 kbps MP3 for good quality/size balance
        auto_extract_audio: true,         // Automatically extract audio from video
    };

    println!("⚙️ Configuration:");
    println!("  Language: {:?}", config.language);
    println!("  Speaker ID: {}", config.identify_speakers);
    println!("  Word timing: {}", config.word_timestamps);
    println!("  Model: {:?}", config.model);
    println!("  Audio extraction: {} ({} kbps)", config.auto_extract_audio, config.audio_bitrate);

    // Method 1: Simple subtitle extraction (automatically extracts audio if needed)
    println!("\n📝 Method 1: Simple Subtitle Extraction");
    println!("----------------------------------------");
    
    match extract_subtitles(api_key.clone(), video_path, Some(config.clone())).await {
        Ok(subtitles) => {
            println!("✅ Extracted {} subtitle segments", subtitles.len());
            
            // Show first few subtitles
            for (i, subtitle) in subtitles.iter().take(3).enumerate() {
                println!("  {}: {:.2}s - {:.2}s: \"{}\"", 
                    i + 1, subtitle.start, subtitle.end, subtitle.text);
                println!("     Confidence: {:.1}%", subtitle.confidence * 100.0);
                if let Some(speaker) = &subtitle.speaker {
                    println!("     Speaker: {}", speaker);
                }
            }
            
            if subtitles.len() > 3 {
                println!("  ... and {} more segments", subtitles.len() - 3);
            }

            // Export to different formats using video filename
            println!("\n💾 Exporting subtitles:");
            
            // Get video directory and filename stem
            let video_dir = video_path.parent().unwrap_or(Path::new("."));
            let video_stem = video_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            
            // Create output filenames based on video name
            let srt_path = video_dir.join(format!("{}-subtitle.srt", video_stem));
            let vtt_path = video_dir.join(format!("{}-subtitle.vtt", video_stem));
            let json_path = video_dir.join(format!("{}-subtitle.json", video_stem));
            
            // Export to SRT format
            let srt_content = SubtitleExporter::to_srt(&subtitles);
            tokio::fs::write(&srt_path, srt_content).await?;
            println!("  ✅ Exported to {}", srt_path.display());
            
            // Export to VTT format  
            let vtt_content = SubtitleExporter::to_vtt(&subtitles);
            tokio::fs::write(&vtt_path, vtt_content).await?;
            println!("  ✅ Exported to {}", vtt_path.display());
            
            // Export to JSON format
            let json_content = SubtitleExporter::to_json(&subtitles)?;
            tokio::fs::write(&json_path, json_content).await?;
            println!("  ✅ Exported to {}", json_path.display());
        }
        Err(e) => {
            println!("❌ Failed to extract subtitles: {}", e);
            return Err(e.into());
        }
    }

  

    Ok(())
} 





#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

       // Get API key from environment variable
       let api_key = env::var("DEEPGRAM_API_KEY")
       .expect("Please set DEEPGRAM_API_KEY environment variable");

   // let video_path = Path::new("/Users/hectorcrean/typescript/ran/public/scene_03_02_base_loop.mp4");

   let video_dir = Path::new("/Users/hectorcrean/rust/video-kit/crates/cutting/processed");
   
       // Process files sequentially
    let glob_pattern = glob::Pattern::new("*.webm")?;
    for entry in walkdir::WalkDir::new(video_dir).into_iter() {
        let entry = entry?;
        if entry.file_type().is_file() && glob_pattern.matches_path(entry.path()) {
            println!("\n📹 Checking: {}", entry.path().display());
            
            // Check if file has audio before processing
            if !has_audio_stream(entry.path()).await {
                println!("🔇 Skipping file (no audio stream): {}", entry.path().display());
                continue;
            }
            
            println!("🎵 Audio detected, processing subtitles...");
            match file_extraction_handler(api_key.clone(), entry.path()).await {
                Ok(()) => println!("✅ Successfully processed: {}", entry.path().display()),
                Err(e) => {
                    println!("⚠️ Skipping file due to error: {}", e);
                    println!("   File: {}", entry.path().display());
                    // Continue processing other files
                    continue;
                }
            }
        }
    }

   Ok(())
}