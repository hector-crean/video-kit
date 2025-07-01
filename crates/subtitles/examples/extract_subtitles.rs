use subtitles::{
    SubtitleExtractor, SmartClipper, SubtitleConfig, SubtitleExporter,
    extract_subtitles, extract_speech_segments, find_optimal_cuts
};
use std::path::Path;
use std::env;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️ Subtitle Extraction Example");
    println!("==============================");

    // Get API key from environment variable
    let api_key = env::var("DEEPGRAM_API_KEY")
        .expect("Please set DEEPGRAM_API_KEY environment variable");

    // Example video path - you can change this to your video file
    let video_path = Path::new("/Users/hectorcrean/rust/video-kit/video_temp/NOV271_Zigakibart_MoA_1080p_HD_240923a.mp4");
    // let video_path = Path::new("/Users/hectorcrean/typescript/ran/public/scene_03_02_base_loop.mp4");
    
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

            // Export to different formats
            println!("\n💾 Exporting subtitles:");
            
            // Export to SRT format
            let srt_content = SubtitleExporter::to_srt(&subtitles);
            tokio::fs::write("output_subtitles.srt", srt_content).await?;
            println!("  ✅ Exported to output_subtitles.srt");
            
            // Export to VTT format  
            let vtt_content = SubtitleExporter::to_vtt(&subtitles);
            tokio::fs::write("output_subtitles.vtt", vtt_content).await?;
            println!("  ✅ Exported to output_subtitles.vtt");
            
            // Export to JSON format
            let json_content = SubtitleExporter::to_json(&subtitles)?;
            tokio::fs::write("output_subtitles.json", json_content).await?;
            println!("  ✅ Exported to output_subtitles.json");
        }
        Err(e) => {
            println!("❌ Failed to extract subtitles: {}", e);
            return Err(e.into());
        }
    }

    // Method 2: Working with Deepgram Utterances directly
    println!("\n🎤 Method 2: Direct Deepgram Utterances");
    println!("--------------------------------------");

    match extract_speech_segments(api_key.clone(), video_path, Some(config.clone())).await {
        Ok(utterances) => {
            println!("✅ Extracted {} utterances", utterances.len());
            
            // Show detailed utterance information
            for (i, utterance) in utterances.iter().take(2).enumerate() {
                println!("  Utterance {}:", i + 1);
                println!("    Time: {:.2}s - {:.2}s", utterance.start, utterance.end);
                println!("    Text: \"{}\"", utterance.transcript);
                println!("    Confidence: {:.1}%", utterance.confidence * 100.0);
                println!("    Channel: {}", utterance.channel);
                if let Some(speaker) = utterance.speaker {
                    println!("    Speaker: {}", speaker);
                }
                println!("    Words: {} words", utterance.words.len());
                
                // Show first few words with timing
                for (wi, word) in utterance.words.iter().take(5).enumerate() {
                    println!("      {}: \"{}\" ({:.2}s-{:.2}s)", 
                        wi + 1, word.word, word.start, word.end);
                }
                if utterance.words.len() > 5 {
                    println!("      ... and {} more words", utterance.words.len() - 5);
                }
                println!();
            }

            // Export utterances directly to subtitle formats
            println!("💾 Exporting utterances to subtitle formats:");
            
            let utterances_srt = SubtitleExporter::utterances_to_srt(&utterances);
            tokio::fs::write("utterances.srt", utterances_srt).await?;
            println!("  ✅ Exported utterances to utterances.srt");
            
            let utterances_vtt = SubtitleExporter::utterances_to_vtt(&utterances);
            tokio::fs::write("utterances.vtt", utterances_vtt).await?;
            println!("  ✅ Exported utterances to utterances.vtt");
        }
        Err(e) => {
            println!("❌ Failed to extract speech segments: {}", e);
            return Err(e.into());
        }
    }

    // Method 3: Smart video clipping with speech analysis
    println!("\n✂️ Method 3: Smart Video Clipping");
    println!("---------------------------------");

    // Define some desired cut segments (start_time, end_time) in seconds
    let desired_segments = vec![
        (0.0, 2.0), 
        (2.0, 17.0),
        (17.0, 22.0),
        (22.0, 58.0),
        (58.0, 67.0),
        (67.0, 77.0),
    ];

    println!("🎯 Finding optimal cut points for {} segments:", desired_segments.len());
    for (i, (start, end)) in desired_segments.iter().enumerate() {
        println!("  Segment {}: {:.1}s - {:.1}s (duration: {:.1}s)", 
            i + 1, start, end, end - start);
    }

    match find_optimal_cuts(
        api_key.clone(), 
        video_path, 
        &desired_segments, 
        Some(config.clone())
    ).await {
        Ok(optimal_cuts) => {
            println!("\n🎯 Optimal cut points found:");
            for (i, cut) in optimal_cuts.iter().enumerate() {
                println!("  Segment {}:", i + 1);
                println!("    Original: {:.2}s - {:.2}s", cut.original_start, cut.original_end);
                println!("    Optimal:  {:.2}s - {:.2}s", cut.optimal_start, cut.optimal_end);
                
                let start_adjustment = (cut.optimal_start - cut.original_start).abs();
                let end_adjustment = (cut.optimal_end - cut.original_end).abs();
                
                if start_adjustment > 0.1 || end_adjustment > 0.1 {
                    println!("    Adjustment: {}", cut.adjustment_reason);
                    println!("    Confidence: start {:.1}%, end {:.1}%", 
                        cut.start_confidence * 100.0, cut.end_confidence * 100.0);
                } else {
                    println!("    ✅ No adjustment needed - cuts are in silent periods");
                }
                println!();
            }
        }
        Err(e) => {
            println!("❌ Failed to find optimal cuts: {}", e);
            return Err(e.into());
        }
    }

    // Method 4: Advanced speech analysis
    println!("\n🧠 Method 4: Advanced Speech Analysis");
    println!("------------------------------------");

    let clipper = SmartClipper::new(api_key.clone())?;

    // Find natural break points (pauses in speech)
    println!("🔍 Finding natural break points...");
    match clipper.find_natural_breaks(video_path, 1.0, &config).await {
        Ok(breaks) => {
            println!("✅ Found {} natural break points:", breaks.len());
            for (i, break_time) in breaks.iter().take(10).enumerate() {
                println!("  Break {}: {:.2}s", i + 1, break_time);
            }
            if breaks.len() > 10 {
                println!("  ... and {} more breaks", breaks.len() - 10);
            }
        }
        Err(e) => {
            println!("❌ Failed to find natural breaks: {}", e);
        }
    }

    // Method 5: Testing different audio quality settings
    println!("\n🎛️ Method 5: Testing Different Audio Quality");
    println!("--------------------------------------------");

    // Test with different bitrates to show file size vs quality tradeoffs
    let test_bitrates = vec![64, 128, 192, 320];
    
    println!("Testing audio extraction with different bitrates:");
    for bitrate in test_bitrates {
        let mut test_config = config.clone();
        test_config.audio_bitrate = bitrate;
        
        println!("\n🔊 Testing {}k bitrate:", bitrate);
        
        match extract_subtitles(api_key.clone(), video_path, Some(test_config)).await {
            Ok(subtitles) => {
                println!("  ✅ Successfully extracted {} segments with {}k bitrate", 
                    subtitles.len(), bitrate);
                
                // Show just the first subtitle for comparison
                if let Some(first) = subtitles.first() {
                    println!("  First segment: \"{:.50}{}\"", 
                        first.text, 
                        if first.text.len() > 50 { "..." } else { "" }
                    );
                }
            }
            Err(e) => {
                println!("  ❌ Failed with {}k bitrate: {}", bitrate, e);
            }
        }
    }

    println!("\n🎉 Example completed successfully!");
    println!("\n💡 Key Features Demonstrated:");
    println!("  1. 🎵 Automatic audio extraction from video files");
    println!("  2. 📏 Configurable audio quality (64k-320k bitrate)");
    println!("  3. 🗂️ Temporary file management (auto-cleanup)");
    println!("  4. 📝 Multiple export formats (SRT, VTT, JSON)");
    println!("  5. ✂️ Smart video clipping based on speech analysis");
    println!("  6. 🔍 Natural break point detection");
    
    println!("\n🔧 Next Steps:");
    println!("  - Set DEEPGRAM_API_KEY environment variable");
    println!("  - Ensure FFmpeg is installed for video processing");
    println!("  - Integrate with cutting crate for automated processing");
    println!("  - Experiment with different audio bitrates for your use case");
    println!("  - Use natural breaks for automatic video segmentation");

    Ok(())
} 