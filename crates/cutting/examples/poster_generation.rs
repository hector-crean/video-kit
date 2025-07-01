use cutting::{Runner, CutVideoOperation};
use cutting::driver::ffmpeg::{FFmpegDriver, FFmpegFileSource, FFmpegFileSink};
use std::ops::Range;
use cutting::driver::StreamOps;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Enhanced Splice with Poster Generation Example");
    
    // Initialize FFmpeg driver
    let driver = FFmpegDriver::new()?;
    
    // Define input and output
    let input_video = "input.mp4"; // Replace with your video file
    let output_video = "output_segment.mp4";
    
    let source = FFmpegFileSource::new(input_video);
    let sink = FFmpegFileSink::new(output_video);
    
    let runner = Runner::new(driver, source, sink);
    
    // Example 1: Single segment splice (generates both video and poster)
    println!("\n📹 Example 1: Single segment splice");
    let single_segment = CutVideoOperation::Splice {
        segments: vec![10.0..30.0], // Extract from 10s to 30s
    };
    
    match runner.execute(single_segment) {
        Ok(_) => {
            println!("✅ Single segment extraction completed!");
            println!("   Generated: output_segment.mp4");
            println!("   Generated: output_segment_poster.png");
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 2: Multiple segments splice (concatenates segments, generates poster from first)
    println!("\n📹 Example 2: Multiple segments splice");
    let multi_segment_sink = FFmpegFileSink::new("output_concatenated.mp4");
    let multi_runner = runner.with_sink(multi_segment_sink);
    
    let multiple_segments = CutVideoOperation::Splice {
        segments: vec![
            5.0..15.0,   // First segment: 5s to 15s
            30.0..45.0,  // Second segment: 30s to 45s  
            60.0..75.0,  // Third segment: 60s to 75s
        ],
    };
    
    match multi_runner.execute(multiple_segments) {
        Ok(_) => {
            println!("✅ Multiple segment concatenation completed!");
            println!("   Generated: output_concatenated.mp4 (concatenated segments)");
            println!("   Generated: output_concatenated_poster.png (from first segment at 5s)");
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 3: Using the streaming API for more control
    println!("\n📹 Example 3: Streaming API");
    let stream_sink = FFmpegFileSink::new("output_stream.mp4");
    let stream_runner = runner.with_sink(stream_sink);
    
    let segments = &[20.0..40.0, 50.0..70.0];
    
    match stream_runner.execute_stream(|stream, driver| {
        stream.splice(driver, segments)
    }) {
        Ok(_) => {
            println!("✅ Streaming API splice completed!");
            println!("   Generated: output_stream.mp4");
            println!("   Generated: output_stream_poster.png");
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    println!("\n🎉 All examples completed!");
    println!("\nNote: Make sure you have a video file named 'input.mp4' in the current directory");
    println!("      or modify the input_video path in the example.");
    
    Ok(())
} 