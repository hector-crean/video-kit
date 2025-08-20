use cutting::{init_macos_app, ExportFormat, LoopMode, ReverseMode, VideoEditCommand, VideoEditor};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    init_macos_app()?;
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("cutting=debug,info")
        .init();

    let input_uri = "file:////Users/hectorcrean/rust/video-kit/cli/assets/NOV271_Zigakibart_MoA_576p_Preview_250728a.mp4";
    
    let mut editor = VideoEditor::new(input_uri.to_string());
    
    // Demonstrate range-based cutting - much more intuitive!
    editor.add_command(VideoEditCommand::cut_range_ms(10500, 15250)); // From 10.5s to 15.25s
    

    // Example 3: Export frame sequence - NEW SIMPLE API for entire video!
    let entire_video_frames = ExportFormat::frame_sequence_entire_video(5.0); // 5fps for entire video
    // This extracts frames from the ENTIRE video at 5fps - so much simpler!
    // Output: ./output/frames/frame_000001.png, frame_000002.png, etc.
    
    // Example 4: Export frame sequence using specific range (for partial video)
    let range_frames_config = ExportFormat::frame_sequence_range_ms_named(1250, 4750, 30.0, "range_frame_%06d".to_string());
    // Extract frames from 1.25s to 4.75s at 30fps
    // Output: ./output/frames/range_frame_000001.png, range_frame_000002.png, etc.
    
    tracing::info!("🎬 Range-Based Video Editing (much more intuitive!):");
    tracing::info!("   - Cut from 10.500s to 15.250s (no mental math needed!)");
    tracing::info!("   - Poster extracted at exactly 2.000s");
    tracing::info!("   - Frame sequence: 1.250s to 4.750s");
    tracing::info!("");
    tracing::info!("✨ New range-based API:");
    tracing::info!("   - cut_range_ms(10500, 15250) = cut from 10.5s to 15.25s");  
    tracing::info!("   - freeze_range_ms(5250, 6750) = freeze from 5.25s to 6.75s");
    tracing::info!("   - frame_sequence_range_ms(1250, 4750, 30.0) = 1.25s to 4.75s");
    tracing::info!("   - No more duration calculations - just specify the range!");
    tracing::info!("");
    tracing::info!("Examples of what gets created:");
    tracing::info!("   Video:    ./output/range_cut_video.mp4");
    tracing::info!("   Poster:   ./output/precise_thumbnail_2000.png");
    tracing::info!("   Frames:   ./output/frames/range_frame_000001.png, ...");
    
    // Uncomment to test actual rendering:
    // For entire video (simple):
    editor.render_to_file(range_frames_config, "./output/frames")?;
    // For specific range:
    // editor.render_to_file(range_frames_config, "./output/frames")?;
    
    Ok(())
}
