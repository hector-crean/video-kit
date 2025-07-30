use cutting::{init_macos_app, ExportFormat, OutputConfig, VideoEditCommand, VideoEditor};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    init_macos_app()?;
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("cutting=debug,info")
        .init();

    let input_uri = "file:////Users/hectorcrean/rust/video-kit/cli/assets/NOV271_Zigakibart_MoA_576p_Preview_250728a.mp4";
    
    let mut editor = VideoEditor::new(input_uri.to_string());
    editor.add_command(VideoEditCommand::cut(10, 30)); // Cut 30 seconds starting at 10s
    
    // Example 1: Export as MP4 video
    let video_config = OutputConfig {
        output_dir: "output".to_string(), // All outputs go in this directory
        format: ExportFormat::mp4_video_named("my_edited_video".to_string()),
    };
    // Creates: ./output/my_edited_video.mp4
    
    // Example 2: Export poster image
    let poster_config = OutputConfig {
        output_dir: "./output".to_string(), // Same directory
        format: ExportFormat::poster_at_named(15, "thumbnail".to_string()),
    };
    // Creates: ./output/thumbnail_15.png
    
    // Example 3: Export frame sequence
    let frames_config = OutputConfig {
        output_dir: "./output/frames".to_string(), // Subdirectory for frames
        format: ExportFormat::frame_sequence_png_named(0, 10, 30.0, "scene_%06d".to_string()),
    };
    // Creates: ./output/frames/scene_000001.png, ./output/frames/scene_000002.png, etc.

    editor.render_to_file(video_config)?;
    
    tracing::info!("✨ The improved API is now consistent:");
    tracing::info!("   - output_dir is ALWAYS a directory");  
    tracing::info!("   - Each export format handles its own filename logic");
    tracing::info!("   - No more confusion about file vs directory paths!");
    tracing::info!("");
    tracing::info!("Examples of what gets created:");
    tracing::info!("   Video:    ./output/my_edited_video.mp4");
    tracing::info!("   Poster:   ./output/thumbnail_15.png");
    tracing::info!("   Frames:   ./output/frames/scene_000001.png, scene_000002.png, ...");
    
    // Uncomment to test actual rendering:
    // editor.render_to_file(video_config)?;
    // editor.render_to_file(poster_config)?; 
    // editor.render_to_file(frames_config)?;
    
    Ok(())
}
