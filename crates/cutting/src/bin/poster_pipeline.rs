use std::path::{Path, PathBuf};
use std::collections::HashMap;
use glob::glob;
use gstreamer::{self as gst, prelude::*, ClockTime, element_error};
use gstreamer_video::{self as gst_video, prelude::*};
use gstreamer_app::{self as gst_app};
use color_eyre::eyre::{Result, Context, Error};
use tracing::{info, warn, error, debug};
use cutting::VideoEditError;
use image::GenericImageView;
use gstreamer::glib;

/// Video file extensions to search for
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v", "webm"];

struct PosterExtractor {
    file_stem: String,
    timestamp: u64, // seconds
    output_dir: PathBuf,
    quality: u8,
}

fn create_poster_pipeline(
    uri: String,
    extractor: PosterExtractor,
) -> Result<gst::Pipeline, Error> {
    gst::init()?;

    // Create our pipeline from a pipeline description string.
    let pipeline = gst::parse::launch(&format!(
        "uridecodebin uri={uri} ! videoconvert ! videoscale ! videorate ! imagefreeze ! appsink name=sink"
    ))?
    .downcast::<gst::Pipeline>()
    .expect("Expected a gst::Pipeline");

    // Get access to the appsink element.
    let appsink = pipeline
        .by_name("sink")
        .expect("Sink element not found")
        .downcast::<gst_app::AppSink>()
        .expect("Sink element is expected to be an appsink!");

    // Don't synchronize on the clock, we want to process as fast as possible.
    appsink.set_property("sync", false);

    // Tell the appsink what format we want.
    appsink.set_caps(Some(
        &gst_video::VideoCapsBuilder::new()
            .format(gst_video::VideoFormat::Rgbx)
            .build(),
    ));

    // Getting data out of the appsink is done by setting callbacks on it.
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                // Pull the sample in question out of the appsink's buffer.
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or_else(|| {
                    element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to get buffer from appsink")
                    );
                    gst::FlowError::Error
                })?;

                // Get the timestamp of this frame
                let timestamp = buffer.pts().unwrap_or(gst::ClockTime::ZERO);
                let current_time = timestamp.seconds() as f64;

                println!("Extracting poster at {:.2}s", current_time);

                let caps = sample.caps().expect("Sample without caps");
                let info = gst_video::VideoInfo::from_caps(caps).expect("Failed to parse caps");

                let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info)
                    .map_err(|_| {
                        element_error!(
                            appsink,
                            gst::ResourceError::Failed,
                            ("Failed to map buffer readable")
                        );
                        gst::FlowError::Error
                    })?;

                // Create a FlatSamples around the borrowed video frame data from GStreamer
                let img = image::FlatSamples::<&[u8]> {
                    samples: frame.plane_data(0).unwrap(),
                    layout: image::flat::SampleLayout {
                        channels: 3,       // RGB
                        channel_stride: 1, // 1 byte from component to component
                        width: frame.width(),
                        width_stride: 4, // 4 byte from pixel to pixel
                        height: frame.height(),
                        height_stride: frame.plane_stride()[0] as usize,
                    },
                    color_hint: Some(image::ColorType::Rgb8),
                };

                // Convert to ImageBuffer for saving
                let image_buffer = image::ImageBuffer::from_fn(
                    frame.width(),
                    frame.height(),
                    |x, y| {
                        let view = img.as_view::<image::Rgb<u8>>().unwrap();
                        view.get_pixel(x, y)
                    }
                );

                // Generate output filename
                let filename = format!("{}-poster.webp", extractor.file_stem);
                let output_path = extractor.output_dir.join(filename);

                // Save the poster as WebP
                image_buffer.save(&output_path).map_err(|err| {
                    element_error!(
                        appsink,
                        gst::ResourceError::Write,
                        (
                            "Failed to write poster file {}: {}",
                            output_path.display(),
                            err
                        )
                    );
                    gst::FlowError::Error
                })?;

                println!("Saved poster to {}", output_path.display());

                // We only want one frame, so return Eos
                Err(gst::FlowError::Eos)
            })
            .build(),
    );

    Ok(pipeline)
}

fn main_loop(pipeline: gst::Pipeline, timestamp: u64) -> Result<(), Error> {
    println!("Setting pipeline to PAUSED state...");
    pipeline.set_state(gst::State::Paused)?;

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    let mut seeked = false;
    
    // Wait for state change to complete with a timeout
    println!("Waiting for pipeline to reach PAUSED state...");
    let (state_result, current_state, pending_state) = pipeline.state(gst::ClockTime::from_seconds(10));
    match state_result {
        Ok(_) => {
            if current_state == gst::State::Paused {
                println!("Pipeline successfully reached PAUSED state");
            } else {
                println!("Pipeline state: current={:?}, pending={:?}", current_state, pending_state);
            }
        }
        Err(e) => {
            eprintln!("Failed to get pipeline state: {:?}", e);
            return Err(color_eyre::eyre::eyre!("Failed to set pipeline to PAUSED state"));
        }
    }

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::AsyncDone(..) => {
                if !seeked {
                    // AsyncDone means that the pipeline has started now and that we can seek
                    println!("Got AsyncDone message, seeking to {}s", timestamp);

                    if pipeline
                        .seek_simple(gst::SeekFlags::FLUSH, timestamp * gst::ClockTime::SECOND)
                        .is_err()
                    {
                        println!("Failed to seek, starting from beginning");
                    }

                    println!("Setting pipeline to PLAYING state...");
                    pipeline.set_state(gst::State::Playing)?;
                    seeked = true;
                } else {
                    println!("Got second AsyncDone message, seek finished");
                }
            }
            MessageView::Eos(..) => {
                println!("Got Eos message, poster extraction complete");
                break;
            }
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;
                return Err(ErrorMessage {
                    src: msg
                        .src()
                        .map(|s| s.path_string())
                        .unwrap_or_else(|| glib::GString::from("UNKNOWN")),
                    error: err.error(),
                    debug: err.debug(),
                }
                .into());
            }
            MessageView::Warning(warn) => {
                println!("Warning from {:?}: {} ({:?})", 
                        msg.src().map(|s| s.path_string()),
                        warn.error(), 
                        warn.debug());
            }
            MessageView::Info(info) => {
                println!("Info from {:?}: {} ({:?})", 
                        msg.src().map(|s| s.path_string()),
                        info.error(), 
                        info.debug());
            }
            _ => (),
        }
    }

    println!("Setting pipeline to NULL state...");
    pipeline.set_state(gst::State::Null)?;

    Ok(())
}

pub struct PosterPipeline {
    file_stem: String,
    input_uri: String,
    timestamp: u64,
    output_dir: PathBuf,
    quality: u8,
}

impl PosterPipeline {
    pub fn new(file_stem: String, input_path: String, timestamp: u64, output_dir: PathBuf, quality: u8) -> Self {
        let uri = if input_path.starts_with("http://") || input_path.starts_with("https://") || input_path.starts_with("file://") {
            input_path
        } else {
            // Convert to absolute path and add file:// prefix
            let absolute_path = std::fs::canonicalize(&input_path)
                .expect("Failed to get absolute path");
            format!("file://{}", absolute_path.display())
        };

        Self { file_stem, input_uri: uri, timestamp, output_dir, quality }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure output directory exists
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir)?;
        }

        let extractor = PosterExtractor {
            file_stem: self.file_stem.clone(),
            timestamp: self.timestamp,
            output_dir: self.output_dir.clone(),
            quality: self.quality,
        };

        create_poster_pipeline(self.input_uri.clone(), extractor)
            .and_then(|pipeline| main_loop(pipeline, self.timestamp))
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn std::error::Error>)
    }
}

/// Find all video files in a directory using glob patterns
fn find_video_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut video_files = Vec::new();
    
    for ext in VIDEO_EXTENSIONS {
        let pattern = directory.join(format!("*.{}", ext));
        let pattern_str = pattern.to_str()
            .ok_or_else(|| color_eyre::eyre::eyre!("Invalid path pattern"))?;
        
        match glob(pattern_str) {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(path) => {
                            if path.is_file() {
                                video_files.push(path);
                            }
                        }
                        Err(e) => warn!("Error accessing file: {}", e),
                    }
                }
            }
            Err(e) => warn!("Invalid glob pattern {}: {}", pattern_str, e),
        }
    }
    
    // Also try case-insensitive patterns
    for ext in VIDEO_EXTENSIONS {
        let pattern = directory.join(format!("*.{}", ext.to_uppercase()));
        let pattern_str = pattern.to_str()
            .ok_or_else(|| color_eyre::eyre::eyre!("Invalid path pattern"))?;
        
        match glob(pattern_str) {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(path) => {
                            if path.is_file() {
                                video_files.push(path);
                            }
                        }
                        Err(e) => warn!("Error accessing file: {}", e),
                    }
                }
            }
            Err(e) => warn!("Invalid glob pattern {}: {}", pattern_str, e),
        }
    }
    
    Ok(video_files)
}

/// Generate posters for all videos in a directory
fn generate_posters_for_directory(
    input_dir: &Path,
    output_dir: &Path,
    timestamp: u64,
    quality: u8,
) -> Result<HashMap<PathBuf, PathBuf>> {
    info!("Scanning directory for video files: {}", input_dir.display());
    
    let video_files = find_video_files(input_dir)?;
    info!("Found {} video files", video_files.len());
    
    let mut results = HashMap::new();
    let mut success_count = 0;
    let mut error_count = 0;
    
    for video_path in video_files {
        info!("Processing video: {}", video_path.display());
        
        let file_stem = video_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        let poster_pipeline = PosterPipeline::new(
            file_stem.to_string(),
            video_path.to_string_lossy().to_string(),
            timestamp,
            output_dir.to_path_buf(),
            quality,
        );

        match poster_pipeline.run() {
            Ok(_) => {
                let poster_path = output_dir.join(format!("{}-poster.webp", file_stem));
                results.insert(video_path.clone(), poster_path);
                success_count += 1;
                info!("Successfully generated poster for {}", video_path.display());
            }
            Err(e) => {
                error!("Failed to generate poster for {}: {}", video_path.display(), e);
                error_count += 1;
            }
        }
    }
    
    info!("Poster generation complete: {} successful, {} failed", success_count, error_count);
    Ok(results)
}

#[derive(Debug, derive_more::Display, derive_more::Error)]
#[display("Received error from {src}: {error} (debug: {debug:?})")]
struct ErrorMessage {
    src: glib::GString,
    error: glib::Error,
    debug: Option<glib::GString>,
}

fn main() -> Result<()> {
    // Initialize color_eyre for better error reporting
    color_eyre::install()?;
    
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Initialize GStreamer
    gst::init().context("Failed to initialize GStreamer")?;
    
    
    let input_dir = Path::new("/Users/hectorcrean/rust/video-kit/crates/cutting/output");
   
    
    let output_dir = Path::new("/Users/hectorcrean/rust/video-kit/crates/cutting/output");
    
    let timestamp_seconds = 0;
    
    let quality = 85;
    
    info!("Starting poster generation pipeline");
    info!("Input directory: {}", input_dir.display());
    info!("Output directory: {}", output_dir.display());
    info!("Timestamp: {} seconds", timestamp_seconds);
    info!("Quality: {}", quality);
    
    // Generate posters for all videos
    let results = generate_posters_for_directory(input_dir, output_dir, timestamp_seconds, quality)?;
    
    // Print summary
    println!("\nPoster generation summary:");
    println!("==========================");
    for (video_path, poster_path) in &results {
        println!("✓ {} -> {}", 
                 video_path.file_name().unwrap().to_str().unwrap(),
                 poster_path.file_name().unwrap().to_str().unwrap());
    }
    
    if results.is_empty() {
        println!("No video files found in {}", input_dir.display());
    } else {
        println!("\nGenerated {} posters successfully", results.len());
    }
    
    Ok(())
}
