// This example demonstrates how to extract a sequence of video frames as PNGs
// from a video file within a specified time range:

// {uridecodebin} - {videoconvert} - {appsink}

// The appsink enforces RGBx so that the image crate can use it. The sample layout is passed
// with the correct stride from GStreamer to the image crate as GStreamer does not necessarily
// produce tightly packed pixels, and in case of RGBx never.

use color_eyre::eyre::Error;
use derive_more::derive::{Display, Error};
use gstreamer::{self as gst,element_error, prelude::*};
use gstreamer_video::{self as gst_video, prelude::*};
use std::sync::{Arc, Mutex};
use gstreamer_video::glib;
use image::GenericImageView;
use gstreamer_app::{self as gst_app};
use gstreamer::glib::translate::IntoGlib;
use glob::glob;
use std::env;

#[derive(Debug, Display, Error)]
#[display("Received error from {src}: {error} (debug: {debug:?})")]
struct ErrorMessage {
    src: glib::GString,
    error: glib::Error,
    debug: Option<glib::GString>,
}

struct FrameExtractor {
    file_stem: String,
    start_time: u64,
    end_time: Option<u64>, // None means extract until end of video
    frame_interval: f64, // seconds between frames
    output_dir: std::path::PathBuf,
    frame_count: Arc<Mutex<u32>>,
    last_frame_time: Arc<Mutex<f64>>,
}

fn create_pipeline(
    uri: String,
    extractor: Arc<FrameExtractor>,
) -> Result<gst::Pipeline, Error> {
    gst::init()?;

    // Create our pipeline from a pipeline description string.
    let pipeline = gst::parse::launch(&format!(
        "uridecodebin uri={uri} ! videoconvert ! appsink name=sink"
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
                let current_time = timestamp.into_glib() as f64 / 1_000_000_000.0;

                // Check if we're past the end time (if specified)
                if let Some(end_time) = extractor.end_time {
                    if current_time > end_time as f64 {
                        println!("Reached end time, stopping extraction");
                        return Err(gst::FlowError::Eos);
                    }
                }

                // Check if we should extract this frame based on timing
                let mut last_time = extractor.last_frame_time.lock().unwrap();
                if current_time - *last_time >= extractor.frame_interval {
                    *last_time = current_time;
                    
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

                    // Get frame number for filename
                    let mut count = extractor.frame_count.lock().unwrap();
                    *count += 1;
                    let frame_num = *count;

                    println!("Extracting frame {} at {:.2}s", frame_num, current_time);

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
                    let filename = format!("{}-{:06}.png", extractor.file_stem, frame_num);
                    let output_path = extractor.output_dir.join(filename);

                    // Save the frame
                    image_buffer.save(&output_path).map_err(|err| {
                        element_error!(
                            appsink,
                            gst::ResourceError::Write,
                            (
                                "Failed to write frame file {}: {}",
                                output_path.display(),
                                err
                            )
                        );
                        gst::FlowError::Error
                    })?;

                    println!("Saved frame to {}", output_path.display());
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

fn main_loop(pipeline: gst::Pipeline, start_time: u64) -> Result<(), Error> {
    println!("Setting pipeline to PAUSED state...");
    pipeline.set_state(gst::State::Paused)?;

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    let mut seeked = false;
    
    // Wait for state change to complete with a timeout
    println!("Waiting for pipeline to reach PAUSED state...");
    match pipeline.state(Some(gst::ClockTime::from_seconds(10))) {
        (Ok(_), gst::State::Paused, _) => {
            println!("Pipeline successfully reached PAUSED state");
        }
        (Ok(_), current_state, pending_state) => {
            println!("Pipeline state: current={:?}, pending={:?}", current_state, pending_state);
        }
        (Err(e), current_state, pending_state) => {
            eprintln!("Failed to get pipeline state: {:?}, current={:?}, pending={:?}", e, current_state, pending_state);
            return Err(color_eyre::eyre::eyre!("Failed to set pipeline to PAUSED state"));
        }
    }

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::AsyncDone(..) => {
                if !seeked {
                    // AsyncDone means that the pipeline has started now and that we can seek
                    println!("Got AsyncDone message, seeking to {}s", start_time);

                    if pipeline
                        .seek_simple(gst::SeekFlags::FLUSH, start_time * gst::ClockTime::SECOND)
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
                println!("Got Eos message, extraction complete");
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




pub struct SequencePipeline {
    file_stem: String,
    input_uri: String,
    start_time: u64,
    end_time: Option<u64>,
    frame_interval: f64,
    output_dir: std::path::PathBuf,
}

impl SequencePipeline {
    pub fn new(file_stem: String, input_path: String, start_time: Option<u64>, end_time: Option<u64>, frame_interval: f64, output_dir: std::path::PathBuf) -> Self {


        let uri = if input_path.starts_with("http://") || input_path.starts_with("https://") || input_path.starts_with("file://") {
            input_path
        } else {
            // Convert to absolute path and add file:// prefix
            let absolute_path = std::fs::canonicalize(&input_path)
                .expect("Failed to get absolute path");
            format!("file://{}", absolute_path.display())
        };


        Self { file_stem, input_uri: uri, start_time: start_time.unwrap_or(0), end_time, frame_interval, output_dir }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure output directory exists
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir)?;
        }

        let extractor = Arc::new(FrameExtractor {
            file_stem: self.file_stem.clone(),
            start_time: self.start_time,
            end_time: self.end_time,
            frame_interval: self.frame_interval,
            output_dir: self.output_dir.clone(),
            frame_count: Arc::new(Mutex::new(0)),
            last_frame_time: Arc::new(Mutex::new(self.start_time as f64)),
        });

        match create_pipeline(self.input_uri.clone(), extractor).and_then(|pipeline| main_loop(pipeline, self.start_time)) {
            Ok(_) => {
                println!("Frame extraction completed successfully!");
            },
            Err(e) => eprintln!("Error! {e}"),
        }

        Ok(())
    }
}



fn main() -> Result<(), Box<dyn std::error::Error>> {

    let search_dir = "/Users/hectorcrean/rust/video-kit/crates/cutting/output";

    // Create glob pattern to find MP4 files with "sequence" in their name
    let pattern = format!("{}/**/*sequence*.mp4", search_dir);
    println!("Searching for files matching pattern: {}", pattern);

    let mut processed_files = 0;
    let mut failed_files = 0;

    // Find all matching files
    for entry in glob(&pattern)? {
        match entry {
            Ok(path) => {
                println!("\n=== Processing: {} ===", path.display());
                
                // Create output directory for this video file
                let file_stem = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                
                let output_dir = std::path::PathBuf::from(search_dir);
                
                // Create output directory if it doesn't exist
                if !output_dir.exists() {
                    std::fs::create_dir_all(&output_dir)?;
                    println!("Created output directory: {}", output_dir.display());
                }

                // Process this video file
                let sequence_pipeline = SequencePipeline::new(
                    file_stem.to_string(),
                    path.to_string_lossy().to_string(),
                    None,
                    None,
                    0.05,
                    output_dir,
                );

                match sequence_pipeline.run() {
                    Ok(_) => {
                        println!("✓ Successfully processed: {}", path.display());
                        processed_files += 1;
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to process {}: {}", path.display(), e);
                        failed_files += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading path: {}", e);
                failed_files += 1;
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Files processed successfully: {}", processed_files);
    println!("Files failed: {}", failed_files);
    
    if processed_files == 0 {
        println!("No files found matching pattern: {}", pattern);
        println!("Make sure:");
        println!("  1. The directory exists");
        println!("  2. There are MP4 files with 'sequence' in their filename");
        println!("  3. The files are accessible");
    }

    Ok(())
}
