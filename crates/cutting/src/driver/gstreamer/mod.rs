use crate::driver::{Driver, DriverError, StreamOperation};
use crate::{Sequentise, sources::{Source, Sink}};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::ops::Range;
use std::path::Path;

/// GStreamer-specific file source
#[derive(Debug, Clone)]
pub struct GStreamerFileSource {
    pub path: String,
}

impl GStreamerFileSource {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Source for GStreamerFileSource {
    fn validate(&self) -> Result<(), DriverError> {
        if Path::new(&self.path).exists() {
            Ok(())
        } else {
            Err(DriverError::Execution(format!("Input file not found: {}", self.path)))
        }
    }
    
    fn description(&self) -> String {
        format!("GStreamer File Source: {}", self.path)
    }
}

/// GStreamer-specific file sink
#[derive(Debug, Clone)]
pub struct GStreamerFileSink {
    pub path: String,
}

impl GStreamerFileSink {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Sink for GStreamerFileSink {
    fn validate(&self) -> Result<(), DriverError> {
        if let Some(parent) = Path::new(&self.path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        }
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("GStreamer File Sink: {}", self.path)
    }
}

/// GStreamer-specific directory sink for frame extraction
#[derive(Debug, Clone)]
pub struct GStreamerDirectorySink {
    pub path: String,
    pub pattern: String,
}

impl GStreamerDirectorySink {
    pub fn new(path: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pattern: pattern.into(),
        }
    }
}

impl Sink for GStreamerDirectorySink {
    fn validate(&self) -> Result<(), DriverError> {
        std::fs::create_dir_all(&self.path)
            .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("GStreamer Directory Sink: {} (pattern: {})", self.path, self.pattern)
    }
}

/// Internal representation of a GStreamer pipeline/stream
/// This represents the operations applied but not yet executed
#[derive(Debug, Clone)]
pub struct GStreamerStream {
    pub operations: Vec<StreamOperation>,
}


impl GStreamerStream {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
    
    fn with_operation(mut self, op: StreamOperation) -> Self {
        self.operations.push(op);
        self
    }
}

/// GStreamer driver implementation
pub struct GStreamerDriver {
    _main_loop: gst::glib::MainLoop,
}

impl GStreamerDriver {
    pub fn new() -> Result<Self, DriverError> {
        gst::init().map_err(|e| DriverError::Initialization(format!("GStreamer init failed: {}", e)))?;
        
        let main_loop = gst::glib::MainLoop::new(None, false);
        
        Ok(Self {
            _main_loop: main_loop,
        })
    }

    /// Create a new driver and verify all required plugins are available
    pub fn new_with_plugin_check() -> Result<Self, DriverError> {
        let driver = Self::new()?;
        Self::check_plugins()?;
        Ok(driver)
    }

    /// Check if GStreamer is available and working
    pub fn is_available() -> bool {
        gst::init().is_ok()
    }

    /// Check if required GStreamer plugins are available
    pub fn check_plugins() -> Result<(), DriverError> {
        let required_plugins = [
            "coreelements",       // filesrc, filesink
            "libav",              // decoding/encoding 
            "videoconvertscale",  // videoconvert for format conversion
            "x264",               // H.264 encoding
            "isomp4",             // MP4 muxing
        ];

        for plugin_name in &required_plugins {
            if gst::Registry::get().find_plugin(plugin_name).is_none() {
                return Err(DriverError::Initialization(
                    format!("Required GStreamer plugin '{}' not found", plugin_name)
                ));
            }
        }

        Ok(())
    }

    fn create_pipeline(&self, name: &str) -> Result<gst::Pipeline, DriverError> {
        Ok(gst::Pipeline::with_name(name))
    }

    fn execute_pipeline(&self, stream: &GStreamerStream, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        let pipeline = self.create_pipeline("video_processing")?;
        
        // Get the source path from the load operation
        let source_path = stream.operations.iter()
            .find_map(|op| match op {
                StreamOperation::Load { source_path } => Some(source_path.clone()),
                _ => None,
            })
            .ok_or_else(|| DriverError::Execution("No source path found".to_string()))?;

        // Build pipeline based on operations
        let has_splice = stream.operations.iter().any(|op| matches!(op, StreamOperation::Splice { .. }));
        let has_extract_frames = stream.operations.iter().any(|op| matches!(op, StreamOperation::ExtractFrames { .. }));
        let has_reverse = stream.operations.iter().any(|op| matches!(op, StreamOperation::Reverse));
        let has_create_loop = stream.operations.iter().any(|op| matches!(op, StreamOperation::CreateLoop));

        if has_extract_frames {
            self.build_frame_extraction_pipeline(&pipeline, &source_path, stream, sink)?;
        } else if has_create_loop {
            self.build_loop_pipeline(&pipeline, &source_path, stream, sink)?;
        } else if has_reverse {
            self.build_reverse_pipeline(&pipeline, &source_path, stream, sink)?;
        } else if has_splice {
            self.build_splice_pipeline(&pipeline, &source_path, stream, sink)?;
        } else {
            self.build_simple_copy_pipeline(&pipeline, &source_path, sink)?;
        }

        self.wait_for_completion(&pipeline)?;
        Ok(())
    }

    fn build_simple_copy_pipeline(&self, pipeline: &gst::Pipeline, source_path: &str, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        // Use a much simpler approach - just basic video transcoding
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", source_path)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create filesrc: {}", e)))?;

        let decodebin = gst::ElementFactory::make("decodebin")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create decodebin: {}", e)))?;

        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create videoconvert: {}", e)))?;

        let x264enc = gst::ElementFactory::make("x264enc")
            .property("speed-preset", 6i32) // Fast preset
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create x264enc: {}", e)))?;

        let mp4mux = gst::ElementFactory::make("mp4mux")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create mp4mux: {}", e)))?;

        let filesink = gst::ElementFactory::make("filesink")
            .property("location", &sink.path)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create filesink: {}", e)))?;

        // Add to pipeline
        pipeline.add_many([&filesrc, &decodebin, &videoconvert, &x264enc, &mp4mux, &filesink])
            .map_err(|e| DriverError::Execution(format!("Failed to add elements: {}", e)))?;

        // Link what we can statically
        filesrc.link(&decodebin)
            .map_err(|e| DriverError::Execution(format!("Failed to link filesrc->decodebin: {}", e)))?;
        videoconvert.link(&x264enc)
            .map_err(|e| DriverError::Execution(format!("Failed to link videoconvert->x264enc: {}", e)))?;
        x264enc.link(&mp4mux)
            .map_err(|e| DriverError::Execution(format!("Failed to link x264enc->mp4mux: {}", e)))?;
        mp4mux.link(&filesink)
            .map_err(|e| DriverError::Execution(format!("Failed to link mp4mux->filesink: {}", e)))?;

        // Handle dynamic pad linking (simplified)
        let videoconvert_weak = videoconvert.downgrade();
        decodebin.connect_pad_added(move |_, src_pad| {
            if let Some(caps) = src_pad.current_caps() {
                if let Some(structure) = caps.structure(0) {
                    if structure.name().starts_with("video/") {
                        if let Some(videoconvert) = videoconvert_weak.upgrade() {
                            let sink_pad = videoconvert.static_pad("sink").unwrap();
                            let _ = src_pad.link(&sink_pad);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    fn build_splice_pipeline(&self, pipeline: &gst::Pipeline, source_path: &str, stream: &GStreamerStream, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        // Get splice segments
        let segments = stream.operations.iter()
            .find_map(|op| match op {
                StreamOperation::Splice { segments } => Some(segments.clone()),
                _ => None,
            })
            .unwrap_or_default();

        if segments.is_empty() {
            return self.build_simple_copy_pipeline(pipeline, source_path, sink);
        }

        // For simplicity, handle the first segment (can be extended for multiple segments)
        let first_segment = &segments[0];

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", source_path)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create filesrc: {}", e)))?;

        let decodebin = gst::ElementFactory::make("decodebin")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create decodebin: {}", e)))?;

        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create videoconvert: {}", e)))?;

        let encoder = gst::ElementFactory::make("x264enc")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create x264enc: {}", e)))?;

        let muxer = gst::ElementFactory::make("mp4mux")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create mp4mux: {}", e)))?;

        let filesink = gst::ElementFactory::make("filesink")
            .property("location", &sink.path)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create filesink: {}", e)))?;

        // Add elements to pipeline
        pipeline.add_many([&filesrc, &decodebin, &videoconvert, &encoder, &muxer, &filesink])
            .map_err(|e| DriverError::Execution(format!("Failed to add elements: {}", e)))?;

        // Link elements
        filesrc.link(&decodebin)
            .map_err(|e| DriverError::Execution(format!("Failed to link: {}", e)))?;

        // Set up seek for the segment (currently unused - would be used in proper seeking implementation)
        let _start_time = gst::ClockTime::from_seconds(first_segment.start as u64);
        let _duration = gst::ClockTime::from_seconds((first_segment.end - first_segment.start) as u64);

        // Handle dynamic linking when pads become available
        let videoconvert_weak = videoconvert.downgrade();
        let encoder_weak = encoder.downgrade();
        let muxer_weak = muxer.downgrade();
        let filesink_weak = filesink.downgrade();

        decodebin.connect_pad_added(move |_, src_pad| {
            let pad_caps = src_pad.current_caps().unwrap();
            let structure = pad_caps.structure(0).unwrap();
            let name = structure.name();

            if name.starts_with("video/") {
                if let (Some(videoconvert), Some(encoder), Some(muxer), Some(filesink)) = 
                    (videoconvert_weak.upgrade(), encoder_weak.upgrade(), muxer_weak.upgrade(), filesink_weak.upgrade()) {
                    let sink_pad = videoconvert.static_pad("sink").unwrap();
                    if src_pad.link(&sink_pad).is_ok() {
                        let _ = videoconvert.link(&encoder);
                        let _ = encoder.link(&muxer);
                        let _ = muxer.link(&filesink);
                    }
                }
            }
        });

        // Note: For proper seeking with segments, we would need to handle this in wait_for_completion
        // or use a different approach with GStreamer's segment handling

        Ok(())
    }

    fn build_frame_extraction_pipeline(&self, pipeline: &gst::Pipeline, source_path: &str, stream: &GStreamerStream, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        // Get frame extraction config
        let (start_time, end_time, config) = stream.operations.iter()
            .find_map(|op| match op {
                StreamOperation::ExtractFrames { start_time, end_time, config } => 
                    Some((*start_time, *end_time, config.clone())),
                _ => None,
            })
            .ok_or_else(|| DriverError::Execution("No frame extraction config found".to_string()))?;

        let fps = config.as_ref().and_then(|c| c.fps).unwrap_or(1.0);
        let format = config.as_ref().and_then(|c| c.format.as_ref()).unwrap_or(&"png".to_string()).clone();

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", source_path)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create filesrc: {}", e)))?;

        let decodebin = gst::ElementFactory::make("decodebin")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create decodebin: {}", e)))?;

        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create videoconvert: {}", e)))?;

        let videorate = gst::ElementFactory::make("videorate")
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create videorate: {}", e)))?;

        let caps_filter = gst::ElementFactory::make("capsfilter")
            .property("caps", gst::Caps::builder("video/x-raw")
                .field("framerate", gst::Fraction::new(fps as i32, 1))
                .build())
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create capsfilter: {}", e)))?;

        let encoder = match format.as_str() {
            "png" => gst::ElementFactory::make("pngenc").build(),
            "jpg" | "jpeg" => gst::ElementFactory::make("jpegenc").build(),
            _ => gst::ElementFactory::make("pngenc").build(),
        }.map_err(|e| DriverError::Execution(format!("Failed to create image encoder: {}", e)))?;

        // Use directory sink pattern for frame extraction
        let output_dir = std::path::Path::new(&sink.path).parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_string_lossy();
        let filename_pattern = format!("{}/frame_%05d.{}", output_dir, format);

        let multifilesink = gst::ElementFactory::make("multifilesink")
            .property("location", filename_pattern)
            .build()
            .map_err(|e| DriverError::Execution(format!("Failed to create multifilesink: {}", e)))?;

        // Add elements to pipeline
        pipeline.add_many([&filesrc, &decodebin, &videoconvert, &videorate, &caps_filter, &encoder, &multifilesink])
            .map_err(|e| DriverError::Execution(format!("Failed to add elements: {}", e)))?;

        // Link static elements
        filesrc.link(&decodebin)
            .map_err(|e| DriverError::Execution(format!("Failed to link: {}", e)))?;

        videoconvert.link(&videorate)
            .and_then(|_| videorate.link(&caps_filter))
            .and_then(|_| caps_filter.link(&encoder))
            .and_then(|_| encoder.link(&multifilesink))
            .map_err(|e| DriverError::Execution(format!("Failed to link video pipeline: {}", e)))?;

        // Handle dynamic linking
        let videoconvert_weak = videoconvert.downgrade();
        decodebin.connect_pad_added(move |_, src_pad| {
            let pad_caps = src_pad.current_caps().unwrap();
            let structure = pad_caps.structure(0).unwrap();
            let name = structure.name();

            if name.starts_with("video/") {
                if let Some(videoconvert) = videoconvert_weak.upgrade() {
                    let sink_pad = videoconvert.static_pad("sink").unwrap();
                    let _ = src_pad.link(&sink_pad);
                }
            }
        });

        Ok(())
    }

    fn build_reverse_pipeline(&self, pipeline: &gst::Pipeline, source_path: &str, _stream: &GStreamerStream, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        // Note: True video reversal in GStreamer is complex and may require custom elements
        // This is a placeholder implementation
        println!("Warning: Video reversal not fully implemented in GStreamer driver");
        self.build_simple_copy_pipeline(pipeline, source_path, sink)
    }

    fn build_loop_pipeline(&self, pipeline: &gst::Pipeline, source_path: &str, _stream: &GStreamerStream, sink: &GStreamerFileSink) -> Result<(), DriverError> {
        // Note: Loop creation requires complex pipeline manipulation
        // This is a placeholder implementation
        println!("Warning: Loop creation not fully implemented in GStreamer driver");
        self.build_simple_copy_pipeline(pipeline, source_path, sink)
    }


    fn wait_for_completion(&self, pipeline: &gst::Pipeline) -> Result<(), DriverError> {
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| DriverError::Execution(format!("Failed to start pipeline: {}", e)))?;

        let bus = pipeline.bus().unwrap();
        
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            use gst::MessageView;
            
            match msg.view() {
                MessageView::Eos(..) => {
                    println!("Pipeline completed successfully");
                    break;
                }
                MessageView::Error(err) => {
                    let error_msg = format!(
                        "Pipeline error: {} (debug: {:?})",
                        err.error(),
                        err.debug()
                    );
                    pipeline.set_state(gst::State::Null).ok();
                    return Err(DriverError::Execution(error_msg));
                }
                _ => {}
            }
        }

        pipeline
            .set_state(gst::State::Null)
            .map_err(|e| DriverError::Execution(format!("Failed to stop pipeline: {}", e)))?;

        Ok(())
    }
}

impl Driver for GStreamerDriver {
    type Source = GStreamerFileSource;
    type Sink = GStreamerFileSink;
    type Stream = GStreamerStream;

    fn load(&self, source: &Self::Source) -> Result<Self::Stream, DriverError> {
        source.validate()?;
        Ok(GStreamerStream::new().with_operation(StreamOperation::Load {
            source_path: source.path.clone(),
        }))
    }
    
    fn splice(&self, stream: Self::Stream, segments: &[Range<f64>]) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::Splice {
            segments: segments.to_vec(),
        }))
    }
    
    fn extract_frames(&self, stream: Self::Stream, sequentise: &Sequentise) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::ExtractFrames { 
            start_time: sequentise.period.start, 
            end_time: sequentise.period.end, 
            config: Some(sequentise.clone()) 
        }))
    }
    
    fn reverse(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::Reverse))
    }
    
    fn create_loop(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::CreateLoop))
    }
    
    fn save(&self, stream: Self::Stream, sink: &Self::Sink) -> Result<(), DriverError> {
        sink.validate()?;
        
        // Check if this is a splice operation to generate poster
        let is_splice_operation = stream.operations.iter()
            .any(|op| matches!(op, StreamOperation::Splice { .. }));
        
        if is_splice_operation {
            // Execute the main pipeline
            self.execute_pipeline(&stream, sink)?;
            
            // Generate poster for splice operations
            // Note: GStreamer poster generation would require a separate pipeline
            // For now, we'll just print the expected poster path
            let poster_path = if sink.path.ends_with(".mp4") {
                sink.path.replace(".mp4", "_poster.png")
            } else if sink.path.ends_with(".mov") {
                sink.path.replace(".mov", "_poster.png")
            } else if sink.path.ends_with(".avi") {
                sink.path.replace(".avi", "_poster.png")
            } else {
                format!("{}_poster.png", sink.path)
            };
            
            println!("✅ Generated video: {}", sink.path);
            println!("🖼️  Poster generation for GStreamer not yet implemented: {}", poster_path);
            println!("   (Use FFmpeg driver for poster generation)");
        } else {
            self.execute_pipeline(&stream, sink)?;
        }
        
        Ok(())
    }
}



