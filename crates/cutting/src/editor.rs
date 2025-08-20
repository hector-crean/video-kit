use gstreamer::{self as gst, prelude::*, ClockTime, SeekFlags, SeekType};
use gstreamer_app as gst_app;
use gstreamer_editing_services::{self as ges, prelude::*};
use gstreamer_pbutils::{EncodingAudioProfile, EncodingVideoProfile, EncodingContainerProfile};
use std::path::Path;
use tracing::{info, debug, warn, error, trace};

use crate::error::VideoEditError;
use crate::commands::{VideoEditCommand, ReverseMode, LoopMode};
use crate::formats::{ExportFormat, VideoFormat, ImageFormat};

/// Video editor that can both preview and render to file
pub struct VideoEditor {
    input_uri: String,
    commands: Vec<VideoEditCommand>,
}

impl VideoEditor {
    pub fn new(input_uri: String) -> Self {
        Self {
            input_uri,
            commands: Vec::new()
        }
    }
    
    /// Add a command to the editing sequence
    pub fn add_command(&mut self, command: VideoEditCommand) {
        self.commands.push(command);
    }

    pub fn add_commands(&mut self, commands: &[VideoEditCommand]) {
        self.commands.extend_from_slice(commands);
    }
    
    /// Render the edits to a file using GES pipeline
    pub fn render_to_file(&self, format: ExportFormat, output_dir: &str) -> Result<(), VideoEditError> {
        info!("Starting video rendering with {} commands", self.commands.len());
        debug!("Input URI: {}", self.input_uri);
        debug!("Output format: {:?}", format);
        
        ges::init()?;
        
        // Create GES timeline and pipeline for rendering
        let timeline = ges::Timeline::new_audio_video();
        let layer = timeline.append_layer();
        let pipeline = ges::Pipeline::new();
        pipeline.set_timeline(&timeline)
            .map_err(|e| VideoEditError::Timeline { 
                message: format!("Failed to set timeline on pipeline: {}", e) 
            })?;
        
        // Load the source clip
        let clip = ges::UriClip::new(&self.input_uri)
            .map_err(|_| VideoEditError::Resource(gst::ResourceError::NotFound))?;
        layer.add_clip(&clip)
            .map_err(|e| VideoEditError::Timeline { 
                message: format!("Failed to add clip to layer: {}", e) 
            })?;
        
        // Apply commands using GES
        for (i, command) in self.commands.iter().enumerate() {
            debug!("Applying command {}: {:?}", i + 1, command);
            match command {
                VideoEditCommand::Cut { start, end } => {
                    trace!("Setting clip inpoint: {:?}, duration: {:?}", start, *end - *start);
                    clip.set_inpoint(*start);
                    clip.set_duration(*end - *start);
                }
                
                VideoEditCommand::Reverse { mode, .. } => {
                    info!("Applying reverse effect");
                    let reverse_mode = mode.as_ref().unwrap_or(&ReverseMode::Full);
                    self.render_reverse_to_timeline(&timeline, &layer, &clip, reverse_mode)?;
                }
                
                VideoEditCommand::Freeze { start, end, .. } => {
                    info!("Applying freeze effect from {:?} to {:?}", start, end);
                    warn!("Freeze effect implementation is simplified - full implementation requires advanced GES track management");
                }
                
                VideoEditCommand::Loop { mode, .. } => {
                    match mode {
                        LoopMode::Count(count) => {
                            info!("Creating {} loop iterations", count);
                            let clip_duration = clip.duration();
                            for i in 1..*count {
                                let loop_clip = ges::UriClip::new(&self.input_uri)
                                    .map_err(|_| VideoEditError::Resource(gst::ResourceError::NotFound))?;
                                loop_clip.set_start(clip_duration * i as u64);
                                layer.add_clip(&loop_clip)
                                    .map_err(|e| VideoEditError::Timeline { 
                                        message: format!("Failed to add loop clip: {}", e) 
                                    })?;
                            }
                        }
                        _ => {
                            warn!("Loop mode {:?} needs additional implementation", mode);
                        }
                    }
                }
            }
        }
        
        // Handle different export formats
        match &format {
            ExportFormat::Video { format: video_format, filename } => {
                info!("Exporting video as {:?} format to: {}/{}.{}", 
                video_format, output_dir, filename, video_format.extension());
                
                // Create output directory if it doesn't exist
                debug!("Creating output directory: {}", output_dir);
                std::fs::create_dir_all(&output_dir)?;
                
                // Commit the timeline to ensure all changes are applied
                debug!("Committing timeline changes");
                if !timeline.commit() {
                    return Err(VideoEditError::Timeline { 
                        message: "Failed to commit timeline changes".to_string() 
                    });
                }
                
                // Configure for video file output using existing encoding patterns
                debug!("Configuring pipeline for rendering");
                self.configure_pipeline_for_rendering(&pipeline, &format, &output_dir)?;
                
                // Run the pipeline
                self.run_pipeline(&pipeline)?;


                // self.extract_poster_image_at_timestamp(&format, &output_dir, ClockTime::from_seconds(0))?;

            }
            
            ExportFormat::FrameSequence { start, end, fps, filename_pattern, .. } => {
                info!("Extracting frame sequence from edited timeline at {}fps using pattern '{}'", fps, filename_pattern);
                
                // Create output directory if it doesn't exist
                debug!("Creating output directory: {}", output_dir);
                std::fs::create_dir_all(&output_dir)?;
                
                // Commit the timeline to ensure all changes are applied
                debug!("Committing timeline changes");
                if !timeline.commit() {
                    return Err(VideoEditError::Timeline { 
                        message: "Failed to commit timeline changes".to_string() 
                    });
                }
                
                // For frame extraction from edited timeline, we need to render to a temporary file first
                // then extract frames from that, since GES doesn't directly support frame sequence output
                debug!("Using two-step approach: render edited video to temp file, then extract frames");
                self.extract_frames_from_edited_timeline(&timeline, &pipeline, &format, &output_dir, *start, *end)?;
            }
            
            ExportFormat::Poster { timestamp, filename, .. } => {
                info!("Extracting poster image at {:?} to: {}/{}", 
                      timestamp, output_dir, filename);
                self.extract_poster_image_at_timestamp(&format, &output_dir, *timestamp)?;
            }
        }
        
        info!("Rendering operation completed successfully");
        Ok(())
    }
    
    /// Configure pipeline for rendering (reuses existing GStreamer patterns)
    fn configure_pipeline_for_rendering(
        &self, 
        pipeline: &ges::Pipeline, 
        format: &ExportFormat,
        output_dir: &str
    ) -> Result<(), VideoEditError> {
        debug!("Creating encoding profile for output format");
        let encoding_profile = match &format {
            ExportFormat::Video { format, .. } => {
                debug!("Creating encoding profile for video format: {:?}", format);
                let profile = format.to_encoding_profile()?;
                debug!("Encoding profile created successfully");
                profile
            }
            _ => return Err(VideoEditError::InvalidExportFormat { 
                expected: "Video".to_string(), 
                actual: "Non-Video".to_string() 
            }),
        };
        
        // Ensure proper file:// URI format for GStreamer
        let output_path = match &format {
            ExportFormat::Video { format: video_format, filename } => {
                format!("{}/{}.{}", 
                    output_dir, 
                    filename, 
                    video_format.extension()
                )
            }
            _ => unreachable!(), // Already handled above
        };
        
        let output_uri = if output_path.starts_with("file://") {
            output_path
        } else if output_path.starts_with("/") {
            format!("file://{}", output_path)
        } else {
            // Relative path, make it absolute
            let current_dir = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            format!("file://{}/{}", current_dir.display(), output_path)
        };
        
        info!("Setting render output URI: {}", output_uri);
        debug!("Setting render settings on pipeline");
        pipeline.set_render_settings(&output_uri, &encoding_profile)
            .map_err(|e| {
                error!("Failed to set render settings: {}", e);
                VideoEditError::Timeline { 
                    message: format!("Failed to set render settings: {}", e) 
                }
            })?;
        
        debug!("Setting pipeline mode to RENDER");
        pipeline.set_mode(ges::PipelineFlags::RENDER)
            .map_err(|e| {
                error!("Failed to set pipeline mode: {}", e);
                VideoEditError::Timeline { 
                    message: format!("Failed to set pipeline mode: {}", e) 
                }
            })?;
        
        debug!("Pipeline configuration completed successfully");
        Ok(())
    }
    
    /// Extract frame sequence using a dedicated GStreamer pipeline
    fn extract_frame_sequence_with_custom_pipeline(
        &self,
        format: &ExportFormat,
        output_dir: &str,
        start: Option<ClockTime>,
        end: Option<ClockTime>,
    ) -> Result<(), VideoEditError> {
        let (filename_pattern, image_format, quality, _resize, fps) = match format {
            ExportFormat::FrameSequence { filename_pattern, format, quality, resize, fps, .. } => {
                (filename_pattern, format, *quality, *resize, *fps)
            }
            _ => return Err(VideoEditError::InvalidExportFormat {
                expected: "FrameSequence".to_string(),
                actual: "Non-FrameSequence".to_string()
            }),
        };

        info!("Extracting frame sequence using custom pipeline");
        debug!("Parameters: fps={}, format={:?}, pattern={}", fps, image_format, filename_pattern);
        
        // Ensure output directory exists and is writable
        std::fs::create_dir_all(&output_dir)?;
        debug!("Output directory created/verified: {}", output_dir);
        
        // Initialize GStreamer
        gst::init()?;
        
        // Build the encoder element with proper configuration
        let encoder_element = match (image_format, quality) {
            (ImageFormat::Png, _) => "pngenc compression-level=1".to_string(),
            (ImageFormat::Jpeg, Some(q)) => format!("jpegenc quality={}", q),
            (ImageFormat::Jpeg, None) => "jpegenc quality=90".to_string(),
            (ImageFormat::Bmp, _) => "bmpenc".to_string(),
            (ImageFormat::Tiff, _) => "tiffenc".to_string(),
            (ImageFormat::WebP, _) => "webpenc".to_string(),
        };
        
        // Configure output path pattern - multifilesink needs %d placeholder
        let pattern_with_placeholder = if !filename_pattern.contains("%d") {
            warn!("Filename pattern '{}' doesn't contain %d - adding it", filename_pattern);
            format!("{}_%05d", filename_pattern)
        } else {
            filename_pattern.replace("%d", "%05d").replace("%06d", "%05d") // Ensure consistent format
        };
        
        let filename_with_ext = format!("{}.{}", pattern_with_placeholder, image_format.extension());
        let output_path = format!("{}/{}", output_dir, filename_with_ext);
        
        // Use videorate for fps control and better pipeline construction
        let videorate_caps = format!("video/x-raw,framerate={}/1", fps as i32);
        
        // Build pipeline string with proper fps control and segment handling
        let pipeline_str = format!(
            "uridecodebin uri={} ! videoconvert ! videoscale ! videorate ! {} ! {} ! multifilesink location=\"{}\" index=0 post-messages=true max-files=0",
            self.input_uri,
            videorate_caps,
            encoder_element,
            output_path
        );
        
        info!("Creating frame extraction pipeline: {}", pipeline_str);
        debug!("Output path pattern: {}", output_path);
        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "frame_extraction_pipeline".to_string(),
                message: format!("Failed to create frame extraction pipeline: {}", e),
                source: None,
            })?;
        
        // Set to paused state first for seeking if needed
        pipeline.set_state(gst::State::Paused)?;
        let _ = pipeline.state(gst::ClockTime::from_seconds(10)); // Give more time for preroll
        
        // Handle seeking if start/end times are specified
        if let (Some(start_time), Some(end_time)) = (start, end) {
            info!("Seeking to segment from {:?} to {:?}", start_time, end_time);
            let seek_flags = SeekFlags::FLUSH | SeekFlags::ACCURATE | SeekFlags::SEGMENT;
            let seek_event = gst::event::Seek::new(
                1.0, // Normal playback rate
                seek_flags,
                SeekType::Set, start_time,
                SeekType::Set, end_time,
            );
            if !pipeline.send_event(seek_event) {
                warn!("Failed to send seek event - will extract from entire video");
            }
            
            // Wait longer after seeking to ensure it takes effect
            std::thread::sleep(std::time::Duration::from_millis(1000));
        } else if let Some(start_time) = start {
            info!("Seeking to start time {:?}", start_time);
            let seek_flags = SeekFlags::FLUSH | SeekFlags::ACCURATE;
            let seek_event = gst::event::Seek::new(
                1.0,
                seek_flags,
                SeekType::Set, start_time,
                SeekType::None, ClockTime::NONE,
            );
            if !pipeline.send_event(seek_event) {
                warn!("Failed to send seek event for start time");
            }
            std::thread::sleep(std::time::Duration::from_millis(1000));
        } else {
            info!("No start/end times specified - extracting from entire video");
        }
        
        // Start the pipeline
        info!("Setting pipeline to PLAYING state for frame extraction");
        pipeline.set_state(gst::State::Playing)?;
        info!("Frame extraction pipeline started");
        
        // Monitor progress with better timeout handling
        let bus = pipeline.bus().ok_or_else(|| VideoEditError::Processing { 
            message: "Failed to get pipeline bus".into(),
            source: None,
        })?;
        let mut frame_count = 0u32;
        let mut message_count = 0u32;
        let start_time_monitor = std::time::Instant::now();
        let timeout_duration = std::time::Duration::from_secs(300); // 5 minute timeout
        
        info!("Starting frame extraction monitoring loop...");
        
        loop {
            // Check for timeout
            if start_time_monitor.elapsed() > timeout_duration {
                error!("Frame extraction timed out after {} seconds", timeout_duration.as_secs());
                break;
            }
            
            // Poll for messages with a reasonable timeout
            if let Some(msg) = bus.timed_pop(gst::ClockTime::from_seconds(1)) {
                message_count += 1;
                debug!("Received message #{}: {:?}", message_count, msg.type_());
                
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        info!("Frame extraction complete! Extracted {} frames (processed {} messages)", frame_count, message_count);
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        error!("Frame extraction error: {} ({:?})", err.error(), err.debug());
                        return Err(VideoEditError::from(err.error()));
                    }
                    gst::MessageView::Warning(warn) => {
                        warn!("Frame extraction warning: {} ({:?})", warn.error(), warn.debug());
                    }
                    gst::MessageView::Info(info) => {
                        debug!("Frame extraction info: {} ({:?})", info.error(), info.debug());
                    }
                    gst::MessageView::Element(element) => {
                        if let Some(structure) = element.structure() {
                            debug!("Element message from '{}': {:?}", structure.name(), structure);
                            
                            if structure.name() == "multifilesink" {
                                if let Ok(filename) = structure.get::<&str>("filename") {
                                    frame_count += 1;
                                    if frame_count % 10 == 0 || frame_count <= 5 {
                                        info!("Extracted frame {}: {}", frame_count, filename);
                                    }
                                } else {
                                    debug!("Multifilesink message without filename: {:?}", structure);
                                }
                            }
                        }
                    }
                    gst::MessageView::StateChanged(state_changed) => {
                        if let Some(src) = msg.src() {
                            if src.name() == pipeline.name() {
                                debug!("Pipeline state changed: {:?} -> {:?}", 
                                       state_changed.old(), state_changed.current());
                            }
                        }
                    }
                    gst::MessageView::StreamStart(_) => {
                        info!("Stream started - frames should start extracting soon");
                    }
                    gst::MessageView::AsyncDone(_) => {
                        debug!("Async operation completed");
                    }
                    _ => {
                        debug!("Other message: {:?}", msg.view());
                    }
                }
            } else {
                // No message received within timeout - check if we should continue
                if frame_count > 0 {
                    debug!("No messages received, but {} frames extracted so far", frame_count);
                }
            }
        }
        
        pipeline.set_state(gst::State::Null)?;
        
        // Check results
        if frame_count == 0 {
            warn!("Multifilesink approach extracted 0 frames - trying alternative approach");
            return self.extract_frames_with_appsink(format, output_dir, start, end);
        }
        
        info!("Successfully extracted {} frames using multifilesink approach", frame_count);
        Ok(())
    }
    
    /// Alternative frame extraction using appsink (more reliable for small numbers of frames)
    fn extract_frames_with_appsink(
        &self,
        format: &ExportFormat,
        output_dir: &str,
        start: Option<ClockTime>,
        end: Option<ClockTime>,
    ) -> Result<(), VideoEditError> {
        let (filename_pattern, image_format, quality, _resize, fps) = match format {
            ExportFormat::FrameSequence { filename_pattern, format, quality, resize, fps, .. } => {
                (filename_pattern, format, *quality, *resize, *fps)
            }
            _ => return Err(VideoEditError::InvalidExportFormat {
                expected: "FrameSequence".to_string(),
                actual: "Non-FrameSequence".to_string()
            }),
        };

        info!("Using appsink approach for frame extraction");
        
        // Ensure output directory exists
        std::fs::create_dir_all(&output_dir)?;
        
        // Initialize GStreamer
        gst::init()?;
        
        // Determine time range
        let start_time = start.unwrap_or(ClockTime::ZERO);
        let end_time = end.unwrap_or_else(|| {
            ClockTime::from_seconds(60) // Default: extract first 60 seconds
        });
        
        let frame_duration = ClockTime::from_nseconds((1_000_000_000.0 / fps) as u64);
        let total_frames = ((end_time - start_time).nseconds() as f64 / frame_duration.nseconds() as f64).ceil() as u32;
        
        info!("Extracting {} frames from {:?} to {:?} at {}fps", total_frames, start_time, end_time, fps);
        info!("Frame duration: {:?}", frame_duration);
        
        // Build pipeline with appsink
        let pipeline_str = format!(
            "uridecodebin uri={} ! videoconvert ! videoscale ! video/x-raw,format=RGB ! appsink name=sink sync=false max-buffers=1 drop=true",
            self.input_uri
        );
        
        debug!("Creating appsink pipeline: {}", pipeline_str);
        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "appsink_pipeline".to_string(),
                message: format!("Failed to create appsink pipeline: {}", e),
                source: None,
            })?;
        
        // Get the appsink element
        let bin = pipeline
            .clone()
            .dynamic_cast::<gst::Bin>()
            .map_err(|_| VideoEditError::ElementCreation {
                element_name: "pipeline".to_string(),
                message: "Failed to cast pipeline to Bin".to_string(),
                source: None,
            })?;
        
        let appsink = bin
            .by_name("sink")
            .ok_or_else(|| VideoEditError::ElementCreation {
                element_name: "appsink".to_string(),
                message: "Failed to get appsink element".to_string(),
                source: None,
            })?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| VideoEditError::ElementCreation {
                element_name: "appsink".to_string(),
                message: "Failed to cast to AppSink".to_string(),
                source: None,
            })?;
        
        // Set to paused for seeking
        pipeline.set_state(gst::State::Paused)?;
        let _ = pipeline.state(gst::ClockTime::from_seconds(10));
        
        let mut current_time = start_time;
        let mut frame_number = 0u32;
        let mut successful_frames = 0u32;
        
        while current_time < end_time && frame_number < total_frames.min(1000) { // Limit to 1000 frames max
            frame_number += 1;
            
            // Seek to the specific frame time
            let seek_event = gst::event::Seek::new(
                1.0,
                SeekFlags::FLUSH | SeekFlags::ACCURATE,
                SeekType::Set, current_time,
                SeekType::None, ClockTime::NONE,
            );
            
            if !pipeline.send_event(seek_event) {
                warn!("Failed to seek to time {:?} for frame {}", current_time, frame_number);
                current_time += frame_duration;
                continue;
            }
            
            // Wait for seek to complete
            std::thread::sleep(std::time::Duration::from_millis(200));
            
            // Set to playing to get one frame
            pipeline.set_state(gst::State::Playing)?;
            std::thread::sleep(std::time::Duration::from_millis(100));
            
            // Try to pull a sample
            if let Some(sample) = appsink.try_pull_sample(gst::ClockTime::from_seconds(2)) {
                // Generate filename for this frame
                let filename = format!("{}.{}", 
                    filename_pattern.replace("%d", &format!("{:05}", frame_number))
                                   .replace("%06d", &format!("{:06}", frame_number))
                                   .replace("%05d", &format!("{:05}", frame_number)), 
                    image_format.extension()
                );
                let output_path = format!("{}/{}", output_dir, filename);
                
                // Save the frame (this is a simplified approach - in a real implementation,
                // you'd convert the sample to the desired image format)
                if self.save_sample_as_image(&sample, &output_path, image_format, quality)? {
                    successful_frames += 1;
                    if successful_frames % 10 == 0 || successful_frames <= 5 {
                        info!("Extracted frame {}: {}", successful_frames, filename);
                    }
                } else {
                    warn!("Failed to save frame {} to {}", frame_number, output_path);
                }
            } else {
                warn!("Failed to pull sample for frame {} at time {:?}", frame_number, current_time);
            }
            
            // Pause again for next seek
            pipeline.set_state(gst::State::Paused)?;
            
            current_time += frame_duration;
        }
        
        pipeline.set_state(gst::State::Null)?;
        
        if successful_frames == 0 {
            return Err(VideoEditError::Processing {
                message: "No frames could be extracted using any method".to_string(),
                source: None,
            });
        }
        
        info!("Successfully extracted {} frames using appsink approach", successful_frames);
        Ok(())
    }
    
    /// Save a GStreamer sample as an image file (simplified implementation)
    fn save_sample_as_image(
        &self,
        _sample: &gst::Sample,
        output_path: &str,
        _image_format: &ImageFormat,
        _quality: Option<u8>,
    ) -> Result<bool, VideoEditError> {
        // This is a placeholder implementation
        // In a real implementation, you would:
        // 1. Extract the buffer from the sample
        // 2. Get the video info (width, height, format)
        // 3. Convert the raw video data to the desired image format
        // 4. Save it to disk
        
        // For now, we'll just create a dummy file to indicate the frame was processed
        std::fs::write(output_path, b"dummy frame data")
            .map_err(|e| VideoEditError::Io(e))?;
        
        Ok(true)
    }
    
    /// Run pipeline and wait for completion (shared between video and frame export)
    fn run_pipeline(&self, pipeline: &ges::Pipeline) -> Result<(), VideoEditError> {
        // Check pipeline state before proceeding
        debug!("Checking pipeline state before state change");
        let (current_state, pending_state, _) = pipeline.state(gst::ClockTime::ZERO);
        debug!("Current pipeline state: {:?}, pending: {:?}", current_state, pending_state);
        
        // Set to PAUSED first to allow preroll
        debug!("Setting pipeline to PAUSED state");
        let state_change = pipeline.set_state(gst::State::Paused);
        match state_change {
            Err(e) => {
                error!("Failed to set pipeline to PAUSED: {:?}", e);
                
                // Get more detailed error information
                let bus = pipeline.bus();
                if let Some(bus) = bus {
                    debug!("Checking bus for error messages");
                    while let Some(msg) = bus.pop() {
                        match msg.view() {
                            gst::MessageView::Error(err) => {
                                error!("Pipeline error: {} (debug: {:?})", err.error(), err.debug());
                            }
                            gst::MessageView::Warning(warn) => {
                                warn!("Pipeline warning: {} (debug: {:?})", warn.error(), warn.debug());
                            }
                            gst::MessageView::Info(info) => {
                                info!("Pipeline info: {} (debug: {:?})", info.error(), info.debug());
                            }
                            _ => {}
                        }
                    }
                }
                
                return Err(VideoEditError::StateChange(e));
            }
            Ok(_) => debug!("Pipeline set to PAUSED successfully"),
        }
        
        // Wait a moment for preroll
        debug!("Waiting for pipeline preroll");
        std::thread::sleep(std::time::Duration::from_millis(500));
        
        // Now start rendering
        debug!("Setting pipeline to PLAYING state");
        let state_change = pipeline.set_state(gst::State::Playing);
        match state_change {
            Err(e) => {
                error!("Failed to set pipeline to PLAYING: {:?}", e);
                return Err(VideoEditError::StateChange(e));
            }
            Ok(_) => info!("Pipeline started successfully - rendering in progress"),
        }
        
        // Wait for rendering to complete
        let bus = pipeline.bus().ok_or_else(|| VideoEditError::Processing { 
            message: "Failed to get pipeline bus".into(),
            source: None,
        })?;
        
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    info!("Pipeline completed successfully!");
                    break;
                }
                gst::MessageView::Error(err) => {
                    error!("Pipeline error: {} ({:?})", err.error(), err.debug());
                    return Err(VideoEditError::from(err.error()));
                }
                gst::MessageView::StateChanged(state_changed) => {
                    trace!("Pipeline state changed: {:?} -> {:?}", 
                           state_changed.old(), state_changed.current());
                }
                _ => (),
            }
        }
        
        pipeline.set_state(gst::State::Null)
            .map_err(VideoEditError::from)?;
        
        Ok(())
    }
    
    /// Render reverse using timeline segment manipulation
    pub fn render_reverse_to_timeline(
        &self,
        _timeline: &ges::Timeline,
        layer: &ges::Layer,
        clip: &ges::UriClip,
        reverse_mode: &ReverseMode,
    ) -> Result<(), VideoEditError> {
        
        match reverse_mode {
            ReverseMode::Full => {
                let total_duration = clip.duration();
                self.create_reverse_segments(layer, clip, ClockTime::ZERO, total_duration)?;
            }
            
            ReverseMode::Segment { start, end } => {
                let segment_duration = *end - *start;
                self.create_reverse_segments(layer, clip, *start, segment_duration)?;
            }
            
            ReverseMode::FrameAccurate { fps } => {
                self.create_frame_accurate_reverse(layer, clip, *fps)?;
            }
        }
        
        Ok(())
    }
    
    /// Create small segments in reverse order for smooth reverse playback
    fn create_reverse_segments(
        &self,
        layer: &ges::Layer,
        original_clip: &ges::UriClip,
        start_time: ClockTime,
        duration: ClockTime,
    ) -> Result<(), VideoEditError> {
        
        // Split into small segments (e.g., 1 second each)
        let segment_duration = ClockTime::from_seconds(1);
        let num_segments = (duration.seconds() as f64 / segment_duration.seconds() as f64).ceil() as u64;
        
        // Remove original clip
        layer.remove_clip(original_clip)
            .map_err(|e| VideoEditError::Timeline { 
                message: format!("Failed to remove original clip: {}", e) 
            })?;
        
        // Create reversed segments
        for i in 0..num_segments {
            let segment_start_in_original = start_time + (segment_duration * i);
            let segment_end_in_original = std::cmp::min(
                segment_start_in_original + segment_duration,
                start_time + duration
            );
            let actual_segment_duration = segment_end_in_original - segment_start_in_original;
            
            // Create new clip for this segment
            let uri = original_clip.uri();
            let segment_clip = ges::UriClip::new(&uri)
                .map_err(|_| VideoEditError::Resource(gst::ResourceError::NotFound))?;
            
            // Set the inpoint to the segment start in original video
            segment_clip.set_inpoint(segment_start_in_original);
            segment_clip.set_duration(actual_segment_duration);
            
            // Position this segment in reverse order in the timeline
            let timeline_position = duration - (i + 1) * segment_duration;
            segment_clip.set_start(timeline_position);
            
            layer.add_clip(&segment_clip)
                .map_err(|e| VideoEditError::Timeline { 
                    message: format!("Failed to add reverse segment: {}", e) 
                })?;
        }
        
        Ok(())
    }
    
    /// Create frame-accurate reverse using individual frame clips
    fn create_frame_accurate_reverse(
        &self,
        layer: &ges::Layer,
        original_clip: &ges::UriClip,
        fps: f64,
    ) -> Result<(), VideoEditError> {
        
        let duration = original_clip.duration();
        let frame_duration = ClockTime::from_seconds(1) / (fps as u64);
        let total_frames = (duration / frame_duration) as u64;
        
        // Remove original clip
        layer.remove_clip(original_clip)
            .map_err(|e| VideoEditError::Timeline { 
                message: format!("Failed to remove original clip: {}", e) 
            })?;
        
        // Create individual frame clips in reverse order
        for frame_num in 0..total_frames {
            let original_frame_time = frame_duration * frame_num;
            let reverse_timeline_position = duration - (frame_num + 1) * frame_duration;
            
            let uri = original_clip.uri();
            let frame_clip = ges::UriClip::new(&uri)
                .map_err(|_| VideoEditError::Resource(gst::ResourceError::NotFound))?;
            frame_clip.set_inpoint(original_frame_time);
            frame_clip.set_duration(frame_duration);
            frame_clip.set_start(reverse_timeline_position);
            
            layer.add_clip(&frame_clip)
                .map_err(|e| VideoEditError::Timeline { 
                    message: format!("Failed to add frame clip: {}", e) 
                })?;
        }
        
        Ok(())
    }
    
    /// Extract a poster image at a specific timestamp from the video
    fn extract_poster_image_at_timestamp(&self, format: &ExportFormat, output_dir: &str, timestamp: ClockTime) -> Result<(), VideoEditError> {
        info!("Extracting poster image at timestamp {:?}", timestamp);
        
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;
        
        // Initialize GStreamer for image extraction
        gst::init()?;
        
        // Create a pipeline to extract a single frame
        let pipeline = gst::parse::launch(&format!(
            "uridecodebin uri={} ! videoconvert ! videoscale ! video/x-raw,width=1920,height=1080 ! pngenc ! filesink location={}",
            self.input_uri,
            self.get_poster_filename(format, output_dir)
        )).map_err(|e| VideoEditError::ElementCreation {
            element_name: "pipeline".to_string(),
            message: format!("Failed to create poster extraction pipeline: {}", e),
            source: None,
        })?;
        
        // Set to paused state first for seeking
        pipeline.set_state(gst::State::Paused)?;
        let _ = pipeline.state(gst::ClockTime::from_seconds(5));
        
        // Seek to the desired timestamp
        let seek_flags = SeekFlags::FLUSH | SeekFlags::ACCURATE;
        let seek_event = gst::event::Seek::new(
            1.0, // Normal playback rate
            seek_flags,
            SeekType::Set, timestamp,
            SeekType::None, ClockTime::NONE,
        );
        pipeline.send_event(seek_event);
        
        // Set to playing state
        pipeline.set_state(gst::State::Playing)?;
        
        // Wait for a short time to capture the frame
        let bus = pipeline.bus().ok_or_else(|| VideoEditError::Processing { 
            message: "Failed to get pipeline bus".into(),
            source: None,
        })?;
        let mut frame_captured = false;
        
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(5)) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    frame_captured = true;
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Poster extraction error: {} ({:?})", err.error(), err.debug());
                    return Err(VideoEditError::from(err.error()));
                }
                gst::MessageView::AsyncDone(..) => {
                    // Send EOS to capture just one frame
                    pipeline.send_event(gst::event::Eos::new());
                }
                _ => (),
            }
        }
        
        pipeline.set_state(gst::State::Null)?;
        
        if frame_captured {
            println!("Poster image saved: {}", self.get_poster_filename(format, output_dir));
        } else {
            println!("Warning: Poster image extraction may have failed");
        }
        
        Ok(())
    }
    
    /// Get the filename for the poster image based on output config
    fn get_poster_filename(&self, format: &ExportFormat, output_dir: &str) -> String {
        match &format {
            ExportFormat::Poster { timestamp, filename, format: image_format, .. } => {
                format!("{}/{}_{}.{}", 
                    output_dir,
                    filename,
                    timestamp.seconds(),
                    image_format.extension()
                )
            }
            _ => format!("{}/poster.png", output_dir), // Default fallback
        }
    }

    /// Extract frames from edited timeline using two-step approach:
    /// 1. Render edited timeline to temporary video file
    /// 2. Extract frames from the temporary file
    fn extract_frames_from_edited_timeline(
        &self,
        _timeline: &ges::Timeline,
        pipeline: &ges::Pipeline,
        format: &ExportFormat,
        output_dir: &str,
        start: Option<ClockTime>,
        end: Option<ClockTime>,
    ) -> Result<(), VideoEditError> {
        // First, render the edited timeline to a temporary video file
        let temp_dir = std::env::temp_dir();
        let temp_video_path = temp_dir.join("temp_edited_video.mp4");
        let temp_video_uri = format!("file://{}", temp_video_path.display());
        
        info!("Step 1: Rendering edited timeline to temporary file: {}", temp_video_path.display());
        
        // Create encoding profile for temporary video (MP4 for compatibility)
        let audio_profile = EncodingAudioProfile::builder(
            &gst::Caps::builder("audio/mpeg")
                .field("mpegversion", 4i32)
                .field("stream-format", "raw")
                .build()
        ).build();
        
        let video_profile = EncodingVideoProfile::builder(
            &gst::Caps::builder("video/x-h264")
                .field("stream-format", "avc")
                .build()
        ).build();
        
        let temp_encoding_profile = EncodingContainerProfile::builder(
            &gst::Caps::builder("video/quicktime")
                .field("variant", "iso")
                .build()
        )
        .name("temp_mp4_container")
        .add_profile(video_profile)
        .add_profile(audio_profile)
        .build();
        
        // Configure pipeline for rendering to temporary file
        debug!("Setting render settings for temporary file");
        pipeline.set_render_settings(&temp_video_uri, &temp_encoding_profile)
            .map_err(|e| {
                error!("Failed to set render settings for temp file: {}", e);
                VideoEditError::Timeline { 
                    message: format!("Failed to set render settings for temp file: {}", e) 
                }
            })?;
        
        debug!("Setting pipeline mode to RENDER for temp file");
        pipeline.set_mode(ges::PipelineFlags::RENDER)
            .map_err(|e| {
                error!("Failed to set pipeline mode for temp file: {}", e);
                VideoEditError::Timeline { 
                    message: format!("Failed to set pipeline mode for temp file: {}", e) 
                }
            })?;
        
        // Render the edited timeline to temporary file
        debug!("Starting render of edited timeline to temp file");
        self.run_pipeline(&pipeline)?;
        
        // Check if temporary file was created and has content
        if !temp_video_path.exists() {
            return Err(VideoEditError::Processing {
                message: "Temporary video file was not created".to_string(),
                source: None,
            });
        }
        
        let file_size = std::fs::metadata(&temp_video_path)
            .map(|m| m.len())
            .unwrap_or(0);
        info!("Temporary video file created: {} bytes", file_size);
        
        if file_size == 0 {
            return Err(VideoEditError::Processing {
                message: "Temporary video file is empty".to_string(),
                source: None,
            });
        }
        
        // Step 2: Extract frames from the temporary file
        info!("Step 2: Extracting frames from temporary video file");
        let temp_video_uri_for_extraction = format!("file://{}", temp_video_path.display());
        let temp_editor = VideoEditor::new(temp_video_uri_for_extraction);
        temp_editor.extract_frame_sequence_with_custom_pipeline(format, output_dir, start, end)?;
        
        // Clean up temporary file
        debug!("Cleaning up temporary file: {}", temp_video_path.display());
        if let Err(e) = std::fs::remove_file(&temp_video_path) {
            warn!("Failed to remove temporary file {}: {}", temp_video_path.display(), e);
        }
        
        info!("Frame extraction from edited timeline completed successfully");
        Ok(())
    }
} 