use gstreamer::{self as gst, prelude::{*}, ClockTime, SeekFlags, SeekType, glib};
use gstreamer_editing_services::{self as ges, prelude::*};
use gstreamer_pbutils::{EncodingAudioProfile, EncodingVideoProfile, EncodingContainerProfile};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use thiserror::Error;
use tracing::{info, debug, warn, error, trace};

// macOS support module
pub mod macos;
pub use macos::{init_macos_app, init_macos_background_app};

/// Comprehensive errors for video editing operations that integrate with GStreamer's error system
#[derive(Error, Debug)]
pub enum VideoEditError {
    /// GStreamer initialization failed
    #[error("GStreamer initialization failed: {0}")]
    GStreamerInit(#[from] glib::Error),
    
    /// GStreamer boolean operation failed
    #[error("GStreamer boolean operation failed: {0}")]
    GStreamerBool(#[from] glib::BoolError),
    
    /// Pipeline state change failed
    #[error("Pipeline state change failed: {0}")]
    StateChange(#[from] gst::StateChangeError),
    
    /// Data flow error in GStreamer pipeline
    #[error("Pipeline data flow error: {0}")]
    Flow(#[from] gst::FlowError),
    
    /// Core GStreamer error (seek, pad, negotiation, etc.)
    #[error("GStreamer core error: {0:?}")]
    Core(gst::CoreError),
    
    /// Resource-related error (file I/O, not found, etc.)
    #[error("GStreamer resource error: {0:?}")]
    Resource(gst::ResourceError),
    
    /// GStreamer Editing Services specific error
    #[error("GES error: {0:?}")]
    GES(ges::Error),
    
    /// Structured GStreamer error message with context
    #[error("GStreamer error: {0}")]
    ErrorMessage(#[from] gst::ErrorMessage),
    
    /// Loggable GStreamer error with debug category
    #[error("GStreamer loggable error: {0}")]
    Loggable(#[from] gst::LoggableError),
    
    /// Timeline operation failed
    #[error("Timeline operation failed: {message}")]
    Timeline { message: String },
    
    /// Encoding profile creation failed
    #[error("Encoding profile creation failed for format '{format}': {error}")]
    EncodingProfile { 
        format: String,
        error: String,
    },
    
    /// File I/O error
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Invalid export format for operation
    #[error("Invalid export format for operation: expected {expected}, got {actual}")]
    InvalidExportFormat { expected: String, actual: String },
    
    /// Video processing failed with structured error
    #[error("Video processing failed: {message}")]
    Processing { 
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    
    /// Seek operation failed with specific seek error
    #[error("Seek operation failed: {message} (core error: {core_error:?})")]
    Seek { 
        message: String,
        core_error: Option<gst::CoreError>,
    },
    
    /// GStreamer element creation failed
    #[error("GStreamer element creation failed for '{element_name}': {message}")]
    ElementCreation { 
        element_name: String,
        message: String,
        #[source]
        source: Option<glib::BoolError>,
    },
    
    /// Pipeline linking failed
    #[error("Pipeline element linking failed: {message} (link result: {link_result:?})")]
    ElementLinking { 
        message: String,
        link_result: Option<gst::PadLinkReturn>,
    },
    
    /// Caps negotiation failed
    #[error("Caps negotiation failed: {message} (core error: {core_error:?})")]
    CapsNegotiation {
        message: String,
        core_error: Option<gst::CoreError>,
    },
    
    /// Clock synchronization error
    #[error("Clock synchronization error: {message} (clock return: {clock_return:?})")]
    Clock {
        message: String,
        clock_return: Option<gst::ClockReturn>,
    },
}

// Manual From implementations for GStreamer error types
impl From<gst::CoreError> for VideoEditError {
    fn from(err: gst::CoreError) -> Self {
        VideoEditError::Core(err)
    }
}

impl From<gst::ResourceError> for VideoEditError {
    fn from(err: gst::ResourceError) -> Self {
        VideoEditError::Resource(err)
    }
}

// Convert from GES Error to our error type
impl From<ges::Error> for VideoEditError {
    fn from(err: ges::Error) -> Self {
        VideoEditError::GES(err)
    }
}

// Convert from PadLinkReturn failures
impl From<gst::PadLinkReturn> for VideoEditError {
    fn from(link_return: gst::PadLinkReturn) -> Self {
        let message = match link_return {
            gst::PadLinkReturn::WrongHierarchy => "Wrong hierarchy for pad linking",
            gst::PadLinkReturn::WasLinked => "Pad was already linked",
            gst::PadLinkReturn::WrongDirection => "Wrong direction for pad linking",
            gst::PadLinkReturn::Noformat => "No compatible format for pad linking",
            gst::PadLinkReturn::Nosched => "No scheduling for pad linking",
            gst::PadLinkReturn::Refused => "Pad linking was refused",
            gst::PadLinkReturn::Ok => "Pad linking succeeded", // This shouldn't happen in error context
        };
        
        VideoEditError::ElementLinking {
            message: message.to_string(),
            link_result: Some(link_return),
        }
    }
}

// Helper trait for converting Results with GStreamer context
pub trait GStreamerResultExt<T> {
    /// Convert a Result to VideoEditError with additional context
    fn with_gst_context(self, context: &str) -> Result<T, VideoEditError>;
}

impl<T, E> GStreamerResultExt<T> for Result<T, E> 
where 
    E: Into<VideoEditError>,
{
    fn with_gst_context(self, context: &str) -> Result<T, VideoEditError> {
        self.map_err(|e| {
            let base_error = e.into();
            match base_error {
                VideoEditError::Processing { message, source } => {
                    VideoEditError::Processing {
                        message: format!("{}: {}", context, message),
                        source,
                    }
                }
                other => other,
            }
        })
    }
} 

/// Top-level video editing commands with file output support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoEditCommand {
    /// Cut a section from the video
    Cut {
        start: ClockTime,
        duration: ClockTime,
    },
    
    /// Reverse video playback
    Reverse {
        flags: SeekFlags,
        rate: f64,
        mode: Option<ReverseMode>, // Added reverse mode
    },
    
    /// Loop video with different options
    Loop {
        mode: LoopMode,
        flags: SeekFlags,
    },
    // Freeze video frame, while original audio continues playing
    Freeze {
        start: ClockTime,      // When to start the freeze
        duration: ClockTime,   // How long to freeze for
        flags: SeekFlags,
        rate: f64,
    },
}

/// Different modes for reverse video playback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReverseMode {
    /// Reverse entire clip
    Full,
    /// Reverse specific time range
    Segment { start: ClockTime, end: ClockTime },
    /// Reverse with frame-level precision (slower but more accurate)
    FrameAccurate { fps: f64 },
}


/// Different looping modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopMode {
    Infinite,
    Count(u32),
    Duration(ClockTime),
    UntilPosition(ClockTime),
    Segment { start: ClockTime, end: ClockTime },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Bmp,
    Tiff,
    WebP,
}

impl ImageFormat {
    fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg", 
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
            ImageFormat::WebP => "webp",
        }
    }
    
    fn caps_string(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Bmp => "image/bmp", 
            ImageFormat::Tiff => "image/tiff",
            ImageFormat::WebP => "image/webp",
        }
    }
}

/// Output configuration for exporting the edited stream
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Output directory where files will be created
    pub output_dir: String,
    /// Export format (video, single frame, or frame sequence)
    pub format: ExportFormat,
}

/// Unified export formats for the edited video stream
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// Export as video file
    Video {
        format: VideoFormat,
        filename: String, // e.g. "my_video" (extension added automatically)
    },
    /// Export single frame as poster image
    Poster {
        format: ImageFormat,
        timestamp: ClockTime, // Which frame to extract
        filename: String, // e.g. "poster" (extension added automatically)
        quality: Option<u8>,
        resize: Option<(u32, u32)>,
    },
    /// Export as sequence of image frames
    FrameSequence {
        format: ImageFormat,
        start: ClockTime,
        duration: ClockTime,
        fps: f64,
        filename_pattern: String, // Pattern like "frame_%06d"
        quality: Option<u8>,
        resize: Option<(u32, u32)>,
    },
}

/// Video export formats using existing GStreamer encoding profiles
#[derive(Debug, Clone)]
pub enum VideoFormat {
    /// WebM container with VP8 video and Opus audio
    WebM,
    /// Matroska container with Theora video and Vorbis audio  
    Matroska,
    /// MP4 container with H.264 video and AAC audio
    Mp4,
    /// Custom format with specific encoding profiles
    Custom {
        container_caps: String,
        video_caps: String,
        audio_caps: String,
    },
}

impl VideoFormat {
    /// Create encoding profile from format using existing GStreamer types
    fn to_encoding_profile(&self) -> Result<EncodingContainerProfile, VideoEditError> {
        match self {
            VideoFormat::WebM => {
                let audio_profile = EncodingAudioProfile::builder(
                    &gst::Caps::builder("audio/x-opus").build()
                ).build();
                
                let video_profile = EncodingVideoProfile::builder(
                    &gst::Caps::builder("video/x-vp8").build()
                ).build();
                
                Ok(EncodingContainerProfile::builder(
                    &gst::Caps::builder("video/webm").build()
                )
                .name("webm_container")
                .add_profile(video_profile)
                .add_profile(audio_profile)
                .build())
            }
            
            VideoFormat::Matroska => {
                let audio_profile = EncodingAudioProfile::builder(
                    &gst::Caps::builder("audio/x-vorbis").build()
                ).build();
                
                let video_profile = EncodingVideoProfile::builder(
                    &gst::Caps::builder("video/x-theora").build()
                ).build();
                
                Ok(EncodingContainerProfile::builder(
                    &gst::Caps::builder("video/x-matroska").build()
                )
                .name("mkv_container")
                .add_profile(video_profile)
                .add_profile(audio_profile)
                .build())
            }
            
            VideoFormat::Mp4 => {
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
                
                Ok(EncodingContainerProfile::builder(
                    &gst::Caps::builder("video/quicktime")
                        .field("variant", "iso")
                        .build()
                )
                .name("mp4_container")
                .add_profile(video_profile)
                .add_profile(audio_profile)
                .build())
            }
            
            VideoFormat::Custom { container_caps, video_caps, audio_caps } => {
                let audio_caps = gst::Caps::from_str(audio_caps)
                    .map_err(|e| VideoEditError::EncodingProfile { 
                        format: "custom audio caps".to_string(), 
                        error: e.to_string() 
                    })?;
                let video_caps = gst::Caps::from_str(video_caps)
                    .map_err(|e| VideoEditError::EncodingProfile { 
                        format: "custom video caps".to_string(), 
                        error: e.to_string() 
                    })?;
                let container_caps = gst::Caps::from_str(container_caps)
                    .map_err(|e| VideoEditError::EncodingProfile { 
                        format: "custom container caps".to_string(), 
                        error: e.to_string() 
                    })?;
                
                let audio_profile = EncodingAudioProfile::builder(&audio_caps).build();
                let video_profile = EncodingVideoProfile::builder(&video_caps).build();
                
                Ok(EncodingContainerProfile::builder(&container_caps)
                .name("custom_container")
                .add_profile(video_profile)
                .add_profile(audio_profile)
                .build())
            }
        }
    }
    
    /// Get file extension for the format
    fn extension(&self) -> &'static str {
        match self {
            VideoFormat::WebM => "webm",
            VideoFormat::Matroska => "mkv", 
            VideoFormat::Mp4 => "mp4",
            VideoFormat::Custom { .. } => "out",
        }
    }
}

/// Video editor that can both preview and render to file
pub struct VideoEditor {
    input_uri: String,
    commands: Vec<VideoEditCommand>,
}

impl VideoEditor {
    pub fn new(input_uri: String) -> Self {
        Self {
            input_uri,
            commands: Vec::new(),
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
    pub fn render_to_file(&self, output_config: OutputConfig) -> Result<(), VideoEditError> {
        info!("Starting video rendering with {} commands", self.commands.len());
        debug!("Input URI: {}", self.input_uri);
        debug!("Output config: {:?}", output_config);
        
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
                VideoEditCommand::Cut { start, duration } => {
                    trace!("Setting clip inpoint: {:?}, duration: {:?}", start, duration);
                    clip.set_inpoint(*start);
                    clip.set_duration(*duration);
                }
                
                VideoEditCommand::Reverse { mode, .. } => {
                    info!("Applying reverse effect");
                    let reverse_mode = mode.as_ref().unwrap_or(&ReverseMode::Full);
                    self.render_reverse_to_timeline(&timeline, &layer, &clip, reverse_mode)?;
                }
                
                VideoEditCommand::Freeze { start, duration, .. } => {
                    info!("Applying freeze effect at {:?} for {:?}", start, duration);
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
        match &output_config.format {
            ExportFormat::Video { format, filename } => {
                info!("Exporting video as {:?} format to: {}/{}.{}", 
                      format, output_config.output_dir, filename, format.extension());
                
                // Create output directory if it doesn't exist
                debug!("Creating output directory: {}", output_config.output_dir);
                std::fs::create_dir_all(&output_config.output_dir)?;
                
                // Commit the timeline to ensure all changes are applied
                debug!("Committing timeline changes");
                if !timeline.commit() {
                    return Err(VideoEditError::Timeline { 
                        message: "Failed to commit timeline changes".to_string() 
                    });
                }
                
                // Configure for video file output using existing encoding patterns
                debug!("Configuring pipeline for rendering");
                self.configure_pipeline_for_rendering(&pipeline, &output_config)?;
                
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
                            info!("Video rendering completed successfully!");
                            break;
                        }
                        gst::MessageView::Error(err) => {
                            error!("Video rendering error: {} ({:?})", err.error(), err.debug());
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
            }
            ExportFormat::Poster { timestamp, filename, .. } => {
                info!("Extracting poster image at {:?} to: {}/{}", 
                      timestamp, output_config.output_dir, filename);
                self.extract_poster_image_at_timestamp(&output_config, *timestamp)?;
            }
            ExportFormat::FrameSequence { start, duration, fps, filename_pattern, .. } => {
                info!("Extracting frame sequence: {}s duration at {}fps using pattern '{}'", 
                      duration.seconds(), fps, filename_pattern);
                self.extract_frames(&self.input_uri, *start, *duration, *fps, &output_config)?;
            }
        }
        
        info!("Rendering operation completed successfully");
        Ok(())
    }
    
    /// Configure pipeline for rendering (reuses existing GStreamer patterns)
    fn configure_pipeline_for_rendering(
        &self, 
        pipeline: &ges::Pipeline, 
        output_config: &OutputConfig
    ) -> Result<(), VideoEditError> {
        debug!("Creating encoding profile for output format");
        let encoding_profile = match &output_config.format {
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
        let output_path = match &output_config.format {
            ExportFormat::Video { format: video_format, filename } => {
                format!("{}/{}.{}", 
                    output_config.output_dir, 
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
    fn extract_poster_image_at_timestamp(&self, output_config: &OutputConfig, timestamp: ClockTime) -> Result<(), VideoEditError> {
        info!("Extracting poster image at timestamp {:?}", timestamp);
        
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_config.output_dir)?;
        
        // Initialize GStreamer for image extraction
        gst::init()?;
        
        // Create a pipeline to extract a single frame
        let pipeline = gst::parse::launch(&format!(
            "uridecodebin uri={} ! videoconvert ! videoscale ! video/x-raw,width=1920,height=1080 ! pngenc ! filesink location={}",
            self.input_uri,
            self.get_poster_filename(output_config)
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
            println!("Poster image saved: {}", self.get_poster_filename(output_config));
        } else {
            println!("Warning: Poster image extraction may have failed");
        }
        
        Ok(())
    }
    
    /// Get the filename for the poster image based on output config
    fn get_poster_filename(&self, output_config: &OutputConfig) -> String {
        match &output_config.format {
            ExportFormat::Poster { timestamp, filename, format: image_format, .. } => {
                format!("{}/{}_{}.{}", 
                    output_config.output_dir,
                    filename,
                    timestamp.seconds(),
                    image_format.extension()
                )
            }
            _ => format!("{}/poster.png", output_config.output_dir), // Default fallback
        }
    }

      /// Extract frames using multifilesink (most efficient for many frames)
    pub fn extract_frames(
        &self,
        input_uri: &str,
        start: ClockTime,
        duration: ClockTime,
        fps: f64,
        output_config: &OutputConfig,
    ) -> Result<(), VideoEditError> {
        
        // Extract frame sequence config from the export format
        let (filename_pattern, image_format, quality, _resize) = match &output_config.format {
            ExportFormat::FrameSequence { filename_pattern, format, quality, resize, .. } => {
                (filename_pattern, format, *quality, *resize)
            }
            _ => return Err(VideoEditError::InvalidExportFormat { expected: "FrameSequence".to_string(), actual: "Non-FrameSequence".to_string() }),
        };
        
        // Use output_dir for frame sequences
        let output_dir = &output_config.output_dir;
        
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)?;
        println!("Created output directory: {}", output_dir);
        
        gst::init()?;
        
        let pipeline = gst::Pipeline::default();
        
        // Source with seeking capability
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", input_uri)
            .build()
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "uridecodebin".to_string(),
                message: "Failed to create URI decode bin".to_string(),
                source: Some(e),
            })?;
        
        // Video processing chain
        let videoconvert = gst::ElementFactory::make("videoconvert").build()
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "videoconvert".to_string(),
                message: "Failed to create video converter".to_string(),
                source: Some(e),
            })?;
        let videoscale = gst::ElementFactory::make("videoscale").build()
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "videoscale".to_string(),
                message: "Failed to create video scaler".to_string(),
                source: Some(e),
            })?;
        
        // Frame rate filter to control extraction rate
        let fps_filter = gst::ElementFactory::make("capsfilter").build()
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "capsfilter".to_string(),
                message: "Failed to create caps filter".to_string(),
                source: Some(e),
            })?;
        let fps_caps = gst::Caps::builder("video/x-raw")
            .field("framerate", gst::Fraction::new(fps as i32, 1))
            .build();
        fps_filter.set_property("caps", &fps_caps);
        
        // Image encoder based on format
        let encoder = match image_format {
            ImageFormat::Png => gst::ElementFactory::make("pngenc").build()
                .map_err(|e| VideoEditError::ElementCreation {
                    element_name: "pngenc".to_string(),
                    message: "Failed to create PNG encoder".to_string(),
                    source: Some(e),
                })?,
            ImageFormat::Jpeg => {
                let enc = gst::ElementFactory::make("jpegenc").build()
                    .map_err(|e| VideoEditError::ElementCreation {
                        element_name: "jpegenc".to_string(),
                        message: "Failed to create JPEG encoder".to_string(),
                        source: Some(e),
                    })?;
                if let Some(quality) = quality {
                    enc.set_property("quality", quality as i32);
                }
                enc
            }
            ImageFormat::Bmp => gst::ElementFactory::make("bmpenc").build()
                .map_err(|e| VideoEditError::ElementCreation {
                    element_name: "bmpenc".to_string(),
                    message: "Failed to create BMP encoder".to_string(),
                    source: Some(e),
                })?,
            _ => gst::ElementFactory::make("pngenc").build()
                .map_err(|e| VideoEditError::ElementCreation {
                    element_name: "pngenc".to_string(),
                    message: "Failed to create PNG encoder (fallback)".to_string(),
                    source: Some(e),
                })?, // Default to PNG
        };
        
        // The key element: multifilesink for saving multiple files
        let multifilesink = gst::ElementFactory::make("multifilesink").build()
            .map_err(|e| VideoEditError::ElementCreation {
                element_name: "multifilesink".to_string(),
                message: "Failed to create multi-file sink".to_string(),
                source: Some(e),
            })?;
        
        // Configure output path pattern
        let filename_with_ext = format!("{}.{}", 
            filename_pattern, 
            image_format.extension()
        );
        let output_path = format!("{}/{}", output_dir, filename_with_ext);
        
        multifilesink.set_property("location", output_path);
        multifilesink.set_property("index", 1i32); // Start numbering from 1
        
        // Add elements to pipeline
        pipeline.add_many([
            &source, &videoconvert, &videoscale, &fps_filter, &encoder, &multifilesink
        ]).map_err(|_| VideoEditError::ElementLinking { 
            message: "Failed to add elements to pipeline".to_string(),
            link_result: None,
        })?;
        
        // Link static elements
        gst::Element::link_many([
            &videoconvert, &videoscale, &fps_filter, &encoder, &multifilesink
        ]).map_err(|_| VideoEditError::ElementLinking { 
            message: "Failed to link pipeline elements".to_string(),
            link_result: None,
        })?;
        
        // Handle dynamic source pads
        let videoconvert_clone = videoconvert.clone();
        source.connect_pad_added(move |_element, pad| {
            if let Some(caps) = pad.current_caps() {
                let structure = caps.structure(0).unwrap();
                if structure.name().starts_with("video/") {
                    let sink_pad = videoconvert_clone.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                    }
                }
            }
        });
        
        // Seek to start position
        pipeline.set_state(gst::State::Paused)?;
        
        // Wait for preroll
        let _ = pipeline.state(gst::ClockTime::from_seconds(5));
        
        // Perform seek to start position
        let seek_flags = SeekFlags::FLUSH | SeekFlags::ACCURATE;
        let end_position = start + duration;
        
        let seek_event = gst::event::Seek::new(
            1.0, // Normal playback rate
            seek_flags,
            SeekType::Set, start,
            SeekType::Set, end_position,
        );
        pipeline.send_event(seek_event);
        
        // Start extraction
        pipeline.set_state(gst::State::Playing)?;
        
        // Monitor progress
        let bus = pipeline.bus().ok_or_else(|| VideoEditError::Processing { 
            message: "Failed to get pipeline bus".into(),
            source: None,
        })?;
        let mut frame_count = 0u32;
        
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("Frame extraction complete! Extracted {} frames", frame_count);
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Extraction error: {} ({:?})", err.error(), err.debug());
                    return Err(VideoEditError::from(err.error()));
                }
                gst::MessageView::Element(element) => {
                    // Monitor multifilesink messages for progress
                    if let Some(structure) = element.structure() {
                        if structure.name() == "multifilesink" {
                            if let Ok(filename) = structure.get::<&str>("filename") {
                                frame_count += 1;
                                println!("Extracted frame: {}", filename);
                            }
                        }
                    }
                }
                _ => (),
            }
        }
        
        pipeline.set_state(gst::State::Null)?;
        Ok(())
    }

}

// Convenience constructors for ExportFormat
impl ExportFormat {
    /// Create a video export with MP4 format
    pub fn mp4_video() -> Self {
        Self::Video {
            format: VideoFormat::Mp4,
            filename: "video".to_string(),
        }
    }
    
    /// Create a video export with MP4 format and custom filename
    pub fn mp4_video_named(filename: String) -> Self {
        Self::Video {
            format: VideoFormat::Mp4,
            filename,
        }
    }
    
    /// Create a video export with WebM format
    pub fn webm_video() -> Self {
        Self::Video {
            format: VideoFormat::WebM,
            filename: "video".to_string(),
        }
    }
    
    /// Create a video export with WebM format and custom filename
    pub fn webm_video_named(filename: String) -> Self {
        Self::Video {
            format: VideoFormat::WebM,
            filename,
        }
    }
    
    /// Create a poster export at the first frame
    pub fn poster_at_start() -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::ZERO,
            filename: "poster".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a poster export at a specific timestamp
    pub fn poster_at(timestamp_seconds: u64) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_seconds(timestamp_seconds),
            filename: "poster".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a poster export at a specific timestamp with custom filename
    pub fn poster_at_named(timestamp_seconds: u64, filename: String) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_seconds(timestamp_seconds),
            filename,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export as PNG images
    pub fn frame_sequence_png(
        start_seconds: u64,
        duration_seconds: u64,
        fps: f64,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: ClockTime::from_seconds(start_seconds),
            duration: ClockTime::from_seconds(duration_seconds),
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with custom filename pattern
    pub fn frame_sequence_png_named(
        start_seconds: u64,
        duration_seconds: u64,
        fps: f64,
        filename_pattern: String,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: ClockTime::from_seconds(start_seconds),
            duration: ClockTime::from_seconds(duration_seconds),
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
}

// Convenience constructors
impl VideoEditCommand {
    pub fn cut(start_seconds: u64, duration_seconds: u64) -> Self {
        Self::Cut {
            start: ClockTime::from_seconds(start_seconds),
            duration: ClockTime::from_seconds(duration_seconds),
        }
    }
    
    pub fn reverse() -> Self {
        Self::Reverse {
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: -1.0,
            mode: None, // Default to full reverse
        }
    }
    
    pub fn reverse_with_mode(mode: ReverseMode) -> Self {
        Self::Reverse {
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: -1.0,
            mode: Some(mode),
        }
    }
    
    pub fn loop_count(count: u32) -> Self {
        Self::Loop {
            mode: LoopMode::Count(count),
            flags: SeekFlags::FLUSH | SeekFlags::KEY_UNIT,
        }
    }

    pub fn freeze(start_seconds: u64, duration_seconds: u64) -> Self {
        Self::Freeze {
            start: ClockTime::from_seconds(start_seconds),
            duration: ClockTime::from_seconds(duration_seconds),
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: 1.0, // Normal audio rate during freeze
        }
    }
} 