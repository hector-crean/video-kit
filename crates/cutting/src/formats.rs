use gstreamer::{self as gst, ClockTime};
use gstreamer_pbutils::{EncodingAudioProfile, EncodingVideoProfile, EncodingContainerProfile};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use crate::error::VideoEditError;
use gstreamer_editing_services::prelude::EncodingProfileBuilder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Bmp,
    Tiff,
    WebP,
}

impl ImageFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg", 
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
            ImageFormat::WebP => "webp",
        }
    }
    
    pub fn caps_string(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Bmp => "image/bmp", 
            ImageFormat::Tiff => "image/tiff",
            ImageFormat::WebP => "image/webp",
        }
    }
}

/// Unified export formats for the edited video stream
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
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
        #[serde(skip_serializing_if = "Option::is_none", default)]
        start: Option<ClockTime>, // Optional start time - defaults to beginning of edited video
        #[serde(skip_serializing_if = "Option::is_none", default)]
        end: Option<ClockTime>,   // Optional end time - defaults to end of edited video
        fps: f64,
        filename_pattern: String, // Pattern like "frame_%06d"
        #[serde(skip_serializing_if = "Option::is_none")]
        quality: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        resize: Option<(u32, u32)>,
    },
}

/// Video export formats using existing GStreamer encoding profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn to_encoding_profile(&self) -> Result<EncodingContainerProfile, VideoEditError> {
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
    pub fn extension(&self) -> &'static str {
        match self {
            VideoFormat::WebM => "webm",
            VideoFormat::Matroska => "mkv", 
            VideoFormat::Mp4 => "mp4",
            VideoFormat::Custom { .. } => "out",
        }
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
    
    /// Create a poster export at a specific timestamp with millisecond precision
    pub fn poster_at_ms(timestamp_ms: u64) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_mseconds(timestamp_ms),
            filename: "poster".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a poster export at a specific timestamp (seconds - kept for compatibility)
    pub fn poster_at(timestamp_seconds: u64) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_seconds(timestamp_seconds),
            filename: "poster".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a poster export at a specific timestamp with custom filename (milliseconds)
    pub fn poster_at_ms_named(timestamp_ms: u64, filename: String) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_mseconds(timestamp_ms),
            filename,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a poster export at a specific timestamp with custom filename (seconds)
    pub fn poster_at_named(timestamp_seconds: u64, filename: String) -> Self {
        Self::Poster {
            format: ImageFormat::Png,
            timestamp: ClockTime::from_seconds(timestamp_seconds),
            filename,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with start/end range (milliseconds)
    pub fn frame_sequence_range_ms(
        start_ms: u64,
        end_ms: u64,
        fps: f64,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_mseconds(start_ms)),
            end: Some(ClockTime::from_mseconds(end_ms)),
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with start/end range (seconds)
    pub fn frame_sequence_range(
        start_seconds: u64,
        end_seconds: u64,
        fps: f64,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_seconds(start_seconds)),
            end: Some(ClockTime::from_seconds(end_seconds)),
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with start/end range and custom pattern (milliseconds)
    pub fn frame_sequence_range_ms_named(
        start_ms: u64,
        end_ms: u64,
        fps: f64,
        filename_pattern: String,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_mseconds(start_ms)),
            end: Some(ClockTime::from_mseconds(end_ms)),
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with start/end range and custom pattern (seconds)
    pub fn frame_sequence_range_named(
        start_seconds: u64,
        end_seconds: u64,
        fps: f64,
        filename_pattern: String,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_seconds(start_seconds)),
            end: Some(ClockTime::from_seconds(end_seconds)),
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export as PNG images with millisecond precision (legacy - uses duration)
    pub fn frame_sequence_png_ms(
        start_ms: u64,
        duration_ms: u64,
        fps: f64,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_mseconds(start_ms)),
            end: Some(ClockTime::from_mseconds(start_ms + duration_ms)),
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export as PNG images (seconds - kept for compatibility)
    pub fn frame_sequence_png(
        start_seconds: u64,
        duration_seconds: u64,
        fps: f64,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_seconds(start_seconds)),
            end: Some(ClockTime::from_seconds(start_seconds + duration_seconds)),
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with custom filename pattern (milliseconds)
    pub fn frame_sequence_png_ms_named(
        start_ms: u64,
        duration_ms: u64,
        fps: f64,
        filename_pattern: String,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_mseconds(start_ms)),
            end: Some(ClockTime::from_mseconds(start_ms + duration_ms)),
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export with custom filename pattern (seconds)
    pub fn frame_sequence_png_named(
        start_seconds: u64,
        duration_seconds: u64,
        fps: f64,
        filename_pattern: String,
    ) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: Some(ClockTime::from_seconds(start_seconds)),
            end: Some(ClockTime::from_seconds(start_seconds + duration_seconds)),
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export for the entire video with default settings
    pub fn frame_sequence_entire_video(fps: f64) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: None, // Extract from beginning
            end: None,   // Extract to end
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export for the entire video with custom filename pattern
    pub fn frame_sequence_entire_video_named(fps: f64, filename_pattern: String) -> Self {
        Self::FrameSequence {
            format: ImageFormat::Png,
            start: None, // Extract from beginning
            end: None,   // Extract to end
            fps,
            filename_pattern,
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export for the entire video with custom format
    pub fn frame_sequence_entire_video_format(fps: f64, format: ImageFormat) -> Self {
        Self::FrameSequence {
            format,
            start: None, // Extract from beginning
            end: None,   // Extract to end
            fps,
            filename_pattern: "frame_%06d".to_string(),
            quality: None,
            resize: None,
        }
    }
    
    /// Create a frame sequence export for the entire video with full customization
    pub fn frame_sequence_entire_video_custom(
        fps: f64, 
        format: ImageFormat,
        filename_pattern: String,
        quality: Option<u8>,
        resize: Option<(u32, u32)>,
    ) -> Self {
        Self::FrameSequence {
            format,
            start: None, // Extract from beginning
            end: None,   // Extract to end
            fps,
            filename_pattern,
            quality,
            resize,
        }
    }
} 