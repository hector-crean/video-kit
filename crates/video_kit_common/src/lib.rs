//! # Video Kit Common - Shared Types and Utilities
//!
//! A foundational library providing shared data structures, utilities, and cross-crate 
//! compatibility helpers for the Video Kit ecosystem.
//!
//! ## Example
//!
//! ```rust
//! use video_kit_common::{TimestampRange, VideoMetadata};
//!
//! // Work with time ranges
//! let range = TimestampRange::new(10.5, 30.2).unwrap();
//! println!("Duration: {:.1}s", range.duration());
//!
//! // Video metadata
//! let metadata = VideoMetadata {
//!     duration: 120.5,
//!     width: 1920,
//!     height: 1080,
//!     framerate: 30.0,
//!     codec: "h264".to_string(),
//!     bitrate: Some(5000000),
//!     pixel_format: Some("yuv420p".to_string()),
//!     color_space: Some("bt709".to_string()),
//! };
//! ```

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use thiserror::Error;

// Re-exports for convenience
pub use chrono::{DateTime, Utc};

/// Result type for video kit operations
pub type Result<T> = std::result::Result<T, VideoKitError>;

/// Standard error type for video kit operations
#[derive(Error, Debug)]
pub enum VideoKitError {
    #[error("Invalid time range: start {start} >= end {end}")]
    InvalidTimeRange { start: f64, end: f64 },
    
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    
    #[error("Unsupported format: {format}")]
    UnsupportedFormat { format: String },
    
    #[error("Configuration validation failed: {details}")]
    ValidationFailed { details: String },
    
    #[error("Invalid value: {message}")]
    InvalidValue { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
}

/// A time range with start and end timestamps in seconds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TimestampRange {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

impl TimestampRange {
    /// Create a new timestamp range
    pub fn new(start: f64, end: f64) -> Result<Self> {
        if start >= end {
            return Err(VideoKitError::InvalidTimeRange { start, end });
        }
        Ok(Self { start, end })
    }
    
    /// Create a range from start time and duration
    pub fn from_start_duration(start: f64, duration: f64) -> Result<Self> {
        if duration <= 0.0 {
            return Err(VideoKitError::InvalidValue {
                message: "Duration must be positive".to_string(),
            });
        }
        Self::new(start, start + duration)
    }
    
    /// Get the duration of this range
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
    
    /// Check if a timestamp is within this range
    pub fn contains(&self, timestamp: f64) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
    
    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &TimestampRange) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// 2D point with floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    /// Translate this point by the given offsets
    pub fn translate(self, dx: f64, dy: f64) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
    
    /// Calculate distance to another point
    pub fn distance_to(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Size with width and height
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    /// Create a new size
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
    
    /// Calculate the area
    pub fn area(&self) -> f64 {
        self.width * self.height
    }
    
    /// Get aspect ratio (width / height)
    pub fn aspect_ratio(&self) -> f64 {
        self.width / self.height
    }
}

/// Rectangle defined by position and size
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Rectangle {
    pub position: Point2D,
    pub size: Size,
}

impl Rectangle {
    /// Create a new rectangle
    pub fn new(position: Point2D, size: Size) -> Self {
        Self { position, size }
    }
    
    /// Check if a point is inside this rectangle
    pub fn contains(&self, point: Point2D) -> bool {
        point.x >= self.position.x &&
        point.x <= self.position.x + self.size.width &&
        point.y >= self.position.y &&
        point.y <= self.position.y + self.size.height
    }
    
    /// Get the area of this rectangle
    pub fn area(&self) -> f64 {
        self.size.area()
    }
}

/// Video metadata information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VideoMetadata {
    /// Duration in seconds
    pub duration: f64,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frame rate (frames per second)
    pub framerate: f64,
    /// Video codec (e.g., "h264", "hevc")
    pub codec: String,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Pixel format (e.g., "yuv420p")
    pub pixel_format: Option<String>,
    /// Color space (e.g., "bt709")
    pub color_space: Option<String>,
}

impl VideoMetadata {
    /// Get aspect ratio as a tuple (width_ratio, height_ratio)
    pub fn aspect_ratio(&self) -> (u32, u32) {
        let gcd = gcd(self.width, self.height);
        (self.width / gcd, self.height / gcd)
    }
    
    /// Check if this is a standard resolution
    pub fn is_standard_resolution(&self) -> bool {
        matches!(
            (self.width, self.height),
            (1920, 1080) | (3840, 2160) | (1280, 720) | (854, 480) | (640, 360)
        )
    }
}

/// Audio metadata information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AudioMetadata {
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Audio codec (e.g., "aac", "mp3")
    pub codec: String,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Channel layout (e.g., "stereo", "5.1")
    pub channel_layout: Option<String>,
}

/// Complete stream information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StreamInfo {
    /// Video stream metadata
    pub video: Option<VideoMetadata>,
    /// Audio stream metadata
    pub audio: Option<AudioMetadata>,
    /// Container format (e.g., "mp4", "avi")
    pub format: String,
    /// File size in bytes
    pub file_size: Option<u64>,
}

/// Utility functions for time formatting
pub mod utils {
    use super::*;
    
    /// Format seconds as HH:MM:SS.mmm
    pub fn format_timestamp(seconds: f64) -> String {
        let total_seconds = seconds as u64;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let secs = total_seconds % 60;
        let millis = ((seconds - total_seconds as f64) * 1000.0) as u32;
        
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
    }
    
    /// Parse timestamp from HH:MM:SS or HH:MM:SS.mmm format
    pub fn parse_timestamp(timestamp: &str) -> Result<f64> {
        let parts: Vec<&str> = timestamp.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(VideoKitError::Parse(
                "Invalid timestamp format. Expected HH:MM:SS or MM:SS".to_string()
            ));
        }
        
        let mut total_seconds = 0.0;
        
        if parts.len() == 3 {
            // HH:MM:SS format
            let hours: f64 = parts[0].parse()
                .map_err(|_| VideoKitError::Parse("Invalid hours".to_string()))?;
            let minutes: f64 = parts[1].parse()
                .map_err(|_| VideoKitError::Parse("Invalid minutes".to_string()))?;
            
            total_seconds += hours * 3600.0 + minutes * 60.0;
            
            // Handle seconds with optional milliseconds
            let seconds_str = parts[2];
            if seconds_str.contains('.') {
                let seconds: f64 = seconds_str.parse()
                    .map_err(|_| VideoKitError::Parse("Invalid seconds".to_string()))?;
                total_seconds += seconds;
            } else {
                let seconds: f64 = seconds_str.parse()
                    .map_err(|_| VideoKitError::Parse("Invalid seconds".to_string()))?;
                total_seconds += seconds;
            }
        } else {
            // MM:SS format
            let minutes: f64 = parts[0].parse()
                .map_err(|_| VideoKitError::Parse("Invalid minutes".to_string()))?;
            let seconds: f64 = parts[1].parse()
                .map_err(|_| VideoKitError::Parse("Invalid seconds".to_string()))?;
            total_seconds = minutes * 60.0 + seconds;
        }
        
        Ok(total_seconds)
    }
    
    /// Format file size in human-readable format
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        
        if bytes == 0 {
            return "0 B".to_string();
        }
        
        let base = 1024_f64;
        let exp = (bytes as f64).log(base).floor() as usize;
        let exp = exp.min(UNITS.len() - 1);
        
        let size = bytes as f64 / base.powi(exp as i32);
        format!("{:.1} {}", size, UNITS[exp])
    }
    
    /// Check if a file extension indicates a video file
    pub fn is_video_file(filename: &str) -> bool {
        if let Some(ext) = std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            matches!(
                ext.to_lowercase().as_str(),
                "mp4" | "avi" | "mov" | "mkv" | "webm" | "flv" | "wmv" | "m4v"
            )
        } else {
            false
        }
    }
    
    /// Get file extension from filename
    pub fn get_file_extension(filename: &str) -> Option<String> {
        std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
    }
    
    /// Ensure output directory exists
    pub fn ensure_output_dir(path: &str) -> Result<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }
}

/// Calculate greatest common divisor
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timestamp_range() {
        let range = TimestampRange::new(10.0, 30.0).unwrap();
        assert_eq!(range.duration(), 20.0);
        assert!(range.contains(15.0));
        assert!(!range.contains(5.0));
        assert!(!range.contains(35.0));
    }
    
    #[test]
    fn test_timestamp_range_overlap() {
        let range1 = TimestampRange::new(10.0, 30.0).unwrap();
        let range2 = TimestampRange::new(25.0, 45.0).unwrap();
        let range3 = TimestampRange::new(35.0, 50.0).unwrap();
        
        assert!(range1.overlaps(&range2));
        assert!(!range1.overlaps(&range3));
    }
    
    #[test]
    fn test_invalid_timestamp_range() {
        assert!(TimestampRange::new(30.0, 10.0).is_err());
        assert!(TimestampRange::new(10.0, 10.0).is_err());
    }
    
    #[test]
    fn test_point_operations() {
        let point = Point2D::new(10.0, 20.0);
        let translated = point.translate(5.0, -3.0);
        
        assert_eq!(translated.x, 15.0);
        assert_eq!(translated.y, 17.0);
        
        let distance = point.distance_to(Point2D::new(13.0, 24.0));
        assert!((distance - 5.0).abs() < 0.001);
    }
    
    #[test]
    fn test_rectangle_contains() {
        let rect = Rectangle::new(
            Point2D::new(10.0, 20.0),
            Size::new(100.0, 50.0)
        );
        
        assert!(rect.contains(Point2D::new(50.0, 30.0)));
        assert!(!rect.contains(Point2D::new(5.0, 30.0)));
        assert_eq!(rect.area(), 5000.0);
    }
    
    #[test]
    fn test_video_metadata_aspect_ratio() {
        let metadata = VideoMetadata {
            duration: 120.0,
            width: 1920,
            height: 1080,
            framerate: 30.0,
            codec: "h264".to_string(),
            bitrate: None,
            pixel_format: None,
            color_space: None,
        };
        
        assert_eq!(metadata.aspect_ratio(), (16, 9));
        assert!(metadata.is_standard_resolution());
    }
    
    #[test]
    fn test_format_timestamp() {
        assert_eq!(utils::format_timestamp(125.5), "00:02:05.500");
        assert_eq!(utils::format_timestamp(3665.0), "01:01:05.000");
    }
    
    #[test]
    fn test_parse_timestamp() {
        assert_eq!(utils::parse_timestamp("00:02:05.500").unwrap(), 125.5);
        assert_eq!(utils::parse_timestamp("01:01:05").unwrap(), 3665.0);
        assert_eq!(utils::parse_timestamp("02:05").unwrap(), 125.0);
    }
    
    #[test]
    fn test_file_utilities() {
        assert!(utils::is_video_file("video.mp4"));
        assert!(!utils::is_video_file("audio.mp3"));
        assert_eq!(utils::get_file_extension("video.mp4"), Some("mp4".to_string()));
    }
    
    #[test]
    fn test_format_file_size() {
        assert_eq!(utils::format_file_size(1536), "1.5 KB");
        assert_eq!(utils::format_file_size(2048576), "2.0 MB");
    }
}
