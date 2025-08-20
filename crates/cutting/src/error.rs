use gstreamer::{self as gst, glib};
use gstreamer_editing_services::{self as ges};
use thiserror::Error;

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