#[cfg(feature = "gstreamer")]
pub mod gstreamer;

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

use std::ops::Range;

use crate::{Sequentise, sources::{Source, Sink}};

#[derive(thiserror::Error, Debug)]
pub enum DriverError {
    #[error("Failed to initialize driver: {0}")]
    Initialization(String),
    #[error("Graph execution failed: {0}")]
    Execution(String),
}

/// A trait for a backend that can build and execute video processing pipelines.
/// Each backend defines its own Source, Sink, and Stream types.
/// 
/// The Stream type represents an internal video/audio stream with operations
/// applied but not yet materialized to a sink. This allows for operation chaining.
pub trait Driver {
    /// The type of source this driver works with
    type Source: Source;
    /// The type of sink this driver works with  
    type Sink: Sink;
    /// The internal representation of a video/audio stream with operations applied
    type Stream: Clone;

    /// Load a source into the driver's internal stream representation
    fn load(&self, source: &Self::Source) -> Result<Self::Stream, DriverError>;
    
    /// Apply a splice operation to the stream, returning a new stream
    fn splice(&self, stream: Self::Stream, segments: &[Range<f64>]) -> Result<Self::Stream, DriverError>;
    
    /// Apply frame extraction to the stream (changes the stream to image sequence)
    fn extract_frames(&self, stream: Self::Stream, sequentise: &Sequentise) -> Result<Self::Stream, DriverError>;
    
    /// Apply reverse operation to the stream
    fn reverse(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError>;
    
    /// Apply loop creation to the stream
    fn create_loop(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError>;
    
    /// Materialize the stream to a sink (execute the pipeline)
    fn save(&self, stream: Self::Stream, sink: &Self::Sink) -> Result<(), DriverError>;
}

/// Extension trait providing a fluent API for operation chaining
pub trait StreamOps<D: Driver>: Sized {
    /// Apply a splice operation
    fn splice(self, driver: &D, segments: &[Range<f64>]) -> Result<Self, DriverError>;
    
    /// Extract frames in a time range  
    fn extract_frames(self, driver: &D, sequentise: &Sequentise) -> Result<Self, DriverError>;
    
    /// Reverse the stream
    fn reverse(self, driver: &D) -> Result<Self, DriverError>;
    
    /// Create a loop
    fn create_loop(self, driver: &D) -> Result<Self, DriverError>;
    
    /// Save to a sink
    fn save(self, driver: &D, sink: &D::Sink) -> Result<(), DriverError>;
}

/// Blanket implementation for all driver stream types
impl<D: Driver> StreamOps<D> for D::Stream {
    fn splice(self, driver: &D, segments: &[Range<f64>]) -> Result<Self, DriverError> {
        driver.splice(self, segments)
    }
    
    fn extract_frames(self, driver: &D, sequentise: &Sequentise) -> Result<Self, DriverError> {
        driver.extract_frames(self, sequentise)
    }
    
    fn reverse(self, driver: &D) -> Result<Self, DriverError> {
        driver.reverse(self)
    }
    
    fn create_loop(self, driver: &D) -> Result<Self, DriverError> {
        driver.create_loop(self)
    }
    
    fn save(self, driver: &D, sink: &D::Sink) -> Result<(), DriverError> {
        driver.save(self, sink)
    }
}





#[derive(Debug, Clone)]
pub enum StreamOperation {
    Load { source_path: String },
    Splice { segments: Vec<Range<f64>> },
    ExtractFrames { start_time: f64, end_time: f64, config: Option<Sequentise> },
    Reverse,
    CreateLoop,
}

