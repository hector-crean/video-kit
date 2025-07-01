pub mod driver;
pub mod sources;

use std::ops::Range;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, EnumIter, IntoStaticStr, VariantNames};
use thiserror::Error;
use color_eyre::eyre::Result;
use sources::{Source, Sink};
use driver::{Driver, StreamOps};


#[derive(Error, Debug)]
pub enum CutError {
    #[error("Backend error: {0}")]
    Backend(#[from] driver::DriverError),
    #[error("Invalid command")]
    InvalidCommand,
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("GStreamer initialization failed")]
    GStreamerInit(#[from] gstreamer::glib::Error),
}



#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct Sequentise {
    pub period: Range<f64>,
    pub fps: Option<f64>,
    pub format: Option<String>,
    pub quality: Option<u8>,
}


#[derive(
    Debug, Clone,
    Serialize, Deserialize, JsonSchema,
    Display, EnumString, EnumIter, VariantNames, IntoStaticStr,
    PartialEq
)]
#[serde(tag = "type", content = "params", rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum CutVideoOperation {
    /// Splice video into segments based on a timeline
    Splice {
        segments: Vec<Range<f64>>,
    },
    /// Extract a clip into a sequence of images
    Sequentise(Sequentise),
    /// Reverse a video clip
    Reverse,
    /// Create a seamless loop by reversing and concatenating
    CreateLoop,
  
}



/// High-level video processing manager that provides both
/// imperative command-based API and fluent streaming API
pub struct Runner<D: driver::Driver> {
    driver: D,
    source: D::Source,
    sink: D::Sink,
}

impl<D: driver::Driver> Runner<D> {
    pub fn new(driver: D, source: D::Source, sink: D::Sink) -> Self {
        Self {
            driver,
            source,
            sink,
        }
    }

    pub fn with_source(mut self, source: D::Source) -> Self {
        self.source = source;
        self
    }

    pub fn with_sink(mut self, sink: D::Sink) -> Self {
        self.sink = sink;
        self
    }

    /// High-level command-based API (for backwards compatibility and simple use cases)
    pub fn execute(&self, command: CutVideoOperation) -> Result<(), CutError> {
        let stream = self.driver.load(&self.source)?;
        
        match command {
            CutVideoOperation::Splice { segments } => {
                stream
                    .splice(&self.driver, &segments)?
                    .save(&self.driver, &self.sink)?;
            }
            CutVideoOperation::Sequentise(sequentise) => {
                stream
                    .extract_frames(&self.driver, &sequentise)?
                    .save(&self.driver, &self.sink)?;
            }
            CutVideoOperation::Reverse => {
                stream
                    .reverse(&self.driver)?
                    .save(&self.driver, &self.sink)?;
            }
            CutVideoOperation::CreateLoop => {
                stream
                    .create_loop(&self.driver)?
                    .save(&self.driver, &self.sink)?;
            }
        }

        Ok(())
    }

  

    /// Low-level streaming API for complex pipeline building
    pub fn stream(&self) -> Result<D::Stream, CutError> {
        Ok(self.driver.load(&self.source)?)
    }

    /// Execute a custom stream processing function
    pub fn execute_stream<F>(&self, func: F) -> Result<(), CutError>
    where
        F: FnOnce(D::Stream, &D) -> Result<D::Stream, driver::DriverError>,
    {
        let stream = self.driver.load(&self.source)?;
        let processed_stream = func(stream, &self.driver)?;
        processed_stream.save(&self.driver, &self.sink)?;
        Ok(())
    }

    /// Get reference to the driver for advanced usage
    pub fn driver(&self) -> &D {
        &self.driver
    }

    /// Get reference to the current source
    pub fn source(&self) -> &D::Source {
        &self.source
    }

    /// Get reference to the current sink
    pub fn sink(&self) -> &D::Sink {
        &self.sink
    }
}



#[cfg(feature = "gstreamer")]
impl Runner<driver::gstreamer::GStreamerDriver> {
    pub fn gstreamer_default(input: &str, output: &str) -> Result<Self, CutError> {
        use driver::gstreamer::{GStreamerDriver, GStreamerFileSource, GStreamerFileSink};
        
        let driver = GStreamerDriver::new_with_plugin_check()?;
        let source = GStreamerFileSource::new(input);
        let sink = GStreamerFileSink::new(output);
        Ok(Self::new(driver, source, sink))
    }

    pub fn gstreamer_available() -> bool {
        use driver::gstreamer::GStreamerDriver;
        GStreamerDriver::is_available()
    }
}

#[cfg(feature = "ffmpeg")]
impl Runner<driver::ffmpeg::FFmpegDriver> {
    pub fn ffmpeg_default(input: &str, output: &str) -> Result<Self, CutError> {
        use driver::ffmpeg::{FFmpegDriver, FFmpegFileSource, FFmpegFileSink};
        
        let driver = FFmpegDriver::new()?;
        let source = FFmpegFileSource::new(input);
        let sink = FFmpegFileSink::new(output);
        Ok(Self::new(driver, source, sink))
    }
    
    pub fn ffmpeg_with_path(input: &str, output: &str, ffmpeg_path: &str) -> Result<Self, CutError> {
        use driver::ffmpeg::{FFmpegDriver, FFmpegFileSource, FFmpegFileSink};
        
        let driver = FFmpegDriver::with_path(ffmpeg_path)?;
        let source = FFmpegFileSource::new(input);
        let sink = FFmpegFileSink::new(output);
        Ok(Self::new(driver, source, sink))
    }
}
