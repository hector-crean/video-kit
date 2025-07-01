use std::fmt::Debug;
use crate::driver::DriverError;

/// Generic trait for data sources (input streams)
/// Different backends can implement this for their specific source types
pub trait Source: Debug + Clone {
    /// Type-specific validation that the source is accessible
    fn validate(&self) -> Result<(), DriverError>;
    
    /// Get a human-readable description of this source
    fn description(&self) -> String;
}

/// Generic trait for data sinks (output destinations)
/// Different backends can implement this for their specific sink types
pub trait Sink: Debug + Clone {
    /// Type-specific validation that the sink can be written to
    fn validate(&self) -> Result<(), DriverError>;
    
    /// Get a human-readable description of this sink
    fn description(&self) -> String;
}

/// A simple file-based source implementation
#[derive(Debug, Clone)]
pub struct FileSource {
    pub path: String,
}

impl FileSource {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Source for FileSource {
    fn validate(&self) -> Result<(), DriverError> {
        if std::path::Path::new(&self.path).exists() {
            Ok(())
        } else {
            Err(DriverError::Execution(format!("Input file not found: {}", self.path)))
        }
    }
    
    fn description(&self) -> String {
        format!("File: {}", self.path)
    }
}

/// A simple file-based sink implementation
#[derive(Debug, Clone)]
pub struct FileSink {
    pub path: String,
}

impl FileSink {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Sink for FileSink {
    fn validate(&self) -> Result<(), DriverError> {
        // Check if parent directory exists, create if needed
        if let Some(parent) = std::path::Path::new(&self.path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        }
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("File: {}", self.path)
    }
}

/// A directory-based sink for multiple files (e.g., frame extraction)
#[derive(Debug, Clone)]
pub struct DirectorySink {
    pub path: String,
    pub file_pattern: String,
}

impl DirectorySink {
    pub fn new(path: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            file_pattern: pattern.into(),
        }
    }
}

impl Sink for DirectorySink {
    fn validate(&self) -> Result<(), DriverError> {
        std::fs::create_dir_all(&self.path)
            .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("Directory: {} (pattern: {})", self.path, self.file_pattern)
    }
} 