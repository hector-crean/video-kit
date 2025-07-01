
use cutting::CutVideoOperation;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;




#[derive(Error, Debug)]
pub enum VideoKitError {
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),
    #[error(transparent)]
    TomlDeError(#[from] toml::de::Error),
    #[error(transparent)]
    TomlSerError(#[from] toml::ser::Error),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error("Missing 'path' or 'video.input_path' field")]
    MissingPath,
    #[error("Missing 'output_dir' or 'video.output_dir' field")]
    MissingOutputDir,
    #[error("Unsupported file format. Please use .toml or .json files")]
    UnsupportedFileFormat,
}


pub enum VideoKitOperation {
    Cut(CutVideoOperation),
}






/// Enhanced clip definition with advanced processing capabilities
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Clip {
    pub name: String,
    pub description: Option<String>,
    pub operation: Option<CutVideoOperation>,
}

/// Enhanced timeline configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct InputVideo {
    pub path: String,
    pub output_dir: String,
    pub clips: Vec<Clip>,
}



#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct VideoSection {
    pub input_path: String,
    pub output_dir: String,
}

impl InputVideo {
    /// Load InputVideo configuration from a TOML file
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self, VideoKitError> {
        let content = fs::read_to_string(path)?;
        Self::from_toml(&content)
    }

    /// Load InputVideo configuration from TOML string
    pub fn from_toml(content: &str) -> Result<Self, VideoKitError> {
        let input: InputVideo = toml::from_str(content)?;
        
        

        Ok(input)
    }

    /// Load InputVideo configuration from a JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, VideoKitError> {
        let content = fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Load InputVideo configuration from JSON string
    pub fn from_json(content: &str) -> Result<Self, VideoKitError> {
        let input_video: InputVideo = serde_json::from_str(content)?;
        Ok(input_video)
    }

    /// Auto-detect file format and load configuration
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, VideoKitError> {
        let path_ref = path.as_ref();
        match path_ref.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => Self::from_toml_file(path),
            Some("json") => Self::from_json_file(path),
            _ => Err(VideoKitError::UnsupportedFileFormat),
        }
    }

    /// Save InputVideo configuration to a TOML file
    pub fn to_toml_file<P: AsRef<Path>>(&self, path: P) -> Result<(), VideoKitError> {
        let content = self.to_toml()?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Convert InputVideo to TOML string
    pub fn to_toml(&self) -> Result<String, VideoKitError> {
        let toml = toml::to_string_pretty(&self)?;
        Ok(toml)
    }

    /// Save InputVideo configuration to a JSON file
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<(), VideoKitError> {
        let content = self.to_json()?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Convert InputVideo to JSON string
    pub fn to_json(&self) -> Result<String, VideoKitError> {
       
        Ok(serde_json::to_string_pretty(&self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

  
 
}



